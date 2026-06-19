from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
import re
import urllib.parse
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import effective_openai_api_key, settings
from app.core.utils import content_hash, ensure_dir, read_json, utc_now_iso, write_json
from app.crawlers.fetcher import PublicFetcher, infer_kind
from app.crawlers.models import FetchError, FetchRunResult, RawDocument
from app.geo.geocoder import Geocoder
from app.llm.client import LLMExtractor, MissingLLMKey
from app.llm.document_filter import DocumentRelevanceFilter
from app.llm.openai_client import (
    OpenAINonRecoverableError,
    is_openai_non_recoverable_error,
    summarize_openai_error,
)
from app.llm.schemas import GovernmentEvent
from app.parsers.dispatcher import parse_raw_document
from app.parsers.models import ParsedDocument
from app.renderers.report import RenderedOutputs, render_outputs, update_latest
from app.renderers.residential_report import render_residential_outputs
from app.scoring.candidates import generate_candidates
from app.scoring.residential import ResidentialCandidateScore, score_residential_candidates
from app.scoring.scorer import CandidateScore, score_candidates
from app.sources.discovery import DiscoveryRun, SourceScoutAgent
from app.sources.registry import SourceRegistry, load_shanghai_registry


class LandScoutAgentState(BaseModel):
    run_id: str
    raw_documents: list[RawDocument] = Field(default_factory=list)
    parsed_documents: list[ParsedDocument] = Field(default_factory=list)
    events: list[GovernmentEvent] = Field(default_factory=list)
    scores: list[CandidateScore] = Field(default_factory=list)
    residential_scores: list[ResidentialCandidateScore] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    visited_sources: list[str] = Field(default_factory=list)
    discovered_sources: list[dict[str, Any]] = Field(default_factory=list)
    api_enrichment_enabled: bool = False
    outputs: RenderedOutputs | None = None


class LandScoutAgent:
    def __init__(self, registry: SourceRegistry | None = None) -> None:
        self.registry = registry or load_shanghai_registry()

    def run_landscout_demo(self, *, live: bool, days: int, top_k: int) -> LandScoutAgentState:
        if live and not effective_openai_api_key():
            raise MissingLLMKey(
                "OPENAI_API_KEY is required for live LLM extraction. "
                "Set it in .env or fill OpenAI API Key in the web page before running the LandScout Agent live demo."
            )
        run_id = make_run_id()
        run_dir = run_data_dir(run_id)
        output_dir = run_output_dir(run_id)
        ensure_dir(run_dir)
        ensure_dir(output_dir)
        if live:
            fetch_result = self._fetch_live(run_id, run_dir, days=days)
            state = LandScoutAgentState(
                run_id=run_id,
                raw_documents=fetch_result.documents,
                errors=[error.model_dump(mode="json") for error in fetch_result.errors],
                visited_sources=fetch_result.visited_sources,
                api_enrichment_enabled=True,
            )
            save_state(state)
            try:
                state = self._parse_extract_score_render(
                    state=state,
                    live=True,
                    days=days,
                    top_k=top_k,
                    output_dir=output_dir,
                )
            except (MissingLLMKey, OpenAINonRecoverableError) as exc:
                state.errors.append({"source_id": "llm", "url": "", "reason": str(exc), "status_code": None})
                save_state(state)
                (output_dir / "pipeline.log").write_text(str(exc), encoding="utf-8")
                raise
            return state

        state = self.load_fixture_documents(run_id, run_dir, fixture_dir=Path("tests") / "fixtures")
        return self._parse_extract_score_render(state=state, live=False, days=days, top_k=top_k, output_dir=output_dir)

    def run_shanghai_demo(self, *, live: bool, days: int, top_k: int) -> LandScoutAgentState:
        return self.run_landscout_demo(live=live, days=days, top_k=top_k)

    def recommend_residential(
        self,
        *,
        live: bool,
        days: int,
        top_k: int,
        source_limit: int = 12,
        discover_sources: bool = False,
        dynamic_source_limit: int = 5,
        discovery_query_limit: int = 6,
        include_non_official_sources: bool = False,
        amap_key: str | None = None,
    ) -> LandScoutAgentState:
        if live and not effective_openai_api_key():
            raise MissingLLMKey(
                "OPENAI_API_KEY is required for live residential recommendation. "
                "Set it in .env, fill OpenAI API Key in the web page, or run without --live for a fixture-backed demo."
            )
        run_id = make_run_id()
        run_dir = run_data_dir(run_id)
        output_dir = run_output_dir(run_id)
        ensure_dir(run_dir)
        ensure_dir(output_dir)
        if live:
            active_registry = self.registry
            effective_source_limit = source_limit
            discovered_sources: list[dict[str, Any]] = []
            source_scout_errors: list[dict[str, Any]] = []
            if discover_sources:
                discovery_run = self._discover_live_sources(
                    max_sources=dynamic_source_limit,
                    query_limit=discovery_query_limit,
                    include_non_official=include_non_official_sources,
                )
                discovered_sources = [candidate.model_dump(mode="json") for candidate in discovery_run.candidates]
                source_scout_errors = [
                    {"source_id": "source_scout", "url": "", "reason": error, "status_code": None}
                    for error in discovery_run.errors
                ]
                dynamic_sources = [
                    candidate.to_source_config(priority=90 + idx)
                    for idx, candidate in enumerate(discovery_run.candidates)
                ]
                active_registry = SourceRegistry(self.registry.select(source_limit)).merged(dynamic_sources)
                effective_source_limit = source_limit + len(dynamic_sources)
            fetch_result = self._fetch_live(
                run_id,
                run_dir,
                source_limit=effective_source_limit,
                registry=active_registry,
                days=days,
            )
            state = LandScoutAgentState(
                run_id=run_id,
                raw_documents=fetch_result.documents,
                errors=[*source_scout_errors, *[error.model_dump(mode="json") for error in fetch_result.errors]],
                visited_sources=fetch_result.visited_sources,
                discovered_sources=discovered_sources,
                api_enrichment_enabled=True,
            )
            save_state(state)
        else:
            state = self.load_fixture_documents(run_id, run_dir, fixture_dir=Path("tests") / "fixtures")
        try:
            return self._parse_extract_score_render(
                state=state,
                live=live,
                days=days,
                top_k=top_k,
                output_dir=output_dir,
                residential=True,
                amap_key=amap_key,
            )
        except (MissingLLMKey, OpenAINonRecoverableError) as exc:
            state.errors.append({"source_id": "llm", "url": "", "reason": str(exc), "status_code": None})
            save_state(state)
            (output_dir / "pipeline.log").write_text(str(exc), encoding="utf-8")
            raise

    def fetch_source(self, *, source_id: str, pages: int) -> LandScoutAgentState:
        run_id = make_run_id()
        run_dir = run_data_dir(run_id)
        fetcher = PublicFetcher(self.registry, run_id=run_id, run_dir=run_dir)
        try:
            result = fetcher.fetch_source(source_id, pages=pages)
        finally:
            fetcher.close()
        state = LandScoutAgentState(
            run_id=run_id,
            raw_documents=result.documents,
            errors=[error.model_dump(mode="json") for error in result.errors],
            visited_sources=result.visited_sources,
        )
        parsed = []
        for raw in state.raw_documents:
            try:
                parsed.append(parse_raw_document(raw))
            except Exception as exc:
                state.errors.append({"source_id": raw.source_id, "url": raw.url, "reason": f"parse failed: {exc}"})
        state.parsed_documents = parsed
        save_state(state)
        return state

    def discover_source(self, *, source_id: str) -> FetchRunResult:
        run_id = make_run_id()
        fetcher = PublicFetcher(self.registry, run_id=run_id, run_dir=run_data_dir(run_id))
        try:
            return fetcher.discover_source(source_id)
        finally:
            fetcher.close()

    def discover_web_sources(
        self,
        *,
        max_sources: int = 5,
        query_limit: int = 6,
        include_non_official: bool = False,
    ) -> DiscoveryRun:
        return self._discover_live_sources(
            max_sources=max_sources,
            query_limit=query_limit,
            include_non_official=include_non_official,
        )

    def score_run(self, *, run_id: str, days: int = 540) -> LandScoutAgentState:
        state = load_state(run_id)
        if not state.events:
            raise RuntimeError(f"Run {run_id} has no extracted events. Run the LandScout Agent demo first.")
        state.events = [event for event in state.events if not event.needs_review]
        if not state.events:
            raise RuntimeError(f"Run {run_id} has no valid extracted events after review filtering.")
        state.events = Geocoder().geocode_many(state.events)
        candidates = generate_candidates(state.events)
        state.scores = score_candidates(candidates, state.events, days=days)
        if state.residential_scores:
            state.residential_scores = score_residential_candidates(candidates, state.events, days=days)
        save_state(state)
        return state

    def render_run(self, *, run_id: str, top_k: int = 8) -> LandScoutAgentState:
        state = load_state(run_id)
        if not state.scores:
            state = self.score_run(run_id=run_id)
        output_dir = run_output_dir(run_id)
        if state.residential_scores:
            state.outputs = render_residential_outputs(
                run_id=state.run_id,
                output_dir=output_dir,
                events=state.events,
                residential_scores=state.residential_scores,
                base_scores=state.scores,
                raw_documents=state.raw_documents,
                parsed_documents=state.parsed_documents,
                errors=state.errors,
                top_k=top_k,
                visited_sources=state.visited_sources,
                discovered_sources=state.discovered_sources,
                api_enrichment=state.api_enrichment_enabled,
            )
        else:
            state.outputs = render_outputs(
                run_id=state.run_id,
                output_dir=output_dir,
                events=state.events,
                scores=state.scores,
                raw_documents=state.raw_documents,
                parsed_documents=state.parsed_documents,
                errors=state.errors,
                top_k=top_k,
                visited_sources=state.visited_sources,
                discovered_sources=state.discovered_sources,
            )
            update_latest(output_dir)
        save_state(state)
        return state

    def load_fixture_documents(self, run_id: str, run_dir: Path, fixture_dir: Path) -> LandScoutAgentState:
        fixture_dir = Path(fixture_dir)
        raw_dir = ensure_dir(run_dir / "raw" / "fixtures")
        raw_documents: list[RawDocument] = []
        for path in fixture_dir.iterdir():
            if path.is_dir() or path.name.startswith("."):
                continue
            data = path.read_bytes()
            url = f"fixture://{path.name}"
            digest = content_hash(data)
            target = raw_dir / path.name
            shutil.copy2(path, target)
            raw_documents.append(
                RawDocument(
                    id=digest[:16],
                    source_id="fixture",
                    url=url,
                    fetched_at=utc_now_iso(),
                    content_hash=digest,
                    path=str(target),
                    kind=infer_kind(path.name, ""),
                    status_code=200,
                    content_type="",
                )
            )
        state = LandScoutAgentState(run_id=run_id, raw_documents=raw_documents, visited_sources=["fixture"])
        save_state(state)
        return state

    def _fetch_live(
        self,
        run_id: str,
        run_dir: Path,
        source_limit: int = 12,
        registry: SourceRegistry | None = None,
        days: int | None = None,
    ) -> FetchRunResult:
        fetcher = PublicFetcher(registry or self.registry, run_id=run_id, run_dir=run_dir)
        try:
            return fetcher.fetch_many(source_limit=source_limit, days=days)
        finally:
            fetcher.close()

    def _discover_live_sources(
        self,
        *,
        max_sources: int,
        query_limit: int,
        include_non_official: bool,
    ) -> DiscoveryRun:
        existing_urls = {str(url) for source in self.registry.sources for url in source.base_urls}
        existing_hosts = {
            urllib.parse.urlparse(str(url)).netloc.split(":", 1)[0].lower()
            for source in self.registry.sources
            for url in source.base_urls
        }
        return SourceScoutAgent().discover(
            max_sources=max_sources,
            query_limit=query_limit,
            include_non_official=include_non_official,
            existing_urls=existing_urls,
            existing_hosts=existing_hosts,
            use_llm_queries=True,
        )

    def _parse_extract_score_render(
        self,
        *,
        state: LandScoutAgentState,
        live: bool,
        days: int,
        top_k: int,
        output_dir: Path,
        residential: bool = False,
        amap_key: str | None = None,
    ) -> LandScoutAgentState:
        parsed_documents: list[ParsedDocument] = []
        for raw in state.raw_documents:
            try:
                parsed_documents.append(parse_raw_document(raw))
            except Exception as exc:
                state.errors.append({"source_id": raw.source_id, "url": raw.url, "reason": f"parse failed: {exc}"})
        state.parsed_documents = parsed_documents
        save_state(state)

        triage = DocumentRelevanceFilter(live=live)
        documents_for_extraction: list[ParsedDocument] = []
        for document in parsed_documents:
            try:
                relevance = triage.classify(document)
            except OpenAINonRecoverableError as exc:
                state.errors.append({"source_id": "llm", "url": document.url, "reason": str(exc), "stage": "document_triage"})
                save_state(state)
                raise
            except Exception as exc:
                if is_openai_non_recoverable_error(exc):
                    error = OpenAINonRecoverableError(summarize_openai_error(exc))
                    state.errors.append({"source_id": "llm", "url": document.url, "reason": str(error), "stage": "document_triage"})
                    save_state(state)
                    raise error from exc
                relevance = None
                document.metadata["relevance"] = {
                    "should_extract": True,
                    "relevance_score": 1.0,
                    "categories": [],
                    "reason": f"triage failed open: {exc}",
                }
                documents_for_extraction.append(document)
                continue
            document.metadata["relevance"] = relevance.model_dump(mode="json")
            if relevance.should_extract:
                documents_for_extraction.append(document)
            else:
                state.errors.append(
                    {
                        "source_id": document.source_id,
                        "url": document.url,
                        "reason": f"document triage skipped: {relevance.reason}",
                        "stage": "document_triage",
                    }
                )
        state.parsed_documents = parsed_documents
        save_state(state)

        extractor = LLMExtractor(live=live, allow_heuristic=not live)
        events: list[GovernmentEvent] = []
        for document in documents_for_extraction:
            try:
                extraction = extractor.extract(document)
            except MissingLLMKey:
                raise
            except OpenAINonRecoverableError as exc:
                state.errors.append({"source_id": "llm", "url": document.url, "reason": str(exc), "stage": "extraction"})
                save_state(state)
                raise
            except Exception as exc:
                if is_openai_non_recoverable_error(exc):
                    error = OpenAINonRecoverableError(summarize_openai_error(exc))
                    state.errors.append({"source_id": "llm", "url": document.url, "reason": str(error), "stage": "extraction"})
                    save_state(state)
                    raise error from exc
                state.errors.append(
                    {
                        "source_id": document.source_id,
                        "url": document.url,
                        "reason": f"extract failed: {exc}",
                    }
                )
                save_state(state)
                continue
            events.extend([event for event in extraction.events if not event.needs_review])
            for note in extraction.review_notes:
                state.errors.append({"source_id": document.source_id, "url": document.url, "reason": note})
            save_state(state)
        if live and len(events) < 20:
            state.errors.append(
                {
                    "source_id": "pipeline",
                    "url": "",
                    "reason": f"Live extraction produced {len(events)} events, below the 20-event target; see source errors and evidence pack.",
                }
            )
        state.events = Geocoder(amap_key=amap_key).geocode_many(events)
        candidates = generate_candidates(state.events)
        state.scores = score_candidates(candidates, state.events, days=days)
        if residential:
            state.residential_scores = score_residential_candidates(candidates, state.events, days=days)
            state.outputs = render_residential_outputs(
                run_id=state.run_id,
                output_dir=output_dir,
                events=state.events,
                residential_scores=state.residential_scores,
                base_scores=state.scores,
                raw_documents=state.raw_documents,
                parsed_documents=state.parsed_documents,
                errors=state.errors,
                top_k=top_k,
                visited_sources=state.visited_sources,
                discovered_sources=state.discovered_sources,
                api_enrichment=state.api_enrichment_enabled,
            )
        else:
            state.outputs = render_outputs(
                run_id=state.run_id,
                output_dir=output_dir,
                events=state.events,
                scores=state.scores,
                raw_documents=state.raw_documents,
                parsed_documents=state.parsed_documents,
                errors=state.errors,
                top_k=top_k,
                visited_sources=state.visited_sources,
                discovered_sources=state.discovered_sources,
            )
            update_latest(output_dir)
        save_state(state)
        return state


def make_run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"


def run_data_dir(run_id: str) -> Path:
    validate_run_id(run_id)
    return ensure_dir(settings.data_dir / "runs" / run_id)


def run_output_dir(run_id: str) -> Path:
    validate_run_id(run_id)
    return ensure_dir(settings.outputs_dir / "shanghai" / run_id)


def state_path(run_id: str, *, create: bool = True) -> Path:
    validate_run_id(run_id)
    run_dir = ensure_dir(settings.data_dir / "runs" / run_id) if create else settings.data_dir / "runs" / run_id
    return run_dir / "pipeline_state.json"


PipelineState = LandScoutAgentState
ShanghaiSignalPipeline = LandScoutAgent


def save_state(state: LandScoutAgentState) -> None:
    write_json(state_path(state.run_id), state.model_dump(mode="json"))


def load_state(run_id: str) -> LandScoutAgentState:
    path = state_path(run_id, create=False)
    if not path.exists():
        latest_state = settings.data_dir / "runs" / run_id / "pipeline_state.json"
        if latest_state.exists():
            path = latest_state
        else:
            raise FileNotFoundError(f"No pipeline_state.json found for run_id={run_id}")
    return LandScoutAgentState.model_validate(read_json(path))


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[0-9]{8}T[0-9]{12}Z_[0-9a-f]{8}", run_id):
        raise ValueError(f"Invalid run_id format: {run_id}")
