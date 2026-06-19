from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.core.config import settings
from app.core.utils import write_json
from app.llm.openai_client import build_openai_client
from app.parsers.models import ParsedDocument


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceCrawlerHint(StrictModel):
    source_id: str
    observed_documents: int
    likely_list_pages: list[str]
    likely_detail_pages: list[str]
    likely_attachment_urls: list[str]
    recommended_access_mode: str
    recommended_keywords: list[str]
    crawler_notes: list[str]


class CrawlerHintPlan(StrictModel):
    hints: list[SourceCrawlerHint]


def write_crawler_hints(
    path: Path,
    *,
    parsed_documents: list[ParsedDocument],
    use_openai: bool,
) -> Path:
    plan = generate_crawler_hints(parsed_documents, use_openai=use_openai)
    write_json(path, plan.model_dump(mode="json"))
    return path


def generate_crawler_hints(parsed_documents: list[ParsedDocument], *, use_openai: bool) -> CrawlerHintPlan:
    if use_openai and settings.openai_api_key and parsed_documents:
        try:
            return generate_crawler_hints_openai(parsed_documents)
        except Exception:
            pass
    return generate_crawler_hints_heuristic(parsed_documents)


def generate_crawler_hints_openai(parsed_documents: list[ParsedDocument]) -> CrawlerHintPlan:
    grouped = summarize_documents_by_source(parsed_documents)
    client = build_openai_client()
    response = client.responses.create(
        model=settings.openai_fast_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are the Crawler Strategy Agent for LandScout Agent. "
                    "Inspect observed public pages, links, attachments, parsers, and relevance metadata. "
                    "Recommend how to improve future crawling for each source. Do not suggest bypassing login, captcha, rate limits, or robots.txt."
                ),
            },
            {"role": "user", "content": json.dumps(grouped, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "CrawlerHintPlan",
                "schema": CrawlerHintPlan.model_json_schema(),
                "strict": True,
            }
        },
    )
    return CrawlerHintPlan.model_validate(json.loads(response.output_text))


def generate_crawler_hints_heuristic(parsed_documents: list[ParsedDocument]) -> CrawlerHintPlan:
    grouped: dict[str, list[ParsedDocument]] = defaultdict(list)
    for document in parsed_documents:
        grouped[document.source_id].append(document)

    hints: list[SourceCrawlerHint] = []
    for source_id, documents in grouped.items():
        list_pages: list[str] = []
        detail_pages: list[str] = []
        attachment_urls: list[str] = []
        keyword_counter: Counter[str] = Counter()
        needs_playwright = False
        for document in documents:
            link_count = len(document.links)
            attachment_count = len(document.attachments)
            relevance = document.metadata.get("relevance") or {}
            relevance = relevance if isinstance(relevance, dict) else {}
            if link_count >= 10 or "index" in document.url.lower():
                list_pages.append(document.url)
            if 0 < link_count < 10 or relevance.get("should_extract"):
                detail_pages.append(document.url)
            for attachment in document.attachments[:8]:
                url = attachment.get("url") or attachment.get("href") or ""
                if url:
                    attachment_urls.append(url)
            if document.parser == "html" and not document.text.strip() and link_count == 0:
                needs_playwright = True
            for category in relevance.get("categories", []):
                keyword_counter[str(category)] += 1

        notes = [
            "Prefer list pages with stable public links, then fetch detail pages and public attachments.",
            "Keep rate limits and robots.txt checks; do not bypass login or captcha pages.",
        ]
        if attachment_urls:
            notes.append("Attachments are important for this source; prioritize PDF/Excel/Word downloads.")
        if needs_playwright:
            notes.append("Observed sparse HTML; Playwright rendering or public XHR discovery may improve recall.")

        hints.append(
            SourceCrawlerHint(
                source_id=source_id,
                observed_documents=len(documents),
                likely_list_pages=list(dict.fromkeys(list_pages))[:8],
                likely_detail_pages=list(dict.fromkeys(detail_pages))[:8],
                likely_attachment_urls=list(dict.fromkeys(attachment_urls))[:12],
                recommended_access_mode="http_then_playwright" if needs_playwright else "http",
                recommended_keywords=[item for item, _ in keyword_counter.most_common(10)],
                crawler_notes=notes,
            )
        )
    return CrawlerHintPlan(hints=hints)


def summarize_documents_by_source(parsed_documents: list[ParsedDocument]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ParsedDocument]] = defaultdict(list)
    for document in parsed_documents:
        grouped[document.source_id].append(document)
    summary: list[dict[str, Any]] = []
    for source_id, documents in grouped.items():
        summary.append(
            {
                "source_id": source_id,
                "documents": [
                    {
                        "url": document.url,
                        "title": document.title,
                        "parser": document.parser,
                        "link_count": len(document.links),
                        "attachment_count": len(document.attachments),
                        "row_count": len(document.rows),
                        "relevance": document.metadata.get("relevance"),
                        "sample_links": document.links[:6],
                        "sample_attachments": document.attachments[:6],
                    }
                    for document in documents[:20]
                ],
            }
        )
    return summary
