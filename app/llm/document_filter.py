from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import effective_openai_api_key, settings
from app.core.utils import short_text
from app.llm.client import MissingLLMKey
from app.llm.openai_client import (
    OpenAINonRecoverableError,
    build_openai_client,
    is_openai_non_recoverable_error,
    summarize_openai_error,
)
from app.parsers.models import ParsedDocument


RELEVANCE_KEYWORDS = (
    "住宅",
    "居住",
    "商品房",
    "租赁住房",
    "保障性住房",
    "土地",
    "地块",
    "出让",
    "控规",
    "控制性详细规划",
    "详细规划",
    "规划许可",
    "批复",
    "重大项目",
    "产业",
    "园区",
    "招商",
    "签约",
    "轨道交通",
    "机场",
    "学校",
    "医院",
    "公共服务",
    "新城",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentRelevance(StrictModel):
    should_extract: bool
    relevance_score: float = Field(ge=0, le=1)
    categories: list[str]
    reason: str


@dataclass
class DocumentRelevanceFilter:
    live: bool = True

    def classify(self, document: ParsedDocument) -> DocumentRelevance:
        obvious = classify_obvious_irrelevant(document)
        if obvious:
            return obvious
        if self.live:
            if not effective_openai_api_key():
                raise MissingLLMKey("OPENAI_API_KEY is required for live document relevance filtering.")
            try:
                return self._classify_openai(document)
            except Exception as exc:
                if is_openai_non_recoverable_error(exc):
                    raise OpenAINonRecoverableError(summarize_openai_error(exc)) from exc
                return classify_document_heuristic(document)
        return classify_document_heuristic(document)

    def _classify_openai(self, document: ParsedDocument) -> DocumentRelevance:
        client = build_openai_client()
        response = client.responses.create(
            model=settings.openai_fast_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the Document Triage Agent for LandScout Agent. "
                        "Decide whether a public Shanghai document is worth sending to a deeper extraction model. "
                        "Prefer recall over precision: keep documents about land, planning, residential supply, major projects, "
                        "industrial investment, transport, schools, hospitals, public services, or new-town development. "
                        "Reject navigation pages, boilerplate, login/captcha pages, and unrelated notices."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"source_id: {document.source_id}\n"
                        f"url: {document.url}\n"
                        f"title: {document.title}\n"
                        f"date: {document.date}\n\n"
                        f"text excerpt:\n{document.text[:12000]}"
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "DocumentRelevance",
                    "schema": DocumentRelevance.model_json_schema(),
                    "strict": True,
                }
            },
        )
        return DocumentRelevance.model_validate(json.loads(response.output_text))


def classify_document_heuristic(document: ParsedDocument) -> DocumentRelevance:
    text = f"{document.title}\n{short_text(document.text, 6000)}"
    matched = [keyword for keyword in RELEVANCE_KEYWORDS if keyword in text]
    score = min(1.0, 0.18 + len(matched) * 0.08)
    should_extract = bool(matched) or bool(document.rows)
    if document.parser in {"excel", "csv", "json"} and document.rows:
        should_extract = True
        score = max(score, 0.65)
    return DocumentRelevance(
        should_extract=should_extract,
        relevance_score=round(score if should_extract else min(score, 0.2), 3),
        categories=matched[:8],
        reason="keyword/structured-data heuristic" if should_extract else "no relevant signal keywords found",
    )


def classify_obvious_irrelevant(document: ParsedDocument) -> DocumentRelevance | None:
    if document.rows or document.attachments:
        return None
    text = f"{document.title}\n{short_text(document.text, 3000)}".strip()
    matched = [keyword for keyword in RELEVANCE_KEYWORDS if keyword in text]
    if matched:
        return None
    compact_text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact_text) < 80:
        return DocumentRelevance(
            should_extract=False,
            relevance_score=0.0,
            categories=[],
            reason="obvious empty or too-short document",
        )
    if document.parser == "html" and len(document.links) >= 20 and len(compact_text) < 1600:
        return DocumentRelevance(
            should_extract=False,
            relevance_score=0.05,
            categories=[],
            reason="obvious navigation/index page without signal keywords",
        )
    return None
