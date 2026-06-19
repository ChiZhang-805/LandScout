from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.utils import short_text, write_json
from app.llm.openai_client import build_openai_client
from app.llm.schemas import GovernmentEvent
from app.parsers.models import ParsedDocument


class EmbeddingRecord(BaseModel):
    id: str
    kind: str
    source_id: str
    source_url: str
    title: str
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_embedding_records(
    parsed_documents: list[ParsedDocument],
    events: list[GovernmentEvent],
) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for document in parsed_documents:
        text = short_text(document.text, 1800)
        if not text:
            continue
        records.append(
            EmbeddingRecord(
                id=f"doc:{document.content_hash or document.url}",
                kind="document",
                source_id=document.source_id,
                source_url=document.url,
                title=document.title or document.url,
                text=text,
                metadata={
                    "parser": document.parser,
                    "date": document.date,
                    "relevance": document.metadata.get("relevance"),
                },
            )
        )
    for event in events:
        quote = event.evidence[0].quote if event.evidence else ""
        text = short_text(f"{event.title}\n{event.summary}\n{quote}", 1200)
        if not text:
            continue
        records.append(
            EmbeddingRecord(
                id=f"event:{event.id}",
                kind="event",
                source_id=event.source_id,
                source_url=event.source_url,
                title=event.title,
                text=text,
                metadata={
                    "event_type": event.event_type.value,
                    "district": event.district,
                    "event_date": event.event_date,
                    "lat": event.lat,
                    "lon": event.lon,
                    "geo_confidence": event.geo_confidence,
                },
            )
        )
    return records


def write_embedding_index(
    path: Path,
    parsed_documents: list[ParsedDocument],
    events: list[GovernmentEvent],
    *,
    use_openai: bool,
) -> Path:
    records = build_embedding_records(parsed_documents, events)
    embedding_error = ""
    if use_openai and settings.openai_api_key and records:
        try:
            embed_records(records)
        except Exception as exc:
            embedding_error = str(exc)
    write_json(
        path,
        {
            "model": settings.openai_embedding_model if use_openai and settings.openai_api_key else "",
            "record_count": len(records),
            "embedding_status": embedding_status(records),
            "embedding_error": embedding_error,
            "records": [record.model_dump(mode="json") for record in records],
        },
    )
    return path


def embedding_status(records: list[EmbeddingRecord]) -> str:
    if not records:
        return "no_records"
    embedded_count = sum(1 for record in records if record.embedding)
    if embedded_count == len(records):
        return "complete"
    if embedded_count:
        return "partial"
    return "not_generated"


def embed_records(records: list[EmbeddingRecord], batch_size: int = 96) -> None:
    client = build_openai_client()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=[record.text for record in batch],
        )
        for record, embedding in zip(batch, response.data, strict=False):
            record.embedding = list(embedding.embedding)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)
