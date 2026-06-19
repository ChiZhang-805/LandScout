from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


RawKind = Literal["html", "json", "csv", "pdf", "excel", "word", "text", "binary"]


class RawDocument(BaseModel):
    id: str
    source_id: str
    url: str
    fetched_at: str
    content_hash: str
    path: str
    kind: RawKind
    status_code: int | None = None
    content_type: str = ""
    parent_url: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def file_path(self) -> Path:
        return Path(self.path)


class FetchError(BaseModel):
    source_id: str
    url: str
    reason: str
    status_code: int | None = None


class FetchRunResult(BaseModel):
    run_id: str
    documents: list[RawDocument] = Field(default_factory=list)
    errors: list[FetchError] = Field(default_factory=list)
    visited_sources: list[str] = Field(default_factory=list)
    discovered_urls: list[str] = Field(default_factory=list)
