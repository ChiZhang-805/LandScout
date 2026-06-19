from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    source_id: str
    url: str
    title: str = ""
    date: str | None = None
    text: str = ""
    links: list[dict[str, str]] = Field(default_factory=list)
    attachments: list[dict[str, str]] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[list[list[str]]] = Field(default_factory=list)
    content_hash: str = ""
    fetched_at: str = ""
    raw_path: str = ""
    parser: str = ""
    needs_ocr: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

