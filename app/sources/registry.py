from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.core.config import settings


AccessMode = Literal["http", "http_then_playwright", "playwright_with_network_discovery"]


class SourceConfig(BaseModel):
    id: str
    name: str
    base_urls: list[HttpUrl]
    access_mode: AccessMode = "http"
    priority: int = 100
    max_pages: int = 2
    delay: float = 1.0
    keywords: list[str] = Field(default_factory=list)
    attachment_types: list[str] = Field(default_factory=list)
    official: bool = True
    notes: str = ""

    @field_validator("attachment_types")
    @classmethod
    def normalize_attachment_types(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            item = item.lower().strip()
            normalized.append(item if item.startswith(".") else f".{item}")
        return normalized

    @property
    def urls(self) -> list[str]:
        return [str(url) for url in self.base_urls]


class SourceRegistry:
    def __init__(self, sources: list[SourceConfig]) -> None:
        seen_ids: set[str] = set()
        duplicate_ids: list[str] = []
        for source in sources:
            if source.id in seen_ids:
                duplicate_ids.append(source.id)
            seen_ids.add(source.id)
        if duplicate_ids:
            raise ValueError(f"Duplicate source ids configured: {', '.join(sorted(set(duplicate_ids)))}")
        self.sources = sorted(sources, key=lambda item: (item.priority, item.id))
        self._by_id = {source.id: source for source in self.sources}

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "SourceRegistry":
        path = Path(path or settings.source_config_path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Source config must be a mapping in {path}")
        sources = [SourceConfig.model_validate(item) for item in payload.get("sources", [])]
        if not sources:
            raise ValueError(f"No sources configured in {path}")
        return cls(sources)

    def get(self, source_id: str) -> SourceConfig:
        try:
            return self._by_id[source_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._by_id))
            raise KeyError(f"Unknown source_id '{source_id}'. Known sources: {known}") from exc

    def select(self, limit: int | None = None) -> list[SourceConfig]:
        if limit is None:
            return list(self.sources)
        if limit <= 0:
            raise ValueError("source limit must be a positive integer")
        return self.sources[:limit]

    def merged(self, sources: list[SourceConfig]) -> "SourceRegistry":
        merged_sources = [*self.sources]
        seen_ids = {source.id for source in merged_sources}
        seen_urls = {str(url) for source in merged_sources for url in source.base_urls}
        for source in sources:
            source_urls = {str(url) for url in source.base_urls}
            if source.id in seen_ids or source_urls & seen_urls:
                continue
            merged_sources.append(source)
            seen_ids.add(source.id)
            seen_urls.update(source_urls)
        return SourceRegistry(merged_sources)


def load_shanghai_registry() -> SourceRegistry:
    return SourceRegistry.from_yaml(settings.source_config_path)
