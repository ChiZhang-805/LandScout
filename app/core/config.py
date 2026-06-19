from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.core.branding import PRODUCT_DISPLAY_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def default_memory_safe_mode() -> bool:
    return env_bool("LANDSCOUT_MEMORY_SAFE_MODE", bool(os.getenv("RENDER")))


class Settings(BaseModel):
    app_name: str = Field(default_factory=lambda: os.getenv("APP_NAME", PRODUCT_DISPLAY_NAME))
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    )
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    openai_fast_model: str = Field(default_factory=lambda: os.getenv("OPENAI_FAST_MODEL", "gpt-4.1-mini"))
    openai_embedding_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    openai_proxy: str = Field(default_factory=lambda: os.getenv("OPENAI_PROXY", ""))
    amap_key: str = Field(default_factory=lambda: os.getenv("AMAP_KEY", ""))
    user_agent: str = Field(
        default_factory=lambda: os.getenv(
            "USER_AGENT",
            "landscout-agent/0.1 (+public-source-research)",
        )
    )
    request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    )
    memory_safe_mode: bool = Field(default_factory=default_memory_safe_mode)
    disable_playwright: bool = Field(
        default_factory=lambda: env_bool("LANDSCOUT_DISABLE_PLAYWRIGHT", default_memory_safe_mode())
    )
    live_source_limit_cap: int = Field(
        default_factory=lambda: env_int(
            "LANDSCOUT_LIVE_SOURCE_LIMIT_CAP",
            12 if default_memory_safe_mode() else 100,
        )
    )
    live_source_batch_size: int = Field(
        default_factory=lambda: env_int(
            "LANDSCOUT_LIVE_SOURCE_BATCH_SIZE",
            3 if default_memory_safe_mode() else 0,
        )
    )
    max_response_bytes: int = Field(
        default_factory=lambda: env_int(
            "LANDSCOUT_MAX_RESPONSE_BYTES",
            1_500_000 if default_memory_safe_mode() else 8_000_000,
        )
    )
    max_raw_documents: int = Field(
        default_factory=lambda: env_int(
            "LANDSCOUT_MAX_RAW_DOCUMENTS",
            40 if default_memory_safe_mode() else 200,
        )
    )
    max_document_chars: int = Field(
        default_factory=lambda: env_int(
            "LANDSCOUT_MAX_DOCUMENT_CHARS",
            20_000 if default_memory_safe_mode() else 60_000,
        )
    )
    max_extraction_documents: int = Field(
        default_factory=lambda: env_int(
            "LANDSCOUT_MAX_EXTRACTION_DOCUMENTS",
            18 if default_memory_safe_mode() else 80,
        )
    )
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    data_dir: Path = PROJECT_ROOT / "data"
    source_config_path: Path = PROJECT_ROOT / "app" / "sources" / "configs" / "shanghai_sources.yml"


settings = Settings()


_request_openai_api_key: ContextVar[str | None] = ContextVar(
    "request_openai_api_key",
    default=None,
)


def effective_openai_api_key() -> str:
    return (_request_openai_api_key.get() or settings.openai_api_key).strip()


@contextmanager
def use_request_openai_api_key(api_key: str | None) -> Iterator[None]:
    cleaned = (api_key or "").strip()
    token = _request_openai_api_key.set(cleaned) if cleaned else None
    try:
        yield
    finally:
        if token is not None:
            _request_openai_api_key.reset(token)
