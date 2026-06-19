from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.core.branding import PRODUCT_DISPLAY_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)


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
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    data_dir: Path = PROJECT_ROOT / "data"
    source_config_path: Path = PROJECT_ROOT / "app" / "sources" / "configs" / "shanghai_sources.yml"


settings = Settings()
