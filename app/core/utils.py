from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def content_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(value: str, max_length: int = 96) -> str:
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value).strip("._")
    return (value or "document")[:max_length]


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def short_text(value: str, limit: int = 200) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def leading_label(value: Any, fallback: str = "未分配") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return text.split("：", 1)[0].split(":", 1)[0].strip() or fallback


def error_reason(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("reason") or "")
    return str(getattr(error, "reason", "") or "")


def error_stage(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("stage") or "")
    return str(getattr(error, "stage", "") or "")


def is_document_triage_skip(error: Any) -> bool:
    reason = error_reason(error)
    return error_stage(error) == "document_triage" or reason.startswith("document triage skipped:")


def actionable_errors(errors: list[Any]) -> list[Any]:
    return [error for error in errors if not is_document_triage_skip(error)]


def document_triage_skip_count(errors: list[Any]) -> int:
    return sum(1 for error in errors if is_document_triage_skip(error))
