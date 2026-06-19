from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings
from app.core.utils import ensure_dir
from app.llm.client import build_extraction_prompt
from app.llm.openai_client import build_openai_client
from app.llm.schemas import LLMGovernmentSignalExtraction
from app.parsers.models import ParsedDocument


def write_extraction_batch_requests(
    path: Path,
    documents: list[ParsedDocument],
    *,
    model: str | None = None,
) -> Path:
    ensure_dir(path.parent)
    schema = LLMGovernmentSignalExtraction.model_json_schema()
    selected_documents = documents_selected_for_extraction(documents)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for idx, document in enumerate(selected_documents):
            request = {
                "custom_id": f"extract-{idx:06d}-{safe_custom_id(document.source_id)}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model or settings.openai_model,
                    "input": [
                        {
                            "role": "system",
                            "content": (
                                "Extract only factual, source-backed Shanghai government real-estate signals. "
                                "Every evidence.quote must be an exact substring from the supplied document. "
                                "Use null for unknown optional values, [] for empty arrays, and include every field."
                            ),
                        },
                        {"role": "user", "content": build_extraction_prompt(document)},
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "GovernmentSignalExtraction",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    return path


def documents_selected_for_extraction(documents: list[ParsedDocument]) -> list[ParsedDocument]:
    selected: list[ParsedDocument] = []
    for document in documents:
        relevance = document.metadata.get("relevance")
        if isinstance(relevance, dict) and relevance.get("should_extract") is False:
            continue
        selected.append(document)
    return selected


def submit_batch_request_file(path: Path, *, completion_window: str = "24h") -> str:
    if not path.exists():
        raise FileNotFoundError(f"Batch request file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Batch request file is empty: {path}")

    client = build_openai_client()
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    return batch.id


def safe_custom_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:48] or "document"
