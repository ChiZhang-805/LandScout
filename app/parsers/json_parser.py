from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.parsers.html import decode_html
from app.parsers.models import ParsedDocument


def parse_json(
    content: bytes | str,
    *,
    url: str,
    source_id: str = "",
    content_hash: str = "",
    fetched_at: str = "",
    raw_path: str = "",
) -> ParsedDocument:
    text = decode_html(content)
    rows: list[dict[str, Any]] = []
    try:
        payload = json.loads(text)
        flattened = flatten(payload)
        rows = extract_record_rows(payload)
        row_text = "\n".join(format_record(row) for row in rows[:200])
        flat_text = "\n".join(f"{key}: {value}" for key, value in flattened[:1000])
        output_text = "\n".join(part for part in (row_text, flat_text) if part)
    except Exception as exc:
        payload = None
        output_text = f"[JSON parse failed: {exc}]\n{text[:2000]}"
    return ParsedDocument(
        source_id=source_id,
        url=url,
        title=Path(str(raw_path or url)).name or "JSON document",
        text=output_text,
        rows=rows,
        content_hash=content_hash,
        fetched_at=fetched_at,
        raw_path=raw_path,
        parser="json",
        metadata={"json": payload is not None, "row_count": len(rows)},
    )


def flatten(value: Any, prefix: str = "", *, limit: int = 1000) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []

    def walk(node: Any, node_prefix: str) -> None:
        if len(items) >= limit:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                if len(items) >= limit:
                    break
                child_prefix = f"{node_prefix}.{key}" if node_prefix else str(key)
                walk(child, child_prefix)
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                if len(items) >= limit:
                    break
                walk(child, f"{node_prefix}[{idx}]")
        else:
            items.append((node_prefix, node))

    walk(value, prefix)
    return items


def extract_record_rows(value: Any, *, limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if len(rows) >= limit:
            return
        if isinstance(node, list):
            dict_items = [item for item in node if isinstance(item, dict)]
            if dict_items:
                added = 0
                for item in dict_items:
                    if len(rows) >= limit:
                        break
                    row = compact_record(item)
                    if is_record_like(row):
                        rows.append(row)
                        added += 1
                if added:
                    return
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            row = compact_record(node)
            if is_record_like(row):
                rows.append(row)
                if len(rows) >= limit:
                    return
            for child in node.values():
                walk(child)

    walk(value)
    return rows


def compact_record(value: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, child in value.items():
        if isinstance(child, (dict, list)):
            continue
        if child is None or child == "":
            continue
        row[str(key)] = child
    return row


def is_record_like(row: dict[str, Any]) -> bool:
    if len(row) < 2:
        return False
    keys = {key.lower() for key in row}
    wrapper_keys = {"code", "status", "success", "message", "msg", "total", "page", "pages", "size"}
    if keys and keys.issubset(wrapper_keys):
        return False
    hints = (
        "id",
        "title",
        "name",
        "date",
        "time",
        "project",
        "address",
        "district",
        "area",
        "amount",
        "url",
        "content",
        "公告",
        "项目",
        "名称",
        "日期",
        "时间",
        "地块",
        "区域",
    )
    return any(any(hint in key for hint in hints) for key in keys)


def format_record(row: dict[str, Any]) -> str:
    return " | ".join(f"{key}: {value}" for key, value in row.items())
