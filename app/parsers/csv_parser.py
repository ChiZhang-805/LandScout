from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from app.parsers.excel import dedupe_headers, markdown_table
from app.parsers.html import decode_html
from app.parsers.models import ParsedDocument


def parse_csv(
    content: bytes | str,
    *,
    url: str,
    source_id: str = "",
    content_hash: str = "",
    fetched_at: str = "",
    raw_path: str = "",
) -> ParsedDocument:
    text = decode_html(content)
    rows = parse_csv_rows(text)
    title = csv_title(raw_path=raw_path, url=url)
    if not rows:
        return ParsedDocument(
            source_id=source_id,
            url=url,
            title=title or "CSV document",
            text=text[:200000],
            content_hash=content_hash,
            fetched_at=fetched_at,
            raw_path=raw_path,
            parser="csv",
            metadata={"row_count": 0},
        )

    headers = dedupe_headers(rows[0])
    data_rows = rows[1:]
    mapped_rows: list[dict[str, Any]] = []
    text_lines = [markdown_table([headers] + data_rows[:30])]
    for row in data_rows[:5000]:
        mapped = {
            headers[idx] if idx < len(headers) else f"col_{idx + 1}": row[idx] if idx < len(row) else ""
            for idx in range(max(len(headers), len(row)))
        }
        mapped = {key: value for key, value in mapped.items() if key and value != ""}
        if mapped:
            mapped_rows.append(mapped)
            text_lines.append(" | ".join(str(value) for value in mapped.values()))

    return ParsedDocument(
        source_id=source_id,
        url=url,
        title=title or "CSV document",
        text="\n".join(text_lines),
        rows=mapped_rows,
        tables=[[headers] + data_rows[:100]],
        content_hash=content_hash,
        fetched_at=fetched_at,
        raw_path=raw_path,
        parser="csv",
        metadata={"row_count": len(mapped_rows)},
    )


def parse_csv_rows(text: str) -> list[list[str]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(StringIO(text), dialect)
    return [[clean_csv_cell(cell) for cell in row] for row in reader if any(cell.strip() for cell in row)]


def clean_csv_cell(value: str) -> str:
    return value.strip().lstrip("\ufeff")


def csv_title(*, raw_path: str, url: str) -> str:
    if raw_path:
        return Path(raw_path).name
    parsed = urlparse(url)
    candidate = parsed.path or parsed.netloc or url
    return Path(unquote(candidate)).name
