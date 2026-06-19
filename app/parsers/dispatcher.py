from __future__ import annotations

from pathlib import Path

from app.crawlers.models import RawDocument
from app.parsers.csv_parser import parse_csv
from app.parsers.excel import parse_excel
from app.parsers.html import decode_html, parse_html
from app.parsers.json_parser import parse_json
from app.parsers.models import ParsedDocument
from app.parsers.pdf import parse_pdf
from app.parsers.word import parse_word


def parse_raw_document(raw: RawDocument) -> ParsedDocument:
    path = Path(raw.path)
    data = path.read_bytes()
    kwargs = {
        "url": raw.url,
        "source_id": raw.source_id,
        "content_hash": raw.content_hash,
        "fetched_at": raw.fetched_at,
        "raw_path": raw.path,
    }
    if raw.kind == "html" or raw.content_type.startswith("text/html"):
        return parse_html(data, **kwargs)
    if raw.kind == "json":
        return parse_json(data, **kwargs)
    if raw.kind == "csv" or raw.content_type in {"text/csv", "application/csv"}:
        return parse_csv(data, **kwargs)
    if raw.kind == "pdf":
        return parse_pdf(data, **kwargs)
    if raw.kind == "excel":
        return parse_excel(data, **kwargs)
    if raw.kind == "word":
        return parse_word(path, **kwargs)
    return ParsedDocument(
        source_id=raw.source_id,
        url=raw.url,
        title=path.name,
        text=decode_html(data[:200000]),
        content_hash=raw.content_hash,
        fetched_at=raw.fetched_at,
        raw_path=raw.path,
        parser="text",
    )
