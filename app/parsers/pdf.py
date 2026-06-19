from __future__ import annotations

from io import BytesIO
from pathlib import Path

from app.parsers.models import ParsedDocument


def parse_pdf(
    content_or_path: bytes | str | Path,
    *,
    url: str,
    source_id: str = "",
    content_hash: str = "",
    fetched_at: str = "",
    raw_path: str = "",
) -> ParsedDocument:
    if isinstance(content_or_path, bytes):
        data = content_or_path
    else:
        data = Path(content_or_path).read_bytes()

    text_parts: list[str] = []
    table_data: list[list[list[str]]] = []
    page_count = 0

    try:
        import fitz

        with fitz.open(stream=data, filetype="pdf") as document:
            page_count = document.page_count
            for page in document:
                text_parts.append(page.get_text("text"))
    except Exception as exc:
        text_parts.append(f"[PyMuPDF extraction failed: {exc}]")

    try:
        import pdfplumber

        with pdfplumber.open(BytesIO(data)) as pdf:
            page_count = page_count or len(pdf.pages)
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    table_data.append([[str(cell or "").strip() for cell in row] for row in table])
    except Exception as exc:
        text_parts.append(f"[pdfplumber table extraction failed: {exc}]")

    text = "\n".join(part.strip() for part in text_parts if part.strip())
    needs_ocr = page_count > 0 and len(text) / max(page_count, 1) < 30
    return ParsedDocument(
        source_id=source_id,
        url=url,
        title=Path(str(raw_path or url)).name or "PDF document",
        text=text,
        tables=table_data,
        content_hash=content_hash,
        fetched_at=fetched_at,
        raw_path=raw_path,
        parser="pdf",
        needs_ocr=needs_ocr,
        metadata={"page_count": page_count, "table_count": len(table_data)},
    )

