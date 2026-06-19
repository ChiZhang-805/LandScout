from __future__ import annotations

import re
import urllib.parse

from bs4 import BeautifulSoup, UnicodeDammit

from app.llm.normalization import normalize_date
from app.parsers.models import ParsedDocument


ATTACHMENT_EXTENSIONS = (".pdf", ".xlsx", ".xls", ".docx", ".doc", ".csv", ".json")


def parse_html(
    content: bytes | str,
    *,
    url: str,
    source_id: str = "",
    content_hash: str = "",
    fetched_at: str = "",
    raw_path: str = "",
) -> ParsedDocument:
    html = decode_html(content)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        heading = soup.find(["h1", "h2", "h3"])
        title = heading.get_text(" ", strip=True) if heading else ""

    links: list[dict[str, str]] = []
    attachments: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", "")).strip()
        if not href or href.startswith(("javascript:", "mailto:", "#")):
            continue
        absolute = urllib.parse.urljoin(url, href)
        text = anchor.get_text(" ", strip=True)
        item = {"url": absolute, "text": text}
        links.append(item)
        lowered = urllib.parse.unquote(absolute).lower()
        if lowered.endswith(ATTACHMENT_EXTENSIONS) or any(ext in lowered for ext in ATTACHMENT_EXTENSIONS):
            attachments.append(item)

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    date = extract_date(text) or extract_date(title)
    return ParsedDocument(
        source_id=source_id,
        url=url,
        title=title,
        date=date,
        text=text,
        links=links,
        attachments=attachments,
        content_hash=content_hash,
        fetched_at=fetched_at,
        raw_path=raw_path,
        parser="html",
    )


def decode_html(content: bytes | str) -> str:
    if isinstance(content, str):
        return content
    declared_encoding = detect_declared_encoding(content[:4096])
    if declared_encoding:
        try:
            return content.decode(declared_encoding)
        except (LookupError, UnicodeDecodeError):
            pass
    for encoding in ("utf-8", "gb18030", "gbk"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    decoded = UnicodeDammit(content, is_html=True).unicode_markup
    if decoded:
        return decoded
    return content.decode("utf-8", errors="ignore")


def detect_declared_encoding(sample: bytes) -> str | None:
    patterns = (
        rb"<meta[^>]+charset\s*=\s*[\"']?\s*([a-zA-Z0-9_\-]+)",
        rb"<\?xml[^>]+encoding\s*=\s*[\"']\s*([a-zA-Z0-9_\-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, sample, flags=re.IGNORECASE)
        if match:
            return match.group(1).decode("ascii", errors="ignore").lower()
    return None


def extract_date(text: str) -> str | None:
    patterns = [
        r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?",
        r"(20\d{2})\s*年\s*(\d{1,2})\s*月",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "")
        if match:
            return normalize_date(match.group(0))
    return None
