from __future__ import annotations

import json
import mimetypes
import os
import re
import ssl
import time
import urllib.parse
import zipfile
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from app.core.config import settings
from app.core.utils import content_hash, ensure_dir, safe_filename, utc_now_iso
from app.crawlers.models import FetchError, FetchRunResult, RawDocument, RawKind
from app.crawlers.robots import RobotsCache
from app.parsers.html import decode_html
from app.sources.registry import SourceConfig, SourceRegistry


BLOCK_PATTERNS = ("验证码", "登录后", "请登录", "forbidden", "access denied", "captcha")
HTML_TYPES = ("text/html", "application/xhtml+xml")
JSON_TYPES = ("application/json", "text/json")
TEXT_DOCUMENT_TYPES = ("text/plain", "text/xml", "application/xml")
PUBLIC_DOCUMENT_KINDS = {"html", "json", "csv", "pdf", "excel", "word", "text"}
COMMON_CHROMIUM_EXECUTABLES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)


class PublicFetcher:
    def __init__(
        self,
        registry: SourceRegistry,
        run_id: str,
        run_dir: Path,
        user_agent: str | None = None,
    ) -> None:
        self.registry = registry
        self.run_id = run_id
        self.run_dir = ensure_dir(run_dir)
        self.raw_dir = ensure_dir(self.run_dir / "raw")
        self.user_agent = user_agent or settings.user_agent
        self.robots = RobotsCache(self.user_agent)
        self.last_domain_access: dict[str, float] = {}
        self.seen_hashes: set[str] = set()
        self.client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
            timeout=settings.request_timeout_seconds,
            verify=build_ssl_context(),
        )

    def close(self) -> None:
        self.client.close()

    def fetch_source(self, source_id: str, pages: int | None = None, days: int | None = None) -> FetchRunResult:
        source = self.registry.get(source_id)
        result = FetchRunResult(run_id=self.run_id, visited_sources=[source.id])
        page_budget = pages or source.max_pages
        cutoff_date = cutoff_date_from_days(days)
        source_dir = ensure_dir(self.raw_dir / source.id)
        urls = source.urls[:page_budget]

        for url in urls:
            doc = self._fetch_public_url(source, url, source_dir)
            if isinstance(doc, FetchError):
                result.errors.append(doc)
                continue
            if doc:
                result.documents.append(doc)
                result.discovered_urls.append(url)
                if doc.kind == "html":
                    self._fetch_related_from_html(source, doc, source_dir, page_budget, result, cutoff_date=cutoff_date)

            if source.access_mode in {"http_then_playwright", "playwright_with_network_discovery"}:
                should_render = source.access_mode == "playwright_with_network_discovery" or (
                    doc is not None and not isinstance(doc, FetchError) and doc.kind == "html" and doc.file_path.stat().st_size < 5000
                )
                if should_render:
                    rendered = self._fetch_with_playwright(source, url, source_dir)
                    result.documents.extend(rendered.documents)
                    result.errors.extend(rendered.errors)
                    result.discovered_urls.extend(rendered.discovered_urls)

        return result

    def discover_source(self, source_id: str) -> FetchRunResult:
        source = self.registry.get(source_id)
        result = FetchRunResult(run_id=self.run_id, visited_sources=[source.id])
        source_dir = ensure_dir(self.raw_dir / source.id)
        for url in source.urls[: source.max_pages]:
            if source.access_mode == "playwright_with_network_discovery":
                discovered = self._fetch_with_playwright(source, url, source_dir)
                result.documents.extend(discovered.documents)
                result.errors.extend(discovered.errors)
                result.discovered_urls.extend(discovered.discovered_urls)
            else:
                doc = self._fetch_public_url(source, url, source_dir)
                if isinstance(doc, FetchError):
                    result.errors.append(doc)
                elif doc:
                    result.documents.append(doc)
                    result.discovered_urls.extend(self._extract_public_links(doc, source))
        return result

    def fetch_many(self, source_limit: int | None = None, pages: int | None = None, days: int | None = None) -> FetchRunResult:
        aggregate = FetchRunResult(run_id=self.run_id)
        for source in self.registry.select(limit=source_limit):
            result = self.fetch_source(source.id, pages=pages, days=days)
            aggregate.documents.extend(result.documents)
            aggregate.errors.extend(result.errors)
            aggregate.visited_sources.extend(result.visited_sources)
            aggregate.discovered_urls.extend(result.discovered_urls)
        aggregate.visited_sources = list(dict.fromkeys(aggregate.visited_sources))
        aggregate.discovered_urls = list(dict.fromkeys(aggregate.discovered_urls))
        return aggregate

    def _fetch_related_from_html(
        self,
        source: SourceConfig,
        doc: RawDocument,
        source_dir: Path,
        page_budget: int,
        result: FetchRunResult,
        cutoff_date: date | None = None,
    ) -> None:
        html = read_html_document(doc)
        soup = BeautifulSoup(html, "html.parser")
        candidates = self._candidate_links(source, doc.url, soup, cutoff_date=cutoff_date)
        detail_limit = max(4, page_budget * 4)
        attachment_limit = max(12, page_budget * 6)
        details_seen = 0
        attachments_seen = 0

        for url, text in candidates:
            if self._is_attachment_url(source, url, text):
                if attachments_seen >= attachment_limit:
                    continue
                attachments_seen += 1
            else:
                if details_seen >= detail_limit:
                    continue
                details_seen += 1
            child = self._fetch_public_url(source, url, source_dir, parent_url=doc.url)
            if isinstance(child, FetchError):
                result.errors.append(child)
            elif child:
                result.documents.append(child)
                result.discovered_urls.append(url)
                if child.kind == "html":
                    self._fetch_attachments_from_detail(
                        source,
                        child,
                        source_dir,
                        result,
                        attachment_limit=max(6, page_budget * 2),
                        cutoff_date=cutoff_date,
                    )

    def _fetch_attachments_from_detail(
        self,
        source: SourceConfig,
        detail_doc: RawDocument,
        source_dir: Path,
        result: FetchRunResult,
        attachment_limit: int,
        cutoff_date: date | None = None,
    ) -> None:
        html = read_html_document(detail_doc)
        soup = BeautifulSoup(html, "html.parser")
        fetched = 0
        for url, text in self._candidate_links(source, detail_doc.url, soup, cutoff_date=cutoff_date):
            if not self._is_attachment_url(source, url, text):
                continue
            if fetched >= attachment_limit:
                break
            fetched += 1
            attachment = self._fetch_public_url(source, url, source_dir, parent_url=detail_doc.url)
            if isinstance(attachment, FetchError):
                result.errors.append(attachment)
            elif attachment:
                result.documents.append(attachment)
                result.discovered_urls.append(url)

    def _fetch_public_url(
        self,
        source: SourceConfig,
        url: str,
        source_dir: Path,
        parent_url: str | None = None,
    ) -> RawDocument | FetchError | None:
        if not self.robots.allowed(url):
            return FetchError(source_id=source.id, url=url, reason="robots.txt disallows this URL")

        self._domain_delay(url, source.delay)
        try:
            response = self.client.get(url)
        except httpx.HTTPError as exc:
            return FetchError(source_id=source.id, url=url, reason=f"http error: {exc}")

        if response.status_code in {401, 403, 429}:
            return FetchError(
                source_id=source.id,
                url=url,
                reason="access limited by source; crawler stopped for this URL",
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            return FetchError(
                source_id=source.id,
                url=url,
                reason=f"http status {response.status_code}; response body was not saved as a public document",
                status_code=response.status_code,
            )

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        filename_hint = filename_from_content_disposition(response.headers.get("content-disposition", ""))
        body = response.content
        if self._looks_blocked(body, content_type):
            return FetchError(
                source_id=source.id,
                url=url,
                reason="captcha/login/access-limit marker found; no bypass attempted",
                status_code=response.status_code,
            )
        kind = refine_kind_from_body(infer_kind(url, content_type, filename_hint=filename_hint), body)
        if kind not in PUBLIC_DOCUMENT_KINDS:
            return FetchError(
                source_id=source.id,
                url=url,
                reason=f"unsupported content kind '{kind}'; response body was not saved as an analyzable public document",
                status_code=response.status_code,
            )
        return self._save_raw_document(
            source,
            url,
            body,
            content_type,
            response.status_code,
            source_dir,
            parent_url,
            filename_hint=filename_hint,
            kind_override=kind,
        )

    def _save_raw_document(
        self,
        source: SourceConfig,
        url: str,
        body: bytes,
        content_type: str,
        status_code: int | None,
        source_dir: Path,
        parent_url: str | None = None,
        metadata: dict | None = None,
        filename_hint: str = "",
        kind_override: RawKind | None = None,
    ) -> RawDocument | None:
        digest = content_hash(body)
        if digest in self.seen_hashes:
            return None
        self.seen_hashes.add(digest)
        kind = kind_override or infer_kind(url, content_type, filename_hint=filename_hint)
        suffix = kind_suffix(url, kind, filename_hint=filename_hint)
        file_name = f"{safe_filename(url)}_{digest[:12]}{suffix}"
        path = source_dir / file_name
        path.write_bytes(body)
        metadata = {**(metadata or {})}
        if filename_hint:
            metadata["filename_hint"] = filename_hint
        return RawDocument(
            id=digest[:16],
            source_id=source.id,
            url=url,
            fetched_at=utc_now_iso(),
            content_hash=digest,
            path=str(path),
            kind=kind,
            status_code=status_code,
            content_type=content_type,
            parent_url=parent_url,
            metadata=metadata,
        )

    def _fetch_with_playwright(self, source: SourceConfig, url: str, source_dir: Path) -> FetchRunResult:
        result = FetchRunResult(run_id=self.run_id, visited_sources=[source.id])
        if not self.robots.allowed(url):
            result.errors.append(FetchError(source_id=source.id, url=url, reason="robots.txt disallows this URL"))
            return result
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            result.errors.append(FetchError(source_id=source.id, url=url, reason=f"Playwright unavailable: {exc}"))
            return result

        self._domain_delay(url, source.delay)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(**chromium_launch_options())
                page = browser.new_page(user_agent=self.user_agent)

                def on_response(response) -> None:  # type: ignore[no-untyped-def]
                    try:
                        response_url = response.url
                        response_type = response.headers.get("content-type", "").split(";")[0].lower()
                        if not _is_public_same_origin_or_api(url, response_url):
                            return
                        if response.status in {401, 403, 429}:
                            result.errors.append(
                                FetchError(
                                    source_id=source.id,
                                    url=response_url,
                                    reason="network response access limited",
                                    status_code=response.status,
                                )
                            )
                            return
                        if response.status >= 400:
                            result.errors.append(
                                FetchError(
                                    source_id=source.id,
                                    url=response_url,
                                    reason=f"network response status {response.status}; body ignored",
                                    status_code=response.status,
                                )
                            )
                            return
                        filename_hint = filename_from_content_disposition(response.headers.get("content-disposition", ""))
                        response_kind = infer_kind(response_url, response_type, filename_hint=filename_hint)
                        if response_kind in PUBLIC_DOCUMENT_KINDS or response_type in {"", "application/octet-stream"}:
                            body = response.body()
                            response_kind = refine_kind_from_body(response_kind, body)
                        if response_kind in PUBLIC_DOCUMENT_KINDS:
                            if self._looks_blocked(body, response_type):
                                result.errors.append(
                                    FetchError(
                                        source_id=source.id,
                                        url=response_url,
                                        reason="network response contains captcha/login/access-limit marker; body ignored",
                                        status_code=response.status,
                                    )
                                )
                                return
                            saved = self._save_raw_document(
                                source,
                                response_url,
                                body,
                                response_type,
                                response.status,
                                source_dir,
                                parent_url=url,
                                metadata={
                                    "discovered_by": "playwright_network",
                                    "network_content_type": response_type,
                                    "network_response_url": response_url,
                                },
                                filename_hint=filename_hint,
                                kind_override=response_kind,
                            )
                            if saved:
                                result.documents.append(saved)
                                result.discovered_urls.append(response_url)
                    except Exception:
                        return

                page.on("response", on_response)
                timeout_ms = int(settings.request_timeout_seconds * 1000)
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5000))
                except Exception:
                    pass
                body = page.content().encode("utf-8", errors="ignore")
                if self._looks_blocked(body, "text/html"):
                    result.errors.append(
                        FetchError(
                            source_id=source.id,
                            url=url,
                            reason="rendered page contains captcha/login/access-limit marker; body ignored",
                            status_code=200,
                        )
                    )
                    browser.close()
                    return result
                saved = self._save_raw_document(
                    source,
                    url,
                    body,
                    "text/html",
                    200,
                    source_dir,
                    metadata={"rendered_by": "playwright"},
                )
                if saved:
                    result.documents.append(saved)
                    result.discovered_urls.append(url)
                browser.close()
        except Exception as exc:
            result.errors.append(FetchError(source_id=source.id, url=url, reason=f"Playwright fetch failed: {exc}"))
        return result

    def _candidate_links(
        self,
        source: SourceConfig,
        base_url: str,
        soup: BeautifulSoup,
        cutoff_date: date | None = None,
    ) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen_urls: set[str] = set()
        base_host = urllib.parse.urlparse(base_url).netloc
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href", "")).strip()
            if href.startswith(("javascript:", "mailto:", "#")):
                continue
            url = urllib.parse.urljoin(base_url, href)
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != base_host:
                continue
            text = anchor.get_text(" ", strip=True)
            lowered = (url + " " + text).lower()
            is_attachment = self._is_attachment_url(source, url, text)
            keyword_hit = any(keyword.lower() in lowered for keyword in source.keywords)
            link_date = extract_date_hint(url, text)
            if cutoff_date and link_date and link_date < cutoff_date:
                continue
            if (is_attachment or keyword_hit) and url not in seen_urls:
                candidates.append((url, text))
                seen_urls.add(url)
        return sorted(
            candidates,
            key=lambda item: (
                0 if extract_date_hint(item[0], item[1]) else 1,
                -(extract_date_hint(item[0], item[1]) or date.min).toordinal(),
            ),
        )

    def _extract_public_links(self, doc: RawDocument, source: SourceConfig) -> list[str]:
        if doc.kind != "html":
            return []
        soup = BeautifulSoup(read_html_document(doc), "html.parser")
        return [url for url, _ in self._candidate_links(source, doc.url, soup)]

    def _is_attachment_url(self, source: SourceConfig, url: str, text: str = "") -> bool:
        lowered = urllib.parse.unquote(url).lower() + " " + text.lower()
        return any(ext in lowered for ext in source.attachment_types)

    def _domain_delay(self, url: str, delay: float) -> None:
        domain = urllib.parse.urlparse(url).netloc
        last = self.last_domain_access.get(domain)
        if last is not None:
            remaining = delay - (time.time() - last)
            if remaining > 0:
                time.sleep(remaining)
        self.last_domain_access[domain] = time.time()

    def _looks_blocked(self, body: bytes, content_type: str) -> bool:
        if content_type and content_type not in HTML_TYPES and content_type not in JSON_TYPES and not content_type.startswith("text/"):
            return False
        sample = decode_html(body[:5000]).lower()
        if content_type in JSON_TYPES or sample.lstrip().startswith(("{", "[")):
            try:
                sample += "\n" + json.dumps(json.loads(sample), ensure_ascii=False).lower()
            except Exception:
                pass
        return any(pattern.lower() in sample for pattern in BLOCK_PATTERNS)


def chromium_launch_options() -> dict[str, object]:
    options: dict[str, object] = {"headless": True}
    configured_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "").strip().strip('"')
    for candidate in (configured_path, *COMMON_CHROMIUM_EXECUTABLES):
        if candidate and Path(candidate).exists():
            options["executable_path"] = candidate
            break
    return options


def infer_kind(url: str, content_type: str, filename_hint: str = "") -> RawKind:
    parsed = urllib.parse.urlparse(url)
    url_candidate = parsed.path or parsed.netloc or url
    candidates = [
        urllib.parse.unquote(url_candidate).lower(),
        urllib.parse.unquote(filename_hint).lower(),
    ]
    if content_type in JSON_TYPES or has_suffix(candidates, ".json"):
        return "json"
    if content_type in {"text/csv", "application/csv"} or has_suffix(candidates, ".csv"):
        return "csv"
    if content_type in HTML_TYPES or has_suffix(candidates, (".html", ".htm", ".jhtml", "/")):
        return "html"
    if "pdf" in content_type or has_suffix(candidates, ".pdf"):
        return "pdf"
    if has_suffix(candidates, (".xlsx", ".xls")) or "spreadsheet" in content_type or "excel" in content_type:
        return "excel"
    if has_suffix(candidates, (".docx", ".doc")) or "word" in content_type:
        return "word"
    if content_type in TEXT_DOCUMENT_TYPES:
        return "text"
    return "binary"


def refine_kind_from_body(kind: RawKind, body: bytes) -> RawKind:
    sample = body[:4096].lstrip()
    if sample.startswith(b"%PDF"):
        return "pdf"
    if sample.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "excel"
    if sample.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(BytesIO(body)) as archive:
                names = set(archive.namelist())
        except Exception:
            return kind
        if any(name.startswith("xl/") for name in names):
            return "excel"
        if any(name.startswith("word/") for name in names):
            return "word"
    if kind in {"binary", "text"}:
        text = decode_html(sample[:1000]).lstrip().lower()
        if text.startswith(("<!doctype html", "<html")):
            return "html"
        if text.startswith(("{", "[")):
            return "json"
    return kind


def has_suffix(candidates: list[str], suffixes: str | tuple[str, ...]) -> bool:
    return any(candidate.endswith(suffixes) for candidate in candidates if candidate)


def kind_suffix(url: str, kind: RawKind, filename_hint: str = "") -> str:
    if filename_hint:
        hinted_suffix = Path(urllib.parse.unquote(filename_hint)).suffix
        if hinted_suffix:
            return hinted_suffix[:16]
    parsed = urllib.parse.urlparse(url)
    suffix = Path(urllib.parse.unquote(parsed.path)).suffix
    if suffix:
        return suffix[:16]
    guessed = mimetypes.guess_extension(kind) if "/" in kind else None
    if guessed:
        return guessed
    return {
        "html": ".html",
        "json": ".json",
        "csv": ".csv",
        "pdf": ".pdf",
        "excel": ".xlsx",
        "word": ".docx",
        "text": ".txt",
        "binary": ".bin",
    }[kind]


def _is_public_same_origin_or_api(page_url: str, response_url: str) -> bool:
    page = urllib.parse.urlparse(page_url)
    response = urllib.parse.urlparse(response_url)
    if response.scheme not in {"http", "https"}:
        return False
    if response.netloc == page.netloc:
        return True
    return is_trusted_public_host(response.netloc)


def is_trusted_public_host(host: str) -> bool:
    host = host.split(":", 1)[0].lower()
    trusted_suffixes = (
        "sh.gov.cn",
        "shanghai.gov.cn",
        "data.sh.gov.cn",
        "lingang.gov.cn",
    )
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in trusted_suffixes)


def build_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    legacy_option = getattr(ssl, "OP_LEGACY_SERVER_CONNECT", None)
    if legacy_option is not None:
        context.options |= legacy_option
    return context


def read_html_document(doc: RawDocument) -> str:
    return decode_html(doc.file_path.read_bytes())


def cutoff_date_from_days(days: int | None) -> date | None:
    if not days or days <= 0:
        return None
    return (datetime.now(UTC) - timedelta(days=days)).date()


def extract_date_hint(url: str, text: str = "") -> date | None:
    haystack = urllib.parse.unquote(f"{url} {text}")
    patterns = (
        r"(?<!\d)(20\d{2})[-_/年.](\d{1,2})[-_/月.](\d{1,2})(?:日)?(?!\d)",
        r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)",
    )
    for pattern in patterns:
        match = re.search(pattern, haystack)
        if not match:
            continue
        year, month, day = (int(part) for part in match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            continue
    return None


def filename_from_content_disposition(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"filename\*\s*=\s*([^;]+)", value, flags=re.IGNORECASE)
    if match:
        token = match.group(1).strip().strip('"')
        if "''" in token:
            token = token.split("''", 1)[1]
        return Path(urllib.parse.unquote(token)).name
    match = re.search(r"filename\s*=\s*([^;]+)", value, flags=re.IGNORECASE)
    if not match:
        return ""
    token = match.group(1).strip().strip('"')
    return Path(urllib.parse.unquote(token)).name
