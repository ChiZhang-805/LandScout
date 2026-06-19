from pathlib import Path
import sys
import types
from datetime import date

from bs4 import BeautifulSoup
import httpx

from app.crawlers.fetcher import (
    PublicFetcher,
    _is_public_same_origin_or_api,
    chromium_launch_options,
    filename_from_content_disposition,
    refine_kind_from_body,
    infer_kind,
    is_trusted_public_host,
    kind_suffix,
)
from app.crawlers.models import FetchError, RawDocument
from app.sources.registry import SourceConfig, SourceRegistry


def test_fetcher_does_not_save_http_error_pages(tmp_path, monkeypatch):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/missing.html"],
        keywords=["项目"],
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path)
    monkeypatch.setattr(fetcher.robots, "allowed", lambda url: True)
    monkeypatch.setattr(fetcher, "_domain_delay", lambda url, delay: None)
    monkeypatch.setattr(
        fetcher.client,
        "get",
        lambda url: httpx.Response(404, content=b"<html>not found</html>"),
    )

    try:
        result = fetcher.fetch_source("test", pages=1)
    finally:
        fetcher.close()

    assert result.documents == []
    assert len(result.errors) == 1
    assert isinstance(result.errors[0], FetchError)
    assert result.errors[0].status_code == 404


def test_fetcher_does_not_save_unsupported_binary_documents(tmp_path, monkeypatch):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/logo.png"],
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path)
    monkeypatch.setattr(fetcher.robots, "allowed", lambda url: True)
    monkeypatch.setattr(fetcher, "_domain_delay", lambda url, delay: None)
    monkeypatch.setattr(
        fetcher.client,
        "get",
        lambda url: httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG\r\n"),
    )

    try:
        result = fetcher.fetch_source("test", pages=1)
    finally:
        fetcher.close()

    assert result.documents == []
    assert len(result.errors) == 1
    assert "unsupported content kind" in result.errors[0].reason


def test_fetcher_uses_magic_bytes_for_octet_stream_documents(tmp_path, monkeypatch):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/download?id=1"],
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path)
    monkeypatch.setattr(fetcher.robots, "allowed", lambda url: True)
    monkeypatch.setattr(fetcher, "_domain_delay", lambda url, delay: None)
    monkeypatch.setattr(
        fetcher.client,
        "get",
        lambda url: httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"%PDF-1.7\nbody"),
    )

    try:
        result = fetcher.fetch_source("test", pages=1)
    finally:
        fetcher.close()

    assert result.errors == []
    assert len(result.documents) == 1
    assert result.documents[0].kind == "pdf"
    assert Path(result.documents[0].path).suffix == ".pdf"


def test_fetcher_uses_content_disposition_filename_for_kind_and_suffix():
    header = "attachment; filename*=UTF-8''%E9%87%8D%E5%A4%A7%E9%A1%B9%E7%9B%AE.xls"
    filename = filename_from_content_disposition(header)

    assert filename == "重大项目.xls"
    assert infer_kind("https://example.sh.gov.cn/download?id=1", "application/octet-stream", filename) == "excel"
    assert kind_suffix("https://example.sh.gov.cn/download?id=1", "excel", filename_hint=filename) == ".xls"


def test_infer_kind_detects_plain_file_name_without_content_type():
    assert infer_kind("sample_shanghai_signals.html", "") == "html"
    assert infer_kind("fixture://sample_shanghai_signals.html", "") == "html"
    assert infer_kind("projects.csv", "") == "csv"
    assert infer_kind("https://example.sh.gov.cn/site.css", "text/css") == "binary"
    assert infer_kind("https://example.sh.gov.cn/notice.txt", "text/plain") == "text"


def test_refine_kind_detects_json_in_text_response():
    assert refine_kind_from_body("text", b'{"records":[{"project":"land"}]}') == "json"
    assert refine_kind_from_body("binary", b' [ {"project":"land"} ] ') == "json"


def test_fetcher_saves_download_with_content_disposition_suffix(tmp_path):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/download?id=1"],
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path)
    saved = fetcher._save_raw_document(
        source,
        "https://example.sh.gov.cn/download?id=1",
        b"excel bytes",
        "application/octet-stream",
        200,
        tmp_path,
        filename_hint="重大项目.xls",
    )

    try:
        assert saved is not None
        assert saved.kind == "excel"
        assert saved.metadata["filename_hint"] == "重大项目.xls"
        assert Path(saved.path).suffix == ".xls"
    finally:
        fetcher.close()


def test_fetcher_extracts_keyword_links_from_gbk_html(tmp_path):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/index.html"],
        keywords=["重大项目"],
    )
    html_path = tmp_path / "index.html"
    html_path.write_bytes(
        """
        <html><head><meta charset="gbk"></head>
        <body><a href="/detail.html">重大项目清单</a></body></html>
        """.encode("gbk")
    )
    raw = RawDocument(
        id="raw",
        source_id="test",
        url="https://example.sh.gov.cn/index.html",
        fetched_at="2026-06-17T00:00:00Z",
        content_hash="hash",
        path=str(html_path),
        kind="html",
        status_code=200,
        content_type="text/html",
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path / "run")

    try:
        links = fetcher._extract_public_links(raw, source)
    finally:
        fetcher.close()

    assert links == ["https://example.sh.gov.cn/detail.html"]


def test_fetcher_detects_gbk_block_markers(tmp_path):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/index.html"],
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path)

    try:
        blocked = fetcher._looks_blocked("请输入验证码后访问".encode("gbk"), "text/html")
    finally:
        fetcher.close()

    assert blocked is True


def test_fetcher_detects_json_block_markers(tmp_path):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/api"],
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path)

    try:
        blocked = fetcher._looks_blocked(b'{"message":"\\u8bf7\\u767b\\u5f55\\u540e\\u8bbf\\u95ee"}', "application/json")
    finally:
        fetcher.close()

    assert blocked is True


def test_playwright_rendered_block_page_is_not_saved(tmp_path, monkeypatch):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/index.html"],
        access_mode="playwright_with_network_discovery",
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path)
    monkeypatch.setattr(fetcher.robots, "allowed", lambda url: True)
    monkeypatch.setattr(fetcher, "_domain_delay", lambda url, delay: None)

    class FakePage:
        def on(self, name, callback):
            return None

        def goto(self, url, wait_until, timeout):
            return None

        def content(self):
            return "<html><body>请输入验证码后访问</body></html>"

    class FakeBrowser:
        def __init__(self):
            self.closed = False

        def new_page(self, user_agent):
            return FakePage()

        def close(self):
            self.closed = True

    class FakeChromium:
        def launch(self, **kwargs):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = lambda: FakePlaywright()
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)

    try:
        result = fetcher._fetch_with_playwright(source, source.urls[0], tmp_path)
    finally:
        fetcher.close()

    assert result.documents == []
    assert len(result.errors) == 1
    assert "body ignored" in result.errors[0].reason


def test_playwright_saves_dom_content_even_if_networkidle_times_out(tmp_path, monkeypatch):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/index.html"],
        access_mode="playwright_with_network_discovery",
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path)
    monkeypatch.setattr(fetcher.robots, "allowed", lambda url: True)
    monkeypatch.setattr(fetcher, "_domain_delay", lambda url, delay: None)

    class FakePage:
        def __init__(self):
            self.goto_wait_until = ""
            self.waited_for_networkidle = False

        def on(self, name, callback):
            return None

        def goto(self, url, wait_until, timeout):
            self.goto_wait_until = wait_until
            return None

        def wait_for_load_state(self, state, timeout):
            self.waited_for_networkidle = state == "networkidle"
            raise TimeoutError("networkidle timeout")

        def content(self):
            return "<html><body><h1>重大项目公告</h1><p>临港产业项目和住宅配套。</p></body></html>"

    fake_page = FakePage()

    class FakeBrowser:
        def new_page(self, user_agent):
            return fake_page

        def close(self):
            return None

    class FakeChromium:
        def launch(self, **kwargs):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = lambda: FakePlaywright()
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)

    try:
        result = fetcher._fetch_with_playwright(source, source.urls[0], tmp_path)
    finally:
        fetcher.close()

    assert result.errors == []
    assert len(result.documents) == 1
    assert result.documents[0].kind == "html"
    assert fake_page.goto_wait_until == "domcontentloaded"
    assert fake_page.waited_for_networkidle is True
    assert "重大项目公告" in Path(result.documents[0].path).read_text(encoding="utf-8")


def test_chromium_launch_options_uses_configured_browser(tmp_path, monkeypatch):
    browser_path = tmp_path / "chrome.exe"
    browser_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", str(browser_path))

    options = chromium_launch_options()

    assert options["headless"] is True
    assert options["executable_path"] == str(browser_path)


def test_candidate_links_are_deduped_by_url(tmp_path):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/index.html"],
        keywords=["项目"],
    )
    html_path = tmp_path / "index.html"
    html_path.write_text(
        """
        <html><body>
          <a href="/detail.html">重大项目</a>
          <a href="/detail.html">项目详情</a>
        </body></html>
        """,
        encoding="utf-8",
    )
    raw = RawDocument(
        id="raw",
        source_id="test",
        url="https://example.sh.gov.cn/index.html",
        fetched_at="2026-06-17T00:00:00Z",
        content_hash="hash",
        path=str(html_path),
        kind="html",
        status_code=200,
        content_type="text/html",
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path / "run")

    try:
        links = fetcher._extract_public_links(raw, source)
    finally:
        fetcher.close()

    assert links == ["https://example.sh.gov.cn/detail.html"]


def test_candidate_links_skip_obviously_old_dated_urls(tmp_path):
    source = SourceConfig(
        id="test",
        name="Test Source",
        base_urls=["https://example.sh.gov.cn/index.html"],
        keywords=["project"],
    )
    soup = BeautifulSoup(
        """
        <html><body>
          <a href="/20250101/old.html">project old</a>
          <a href="/20260601/new.html">project new</a>
          <a href="/undated.html">project undated</a>
        </body></html>
        """,
        "html.parser",
    )
    fetcher = PublicFetcher(SourceRegistry([source]), run_id="run", run_dir=tmp_path / "run")

    try:
        links = fetcher._candidate_links(
            source,
            "https://example.sh.gov.cn/index.html",
            soup,
            cutoff_date=date(2026, 1, 1),
        )
    finally:
        fetcher.close()

    assert links == [
        ("https://example.sh.gov.cn/20260601/new.html", "project new"),
        ("https://example.sh.gov.cn/undated.html", "project undated"),
    ]


def test_public_host_matching_rejects_domain_spoofing():
    assert is_trusted_public_host("fgw.sh.gov.cn")
    assert is_trusted_public_host("data.sh.gov.cn")
    assert not is_trusted_public_host("sh.gov.cn.evil.example")
    assert not is_trusted_public_host("not-shanghai.example")
    assert _is_public_same_origin_or_api(
        "https://fgw.sh.gov.cn/page.html",
        "https://cdn.fgw.sh.gov.cn/api.json",
    )
    assert not _is_public_same_origin_or_api(
        "https://fgw.sh.gov.cn/page.html",
        "https://sh.gov.cn.evil.example/api.json",
    )
