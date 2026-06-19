from app.sources.discovery import (
    DiscoveryRun,
    SearchResult,
    SourceDiscoveryCandidate,
    SourceScoutAgent,
    parse_duckduckgo_html,
    rank_search_results,
)
from app.crawlers.models import FetchRunResult
from app.pipeline.orchestrator import LandScoutAgent
from app.sources.registry import SourceConfig, SourceRegistry


def test_source_discovery_ranks_official_candidates_and_skips_existing_hosts():
    results = [
        SearchResult(
            query="上海 规划公示 住宅 用地",
            title="浦东新区规划土地公示 住宅用地项目公告",
            url="https://www.pudong.gov.cn/zwgk/planning/notice.html",
            snippet="住宅 用地 地块 规划公示 重大项目",
        ),
        SearchResult(
            query="上海 规划公示 住宅 用地",
            title="搜索引擎缓存",
            url="https://www.baidu.com/link?x=1",
            snippet="住宅 用地",
        ),
        SearchResult(
            query="上海 规划公示 住宅 用地",
            title="已有来源",
            url="https://fgw.sh.gov.cn/fgw_zdjsxmqd/index.html",
            snippet="重大项目 清单",
        ),
    ]

    candidates = rank_search_results(
        results,
        max_sources=5,
        include_non_official=False,
        existing_urls={"https://fgw.sh.gov.cn/fgw_zdjsxmqd/index.html"},
        existing_hosts=set(),
    )

    assert len(candidates) == 1
    assert candidates[0].official is True
    assert candidates[0].host == "www.pudong.gov.cn"
    assert "住宅" in candidates[0].matched_keywords


def test_source_discovery_candidate_converts_to_source_config():
    result = SearchResult(
        query="上海 轨道交通 新建工程 规划 公示 官方",
        title="上海交通委轨道交通建设规划公示",
        url="https://jtw.sh.gov.cn/gsgg/example.html",
        snippet="轨道交通 新建工程 规划 公示",
    )

    candidate = rank_search_results(
        [result],
        max_sources=1,
        include_non_official=False,
        existing_urls=set(),
        existing_hosts=set(),
    )[0]
    source = candidate.to_source_config(priority=90)

    assert source.id.startswith("dynamic_")
    assert source.access_mode == "http_then_playwright"
    assert source.max_pages >= len(source.base_urls)
    assert ".pdf" in source.attachment_types
    assert "轨道交通" in source.keywords


def test_duckduckgo_html_parser_unwraps_result_urls():
    html = """
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.pudong.gov.cn%2Fnotice.html">浦东规划公示</a>
      <a class="result__snippet">住宅 用地 规划 公示</a>
    </div>
    """

    results = parse_duckduckgo_html(html, query="q", limit=5)

    assert results[0].url == "https://www.pudong.gov.cn/notice.html"
    assert results[0].title == "浦东规划公示"


def test_source_scout_uses_search_client_and_query_plan(monkeypatch):
    monkeypatch.setattr("app.sources.discovery.settings.openai_api_key", "")

    class FakeSearchClient:
        def search(self, query, *, limit):  # type: ignore[no-untyped-def]
            return [
                SearchResult(
                    query=query,
                    title="青浦区产业项目规划公示",
                    url="https://www.shqp.gov.cn/notice.html",
                    snippet="产业 项目 规划 公示 新城",
                )
            ]

    monkeypatch.setattr("app.sources.discovery.plan_discovery_queries", lambda use_llm: ["q"])

    run = SourceScoutAgent(search_client=FakeSearchClient()).discover(max_sources=3, query_limit=2)

    assert run.queries == ["q"]
    assert len(run.candidates) == 1
    assert run.candidates[0].host == "www.shqp.gov.cn"


def test_source_registry_merged_dedupes_dynamic_sources():
    existing = SourceConfig(
        id="existing",
        name="Existing",
        base_urls=["https://example.gov.cn/a.html"],
        priority=10,
    )
    duplicate = SourceConfig(
        id="dynamic_duplicate",
        name="Duplicate",
        base_urls=["https://example.gov.cn/a.html"],
    )
    fresh = SourceConfig(
        id="dynamic_fresh",
        name="Fresh",
        base_urls=["https://fresh.gov.cn/a.html"],
        priority=90,
    )

    merged = SourceRegistry([existing]).merged([duplicate, fresh])

    assert [source.id for source in merged.sources] == ["existing", "dynamic_fresh"]


def test_recommend_residential_injects_dynamic_sources(monkeypatch, tmp_path):
    static_source = SourceConfig(
        id="static",
        name="Static",
        base_urls=["https://static.gov.cn/index.html"],
        priority=10,
    )
    skipped_static_source = SourceConfig(
        id="skipped_static",
        name="Skipped Static",
        base_urls=["https://skipped.gov.cn/index.html"],
        priority=20,
    )
    dynamic_candidate = SourceDiscoveryCandidate(
        id="dynamic_test",
        name="Dynamic Test",
        host="dynamic.gov.cn",
        base_urls=["https://dynamic.gov.cn/notice.html"],
        score=60,
        official=True,
        query="q",
        sample_title="动态规划公示",
        matched_keywords=["规划公示"],
        reason="official host",
    )
    agent = LandScoutAgent(registry=SourceRegistry([static_source, skipped_static_source]))
    captured = {}

    monkeypatch.setattr("app.pipeline.orchestrator.run_data_dir", lambda run_id: tmp_path / "data" / run_id)
    monkeypatch.setattr("app.pipeline.orchestrator.run_output_dir", lambda run_id: tmp_path / "outputs" / run_id)
    monkeypatch.setattr("app.pipeline.orchestrator.save_state", lambda state: None)
    monkeypatch.setattr("app.pipeline.orchestrator.settings.openai_api_key", "test-key")

    def fake_discover(**kwargs):  # type: ignore[no-untyped-def]
        return DiscoveryRun(queries=["q"], candidates=[dynamic_candidate], errors=[])

    def fake_fetch(run_id, run_dir, source_limit=12, registry=None, days=None):  # type: ignore[no-untyped-def]
        captured["source_limit"] = source_limit
        captured["source_ids"] = [source.id for source in registry.sources]
        captured["days"] = days
        return FetchRunResult(run_id=run_id, visited_sources=captured["source_ids"])

    def fake_parse(**kwargs):  # type: ignore[no-untyped-def]
        return kwargs["state"]

    monkeypatch.setattr(agent, "_discover_live_sources", fake_discover)
    monkeypatch.setattr(agent, "_fetch_live", fake_fetch)
    monkeypatch.setattr(agent, "_parse_extract_score_render", fake_parse)

    state = agent.recommend_residential(
        live=True,
        days=540,
        top_k=8,
        source_limit=1,
        discover_sources=True,
        dynamic_source_limit=1,
    )

    assert captured["source_limit"] == 2
    assert captured["days"] == 540
    assert captured["source_ids"] == ["static", "dynamic_test"]
    assert state.discovered_sources[0]["id"] == "dynamic_test"
