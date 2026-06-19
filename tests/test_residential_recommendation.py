from pathlib import Path

from app.core.branding import PRODUCT_DISPLAY_NAME
from app.core.utils import read_json
from app.llm.client import LLMExtractor
from app.llm.schemas import Evidence, EventType, GovernmentEvent, GovernmentSignalExtraction
from app.pipeline.orchestrator import LandScoutAgent, make_run_id, save_state
from app.renderers.residential_report import build_residential_markdown
import app.scoring.residential as residential_scoring
from app.scoring.residential import ResidentialCandidateScore, score_residential_candidates
from app.scoring.candidates import seed_candidates


def test_residential_recommendation_generates_outputs():
    landscout_agent = LandScoutAgent()
    state = landscout_agent.recommend_residential(live=False, days=540, top_k=6)

    assert len(state.events) >= 20
    assert len(state.residential_scores) >= 6
    assert state.residential_scores[0].evidence_count >= 3
    assert state.residential_scores[0].recommendation in {"优先抢先跟踪", "谨慎抢先跟踪", "观察跟踪", "暂不优先"}
    assert state.residential_scores[0].next_action
    assert state.outputs is not None

    recommendation_path = Path(state.outputs.recommendation_md)
    signals_path = Path(state.outputs.signals_json)
    assert recommendation_path.exists()
    assert signals_path.exists()
    assert Path(state.outputs.visual_summary_html).exists()
    assert Path(state.outputs.investment_memo_md).exists()
    assert Path(state.outputs.monitoring_queries_json).exists()
    assert Path(state.outputs.embedding_index_json).exists()
    assert Path(state.outputs.batch_requests_jsonl).exists()
    assert Path(state.outputs.quality_review_json).exists()
    assert Path(state.outputs.crawler_hints_json).exists()
    recommendation_text = recommendation_path.read_text(encoding="utf-8")
    assert "上海住宅开发机会推荐报告" in recommendation_text
    assert "下一步动作" in recommendation_text
    visual_html = Path(state.outputs.visual_summary_html).read_text(encoding="utf-8")
    memo_text = Path(state.outputs.investment_memo_md).read_text(encoding="utf-8")
    assert "上海住宅开发机会可视化摘要" in visual_html
    assert "人口流入潜力" in visual_html
    assert "下一步动作" in visual_html
    assert "下一步动作分布" in visual_html
    assert PRODUCT_DISPLAY_NAME in visual_html
    assert "Agent 工作流与数据管线" in visual_html
    assert "### 6." in memo_text

    signals = read_json(signals_path)
    assert signals["product_name"] == PRODUCT_DISPLAY_NAME
    assert [item["agent"] for item in signals["agent_workflow"]][:2] == ["Source Scout", "Crawler Agent"]
    assert signals["analysis_type"] == "residential_development_recommendation"
    assert signals["candidate_count"] == len(state.residential_scores)
    assert signals["pipeline_metrics"]["evidence_event_count"] == len(state.events)
    assert signals["pipeline_metrics"]["parsed_document_count"] == len(state.parsed_documents)
    assert signals["pipeline_metrics"]["triaged_document_count"] == len(state.parsed_documents)
    assert signals["pipeline_metrics"]["selected_document_count"] >= 1
    assert signals["pipeline_metrics"]["skipped_document_count"] >= 0
    assert signals["event_type_counts"]
    assert signals["source_event_counts"] == {"fixture": len(state.events)}
    assert signals["ai_artifacts"]["investment_memo_md"] == state.outputs.investment_memo_md
    assert signals["ai_artifacts"]["api_enrichment"] is False
    assert signals["ai_artifacts"]["crawler_hints_json"] == state.outputs.crawler_hints_json
    assert signals["top_areas"][0]["residential_development_score"] > 0
    assert signals["top_areas"][0]["future_population_inflow_score"] > 0
    assert "pre_inflow_signal_score" in signals["top_areas"][0]
    assert "land_grab_window_score" in signals["top_areas"][0]
    assert signals["top_areas"][0]["next_action"]
    assert signals["next_action_counts"]
    assert sum(signals["next_action_counts"].values()) == len(signals["top_areas"])

    top_event_sets = [
        tuple(sorted(score["evidence_event_ids"]))
        for score in signals["top_areas"]
        if score["evidence_event_ids"]
    ]
    assert len(top_event_sets) == len(set(top_event_sets))
    top_scores = [score["residential_development_score"] for score in signals["top_areas"]]
    assert top_scores == sorted(top_scores, reverse=True)
    for score in signals["top_areas"]:
        assert score["area"]["evidence_event_ids"] == score["evidence_event_ids"]


def test_render_run_preserves_residential_analysis_type():
    landscout_agent = LandScoutAgent()
    state = landscout_agent.recommend_residential(live=False, days=540, top_k=5)

    rendered = landscout_agent.render_run(run_id=state.run_id, top_k=3)

    assert rendered.outputs is not None
    signals = read_json(Path(rendered.outputs.signals_json))
    assert signals["analysis_type"] == "residential_development_recommendation"
    assert len(signals["top_areas"]) == 3
    assert "residential_development_score" in signals["top_areas"][0]
    assert "future_population_inflow_score" in signals["top_areas"][0]


def test_score_run_recomputes_residential_scores_for_residential_run():
    landscout_agent = LandScoutAgent()
    state = landscout_agent.recommend_residential(live=False, days=540, top_k=5)
    state.residential_scores[0].residential_development_score = -1
    save_state(state)

    scored = landscout_agent.score_run(run_id=state.run_id, days=30)

    assert scored.residential_scores
    assert scored.residential_scores[0].residential_development_score >= 0


def test_run_id_has_subsecond_entropy():
    ids = [make_run_id() for _ in range(10)]
    assert len(ids) == len(set(ids))


def test_residential_report_does_not_recommend_without_evidence():
    scores = score_residential_candidates(seed_candidates(), [])
    markdown = build_residential_markdown("run", [score for score in scores if score.evidence_count > 0], len(scores), [])
    assert "当前公开证据不足" in markdown
    assert "首选抢先跟踪区域" not in markdown
    assert "下一步核查清单" in markdown


def test_residential_report_does_not_overrecommend_low_priority_scores():
    candidate = seed_candidates()[0]
    score = ResidentialCandidateScore(
        area=candidate,
        future_population_inflow_score=18,
        pre_inflow_signal_score=16,
        land_grab_window_score=12,
        demand_driver_score=10,
        transport_public_service_score=8,
        residential_land_access_score=6,
        market_entry_score=4,
        recency_score=20,
        evidence_confidence_score=70,
        residential_supply_pressure=25,
        maturity_penalty=20,
        geo_uncertainty_penalty=10,
        residential_development_score=24,
        opportunity_score=24,
        confidence=0.7,
        evidence_count=3,
        evidence_event_ids=[],
        recommendation="暂不优先",
        next_action="低优先级监测：仅保留政策和供地提醒，暂不投入拿地资源。",
        suggested_product="先做持续监测，暂不建议重资产拿地",
        key_reasons=["附近有少量公开信号，但住宅需求证据仍弱"],
        major_risks=["缺少明确住宅供地证据"],
    )

    markdown = build_residential_markdown("run", [score], 1, [])

    assert "当前没有达到抢先跟踪阈值的优先区域" in markdown
    assert "## 低优先级观察区域" in markdown
    assert "首选抢先跟踪区域" not in markdown
    assert "低优先级监测" in markdown
    assert "下一步核查清单" in markdown


def test_residential_total_score_uses_land_access_and_market_entry(monkeypatch):
    candidate = seed_candidates()[0]
    monkeypatch.setattr(residential_scoring, "residential_land_access_score", lambda counts: 80.0)
    monkeypatch.setattr(residential_scoring, "market_entry_score", lambda counts: 50.0)
    monkeypatch.setattr(residential_scoring, "geo_uncertainty_penalty", lambda events: 0.0)

    scores = residential_scoring.score_residential_candidates([candidate], [])

    assert scores[0].residential_land_access_score == 80.0
    assert scores[0].market_entry_score == 50.0
    assert scores[0].residential_development_score == 6.0
    assert scores[0].next_action.startswith("观察池")


def test_residential_recommendation_uses_evidence_backed_event_count():
    candidate = next(item for item in seed_candidates() if item.name == "张江")
    evidence_event = GovernmentEvent(
        source_id="fixture",
        source_url="fixture://x",
        event_type=EventType.INDUSTRIAL_PROJECT,
        title="张江科学城产业项目",
        lat=candidate.lat,
        lon=candidate.lon,
        evidence=[Evidence(source_id="fixture", url="fixture://x", quote="张江科学城产业项目")],
    )
    unevidenced_events = [
        GovernmentEvent(
            source_id="fixture",
            source_url="fixture://x",
            event_type=event_type,
            title=f"张江无证据信号 {idx}",
            lat=candidate.lat,
            lon=candidate.lon,
        )
        for idx, event_type in enumerate(
            [
                EventType.PLANNING_POLICY,
                EventType.PROJECT_APPROVAL,
                EventType.LAND_SUPPLY,
                EventType.INFRASTRUCTURE,
            ],
            start=1,
        )
    ]

    score = score_residential_candidates([candidate], [evidence_event, *unevidenced_events])[0]

    assert score.evidence_count == 1
    assert score.evidence_event_ids == [evidence_event.id]
    assert score.land_grab_window_score == 0
    assert score.residential_land_access_score == 0
    assert score.recommendation == "观察跟踪"
    assert score.next_action.startswith("观察池")


def test_residential_pipeline_renders_when_extraction_fails(monkeypatch):
    def fail_extract(self, document):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic extraction failure")

    monkeypatch.setattr(LLMExtractor, "extract", fail_extract)
    state = LandScoutAgent().recommend_residential(live=False, days=540, top_k=5)

    assert state.outputs is not None
    assert state.events == []
    assert state.errors
    signals = read_json(Path(state.outputs.signals_json))
    assert signals["top_areas"] == []
    assert "当前公开证据不足" in Path(state.outputs.recommendation_md).read_text(encoding="utf-8")


def test_pipeline_does_not_score_events_marked_needs_review(monkeypatch):
    def reviewed_extract(self, document):  # type: ignore[no-untyped-def]
        return GovernmentSignalExtraction(
            document_classification="test",
            events=[
                GovernmentEvent(
                    source_id=document.source_id,
                    source_url=document.url,
                    event_type=EventType.INFRASTRUCTURE,
                    title="reviewed event",
                    summary="should not score",
                    district="浦东新区",
                    evidence=[
                        Evidence(
                            source_id=document.source_id,
                            url=document.url,
                            quote=document.text[:20],
                            confidence=0.9,
                        )
                    ],
                    needs_review=True,
                    review_reason="synthetic review flag",
                )
            ],
        )

    monkeypatch.setattr(LLMExtractor, "extract", reviewed_extract)
    state = LandScoutAgent().recommend_residential(live=False, days=540, top_k=5)

    assert state.events == []
    assert all(score.evidence_count == 0 for score in state.residential_scores)
    signals = read_json(Path(state.outputs.signals_json))
    assert signals["top_areas"] == []
