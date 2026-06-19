import csv
from pathlib import Path

from app.core.utils import read_json
from app.llm.schemas import Evidence, EventType, GovernmentEvent
from app.pipeline.orchestrator import LandScoutAgent, ShanghaiSignalPipeline
from app.pipeline.orchestrator import load_state
from app.core.config import settings
from app.renderers.report import render_outputs, simple_map_html, update_latest, write_events_csv
from app.scoring.candidates import CandidateArea
from app.scoring.candidates import seed_candidates
from app.scoring.scorer import CandidateScore
from app.scoring.scorer import score_candidates


def test_fixture_pipeline_generates_outputs():
    landscout_agent = LandScoutAgent()
    state = landscout_agent.run_landscout_demo(live=False, days=540, top_k=8)
    assert len(state.events) >= 20
    assert len(state.scores) >= 3
    assert state.outputs is not None
    assert Path(state.outputs.recommendation_md).exists()
    assert Path(state.outputs.opportunity_map_html).exists()
    assert Path(state.outputs.visual_summary_html).exists()
    assert Path(state.outputs.events_csv).exists()
    assert state.scores[0].evidence_count >= 3
    assert "可视化摘要" in Path(state.outputs.visual_summary_html).read_text(encoding="utf-8")
    signals = read_json(Path(state.outputs.signals_json))
    assert signals["pipeline_metrics"]["evidence_event_count"] == len(state.events)
    assert signals["event_type_counts"]
    assert signals["source_event_counts"] == {"fixture": len(state.events)}
    evidence_pack = read_json(Path(state.outputs.evidence_pack_json))
    assert evidence_pack["parsed_documents"][0]["row_count"] == 0
    assert "metadata" in evidence_pack["parsed_documents"][0]


def test_legacy_pipeline_name_points_to_landscout_agent():
    assert ShanghaiSignalPipeline is LandScoutAgent


def test_generic_report_does_not_recommend_without_evidence(tmp_path):
    scores = score_candidates(seed_candidates(), [], days=540)
    outputs = render_outputs(
        run_id="run",
        output_dir=tmp_path / "outputs",
        events=[],
        scores=scores,
        raw_documents=[],
        parsed_documents=[],
        errors=[],
        top_k=8,
        visited_sources=[],
    )

    signals = read_json(Path(outputs.signals_json))
    markdown = Path(outputs.recommendation_md).read_text(encoding="utf-8")
    assert signals["top_areas"] == []
    assert "当前公开证据不足" in markdown
    assert "### 1." not in markdown
    assert Path(outputs.visual_summary_html).exists()


def test_generic_report_outputs_honor_top_k(tmp_path):
    events = [
        GovernmentEvent(
            id=f"event{idx}",
            source_id="fixture",
            source_url=f"fixture://event{idx}",
            event_type=EventType.MAJOR_PROJECT,
            title=f"项目 {idx}",
            evidence=[Evidence(source_id="fixture", url=f"fixture://event{idx}", quote=f"项目 {idx}")],
        )
        for idx in range(6)
    ]
    scores = [
        CandidateScore(
            area=CandidateArea(
                id=f"area{idx}",
                name=f"区域 {idx}",
                lat=31.0 + idx * 0.01,
                lon=121.0 + idx * 0.01,
                description="测试区域",
            ),
            industrial_import_score=10,
            infrastructure_score=10,
            public_service_score=10,
            land_structure_score=10,
            residential_supply_risk=0,
            market_entry_score=10,
            recency_score=10,
            evidence_confidence_score=80,
            geo_uncertainty_penalty=0,
            opportunity_score=50 - idx,
            confidence=0.8,
            evidence_count=1,
            evidence_event_ids=[events[idx].id],
            key_reasons=["产业导入信号增强"],
            major_risks=["需要核查库存去化"],
        )
        for idx in range(6)
    ]

    outputs = render_outputs(
        run_id="run",
        output_dir=tmp_path / "outputs",
        events=events,
        scores=scores,
        raw_documents=[],
        parsed_documents=[],
        errors=[],
        top_k=6,
        visited_sources=[],
    )

    signals = read_json(Path(outputs.signals_json))
    markdown = Path(outputs.recommendation_md).read_text(encoding="utf-8")
    visual = Path(outputs.visual_summary_html).read_text(encoding="utf-8")
    assert len(signals["top_areas"]) == 6
    assert "### 6. 区域 5" in markdown
    assert "展示 Top 6 区域" in visual
    assert "重大项目" in visual
    assert "产业导入信号增强" in visual
    assert "需要核查库存去化" in visual
    assert "letter-spacing:0" in visual
    assert "下一步动作分布" not in visual


def test_events_csv_preserves_zero_values(tmp_path):
    path = tmp_path / "events.csv"
    write_events_csv(
        path,
        [
            GovernmentEvent(
                source_id="fixture",
                source_url="fixture://x",
                event_type=EventType.LAND_SUPPLY,
                title="零值测试",
                amount_wanyuan=0,
                area_sqm=0,
                lat=0,
                lon=0,
            )
        ],
    )

    with path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["title"] == "零值测试"
    assert row["amount_wanyuan"] == "0.0"
    assert row["area_sqm"] == "0.0"
    assert row["lat"] == "0.0"
    assert row["lon"] == "0.0"


def test_map_fallback_only_shows_evidence_backed_events():
    evidenced = GovernmentEvent(
        source_id="fixture",
        source_url="fixture://verified",
        event_type=EventType.INDUSTRIAL_PROJECT,
        title="verified project",
        lat=31.2,
        lon=121.4,
        evidence=[Evidence(source_id="fixture", url="fixture://verified", quote="verified project")],
    )
    unevidenced = GovernmentEvent(
        source_id="fixture",
        source_url="fixture://unverified",
        event_type=EventType.INFRASTRUCTURE,
        title="unverified project",
        lat=31.3,
        lon=121.5,
    )

    html = simple_map_html([evidenced, unevidenced], [])

    assert "verified project" in html
    assert "unverified project" not in html


def test_load_state_rejects_unsafe_run_id_and_does_not_create_missing_dir():
    missing_run_id = "20260617T000000000000Z_abcdef12"
    missing_dir = settings.data_dir / "runs" / missing_run_id

    try:
        load_state(missing_run_id)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing run should raise FileNotFoundError")

    assert not missing_dir.exists()

    try:
        load_state("../outside")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe run_id should be rejected")


def test_update_latest_is_noop_when_source_is_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "outputs_dir", tmp_path / "outputs")
    latest_dir = settings.outputs_dir / "shanghai" / "latest"
    latest_dir.mkdir(parents=True)
    marker = latest_dir / "signals.json"
    marker.write_text("{}", encoding="utf-8")

    result = update_latest(latest_dir)

    assert result == latest_dir.resolve()
    assert marker.exists()


def test_update_latest_replaces_file_target(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "outputs_dir", tmp_path / "outputs")
    source_dir = tmp_path / "run"
    source_dir.mkdir()
    (source_dir / "signals.json").write_text("{}", encoding="utf-8")
    latest_path = settings.outputs_dir / "shanghai" / "latest"
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text("stale", encoding="utf-8")

    update_latest(source_dir)

    assert latest_path.is_dir()
    assert (latest_path / "signals.json").exists()
