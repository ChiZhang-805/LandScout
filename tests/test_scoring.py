from app.geo.geocoder import Geocoder
from app.llm.schemas import Evidence, EventType, GovernmentEvent
from app.scoring.candidates import data_driven_candidates, generate_candidates
from app.scoring.scorer import count_by_type, development_direction, residential_keyword_hits, score_candidates


def make_event(title: str, event_type: EventType) -> GovernmentEvent:
    return GovernmentEvent(
        source_id="fixture",
        source_url="fixture://x",
        event_type=event_type,
        title=title,
        address="张江科学城",
        event_date="2026-03-01",
        evidence=[Evidence(source_id="fixture", url="fixture://x", quote=title, confidence=0.8)],
    )


def test_scoring_requires_evidence_and_ranks_candidate():
    events = [
        make_event("张江科学城产业项目", EventType.INDUSTRIAL_PROJECT),
        make_event("张江科学城轨道交通项目", EventType.INFRASTRUCTURE),
        make_event("张江科学城地块出让", EventType.LAND_SUPPLY),
        make_event("张江科学城学校项目", EventType.PUBLIC_SERVICE),
    ]
    events = Geocoder(amap_key="").geocode_many(events)
    scores = score_candidates(generate_candidates(events), events, days=540)
    assert scores[0].evidence_count >= 3
    assert scores[0].opportunity_score > 0
    assert "张江" in scores[0].area.name


def test_scoring_evidence_event_ids_only_include_evidence_backed_events():
    with_evidence = make_event("张江科学城产业项目", EventType.INDUSTRIAL_PROJECT)
    without_evidence = GovernmentEvent(
        source_id="fixture",
        source_url="fixture://x",
        event_type=EventType.INFRASTRUCTURE,
        title="张江科学城轨道交通项目",
        address="张江科学城",
        event_date="2026-03-01",
    )
    events = Geocoder(amap_key="").geocode_many([with_evidence, without_evidence])

    scores = score_candidates(generate_candidates(events), events, days=540)
    zhangjiang = next(score for score in scores if "张江" in score.area.name)

    assert zhangjiang.evidence_count == 1
    assert zhangjiang.evidence_event_ids == [with_evidence.id]
    assert zhangjiang.area.evidence_event_ids == [with_evidence.id]
    assert zhangjiang.residential_supply_risk == 0


def test_generate_candidates_only_tracks_evidence_backed_event_ids():
    with_evidence = make_event("张江科学城产业项目", EventType.INDUSTRIAL_PROJECT)
    without_evidence = GovernmentEvent(
        source_id="fixture",
        source_url="fixture://x",
        event_type=EventType.INFRASTRUCTURE,
        title="张江科学城轨道交通项目",
        address="张江科学城",
        event_date="2026-03-01",
    )
    events = Geocoder(amap_key="").geocode_many([with_evidence, without_evidence])

    candidate = next(item for item in generate_candidates(events) if "张江" in item.name)

    assert candidate.evidence_event_ids == [with_evidence.id]


def test_data_driven_candidates_ignore_low_confidence_district_fallbacks():
    low_confidence_events = [
        GovernmentEvent(
            source_id="fixture",
            source_url="fixture://x",
            event_type=EventType.MAJOR_PROJECT,
            title=f"broad district event {idx}",
            lat=31.2215,
            lon=121.5441,
            geo_confidence=0.45,
            evidence=[Evidence(source_id="fixture", url="fixture://x", quote=f"quote {idx}")],
        )
        for idx in range(3)
    ]
    precise_events = [
        GovernmentEvent(
            source_id="fixture",
            source_url="fixture://x",
            event_type=EventType.MAJOR_PROJECT,
            title=f"precise event {idx}",
            lat=31.2077 + idx * 0.001,
            lon=121.5999 + idx * 0.001,
            geo_confidence=0.75,
            evidence=[Evidence(source_id="fixture", url="fixture://x", quote=f"quote {idx}")],
        )
        for idx in range(3)
    ]

    assert data_driven_candidates(low_confidence_events) == []
    candidates = data_driven_candidates(precise_events)
    assert len(candidates) == 1
    assert "张江" in candidates[0].name
    assert "数据聚类区域" not in candidates[0].name


def test_empty_counts_do_not_default_to_industrial_direction():
    assert development_direction(count_by_type([])) == "谨慎跟踪，等待更多公开证据"


def test_residential_keyword_hits_checks_title_and_summary():
    events = [
        GovernmentEvent(
            source_id="fixture",
            source_url="fixture://x",
            event_type=EventType.LAND_SUPPLY,
            title="普通地块公告",
            summary="涉及住宅用地和社区配套",
            evidence=[Evidence(source_id="fixture", url="fixture://x", quote="住宅用地")],
        ),
        GovernmentEvent(
            source_id="fixture",
            source_url="fixture://x",
            event_type=EventType.LAND_SUPPLY,
            title="商品住房供应计划",
            summary="普通摘要",
            evidence=[Evidence(source_id="fixture", url="fixture://x", quote="商品住房")],
        ),
    ]

    assert residential_keyword_hits(events) == 2
