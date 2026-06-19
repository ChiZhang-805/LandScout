from app.llm.client import LLMExtractor, classify_event, dedupe_events, to_government_extraction
from app.llm.schemas import Evidence, EventType, GovernmentEvent, LLMGovernmentSignalExtraction
from app.parsers.models import ParsedDocument


def test_live_extraction_schema_is_strict_json_schema():
    schema = LLMGovernmentSignalExtraction.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["document_classification", "events", "review_notes"]
    assert schema["$defs"]["ExtractedGovernmentEvent"]["additionalProperties"] is False
    assert schema["$defs"]["ExtractedEvidence"]["additionalProperties"] is False
    assert schema["$defs"]["ExtractedGovernmentEvent"]["properties"]["evidence"]["minItems"] == 1
    assert schema["$defs"]["ExtractedEvidence"]["properties"]["quote"]["minLength"] == 1


def test_heuristic_extractor_method_is_available():
    assert callable(LLMExtractor(live=False, allow_heuristic=True)._extract_heuristic)


def test_llm_event_dates_are_normalized_before_scoring():
    extraction = LLMGovernmentSignalExtraction.model_validate(
        {
            "document_classification": "test",
            "events": [
                {
                    "event_type": EventType.INFRASTRUCTURE,
                    "title": "轨道交通项目",
                    "summary": "轨道交通项目",
                    "district": "浦东新区",
                    "address": None,
                    "project_name": None,
                    "event_date": "2026年6月17日",
                    "amount_wanyuan": None,
                    "area_sqm": None,
                    "evidence": [{"quote": "轨道交通项目", "confidence": 0.9}],
                    "tags": ["infrastructure"],
                }
            ],
            "review_notes": [],
        }
    )
    document = ParsedDocument(
        source_id="source",
        url="https://example.com",
        title="test",
        text="轨道交通项目",
        content_hash="hash",
        fetched_at="2026-06-17T00:00:00Z",
    )

    converted = to_government_extraction(extraction, document)

    assert converted.events[0].event_date == "2026-06-17"


def test_llm_event_date_falls_back_to_document_date():
    extraction = LLMGovernmentSignalExtraction.model_validate(
        {
            "document_classification": "test",
            "events": [
                {
                    "event_type": EventType.INFRASTRUCTURE,
                    "title": "轨道交通项目",
                    "summary": "轨道交通项目",
                    "district": "浦东新区",
                    "address": None,
                    "project_name": None,
                    "event_date": None,
                    "amount_wanyuan": None,
                    "area_sqm": None,
                    "evidence": [{"quote": "轨道交通项目", "confidence": 0.9}],
                    "tags": ["infrastructure"],
                }
            ],
            "review_notes": [],
        }
    )
    document = ParsedDocument(
        source_id="source",
        url="https://example.com",
        title="test",
        date="2026-05-20",
        text="轨道交通项目",
        content_hash="hash",
        fetched_at="2026-06-17T00:00:00Z",
    )

    converted = to_government_extraction(extraction, document)

    assert converted.events[0].event_date == "2026-05-20"


def test_llm_invalid_event_date_falls_back_to_document_date():
    extraction = LLMGovernmentSignalExtraction.model_validate(
        {
            "document_classification": "test",
            "events": [
                {
                    "event_type": EventType.INFRASTRUCTURE,
                    "title": "轨道交通项目",
                    "summary": "轨道交通项目",
                    "district": "浦东新区",
                    "address": None,
                    "project_name": None,
                    "event_date": "近期",
                    "amount_wanyuan": None,
                    "area_sqm": None,
                    "evidence": [{"quote": "轨道交通项目", "confidence": 0.9}],
                    "tags": ["infrastructure"],
                }
            ],
            "review_notes": [],
        }
    )
    document = ParsedDocument(
        source_id="source",
        url="https://example.com",
        title="test",
        date="2026-05-20",
        text="轨道交通项目",
        content_hash="hash",
        fetched_at="2026-06-17T00:00:00Z",
    )

    converted = to_government_extraction(extraction, document)

    assert converted.events[0].event_date == "2026-05-20"


def test_llm_conversion_fills_missing_units_from_evidence():
    extraction = LLMGovernmentSignalExtraction.model_validate(
        {
            "document_classification": "test",
            "events": [
                {
                    "event_type": EventType.INDUSTRIAL_PROJECT,
                    "title": "张江研发平台",
                    "summary": "张江研发平台",
                    "district": "浦东新区",
                    "address": None,
                    "project_name": None,
                    "event_date": "2026-06-17",
                    "amount_wanyuan": None,
                    "area_sqm": None,
                    "evidence": [{"quote": "张江研发平台总投资12亿元，新增建筑面积8万平方米", "confidence": 0.9}],
                    "tags": ["industrial_project"],
                }
            ],
            "review_notes": [],
        }
    )
    document = ParsedDocument(
        source_id="source",
        url="https://example.com",
        title="test",
        text="张江研发平台总投资12亿元，新增建筑面积8万平方米",
        content_hash="hash",
        fetched_at="2026-06-17T00:00:00Z",
    )

    converted = to_government_extraction(extraction, document)

    assert converted.events[0].amount_wanyuan == 120000
    assert converted.events[0].area_sqm == 80000


def test_heuristic_classification_preserves_early_and_land_signals():
    assert classify_event("张江科学城地块成交结果公示") == EventType.LAND_TRANSACTION
    assert classify_event("轨道交通建设项目获得批复") == EventType.PROJECT_APPROVAL
    assert classify_event("临港新片区国土空间规划发布") == EventType.PLANNING_POLICY
    assert classify_event("张江居住用地出让公告") == EventType.RESIDENTIAL_SUPPLY


def test_heuristic_extractor_skips_page_title_headings():
    document = ParsedDocument(
        source_id="fixture",
        url="fixture://x",
        title="上海公开项目信号测试样本",
        text=(
            "上海公开项目信号测试样本\n"
            "2026年上海公开项目信号测试样本\n"
            "2026年3月，张江科学城启动生物医药研发平台重大建设项目，总投资12亿元。"
        ),
        parser="html",
    )

    extraction = LLMExtractor(live=False, allow_heuristic=True).extract(document)

    assert len(extraction.events) == 1
    assert extraction.events[0].title == "张江科学城启动生物医药研发平台重大建设项目"


def test_dedupe_preserves_same_title_with_different_signal_types_or_evidence():
    base = {
        "source_id": "fixture",
        "source_url": "fixture://x",
        "title": "张江地块",
        "event_date": "2026-06-17",
    }
    land = GovernmentEvent(
        **base,
        event_type=EventType.LAND_SUPPLY,
        evidence=[Evidence(source_id="fixture", url="fixture://x", quote="张江地块出让公告")],
    )
    transaction = GovernmentEvent(
        **base,
        event_type=EventType.LAND_TRANSACTION,
        evidence=[Evidence(source_id="fixture", url="fixture://x", quote="张江地块成交结果")],
    )
    duplicate_land = GovernmentEvent(
        **base,
        event_type=EventType.LAND_SUPPLY,
        evidence=[Evidence(source_id="fixture", url="fixture://x", quote="张江地块出让公告")],
    )

    deduped = dedupe_events([land, transaction, duplicate_land])

    assert [event.event_type for event in deduped] == [EventType.LAND_SUPPLY, EventType.LAND_TRANSACTION]
