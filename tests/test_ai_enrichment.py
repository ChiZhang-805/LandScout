import json

from app.llm.batch import documents_selected_for_extraction, submit_batch_request_file, write_extraction_batch_requests
from app.llm.crawler_hints import CrawlerHintPlan, generate_crawler_hints, write_crawler_hints
from app.llm.document_filter import DocumentRelevance, classify_document_heuristic
from app.llm.embeddings import build_embedding_records, embedding_status, write_embedding_index
from app.llm.investment_memo import InvestmentMemo, generate_investment_memo
from app.llm.monitoring import MonitoringPlan, generate_monitoring_queries
from app.llm.quality import QualityReview, generate_quality_review
from app.llm.schemas import Evidence, EventType, GovernmentEvent
from app.parsers.models import ParsedDocument
from app.scoring.candidates import CandidateArea
from app.scoring.residential import ResidentialCandidateScore
from app.sources.discovery import SourceCandidateAssessmentList


def make_document(text: str = "张江重大项目和住宅用地规划公示") -> ParsedDocument:
    return ParsedDocument(
        source_id="fixture",
        url="fixture://doc",
        title="测试文档",
        text=text,
        content_hash="hash",
        parser="html",
    )


def make_event() -> GovernmentEvent:
    return GovernmentEvent(
        id="event1",
        source_id="fixture",
        source_url="fixture://doc",
        event_type=EventType.MAJOR_PROJECT,
        title="张江重大项目",
        summary="产业导入",
        evidence=[Evidence(source_id="fixture", url="fixture://doc", quote="张江重大项目")],
    )


def make_score() -> ResidentialCandidateScore:
    return ResidentialCandidateScore(
        area=CandidateArea(
            id="seed_1",
            name="张江",
            lat=31.2,
            lon=121.6,
            description="测试片区",
        ),
        future_population_inflow_score=60,
        pre_inflow_signal_score=70,
        land_grab_window_score=50,
        demand_driver_score=40,
        transport_public_service_score=30,
        residential_land_access_score=20,
        market_entry_score=10,
        recency_score=80,
        evidence_confidence_score=90,
        residential_supply_pressure=0,
        maturity_penalty=0,
        geo_uncertainty_penalty=0,
        residential_development_score=55,
        opportunity_score=55,
        confidence=0.9,
        evidence_count=3,
        evidence_event_ids=["event1"],
        recommendation="优先抢先跟踪",
        suggested_product="人才租赁住房",
        key_reasons=["产业导入信号 1 条"],
        major_risks=["需要人工尽调"],
    )


def test_document_relevance_heuristic_keeps_real_estate_signals():
    relevant = classify_document_heuristic(make_document())
    irrelevant = classify_document_heuristic(make_document("普通会议通知"))

    assert relevant.should_extract is True
    assert relevant.relevance_score > irrelevant.relevance_score
    assert irrelevant.should_extract is False


def test_openai_strict_output_schemas_require_all_properties():
    for model in (
        DocumentRelevance,
        InvestmentMemo,
        MonitoringPlan,
        QualityReview,
        CrawlerHintPlan,
        SourceCandidateAssessmentList,
    ):
        schema = model.model_json_schema()
        assert set(schema["required"]) == set(schema["properties"])
        for definition in schema.get("$defs", {}).values():
            if definition.get("type") == "object":
                assert definition.get("additionalProperties") is False
                assert set(definition["required"]) == set(definition["properties"])


def test_embedding_index_writes_records_without_openai(tmp_path):
    path = tmp_path / "embedding_index.json"

    write_embedding_index(path, [make_document()], [make_event()], use_openai=False)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["record_count"] == 2
    assert payload["records"][0]["embedding"] is None
    assert {record.kind for record in build_embedding_records([make_document()], [make_event()])} == {"document", "event"}


def test_embedding_index_records_api_error_without_failing(tmp_path, monkeypatch):
    path = tmp_path / "embedding_index.json"

    monkeypatch.setattr("app.llm.embeddings.settings.openai_api_key", "test-key")

    def fail_embed(records):  # type: ignore[no-untyped-def]
        raise RuntimeError("embedding down")

    monkeypatch.setattr("app.llm.embeddings.embed_records", fail_embed)

    write_embedding_index(path, [make_document()], [make_event()], use_openai=True)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["embedding_status"] == "not_generated"
    assert "embedding down" in payload["embedding_error"]


def test_embedding_status_distinguishes_partial_vectors():
    records = build_embedding_records([make_document()], [make_event()])
    records[0].embedding = [1.0, 0.0]

    assert embedding_status([]) == "no_records"
    assert embedding_status(records) == "partial"
    records[1].embedding = [0.0, 1.0]
    assert embedding_status(records) == "complete"


def test_batch_request_export_uses_responses_endpoint(tmp_path):
    path = tmp_path / "batch_requests.jsonl"

    write_extraction_batch_requests(path, [make_document()], model="gpt-test")
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert line["url"] == "/v1/responses"
    assert line["body"]["model"] == "gpt-test"
    assert line["body"]["text"]["format"]["strict"] is True


def test_batch_request_export_skips_triage_rejected_documents(tmp_path):
    selected = make_document("张江重大项目")
    skipped = make_document("普通会议通知")
    skipped.metadata["relevance"] = {"should_extract": False, "reason": "not relevant"}
    path = tmp_path / "batch_requests.jsonl"

    write_extraction_batch_requests(path, [selected, skipped], model="gpt-test")
    lines = path.read_text(encoding="utf-8").splitlines()

    assert documents_selected_for_extraction([selected, skipped]) == [selected]
    assert len(lines) == 1
    assert "张江重大项目" in lines[0]


def test_submit_batch_rejects_empty_request_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    try:
        submit_batch_request_file(path)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty batch request file should be rejected before API upload")


def test_enrichment_agents_have_heuristic_fallbacks():
    score = make_score()
    event = make_event()

    memo = generate_investment_memo([score], [event], use_openai=False)
    monitoring = generate_monitoring_queries([score], [event], use_openai=False)
    review = generate_quality_review([score], [event], use_openai=False)

    assert memo.area_memos[0].area_name == "张江"
    assert "投拓预筛" in memo.executive_summary
    assert monitoring.queries
    assert monitoring.queries[0].cadence == "每周"
    assert review.overall_score > 0
    assert "公开信号" in review.residual_risks[0]


def test_crawler_hints_generate_source_level_strategy(tmp_path):
    document = make_document()
    document.url = "https://example.gov.cn/index.html"
    document.links = [{"url": f"https://example.gov.cn/detail-{idx}.html", "text": "规划公示"} for idx in range(12)]
    document.attachments = [{"url": "https://example.gov.cn/a.pdf", "text": "附件"}]
    document.metadata["relevance"] = {"should_extract": True, "categories": ["规划公示"]}
    path = tmp_path / "crawler_hints.json"

    write_crawler_hints(path, parsed_documents=[document], use_openai=False)
    plan = generate_crawler_hints([document], use_openai=False)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["hints"][0]["source_id"] == "fixture"
    assert plan.hints[0].likely_list_pages
    assert plan.hints[0].likely_attachment_urls == ["https://example.gov.cn/a.pdf"]
