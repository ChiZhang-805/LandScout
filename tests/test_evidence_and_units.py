import pytest

from app.llm.normalization import normalize_date, parse_amount_wanyuan, parse_area_sqm
from app.llm.schemas import Evidence, EventType, ExtractedEvidence, GovernmentEvent, GovernmentSignalExtraction
from app.llm.validation import quote_in_text, validate_evidence_quotes


def test_evidence_quote_validation():
    text = "张江科学城启动生物医药研发平台重大建设项目，总投资12亿元。"
    extraction = GovernmentSignalExtraction(
        document_classification="major_projects",
        events=[
            GovernmentEvent(
                source_id="fixture",
                source_url="fixture://x",
                event_type=EventType.MAJOR_PROJECT,
                title="张江科学城",
                evidence=[Evidence(source_id="fixture", url="fixture://x", quote="生物医药研发平台重大建设项目")],
            )
        ],
    )
    valid, errors = validate_evidence_quotes(extraction, text)
    assert valid
    assert errors == []


def test_blank_evidence_quotes_are_invalid():
    assert not quote_in_text("   ", "foo bar")
    assert not quote_in_text("\n\t", "foo bar")
    with pytest.raises(ValueError):
        Evidence(source_id="fixture", url="fixture://x", quote="   ")
    with pytest.raises(ValueError):
        ExtractedEvidence(quote="   ", confidence=0.9)


def test_unit_conversion_and_date_normalization():
    assert parse_amount_wanyuan("总投资12亿元") == 120000
    assert parse_amount_wanyuan("总投资1.5万亿元") == 150000000
    assert parse_amount_wanyuan("投资3000万元") == 3000
    assert parse_amount_wanyuan("投资3000万") == 3000
    assert parse_amount_wanyuan("新增建筑面积8万平方米") is None
    assert parse_amount_wanyuan("新增建筑面积8万㎡") is None
    assert parse_area_sqm("面积8万平方米") == 80000
    assert parse_area_sqm("面积8万㎡") == 80000
    assert parse_area_sqm("面积8万平米") == 80000
    assert parse_area_sqm("面积1.2平方公里") == 1200000
    assert parse_area_sqm("面积1.2平方千米") == 1200000
    assert parse_area_sqm("面积32000㎡") == 32000
    assert parse_area_sqm("面积1.5万亩") == 10000000.5
    assert parse_area_sqm("面积10亩") == 6666.667
    assert normalize_date("2026年3月5日") == "2026-03-05"
