import json

from app.parsers.json_parser import parse_json


def test_json_parser_extracts_structured_rows_from_api_payload():
    payload = {
        "data": {
            "records": [
                {"project": "Zhangjiang land", "date": "2026-06-01", "empty": ""},
                {"project": "Lingang housing", "date": "2026-06-02", "nested": {"ignored": True}},
            ]
        }
    }

    parsed = parse_json(
        json.dumps(payload).encode("utf-8"),
        url="https://example.sh.gov.cn/api/list",
        source_id="test",
    )

    assert parsed.parser == "json"
    assert len(parsed.rows) == 2
    assert parsed.rows[0] == {"project": "Zhangjiang land", "date": "2026-06-01"}
    assert "project: Zhangjiang land" in parsed.text
    assert parsed.metadata["row_count"] == 2


def test_json_parser_extracts_single_detail_record():
    payload = {"data": {"title": "Planning notice", "date": "2026-06-03", "content": "land supply"}}

    parsed = parse_json(
        json.dumps(payload).encode("utf-8"),
        url="https://example.sh.gov.cn/api/detail",
        source_id="test",
    )

    assert parsed.rows == [{"title": "Planning notice", "date": "2026-06-03", "content": "land supply"}]


def test_json_parser_walks_single_wrapper_inside_list():
    payload = [{"records": [{"title": "Land notice", "date": "2026-06-04"}]}]

    parsed = parse_json(
        json.dumps(payload).encode("utf-8"),
        url="https://example.sh.gov.cn/api/wrapped",
        source_id="test",
    )

    assert parsed.rows == [{"title": "Land notice", "date": "2026-06-04"}]
