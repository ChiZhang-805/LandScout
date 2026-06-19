from app.core.utils import content_hash
from app.crawlers.fetcher import infer_kind
from app.crawlers.models import RawDocument
from app.parsers.csv_parser import parse_csv
from app.parsers.dispatcher import parse_raw_document


def test_csv_parser_preserves_rows_and_text():
    parsed = parse_csv(
        "项目名称,所在区,总投资\n临港产业项目,浦东新区,30亿元\n".encode("utf-8"),
        url="fixture://projects.csv",
        source_id="fixture",
    )

    assert parsed.parser == "csv"
    assert parsed.rows[0]["项目名称"] == "临港产业项目"
    assert "30亿元" in parsed.text
    assert parsed.metadata["row_count"] == 1


def test_dispatcher_parses_csv_raw_document(tmp_path):
    data = "项目名称,所在区\n张江研发平台,浦东新区\n".encode("utf-8")
    path = tmp_path / "projects.csv"
    path.write_bytes(data)
    raw = RawDocument(
        id="raw",
        source_id="fixture",
        url="https://example.sh.gov.cn/projects.csv",
        fetched_at="2026-06-17T00:00:00Z",
        content_hash=content_hash(data),
        path=str(path),
        kind=infer_kind(str(path), "text/csv"),
        status_code=200,
        content_type="text/csv",
    )

    parsed = parse_raw_document(raw)

    assert raw.kind == "csv"
    assert parsed.parser == "csv"
    assert parsed.rows[0]["项目名称"] == "张江研发平台"


def test_csv_parser_strips_utf8_bom_from_header():
    parsed = parse_csv(
        "\ufeff项目名称,所在区\n东方枢纽项目,浦东新区\n",
        url="fixture://projects.csv",
        source_id="fixture",
    )

    assert parsed.rows[0]["项目名称"] == "东方枢纽项目"
    assert "\ufeff项目名称" not in parsed.rows[0]
