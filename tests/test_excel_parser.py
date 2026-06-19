from io import BytesIO

import pandas as pd
from openpyxl import Workbook

from app.parsers.excel import parse_excel


def test_excel_parser_detects_header_and_rows():
    wb = Workbook()
    ws = wb.active
    ws.title = "重大项目"
    ws.append(["说明", "", ""])
    ws.append(["项目名称", "所在区", "总投资"])
    ws.append(["张江科学城研发平台", "浦东新区", "12亿元"])
    stream = BytesIO()
    wb.save(stream)
    parsed = parse_excel(stream.getvalue(), url="fixture://major.xlsx", source_id="fixture")
    assert parsed.rows[0]["项目名称"] == "张江科学城研发平台"
    assert "12亿元" in parsed.text
    assert parsed.metadata["row_count"] == 1


def test_legacy_xls_parser_uses_pandas_when_available(monkeypatch):
    def fake_read_excel(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "Sheet1": pd.DataFrame(
                [
                    ["说明", "", ""],
                    ["项目名称", "所在区", "总投资"],
                    ["临港产业项目", "浦东新区", "30亿元"],
                ]
            )
        }

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    parsed = parse_excel(b"legacy", url="fixture://major.xls", source_id="fixture")

    assert parsed.metadata["legacy_xls"] is True
    assert parsed.rows[0]["项目名称"] == "临港产业项目"
    assert "30亿元" in parsed.text


def test_legacy_xls_parser_detects_magic_bytes_without_suffix(monkeypatch):
    def fake_read_excel(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["engine"] == "xlrd"
        return {
            "Sheet1": pd.DataFrame(
                [
                    ["项目名称", "所在区"],
                    ["无后缀重大项目", "浦东新区"],
                ]
            )
        }

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    parsed = parse_excel(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1legacy",
        url="https://example.sh.gov.cn/download?id=123",
        source_id="fixture",
    )

    assert parsed.metadata["legacy_xls"] is True
    assert parsed.rows[0]["项目名称"] == "无后缀重大项目"
