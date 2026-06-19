from app.parsers.html import parse_html


def test_html_parser_extracts_title_date_links_and_attachments():
    html = """
    <html><head><title>上海土地公告</title></head>
    <body><h1>土地出让</h1><p>发布日期：2026年3月5日 张江地块出让。</p>
    <a href="/files/a.xlsx">附件</a></body></html>
    """
    parsed = parse_html(html, url="https://example.sh.gov.cn/news/index.html", source_id="test")
    assert parsed.title == "上海土地公告"
    assert parsed.date == "2026-03-05"
    assert "张江地块出让" in parsed.text
    assert parsed.attachments[0]["url"] == "https://example.sh.gov.cn/files/a.xlsx"


def test_html_parser_decodes_gbk_pages():
    html = """
    <html><head><meta charset="gbk"><title>上海重大项目</title></head>
    <body><p>发布日期：2026年3月5日 临港产业项目签约</p></body></html>
    """.encode("gbk")

    parsed = parse_html(html, url="https://example.sh.gov.cn/news/index.html", source_id="test")

    assert parsed.title == "上海重大项目"
    assert parsed.date == "2026-03-05"
    assert "临港产业项目签约" in parsed.text
