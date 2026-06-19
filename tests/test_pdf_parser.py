from app.parsers.pdf import parse_pdf


def test_pdf_parser_extracts_text():
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Zhangjiang major construction project investment 12 billion RMB")
    data = doc.tobytes()
    doc.close()

    parsed = parse_pdf(data, url="fixture://sample.pdf", source_id="fixture")
    assert "Zhangjiang major construction project" in parsed.text
    assert parsed.metadata["page_count"] == 1
