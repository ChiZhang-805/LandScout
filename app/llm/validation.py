from __future__ import annotations

from app.llm.schemas import GovernmentSignalExtraction


def quote_in_text(quote: str, text: str) -> bool:
    quote = quote.strip()
    if not quote:
        return False
    if quote in text:
        return True
    compact_quote = "".join(quote.split())
    if not compact_quote:
        return False
    compact_text = "".join(text.split())
    return compact_quote in compact_text


def validate_evidence_quotes(extraction: GovernmentSignalExtraction, source_text: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for event in extraction.events:
        if not event.evidence:
            errors.append(f"{event.title}: missing evidence")
            continue
        for evidence in event.evidence:
            if not quote_in_text(evidence.quote, source_text):
                errors.append(f"{event.title}: quote not found in source text")
    return not errors, errors
