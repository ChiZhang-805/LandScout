from __future__ import annotations

import re
from datetime import date


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})?\s*日?", text)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3) or "1"
        return _safe_date(year, month, day)
    match = re.search(r"(20\d{2})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?", text)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3) or "1"
        return _safe_date(year, month, day)
    match = re.search(r"(20\d{2})", text)
    if match:
        return f"{match.group(1)}-01-01"
    return None


def parse_amount_wanyuan(text: str | None) -> float | None:
    if not text:
        return None
    normalized = text.replace(",", "")
    patterns = [
        (r"([0-9]+(?:\.[0-9]+)?)\s*万\s*(?:亿元|亿)", 10000.0 * 10000.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:亿元|亿)", 10000.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:万元|万(?!\s*(?:平方米|平米|平方公里|亩|公顷|㎡|m2|m²)))", 1.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*元", 0.0001),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, normalized)
        if match:
            return round(float(match.group(1)) * multiplier, 4)
    return None


def parse_area_sqm(text: str | None) -> float | None:
    if not text:
        return None
    normalized = text.replace(",", "")
    patterns = [
        (r"([0-9]+(?:\.[0-9]+)?)\s*万\s*(?:平方米|平米|㎡|m2|m²)", 10000.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:平方公里|平方千米)", 1_000_000.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:平方米|平米|㎡|m2|m²)", 1.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*公顷", 10000.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*万亩", 666.6667 * 10000.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*亩", 666.6667),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, normalized)
        if match:
            return round(float(match.group(1)) * multiplier, 4)
    return None


def _safe_date(year: str, month: str, day: str) -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None
