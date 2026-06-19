from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.core.config import settings
from app.core.utils import short_text
from app.llm.normalization import normalize_date, parse_amount_wanyuan, parse_area_sqm
from app.llm.openai_client import build_openai_client
from app.llm.schemas import (
    Evidence,
    EventType,
    GovernmentEvent,
    GovernmentSignalExtraction,
    LLMGovernmentSignalExtraction,
)
from app.llm.validation import quote_in_text, validate_evidence_quotes
from app.parsers.models import ParsedDocument


class MissingLLMKey(RuntimeError):
    pass


@dataclass
class LLMExtractor:
    live: bool = True
    allow_heuristic: bool = False

    def extract(self, document: ParsedDocument) -> GovernmentSignalExtraction:
        if self.live:
            if not settings.openai_api_key:
                raise MissingLLMKey(
                    "OPENAI_API_KEY is required for live LLM extraction. "
                    "Set it in .env or run fixture tests without --live."
                )
            return self._extract_openai(document)
        if not self.allow_heuristic:
            raise MissingLLMKey("Non-live extraction requires allow_heuristic=True.")
        return self._extract_heuristic(document)

    def _extract_openai(self, document: ParsedDocument) -> GovernmentSignalExtraction:
        client = build_openai_client()
        schema = LLMGovernmentSignalExtraction.model_json_schema()
        prompt = build_extraction_prompt(document)
        last_errors: list[str] = []
        for attempt in range(2):
            if last_errors:
                prompt += "\n\nValidation errors to fix: " + "; ".join(last_errors)
            response = client.responses.create(
                model=settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Extract only factual, source-backed Shanghai government real-estate signals. "
                            "Every evidence.quote must be an exact substring from the supplied document. "
                            "Normalize money to 万元, area to square meters, and dates to ISO. "
                            "Use null for unknown optional values, [] for empty arrays, and include every field."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "GovernmentSignalExtraction",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            data = json.loads(response.output_text)
            llm_extraction = LLMGovernmentSignalExtraction.model_validate(data)
            extraction = to_government_extraction(llm_extraction, document)
            valid, errors = validate_evidence_quotes(extraction, document.text)
            if valid:
                return extraction
            last_errors = errors
        for event in extraction.events:
            if not event_has_valid_evidence(event, document.text):
                event.needs_review = True
                event.review_reason = "; ".join(last_errors[:3])
        extraction.review_notes.extend(last_errors)
        extraction.events = [event for event in extraction.events if not event.needs_review]
        return extraction

    def _extract_heuristic(self, document: ParsedDocument) -> GovernmentSignalExtraction:
        candidates = row_candidates(document) if document.rows else line_candidates(document.text)
        events: list[GovernmentEvent] = []
        for item in candidates:
            if is_low_information_heading(item, document):
                continue
            event_type = classify_event(item)
            if event_type == EventType.OTHER:
                continue
            quote = short_text(item, 180)
            if quote and quote not in document.text:
                quote = find_subquote(quote, document.text)
            if not quote:
                continue
            event = GovernmentEvent(
                source_id=document.source_id,
                source_url=document.url,
                event_type=event_type,
                title=short_text(extract_title(item), 90),
                summary=short_text(item, 240),
                district=extract_district(item),
                address=extract_address(item),
                project_name=extract_project_name(item),
                event_date=normalize_date(item) or document.date,
                amount_wanyuan=parse_amount_wanyuan(item),
                area_sqm=parse_area_sqm(item),
                evidence=[
                    Evidence(
                        source_id=document.source_id,
                        url=document.url,
                        quote=quote,
                        fetched_at=document.fetched_at,
                        content_hash=document.content_hash,
                        confidence=0.72 if document.parser != "html" else 0.68,
                    )
                ],
                tags=tags_for_event(item, event_type),
                raw_doc_hash=document.content_hash,
            )
            events.append(event)
        extraction = GovernmentSignalExtraction(
            document_classification=classify_document(document),
            events=dedupe_events(events),
        )
        valid, errors = validate_evidence_quotes(extraction, document.text)
        if not valid:
            extraction.review_notes.extend(errors)
            for event in extraction.events:
                event.needs_review = True
        return extraction


def build_extraction_prompt(document: ParsedDocument) -> str:
    return (
        f"source_id: {document.source_id}\n"
        f"url: {document.url}\n"
        f"title: {document.title}\n"
        f"date: {document.date}\n\n"
        "Document text:\n"
        f"{document.text[:60000]}"
    )


def classify_document(document: ParsedDocument) -> str:
    text = f"{document.title}\n{document.text[:2000]}"
    if any(token in text for token in ("土地", "地块", "出让")):
        return "land_market"
    if any(token in text for token in ("重大建设项目", "重大工程")):
        return "major_projects"
    if any(token in text for token in ("招标", "中标", "建设工程")):
        return "construction_tender"
    if any(token in text for token in ("招商", "投资", "签约")):
        return "investment_promotion"
    return "government_public_information"


def classify_event(text: str) -> EventType:
    land_hit = any(token in text for token in ("土地", "地块", "用地"))
    land_transaction_hit = any(token in text for token in ("成交", "竞得", "摘牌"))
    land_supply_hit = any(token in text for token in ("出让", "供应", "公告", "挂牌"))
    if any(token in text for token in ("住宅", "商品住房", "居住用地")) and land_hit and (land_supply_hit or land_transaction_hit):
        return EventType.RESIDENTIAL_SUPPLY
    if land_hit and land_transaction_hit:
        return EventType.LAND_TRANSACTION
    if land_hit and land_supply_hit:
        return EventType.LAND_SUPPLY
    if any(token in text for token in ("规划", "专项规划", "行动方案", "国土空间")):
        return EventType.PLANNING_POLICY
    if any(token in text for token in ("批复", "核准", "备案")):
        return EventType.PROJECT_APPROVAL
    if any(token in text for token in ("轨道", "铁路", "机场", "枢纽", "道路", "交通", "隧道", "桥梁")):
        return EventType.INFRASTRUCTURE
    if any(token in text for token in ("学校", "医院", "卫生", "教育", "养老", "公共服务")):
        return EventType.PUBLIC_SERVICE
    if any(token in text for token in ("招商", "签约", "投资促进", "落地")):
        return EventType.INVESTMENT_SIGNING
    if any(token in text for token in ("产业", "园区", "研发", "创新", "科创", "制造", "集成电路", "生物医药")):
        return EventType.INDUSTRIAL_PROJECT
    if any(token in text for token in ("重大工程", "重大建设项目", "开工", "建设项目", "项目")):
        return EventType.MAJOR_PROJECT
    return EventType.OTHER


def line_candidates(text: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 12]
    keyworded = [line for line in lines if classify_event(line) != EventType.OTHER]
    return keyworded[:120]


def is_low_information_heading(text: str, document: ParsedDocument) -> bool:
    title = normalize_heading(document.title)
    if not title:
        return False
    candidate = normalize_heading(text)
    candidate_without_date = normalize_heading(strip_leading_date(text))
    candidate_without_year = re.sub(r"^20\d{2}年?", "", candidate)
    if candidate == title or candidate_without_year == title:
        return True
    return candidate_without_date == title


def normalize_heading(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "").strip()


def row_candidates(document: ParsedDocument) -> list[str]:
    rows = []
    for row in document.rows:
        values = [str(value) for key, value in row.items() if not key.startswith("_") and value not in (None, "")]
        text = " | ".join(values)
        if text:
            rows.append(text)
    return rows[:200]


def extract_district(text: str) -> str | None:
    districts = [
        "浦东新区", "黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区",
        "闵行区", "宝山区", "嘉定区", "金山区", "松江区", "青浦区", "奉贤区", "崇明区",
        "浦东", "临港", "张江", "金桥", "虹桥", "祝桥", "安亭", "南大", "吴淞",
    ]
    for district in districts:
        if district in text:
            return district
    return None


def extract_address(text: str) -> str | None:
    district = extract_district(text)
    if district:
        match = re.search(r"([\u4e00-\u9fffA-Za-z0-9/·\-]{0,20}" + re.escape(district) + r"[\u4e00-\u9fffA-Za-z0-9/·\-]{0,30})", text)
        if match:
            return match.group(1)
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9/·\-]{2,30}(?:镇|街道|园区|新城|片区|地块|枢纽|医院|学校))", text)
    return match.group(1) if match else district


def extract_project_name(text: str) -> str | None:
    text = strip_leading_date(text)
    for delimiter in ("：", ":", "|", "，", ",", "。"):
        if delimiter in text:
            return short_text(text.split(delimiter)[0], 60)
    return short_text(text, 60)


def extract_title(text: str) -> str:
    project = extract_project_name(text)
    return project or text


def strip_leading_date(text: str) -> str:
    return re.sub(
        r"^\s*20\d{2}\s*(?:年|[-/.])\s*\d{1,2}\s*(?:月|[-/.])?\s*(?:\d{1,2}\s*日?)?[，,、\s]*",
        "",
        text,
    ).strip()


def tags_for_event(text: str, event_type: EventType) -> list[str]:
    tags = [event_type.value]
    for token in ("轨道交通", "学校", "医院", "生物医药", "集成电路", "人工智能", "住宅", "产业园", "机场"):
        if token in text:
            tags.append(token)
    return list(dict.fromkeys(tags))


def find_subquote(quote: str, text: str) -> str:
    for size in (120, 80, 40, 20):
        for idx in range(0, max(len(quote) - size + 1, 1), 10):
            part = quote[idx : idx + size]
            if part and part in text:
                return part
    return ""


def dedupe_events(events: list[GovernmentEvent]) -> list[GovernmentEvent]:
    seen: set[tuple[str, str, str | None, EventType, str]] = set()
    result: list[GovernmentEvent] = []
    for event in events:
        quote = event.evidence[0].quote if event.evidence else ""
        key = (event.title, event.source_url, event.event_date, event.event_type, quote)
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


def to_government_extraction(
    extraction: LLMGovernmentSignalExtraction,
    document: ParsedDocument,
) -> GovernmentSignalExtraction:
    events: list[GovernmentEvent] = []
    for item in extraction.events:
        numeric_source = "\n".join(
            [item.title, item.summary, *(evidence.quote for evidence in item.evidence)]
        )
        events.append(
            GovernmentEvent(
                source_id=document.source_id,
                source_url=document.url,
                event_type=item.event_type,
                title=item.title,
                summary=item.summary,
                district=item.district,
                address=item.address,
                project_name=item.project_name,
                event_date=normalize_date(item.event_date) or document.date,
                amount_wanyuan=item.amount_wanyuan if item.amount_wanyuan is not None else parse_amount_wanyuan(numeric_source),
                area_sqm=item.area_sqm if item.area_sqm is not None else parse_area_sqm(numeric_source),
                evidence=[
                    Evidence(
                        source_id=document.source_id,
                        url=document.url,
                        quote=evidence.quote,
                        fetched_at=document.fetched_at,
                        content_hash=document.content_hash,
                        confidence=evidence.confidence,
                    )
                    for evidence in item.evidence
                ],
                tags=item.tags,
                raw_doc_hash=document.content_hash,
            )
        )
    return GovernmentSignalExtraction(
        document_classification=extraction.document_classification,
        events=events,
        review_notes=list(extraction.review_notes),
    )


def event_has_valid_evidence(event: GovernmentEvent, source_text: str) -> bool:
    return bool(event.evidence) and all(quote_in_text(evidence.quote, source_text) for evidence in event.evidence)
