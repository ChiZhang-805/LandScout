from __future__ import annotations

from datetime import date
from statistics import mean

from pydantic import BaseModel, Field

from app.llm.normalization import clamp
from app.llm.schemas import EventType, GovernmentEvent
from app.scoring.candidates import CandidateArea, haversine_m


class CandidateScore(BaseModel):
    area: CandidateArea
    industrial_import_score: float
    infrastructure_score: float
    public_service_score: float
    land_structure_score: float
    residential_supply_risk: float
    market_entry_score: float
    recency_score: float
    evidence_confidence_score: float
    geo_uncertainty_penalty: float
    opportunity_score: float
    confidence: float
    evidence_count: int
    evidence_event_ids: list[str] = Field(default_factory=list)
    development_direction: str = ""
    key_reasons: list[str] = Field(default_factory=list)
    major_risks: list[str] = Field(default_factory=list)


def score_candidates(
    candidates: list[CandidateArea],
    events: list[GovernmentEvent],
    *,
    days: int = 540,
) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for candidate in candidates:
        nearby = [
            event
            for event in events
            if event.lat is not None
            and event.lon is not None
            and haversine_m(candidate.lat, candidate.lon, event.lat, event.lon) <= candidate.radius_m
        ]
        evidence_events = [event for event in nearby if event.evidence]
        counts = count_by_type(evidence_events)
        industrial = clamp(counts[EventType.INDUSTRIAL_PROJECT] * 24 + counts[EventType.INVESTMENT_SIGNING] * 18 + counts[EventType.MAJOR_PROJECT] * 8)
        infrastructure = clamp(counts[EventType.INFRASTRUCTURE] * 28 + counts[EventType.MAJOR_PROJECT] * 5)
        public_service = clamp(counts[EventType.PUBLIC_SERVICE] * 30)
        land_structure = clamp(counts[EventType.LAND_SUPPLY] * 22 + counts[EventType.LAND_TRANSACTION] * 18 + counts[EventType.PLANNING_POLICY] * 10)
        residential_risk = clamp(counts[EventType.RESIDENTIAL_SUPPLY] * 30 + residential_keyword_hits(evidence_events) * 10)
        market_entry = clamp(counts[EventType.LAND_SUPPLY] * 16 + counts[EventType.PROJECT_APPROVAL] * 12 + counts[EventType.LAND_TRANSACTION] * 10)
        recency = recency_score(evidence_events, days)
        evidence_conf = evidence_confidence(evidence_events)
        geo_penalty = clamp(100 - mean([event.geo_confidence * 100 for event in evidence_events]) if evidence_events else 100)
        opportunity = clamp(
            0.25 * industrial
            + 0.18 * infrastructure
            + 0.12 * public_service
            + 0.15 * land_structure
            + 0.12 * market_entry
            + 0.08 * recency
            + 0.10 * evidence_conf
            - 0.15 * residential_risk
            - 0.05 * geo_penalty
        )
        score = CandidateScore(
            area=candidate,
            industrial_import_score=round(industrial, 2),
            infrastructure_score=round(infrastructure, 2),
            public_service_score=round(public_service, 2),
            land_structure_score=round(land_structure, 2),
            residential_supply_risk=round(residential_risk, 2),
            market_entry_score=round(market_entry, 2),
            recency_score=round(recency, 2),
            evidence_confidence_score=round(evidence_conf, 2),
            geo_uncertainty_penalty=round(geo_penalty, 2),
            opportunity_score=round(opportunity, 2),
            confidence=round(evidence_conf / 100, 3),
            evidence_count=len(evidence_events),
            evidence_event_ids=[event.id for event in evidence_events],
            development_direction=development_direction(counts),
            key_reasons=key_reasons(counts, evidence_events),
            major_risks=major_risks(residential_risk, geo_penalty, evidence_events),
        )
        scores.append(score)
    return rank_scores(scores)


def rank_scores(scores: list[CandidateScore]) -> list[CandidateScore]:
    eligible = [score for score in scores if score.evidence_count >= 3]
    ineligible = [score for score in scores if score.evidence_count < 3]
    eligible.sort(key=lambda item: item.opportunity_score, reverse=True)
    ineligible.sort(key=lambda item: item.opportunity_score, reverse=True)
    return eligible + ineligible


def count_by_type(events: list[GovernmentEvent]) -> dict[EventType, int]:
    counts = {event_type: 0 for event_type in EventType}
    for event in events:
        counts[event.event_type] += 1
    return counts


def recency_score(events: list[GovernmentEvent], days: int) -> float:
    if not events:
        return 0.0
    today = date.today()
    values: list[float] = []
    for event in events:
        if not event.event_date:
            values.append(45.0)
            continue
        try:
            event_day = date.fromisoformat(event.event_date[:10])
        except ValueError:
            values.append(45.0)
            continue
        age = max((today - event_day).days, 0)
        values.append(clamp(100 * (1 - min(age, days) / max(days, 1))))
    return mean(values)


def evidence_confidence(events: list[GovernmentEvent]) -> float:
    values: list[float] = []
    for event in events:
        if event.evidence:
            values.append(mean([evidence.confidence for evidence in event.evidence]) * 100)
    return mean(values) if values else 0.0


def residential_keyword_hits(events: list[GovernmentEvent]) -> int:
    return sum(
        1
        for event in events
        if any(token in f"{event.title}\n{event.summary}" for token in ("住宅", "商品住房", "居住用地"))
    )


def development_direction(counts: dict[EventType, int]) -> str:
    demand = counts[EventType.INDUSTRIAL_PROJECT] + counts[EventType.INVESTMENT_SIGNING]
    if demand > 0 and demand >= counts[EventType.INFRASTRUCTURE]:
        return "产业载体、研发办公、园区配套和人才租赁住房"
    if counts[EventType.INFRASTRUCTURE] > 0:
        return "TOD 综合开发、商务配套和公共服务补齐"
    if counts[EventType.LAND_SUPPLY] > 0:
        return "围绕新增土地供应的分期开发和配套商业"
    return "谨慎跟踪，等待更多公开证据"


def key_reasons(counts: dict[EventType, int], events: list[GovernmentEvent]) -> list[str]:
    reasons: list[str] = []
    if counts[EventType.INDUSTRIAL_PROJECT] or counts[EventType.INVESTMENT_SIGNING]:
        reasons.append(f"产业/招商事件 {counts[EventType.INDUSTRIAL_PROJECT] + counts[EventType.INVESTMENT_SIGNING]} 条")
    if counts[EventType.INFRASTRUCTURE]:
        reasons.append(f"交通/基建事件 {counts[EventType.INFRASTRUCTURE]} 条")
    if counts[EventType.PUBLIC_SERVICE]:
        reasons.append(f"学校医院等公共服务事件 {counts[EventType.PUBLIC_SERVICE]} 条")
    if counts[EventType.LAND_SUPPLY] or counts[EventType.LAND_TRANSACTION]:
        reasons.append(f"土地供应/交易事件 {counts[EventType.LAND_SUPPLY] + counts[EventType.LAND_TRANSACTION]} 条")
    if not reasons:
        reasons.append(f"有效证据 {len(events)} 条，暂未形成单项强信号")
    return reasons


def major_risks(residential_risk: float, geo_penalty: float, events: list[GovernmentEvent]) -> list[str]:
    risks: list[str] = []
    if residential_risk >= 40:
        risks.append("住宅供应风险偏高，需核查周边去化与竞品供给。")
    if geo_penalty >= 45:
        risks.append("部分事件仅能定位到区级或板块级，坐标精度有限。")
    if len(events) < 3:
        risks.append("公开证据少于 3 条，不能作为第一推荐。")
    if not risks:
        risks.append("需继续跟踪批复、供地节奏和招拍挂条件变化。")
    return risks
