from __future__ import annotations

from statistics import mean

from pydantic import BaseModel, Field

from app.llm.normalization import clamp
from app.llm.schemas import EventType, GovernmentEvent
from app.scoring.candidates import CandidateArea, haversine_m
from app.scoring.scorer import count_by_type, evidence_confidence, recency_score, residential_keyword_hits


class ResidentialCandidateScore(BaseModel):
    area: CandidateArea
    future_population_inflow_score: float
    pre_inflow_signal_score: float
    land_grab_window_score: float
    demand_driver_score: float
    transport_public_service_score: float
    residential_land_access_score: float
    market_entry_score: float
    recency_score: float
    evidence_confidence_score: float
    residential_supply_pressure: float
    maturity_penalty: float
    geo_uncertainty_penalty: float
    residential_development_score: float
    opportunity_score: float
    confidence: float
    evidence_count: int
    evidence_event_ids: list[str] = Field(default_factory=list)
    recommendation: str = ""
    next_action: str = ""
    suggested_product: str = ""
    key_reasons: list[str] = Field(default_factory=list)
    major_risks: list[str] = Field(default_factory=list)


def score_residential_candidates(
    candidates: list[CandidateArea],
    events: list[GovernmentEvent],
    *,
    days: int = 540,
) -> list[ResidentialCandidateScore]:
    scores: list[ResidentialCandidateScore] = []
    for candidate in candidates:
        nearby = nearby_events(candidate, events)
        evidence_events = [event for event in nearby if event.evidence]
        counts = count_by_type(evidence_events)
        population_inflow = future_population_inflow_score(counts)
        pre_inflow = pre_inflow_signal_score(counts)
        grab_window = land_grab_window_score(counts, evidence_events)
        demand = demand_driver_score(counts)
        transport_public = transport_public_service_score(counts)
        land_access = residential_land_access_score(counts)
        entry = market_entry_score(counts)
        recency = recency_score(evidence_events, days)
        evidence_conf = evidence_confidence(evidence_events)
        supply_pressure = residential_supply_pressure(counts, evidence_events)
        maturity = maturity_penalty(counts)
        geo_penalty = geo_uncertainty_penalty(evidence_events)
        development_score = clamp(
            0.25 * population_inflow
            + 0.22 * pre_inflow
            + 0.16 * grab_window
            + 0.07 * demand
            + 0.05 * transport_public
            + 0.05 * land_access
            + 0.04 * entry
            + 0.06 * recency
            + 0.08 * evidence_conf
            - 0.08 * supply_pressure
            - 0.06 * maturity
            - 0.04 * geo_penalty
        )
        evidence_ids = [event.id for event in evidence_events]
        recommendation = recommendation_label(development_score, len(evidence_events))
        score = ResidentialCandidateScore(
            area=candidate,
            future_population_inflow_score=round(population_inflow, 2),
            pre_inflow_signal_score=round(pre_inflow, 2),
            land_grab_window_score=round(grab_window, 2),
            demand_driver_score=round(demand, 2),
            transport_public_service_score=round(transport_public, 2),
            residential_land_access_score=round(land_access, 2),
            market_entry_score=round(entry, 2),
            recency_score=round(recency, 2),
            evidence_confidence_score=round(evidence_conf, 2),
            residential_supply_pressure=round(supply_pressure, 2),
            maturity_penalty=round(maturity, 2),
            geo_uncertainty_penalty=round(geo_penalty, 2),
            residential_development_score=round(development_score, 2),
            opportunity_score=round(development_score, 2),
            confidence=round(evidence_conf / 100, 3),
            evidence_count=len(evidence_events),
            evidence_event_ids=evidence_ids,
            recommendation=recommendation,
            next_action=next_action_label(
                recommendation,
                len(evidence_events),
                supply_pressure=supply_pressure,
                geo_penalty=geo_penalty,
            ),
            suggested_product=suggested_product(counts),
            key_reasons=key_reasons(counts, evidence_events),
            major_risks=major_risks(counts, evidence_events, supply_pressure, geo_penalty),
        )
        scores.append(score)
    return rank_residential_scores(scores)


def nearby_events(candidate: CandidateArea, events: list[GovernmentEvent]) -> list[GovernmentEvent]:
    return [
        event
        for event in events
        if event.lat is not None
        and event.lon is not None
        and haversine_m(candidate.lat, candidate.lon, event.lat, event.lon) <= candidate.radius_m
    ]


def demand_driver_score(counts: dict[EventType, int]) -> float:
    return clamp(
        counts[EventType.INDUSTRIAL_PROJECT] * 24
        + counts[EventType.INVESTMENT_SIGNING] * 18
        + counts[EventType.MAJOR_PROJECT] * 8
        + counts[EventType.PROJECT_APPROVAL] * 6
    )


def future_population_inflow_score(counts: dict[EventType, int]) -> float:
    return clamp(
        counts[EventType.INDUSTRIAL_PROJECT] * 28
        + counts[EventType.INVESTMENT_SIGNING] * 22
        + counts[EventType.MAJOR_PROJECT] * 10
        + counts[EventType.INFRASTRUCTURE] * 8
        + counts[EventType.PUBLIC_SERVICE] * 6
    )


def pre_inflow_signal_score(counts: dict[EventType, int]) -> float:
    return clamp(
        counts[EventType.PLANNING_POLICY] * 26
        + counts[EventType.PROJECT_APPROVAL] * 22
        + counts[EventType.INFRASTRUCTURE] * 18
        + counts[EventType.MAJOR_PROJECT] * 12
        + counts[EventType.INVESTMENT_SIGNING] * 10
        + counts[EventType.INDUSTRIAL_PROJECT] * 8
    )


def land_grab_window_score(counts: dict[EventType, int], events: list[GovernmentEvent]) -> float:
    return clamp(
        counts[EventType.LAND_SUPPLY] * 22
        + counts[EventType.PLANNING_POLICY] * 18
        + counts[EventType.PROJECT_APPROVAL] * 16
        + counts[EventType.MAJOR_PROJECT] * 8
        - residential_supply_pressure(counts, events) * 0.3
    )


def transport_public_service_score(counts: dict[EventType, int]) -> float:
    return clamp(
        counts[EventType.INFRASTRUCTURE] * 26
        + counts[EventType.PUBLIC_SERVICE] * 24
        + counts[EventType.MAJOR_PROJECT] * 5
    )


def residential_land_access_score(counts: dict[EventType, int]) -> float:
    return clamp(
        counts[EventType.RESIDENTIAL_SUPPLY] * 24
        + counts[EventType.LAND_SUPPLY] * 18
        + counts[EventType.LAND_TRANSACTION] * 14
        + counts[EventType.PLANNING_POLICY] * 10
        + counts[EventType.PROJECT_APPROVAL] * 8
    )


def market_entry_score(counts: dict[EventType, int]) -> float:
    return clamp(
        counts[EventType.LAND_SUPPLY] * 14
        + counts[EventType.PROJECT_APPROVAL] * 10
        + counts[EventType.PLANNING_POLICY] * 8
        + counts[EventType.RESIDENTIAL_SUPPLY] * 6
    )


def residential_supply_pressure(counts: dict[EventType, int], events: list[GovernmentEvent]) -> float:
    return clamp(counts[EventType.RESIDENTIAL_SUPPLY] * 18 + residential_keyword_hits(events) * 8)


def maturity_penalty(counts: dict[EventType, int]) -> float:
    return clamp(
        counts[EventType.RESIDENTIAL_SUPPLY] * 16
        + counts[EventType.LAND_TRANSACTION] * 12
        + counts[EventType.PUBLIC_SERVICE] * 8
    )


def geo_uncertainty_penalty(events: list[GovernmentEvent]) -> float:
    if not events:
        return 100.0
    return clamp(100 - mean([event.geo_confidence * 100 for event in events]))


def recommendation_label(score: float, evidence_count: int) -> str:
    if evidence_count < 3:
        return "观察跟踪"
    if score >= 42:
        return "优先抢先跟踪"
    if score >= 28:
        return "谨慎抢先跟踪"
    return "暂不优先"


def next_action_label(
    recommendation: str,
    evidence_count: int,
    *,
    supply_pressure: float,
    geo_penalty: float,
) -> str:
    if evidence_count < 3:
        return "观察池：补充同区域新增政策、产业、供地证据后再评估。"
    if recommendation == "优先抢先跟踪":
        if supply_pressure >= 35:
            return "投拓预筛：立即核查控规、供地窗口、库存去化和竞品价格，再决定是否立项。"
        return "投拓预筛：立即核查控规、供地窗口、权属边界和拿地测算。"
    if recommendation == "谨慎抢先跟踪":
        if geo_penalty >= 45:
            return "谨慎预筛：先人工核验原文位置和地块边界，再补市场数据。"
        return "重点观察：每周监测控规、供地、产业兑现和配套开工，补齐市场数据后再升级。"
    return "低优先级监测：仅保留政策和供地提醒，暂不投入拿地资源。"


def suggested_product(counts: dict[EventType, int]) -> str:
    demand = counts[EventType.INDUSTRIAL_PROJECT] + counts[EventType.INVESTMENT_SIGNING]
    transport = counts[EventType.INFRASTRUCTURE]
    public = counts[EventType.PUBLIC_SERVICE]
    land = counts[EventType.LAND_SUPPLY] + counts[EventType.RESIDENTIAL_SUPPLY]
    if demand >= 2 and public > 0:
        return "人才租赁住房、刚改住宅和园区生活配套"
    if transport > 0 and public > 0:
        return "TOD 周边改善型住宅、租赁住房和社区商业"
    if land > 0:
        return "围绕新增供地的分期住宅社区和底层生活配套"
    if demand > 0:
        return "服务产业导入人群的租赁住房和小户型产品"
    return "先做持续监测，暂不建议重资产拿地"


def key_reasons(counts: dict[EventType, int], events: list[GovernmentEvent]) -> list[str]:
    reasons: list[str] = []
    demand = counts[EventType.INDUSTRIAL_PROJECT] + counts[EventType.INVESTMENT_SIGNING]
    early = counts[EventType.PLANNING_POLICY] + counts[EventType.PROJECT_APPROVAL] + counts[EventType.MAJOR_PROJECT]
    if demand:
        reasons.append(f"产业/招商导入信号 {demand} 条，可能形成未来新增就业人口和居住需求")
    if early:
        reasons.append(f"人口流入前置信号 {early} 条，来自规划、批复或重大项目")
    if counts[EventType.INFRASTRUCTURE]:
        reasons.append(f"交通/基建改善信号 {counts[EventType.INFRASTRUCTURE]} 条")
    if counts[EventType.PUBLIC_SERVICE]:
        reasons.append(f"学校、医院等公共服务信号 {counts[EventType.PUBLIC_SERVICE]} 条")
    land = counts[EventType.LAND_SUPPLY] + counts[EventType.RESIDENTIAL_SUPPLY]
    if land:
        reasons.append(f"土地供应或涉住宅供地信号 {land} 条")
    if counts[EventType.PROJECT_APPROVAL] or counts[EventType.PLANNING_POLICY]:
        reasons.append(
            f"批复/规划信号 {counts[EventType.PROJECT_APPROVAL] + counts[EventType.PLANNING_POLICY]} 条"
        )
    if not reasons:
        reasons.append(f"附近有效公开证据 {len(events)} 条，但住宅方向证据仍弱")
    return reasons


def major_risks(
    counts: dict[EventType, int],
    events: list[GovernmentEvent],
    supply_pressure: float,
    geo_penalty: float,
) -> list[str]:
    risks: list[str] = []
    risks.append("该评分用于提前发现人口流入前的代理信号，不能替代正式投拓尽调和拿地测算。")
    if supply_pressure >= 35:
        risks.append("住宅供应压力偏高，需补充核查周边库存、去化和竞品价格。")
    if counts[EventType.LAND_SUPPLY] + counts[EventType.RESIDENTIAL_SUPPLY] == 0:
        risks.append("缺少明确住宅供地证据，需核查控规、用地性质和后续招拍挂条件。")
    if geo_penalty >= 45:
        risks.append("部分事件只能定位到区级或板块级，坐标精度有限。")
    if len(events) < 3:
        risks.append("公开证据少于 3 条，不能作为优先拿地依据。")
    risks.append("当前模型只使用政府公开信号，尚未纳入成交价、库存、去化周期、人口迁徙和竞品数据。")
    return list(dict.fromkeys(risks))


def rank_residential_scores(scores: list[ResidentialCandidateScore]) -> list[ResidentialCandidateScore]:
    ordered = sorted(
        scores,
        key=lambda item: (
            item.evidence_count >= 3,
            item.residential_development_score,
            item.pre_inflow_signal_score,
            item.land_grab_window_score,
            item.area.source == "seed",
        ),
        reverse=True,
    )
    result: list[ResidentialCandidateScore] = []
    seen_event_sets: set[tuple[str, ...]] = set()
    seen_locations: set[tuple[float, float]] = set()
    for score in ordered:
        event_key = tuple(sorted(score.evidence_event_ids))
        location_key = (round(score.area.lat, 3), round(score.area.lon, 3))
        if event_key and event_key in seen_event_sets:
            continue
        if location_key in seen_locations and score.area.source == "dbscan":
            continue
        result.append(score)
        if event_key:
            seen_event_sets.add(event_key)
        seen_locations.add(location_key)
    return result
