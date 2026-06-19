from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import effective_openai_api_key, settings
from app.core.utils import write_json
from app.llm.openai_client import build_openai_client
from app.llm.schemas import GovernmentEvent
from app.scoring.residential import ResidentialCandidateScore


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityFinding(StrictModel):
    severity: str
    area_name: str
    issue: str
    recommendation: str


class QualityReview(StrictModel):
    pass_review: bool
    overall_score: float = Field(ge=0, le=100)
    findings: list[QualityFinding]
    residual_risks: list[str]


def write_quality_review(
    path: Path,
    *,
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
    use_openai: bool,
    top_k: int = 8,
) -> Path:
    review = generate_quality_review(scores[:top_k], events, use_openai=use_openai)
    write_json(path, review.model_dump(mode="json"))
    return path


def generate_quality_review(
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
    *,
    use_openai: bool,
) -> QualityReview:
    if use_openai and effective_openai_api_key() and scores:
        try:
            return generate_quality_review_openai(scores, events)
        except Exception:
            pass
    return generate_quality_review_heuristic(scores)


def generate_quality_review_openai(
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
) -> QualityReview:
    payload = {
        "scores": [
            {
                "area_name": score.area.name,
                "score": score.residential_development_score,
                "evidence_count": score.evidence_count,
                "geo_uncertainty_penalty": score.geo_uncertainty_penalty,
                "maturity_penalty": score.maturity_penalty,
                "supply_pressure": score.residential_supply_pressure,
                "recommendation": score.recommendation,
                "key_reasons": score.key_reasons,
                "risks": score.major_risks,
            }
            for score in scores
        ],
        "event_type_counts": event_type_counts(events),
    }
    client = build_openai_client()
    response = client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are the verifier for LandScout Agent. Review whether the residential recommendations are defensible. "
                    "Flag evidence gaps, mature-market overrecommendation, weak pre-inflow logic, high supply pressure, "
                    "or location uncertainty. Use only provided data."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "QualityReview",
                "schema": QualityReview.model_json_schema(),
                "strict": True,
            }
        },
    )
    return QualityReview.model_validate(json.loads(response.output_text))


def generate_quality_review_heuristic(scores: list[ResidentialCandidateScore]) -> QualityReview:
    findings: list[QualityFinding] = []
    for score in scores:
        if score.evidence_count < 3:
            findings.append(
                QualityFinding(
                    severity="medium",
                    area_name=score.area.name,
                    issue="有效证据数低于正常推荐阈值。",
                    recommendation="先放入观察池，等待更多同区域公开信号后再升级。",
                )
            )
        if score.geo_uncertainty_penalty >= 45:
            findings.append(
                QualityFinding(
                    severity="high",
                    area_name=score.area.name,
                    issue="地理定位不确定性较高。",
                    recommendation="人工核验原文位置和地块边界前，不应视为地块级证据。",
                )
            )
        if score.residential_supply_pressure >= 35:
            findings.append(
                QualityFinding(
                    severity="medium",
                    area_name=score.area.name,
                    issue="住宅供应压力偏高。",
                    recommendation="拿地前需补充库存、去化周期和周边竞品项目核查。",
                )
            )
    score_value = max(0.0, 100.0 - len(findings) * 8)
    return QualityReview(
        pass_review=not any(finding.severity == "high" for finding in findings),
        overall_score=round(score_value, 2),
        findings=findings,
        residual_risks=[
            "公开信号不能替代正式投拓尽调和拿地测算。",
            "成交价、库存、去化周期和竞品数据尚未完整接入。",
        ],
    )


def event_type_counts(events: list[GovernmentEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type.value] = counts.get(event.event_type.value, 0) + 1
    return counts
