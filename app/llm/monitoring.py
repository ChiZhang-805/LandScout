from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.core.config import settings
from app.core.utils import write_json
from app.llm.openai_client import build_openai_client
from app.llm.schemas import GovernmentEvent
from app.scoring.residential import ResidentialCandidateScore


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MonitoringQuery(StrictModel):
    area_name: str
    query: str
    purpose: str
    cadence: str


class MonitoringPlan(StrictModel):
    queries: list[MonitoringQuery]


def write_monitoring_queries(
    path: Path,
    *,
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
    use_openai: bool,
    top_k: int = 8,
) -> Path:
    plan = generate_monitoring_queries(scores[:top_k], events, use_openai=use_openai)
    write_json(path, plan.model_dump(mode="json"))
    return path


def generate_monitoring_queries(
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
    *,
    use_openai: bool,
) -> MonitoringPlan:
    if use_openai and settings.openai_api_key and scores:
        try:
            return generate_monitoring_queries_openai(scores, events)
        except Exception:
            pass
    return generate_monitoring_queries_heuristic(scores)


def generate_monitoring_queries_openai(
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
) -> MonitoringPlan:
    payload = {
        "areas": [
            {
                "area_name": score.area.name,
                "score": score.residential_development_score,
                "key_reasons": score.key_reasons,
                "risks": score.major_risks,
            }
            for score in scores[:8]
        ],
        "recent_event_titles": [event.title for event in events[-40:]],
    }
    client = build_openai_client()
    response = client.responses.create(
        model=settings.openai_fast_model,
        input=[
            {
                "role": "system",
                "content": (
                    "Generate precise Chinese web monitoring queries for Shanghai residential land opportunity tracking. "
                    "Queries should target public official information: planning notices, land supply, project approvals, "
                    "transport, schools, hospitals, industrial investment, and district announcements."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "MonitoringPlan",
                "schema": MonitoringPlan.model_json_schema(),
                "strict": True,
            }
        },
    )
    return MonitoringPlan.model_validate(json.loads(response.output_text))


def generate_monitoring_queries_heuristic(scores: list[ResidentialCandidateScore]) -> MonitoringPlan:
    queries: list[MonitoringQuery] = []
    for score in scores:
        area = score.area.name
        for suffix, purpose in [
            (" 控制性详细规划 公示 住宅 用地", "跟踪控规和住宅用地性质变化。"),
            (" 住宅用地 出让 公告", "跟踪可参与的供地窗口。"),
            (" 重大项目 开工 招商 签约", "跟踪人口流入前的产业和就业导入信号。"),
        ]:
            queries.append(
                MonitoringQuery(
                    area_name=area,
                    query=f"上海 {area}{suffix}",
                    purpose=purpose,
                    cadence="每周",
                )
            )
    return MonitoringPlan(queries=queries[:24])
