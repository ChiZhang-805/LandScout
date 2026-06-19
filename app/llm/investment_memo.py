from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.llm.openai_client import build_openai_client
from app.llm.schemas import GovernmentEvent
from app.scoring.residential import ResidentialCandidateScore


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AreaMemo(StrictModel):
    area_name: str
    thesis: str
    causal_chain: list[str]
    land_grab_window: str
    suggested_next_steps: list[str]
    key_risks: list[str]


class InvestmentMemo(StrictModel):
    executive_summary: str
    area_memos: list[AreaMemo]
    cross_area_observations: list[str]
    human_due_diligence_questions: list[str]


def build_investment_memo(
    path: Path,
    *,
    run_id: str,
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
    use_openai: bool,
    top_k: int = 5,
) -> Path:
    memo = generate_investment_memo(scores[:top_k], events, use_openai=use_openai)
    path.write_text(render_memo_markdown(run_id, memo), encoding="utf-8")
    return path


def generate_investment_memo(
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
    *,
    use_openai: bool,
) -> InvestmentMemo:
    if use_openai and settings.openai_api_key and scores:
        try:
            return generate_investment_memo_openai(scores, events)
        except Exception:
            pass
    return generate_investment_memo_heuristic(scores)


def generate_investment_memo_openai(
    scores: list[ResidentialCandidateScore],
    events: list[GovernmentEvent],
) -> InvestmentMemo:
    event_by_id = {event.id: event for event in events}
    payload: list[dict[str, Any]] = []
    for score in scores:
        evidence = [event_by_id[event_id] for event_id in score.evidence_event_ids if event_id in event_by_id][:8]
        payload.append(
            {
                "area": score.area.name,
                "score": score.residential_development_score,
                "recommendation": score.recommendation,
                "key_reasons": score.key_reasons,
                "risks": score.major_risks,
                "evidence": [
                    {
                        "title": event.title,
                        "event_type": event.event_type.value,
                        "date": event.event_date,
                        "district": event.district,
                        "quote": event.evidence[0].quote if event.evidence else "",
                    }
                    for event in evidence
                ],
            }
        )
    client = build_openai_client()
    response = client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an investment memo agent for Chinese residential developers. "
                    "Use only the provided scores and evidence. Explain the causal chain from early public signals "
                    "to possible future population inflow and housing demand. Do not invent facts."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "InvestmentMemo",
                "schema": InvestmentMemo.model_json_schema(),
                "strict": True,
            }
        },
    )
    return InvestmentMemo.model_validate(json.loads(response.output_text))


def generate_investment_memo_heuristic(scores: list[ResidentialCandidateScore]) -> InvestmentMemo:
    area_memos: list[AreaMemo] = []
    for score in scores:
        area_memos.append(
            AreaMemo(
                area_name=score.area.name,
                thesis=f"{score.area.name}住宅开发分为 {score.residential_development_score}，建议级别为{score.recommendation}。",
                causal_chain=[
                    *score.key_reasons[:3],
                    "将早期公开信号转化为投拓预筛问题，并在拿地前补齐人工尽调。",
                ],
                land_grab_window=f"抢先拿地窗口分为 {score.land_grab_window_score}；当前建议级别为{score.recommendation}。",
                suggested_next_steps=[
                    "核查用地性质、控制性详细规划、权属边界和后续招拍挂条件。",
                    "补充周边库存、去化周期、竞品项目、成交价格和目标客群支付能力。",
                    "每周跟踪规划、供地、交通、公共服务和产业项目后续公告。",
                ],
                key_risks=score.major_risks[:4],
            )
        )
    return InvestmentMemo(
        executive_summary="本备忘录基于可追溯公开证据和住宅开发评分生成，用于投拓预筛，不替代正式尽调和拿地测算。",
        area_memos=area_memos,
        cross_area_observations=[
            "优先跟踪在人口明显流入前已经出现规划、产业、交通和供地窗口信号的区域。",
            "低地理置信度或区级泛定位信号只能进入观察池，不能当作地块级确认。",
        ],
        human_due_diligence_questions=[
            "候选地块是否具备住宅、租赁住房或商住混合的用地条件？",
            "同一客群覆盖范围内是否存在当前或即将入市的竞品住宅供应？",
            "交通、学校、医院和商业配套的兑现时间是否覆盖项目销售周期？",
        ],
    )


def render_memo_markdown(run_id: str, memo: InvestmentMemo) -> str:
    lines = [
        "# LandScout Agent 投拓备忘录",
        "",
        f"- run_id: `{run_id}`",
        "",
        "## 执行摘要",
        "",
        memo.executive_summary,
        "",
        "## 区域备忘录",
        "",
    ]
    for idx, area in enumerate(memo.area_memos, start=1):
        lines.extend(
            [
                f"### {idx}. {area.area_name}",
                "",
                f"**判断:** {area.thesis}",
                "",
                "**因果链:**",
                *[f"- {item}" for item in area.causal_chain],
                "",
                f"**拿地窗口:** {area.land_grab_window}",
                "",
                "**下一步:**",
                *[f"- {item}" for item in area.suggested_next_steps],
                "",
                "**风险:**",
                *[f"- {item}" for item in area.key_risks],
                "",
            ]
        )
    lines.extend(
        [
            "## 跨区域观察",
            "",
            *[f"- {item}" for item in memo.cross_area_observations],
            "",
            "## 人工尽调问题",
            "",
            *[f"- {item}" for item in memo.human_due_diligence_questions],
            "",
        ]
    )
    return "\n".join(lines)
