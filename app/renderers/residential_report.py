from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.branding import PRODUCT_DISPLAY_NAME, landscout_agent_workflow_payload
from app.core.utils import (
    actionable_errors,
    document_triage_skip_count,
    ensure_dir,
    leading_label,
    short_text,
    write_json,
)
from app.crawlers.models import FetchError, RawDocument
from app.llm.batch import write_extraction_batch_requests
from app.llm.crawler_hints import write_crawler_hints
from app.llm.embeddings import write_embedding_index
from app.llm.investment_memo import build_investment_memo
from app.llm.monitoring import write_monitoring_queries
from app.llm.quality import write_quality_review
from app.llm.schemas import GovernmentEvent
from app.parsers.models import ParsedDocument
from app.renderers.report import (
    RenderedOutputs,
    build_event_type_counts,
    build_log,
    build_pipeline_metrics,
    build_source_event_counts,
    render_map,
    render_visual_summary,
    update_latest,
    write_events_csv,
)
from app.scoring.residential import ResidentialCandidateScore
from app.scoring.scorer import CandidateScore


def render_residential_outputs(
    *,
    run_id: str,
    output_dir: Path,
    events: list[GovernmentEvent],
    residential_scores: list[ResidentialCandidateScore],
    base_scores: list[CandidateScore],
    raw_documents: list[RawDocument],
    parsed_documents: list[ParsedDocument],
    errors: list[FetchError] | list[dict[str, Any]],
    top_k: int,
    visited_sources: list[str] | None = None,
    discovered_sources: list[dict[str, Any]] | None = None,
    api_enrichment: bool = False,
) -> RenderedOutputs:
    ensure_dir(output_dir)
    top_scores = [score for score in residential_scores if score.evidence_count > 0][:top_k]
    visible_errors = actionable_errors(errors)
    pipeline_metrics = build_pipeline_metrics(
        visited_source_count=len(visited_sources or []),
        raw_document_count=len(raw_documents),
        parsed_document_count=len(parsed_documents),
        event_count=len(events),
        evidence_event_count=sum(1 for event in events if event.evidence),
        error_count=len(visible_errors),
        filtered_document_count=document_triage_skip_count(errors),
    )
    pipeline_metrics.update(build_triage_metrics(parsed_documents))
    recommendation_path = output_dir / "recommendation.md"
    map_path = output_dir / "opportunity_map.html"
    visual_path = output_dir / "visual_summary.html"
    events_path = output_dir / "events.csv"
    signals_path = output_dir / "signals.json"
    evidence_path = output_dir / "evidence_pack.json"
    log_path = output_dir / "pipeline.log"
    memo_path = output_dir / "investment_memo.md"
    monitoring_path = output_dir / "monitoring_queries.json"
    embedding_path = output_dir / "embedding_index.json"
    batch_path = output_dir / "batch_requests.jsonl"
    quality_path = output_dir / "quality_review.json"
    crawler_hints_path = output_dir / "crawler_hints.json"

    write_events_csv(events_path, events)
    write_extraction_batch_requests(batch_path, parsed_documents)
    write_crawler_hints(crawler_hints_path, parsed_documents=parsed_documents, use_openai=api_enrichment)
    write_embedding_index(embedding_path, parsed_documents, events, use_openai=api_enrichment)
    build_investment_memo(
        memo_path,
        run_id=run_id,
        scores=top_scores,
        events=events,
        use_openai=api_enrichment,
        top_k=top_k,
    )
    write_monitoring_queries(monitoring_path, scores=top_scores, events=events, use_openai=api_enrichment, top_k=top_k)
    write_quality_review(quality_path, scores=top_scores, events=events, use_openai=api_enrichment, top_k=top_k)
    write_json(
        signals_path,
        {
            "product_name": PRODUCT_DISPLAY_NAME,
            "agent_workflow": landscout_agent_workflow_payload(),
            "analysis_type": "residential_development_recommendation",
            "run_id": run_id,
            "top_areas": [score.model_dump(mode="json") for score in top_scores],
            "event_count": len(events),
            "candidate_count": len(residential_scores),
            "visited_sources": visited_sources or [],
            "discovered_sources": discovered_sources or [],
            "ai_artifacts": {
                "investment_memo_md": str(memo_path.resolve()),
                "monitoring_queries_json": str(monitoring_path.resolve()),
                "embedding_index_json": str(embedding_path.resolve()),
                "batch_requests_jsonl": str(batch_path.resolve()),
                "quality_review_json": str(quality_path.resolve()),
                "crawler_hints_json": str(crawler_hints_path.resolve()),
                "api_enrichment": api_enrichment,
            },
            "pipeline_metrics": pipeline_metrics,
            "next_action_counts": build_next_action_counts(top_scores),
            "event_type_counts": build_event_type_counts(events),
            "source_event_counts": build_source_event_counts(events),
        },
    )
    write_json(
        evidence_path,
        {
            "product_name": PRODUCT_DISPLAY_NAME,
            "agent_workflow": landscout_agent_workflow_payload(),
            "analysis_type": "residential_development_recommendation",
            "run_id": run_id,
            "discovered_sources": discovered_sources or [],
            "ai_artifacts": {
                "investment_memo_md": str(memo_path.resolve()),
                "monitoring_queries_json": str(monitoring_path.resolve()),
                "embedding_index_json": str(embedding_path.resolve()),
                "batch_requests_jsonl": str(batch_path.resolve()),
                "quality_review_json": str(quality_path.resolve()),
                "crawler_hints_json": str(crawler_hints_path.resolve()),
                "api_enrichment": api_enrichment,
            },
            "raw_documents": [document.model_dump(mode="json") for document in raw_documents],
            "parsed_documents": [
                {
                    "source_id": document.source_id,
                    "url": document.url,
                    "title": document.title,
                    "date": document.date,
                    "content_hash": document.content_hash,
                    "fetched_at": document.fetched_at,
                    "raw_path": document.raw_path,
                    "parser": document.parser,
                    "needs_ocr": document.needs_ocr,
                    "row_count": len(document.rows),
                    "table_count": len(document.tables),
                    "metadata": document.metadata,
                    "text_excerpt": short_text(document.text, 800),
                }
                for document in parsed_documents
            ],
            "events": [event.model_dump(mode="json") for event in events],
            "residential_scores": [score.model_dump(mode="json") for score in residential_scores],
            "base_candidate_scores": [score.model_dump(mode="json") for score in base_scores],
            "pipeline_metrics": pipeline_metrics,
            "next_action_counts": build_next_action_counts(top_scores),
            "event_type_counts": build_event_type_counts(events),
            "source_event_counts": build_source_event_counts(events),
        },
    )
    recommendation_path.write_text(
        build_residential_markdown(run_id, top_scores, len(residential_scores), events),
        encoding="utf-8",
    )
    render_map(map_path, events, top_scores)
    render_visual_summary(
        visual_path,
        run_id=run_id,
        title="上海住宅开发机会可视化摘要",
        events=events,
        scores=top_scores,
        metric_fields=[
            ("residential_development_score", "住宅开发分"),
            ("future_population_inflow_score", "人口流入潜力"),
            ("pre_inflow_signal_score", "人口流入前置信号"),
            ("land_grab_window_score", "抢先拿地窗口"),
            ("demand_driver_score", "需求驱动"),
            ("transport_public_service_score", "交通公共服务"),
            ("residential_land_access_score", "住宅供地准入"),
            ("residential_supply_pressure", "住宅供应压力"),
            ("maturity_penalty", "成熟度扣分"),
        ],
        main_score_field="residential_development_score",
        visited_source_count=len(visited_sources or []),
        raw_document_count=len(raw_documents),
        parsed_document_count=len(parsed_documents),
        error_count=len(visible_errors),
    )
    log_path.write_text(
        build_log(run_id, raw_documents, parsed_documents, events, residential_scores, errors, visited_sources or []),
        encoding="utf-8",
    )
    update_latest(output_dir)
    return RenderedOutputs(
        recommendation_md=str(recommendation_path.resolve()),
        opportunity_map_html=str(map_path.resolve()),
        visual_summary_html=str(visual_path.resolve()),
        events_csv=str(events_path.resolve()),
        signals_json=str(signals_path.resolve()),
        evidence_pack_json=str(evidence_path.resolve()),
        pipeline_log=str(log_path.resolve()),
        investment_memo_md=str(memo_path.resolve()),
        monitoring_queries_json=str(monitoring_path.resolve()),
        embedding_index_json=str(embedding_path.resolve()),
        batch_requests_jsonl=str(batch_path.resolve()),
        quality_review_json=str(quality_path.resolve()),
        crawler_hints_json=str(crawler_hints_path.resolve()),
    )


def build_residential_markdown(
    run_id: str,
    scores: list[ResidentialCandidateScore],
    candidate_count: int,
    events: list[GovernmentEvent],
) -> str:
    lines = [
        "# 上海住宅开发机会推荐报告",
        "",
        f"- run_id: `{run_id}`",
        f"- event_count: {len(events)}",
        f"- candidate_count: {candidate_count}",
        "- analysis_type: residential_development_recommendation",
        "",
        "## 结论摘要",
        "",
    ]
    if not scores:
        lines.extend(
            [
                "当前公开证据不足，暂时不能形成住宅开发推荐区域。",
                "",
                "建议补充住宅用地、成交价、库存、去化周期、人口和竞品数据后重新评估。",
            ]
        )
        lines.extend(build_residential_next_steps())
        return "\n".join(lines)

    top = scores[0]
    if top.evidence_count < 3:
        lines.extend(
            [
                "当前公开证据不足以形成优先推荐区域；以下区域只能作为观察清单，不能作为拿地依据。",
                "",
                "## 观察区域",
                "",
            ]
        )
    elif top.recommendation == "暂不优先" or top.residential_development_score < 28:
        lines.extend(
            [
                f"当前没有达到抢先跟踪阈值的优先区域；最高分区域为 **{top.area.name}**，"
                f"住宅开发分 {top.residential_development_score}，建议级别为 **{top.recommendation}**。",
                "以下区域只能作为低优先级观察清单，不能作为拿地依据。建议等待新增规划、产业、交通、"
                "供地或市场验证信号后再进入投拓筛选。",
                "",
                "## 低优先级观察区域",
                "",
            ]
        )
    else:
        lead_label = "首选抢先跟踪区域" if top.recommendation == "优先抢先跟踪" else "首个谨慎抢先跟踪区域"
        lines.extend(
            [
                f"{lead_label}为 **{top.area.name}**，住宅开发分 {top.residential_development_score}，"
                f"建议级别为 **{top.recommendation}**。该结论优先识别人口正式流入前的政策、产业、交通、"
                "批复和供地信号，用于帮助开发商提前发现潜在居住需求，不替代正式投拓尽调。",
                "",
                "## 推荐区域",
                "",
            ]
        )
    event_by_id = {event.id: event for event in events}
    for idx, score in enumerate(scores, start=1):
        evidence = [event_by_id[event_id] for event_id in score.evidence_event_ids if event_id in event_by_id][:5]
        next_action = getattr(score, "next_action", "")
        score_lines = [
            f"### {idx}. {score.area.name}",
            "",
            f"- 建议级别: {score.recommendation}",
            f"- 住宅开发分: {score.residential_development_score}",
            f"- 置信度: {score.confidence}",
            f"- 具体位置描述: {score.area.description}",
            f"- 建议产品方向: {score.suggested_product}",
        ]
        if next_action:
            score_lines.append(f"- 下一步动作: {next_action}")
        score_lines.extend(
            [
                f"- 未来人口流入潜力分: {score.future_population_inflow_score}",
                f"- 人口流入前置信号分: {score.pre_inflow_signal_score}",
                f"- 抢先拿地窗口分: {score.land_grab_window_score}",
                f"- 需求驱动分: {score.demand_driver_score}",
                f"- 交通与公共服务分: {score.transport_public_service_score}",
                f"- 住宅供地/准入分: {score.residential_land_access_score}",
                f"- 住宅供应压力: {score.residential_supply_pressure}",
                f"- 成熟度/竞争暴露扣分: {score.maturity_penalty}",
                f"- 关键理由: {join_cn(score.key_reasons)}",
                f"- 主要风险: {join_cn(score.major_risks)}",
                f"- 有效证据数: {score.evidence_count}",
                "",
                "证据链:",
            ]
        )
        lines.extend(score_lines)
        for event in evidence:
            quote = event.evidence[0].quote if event.evidence else event.summary
            lines.append(f"- [{event.title}]({event.source_url})：{quote}")
        lines.append("")
    lines.extend(build_residential_next_steps())
    lines.extend(
        [
            "## 评分口径",
            "",
            "住宅开发分优先衡量未来人口流入潜力、人口流入前置信号和抢先拿地窗口；"
            "产业/招商、重大项目、规划批复、交通建设、住宅供地/准入和市场进入信号会加分。"
            "对住宅供应压力、区域成熟度/竞争暴露和地理定位不确定性做扣分。"
            "模型只使用已抓取的政府公开证据，不包含成交价、库存、去化周期、竞品、融资和税费测算。",
        ]
    )
    return "\n".join(lines)


def build_residential_next_steps() -> list[str]:
    return [
        "## 下一步核查清单",
        "",
        "- 控规与用地性质：核查候选地块是否允许住宅、租赁住房或商住混合，以及容积率和配建条件。",
        "- 供地窗口：持续跟踪未来 3-6 个月招拍挂、征收、控规调整、城市更新和土地储备公告。",
        "- 需求验证：核查产业导入岗位规模、通勤半径、落地节奏和目标客群支付能力。",
        "- 市场校验：补充周边库存、去化周期、成交价、竞品供应和租售比数据。",
        "- 兑现节奏：核查轨交、学校、医院和商业配套的批复、开工、竣工时间是否覆盖项目销售周期。",
        "",
    ]


def build_next_action_counts(scores: list[ResidentialCandidateScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for score in scores:
        label = leading_label(getattr(score, "next_action", ""))
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_triage_metrics(parsed_documents: list[ParsedDocument]) -> dict[str, int]:
    triaged = 0
    selected = 0
    skipped = 0
    for document in parsed_documents:
        relevance = document.metadata.get("relevance")
        if not isinstance(relevance, dict):
            continue
        triaged += 1
        if relevance.get("should_extract") is False:
            skipped += 1
        else:
            selected += 1
    return {
        "triaged_document_count": triaged,
        "selected_document_count": selected,
        "skipped_document_count": skipped,
    }


def join_cn(items: list[str]) -> str:
    cleaned = [item.rstrip("。；; ") for item in items if item]
    if not cleaned:
        return ""
    return "；".join(cleaned) + "。"
