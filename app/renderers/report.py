from __future__ import annotations

import csv
import html
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.branding import PRODUCT_DISPLAY_NAME, PRODUCT_TAGLINE, landscout_agent_workflow_payload
from app.core.config import settings
from app.core.utils import (
    actionable_errors,
    document_triage_skip_count,
    ensure_dir,
    leading_label,
    short_text,
    write_json,
)
from app.crawlers.models import FetchError, RawDocument
from app.llm.schemas import GovernmentEvent
from app.parsers.models import ParsedDocument
from app.scoring.scorer import CandidateScore


EVENT_TYPE_LABELS = {
    "land_supply": "土地供应",
    "land_transaction": "土地成交",
    "major_project": "重大项目",
    "infrastructure": "交通基建",
    "industrial_project": "产业项目",
    "public_service": "公共服务",
    "investment_signing": "招商签约",
    "project_approval": "项目批复",
    "planning_policy": "规划政策",
    "residential_supply": "住宅供应",
    "other": "其他",
}


class RenderedOutputs(BaseModel):
    recommendation_md: str
    opportunity_map_html: str
    visual_summary_html: str = ""
    events_csv: str
    signals_json: str
    evidence_pack_json: str
    pipeline_log: str
    investment_memo_md: str = ""
    monitoring_queries_json: str = ""
    embedding_index_json: str = ""
    batch_requests_jsonl: str = ""
    quality_review_json: str = ""
    crawler_hints_json: str = ""


def render_outputs(
    *,
    run_id: str,
    output_dir: Path,
    events: list[GovernmentEvent],
    scores: list[CandidateScore],
    raw_documents: list[RawDocument],
    parsed_documents: list[ParsedDocument],
    errors: list[FetchError] | list[dict[str, Any]],
    top_k: int,
    visited_sources: list[str] | None = None,
    discovered_sources: list[dict[str, Any]] | None = None,
) -> RenderedOutputs:
    ensure_dir(output_dir)
    top_scores = [score for score in scores if score.evidence_count > 0][:top_k]
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
    recommendation_path = output_dir / "recommendation.md"
    map_path = output_dir / "opportunity_map.html"
    visual_path = output_dir / "visual_summary.html"
    events_path = output_dir / "events.csv"
    signals_path = output_dir / "signals.json"
    evidence_path = output_dir / "evidence_pack.json"
    log_path = output_dir / "pipeline.log"

    write_events_csv(events_path, events)
    write_json(
        signals_path,
        {
            "product_name": PRODUCT_DISPLAY_NAME,
            "agent_workflow": landscout_agent_workflow_payload(),
            "run_id": run_id,
            "top_areas": [score.model_dump(mode="json") for score in top_scores],
            "event_count": len(events),
            "candidate_count": len(scores),
            "visited_sources": visited_sources or [],
            "discovered_sources": discovered_sources or [],
            "pipeline_metrics": pipeline_metrics,
            "event_type_counts": build_event_type_counts(events),
            "source_event_counts": build_source_event_counts(events),
        },
    )
    write_json(
        evidence_path,
        {
            "product_name": PRODUCT_DISPLAY_NAME,
            "agent_workflow": landscout_agent_workflow_payload(),
            "run_id": run_id,
            "discovered_sources": discovered_sources or [],
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
            "pipeline_metrics": pipeline_metrics,
            "event_type_counts": build_event_type_counts(events),
            "source_event_counts": build_source_event_counts(events),
        },
    )
    recommendation_path.write_text(build_markdown(run_id, top_scores, len(scores), events), encoding="utf-8")
    render_map(map_path, events, top_scores)
    render_visual_summary(
        visual_path,
        run_id=run_id,
        title="上海房地产开发选址可视化摘要",
        events=events,
        scores=top_scores,
        metric_fields=[
            ("opportunity_score", "机会总分"),
            ("industrial_import_score", "产业导入"),
            ("infrastructure_score", "交通基建"),
            ("public_service_score", "公共服务"),
            ("land_structure_score", "土地结构"),
            ("market_entry_score", "市场进入"),
            ("residential_supply_risk", "住宅供应风险"),
        ],
        main_score_field="opportunity_score",
        visited_source_count=len(visited_sources or []),
        raw_document_count=len(raw_documents),
        parsed_document_count=len(parsed_documents),
        error_count=len(visible_errors),
    )
    log_path.write_text(build_log(run_id, raw_documents, parsed_documents, events, scores, errors, visited_sources or []), encoding="utf-8")
    return RenderedOutputs(
        recommendation_md=str(recommendation_path.resolve()),
        opportunity_map_html=str(map_path.resolve()),
        visual_summary_html=str(visual_path.resolve()),
        events_csv=str(events_path.resolve()),
        signals_json=str(signals_path.resolve()),
        evidence_pack_json=str(evidence_path.resolve()),
        pipeline_log=str(log_path.resolve()),
    )


def update_latest(run_output_dir: Path) -> Path:
    source_dir = run_output_dir.resolve()
    latest_dir = (settings.outputs_dir / "shanghai" / "latest").resolve()
    if source_dir == latest_dir:
        return latest_dir
    ensure_dir(latest_dir.parent)
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_dir() and not latest_dir.is_symlink():
            shutil.rmtree(latest_dir)
        else:
            latest_dir.unlink()
    shutil.copytree(source_dir, latest_dir)
    return latest_dir


def write_events_csv(path: Path, events: list[GovernmentEvent]) -> None:
    ensure_dir(path.parent)
    fields = [
        "id",
        "event_type",
        "title",
        "district",
        "address",
        "project_name",
        "event_date",
        "amount_wanyuan",
        "area_sqm",
        "lat",
        "lon",
        "geo_confidence",
        "source_id",
        "source_url",
        "evidence_quote",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "id": event.id,
                    "event_type": event.event_type.value,
                    "title": event.title,
                    "district": event.district or "",
                    "address": event.address or "",
                    "project_name": event.project_name or "",
                    "event_date": event.event_date or "",
                    "amount_wanyuan": csv_optional(event.amount_wanyuan),
                    "area_sqm": csv_optional(event.area_sqm),
                    "lat": csv_optional(event.lat),
                    "lon": csv_optional(event.lon),
                    "geo_confidence": event.geo_confidence,
                    "source_id": event.source_id,
                    "source_url": event.source_url,
                    "evidence_quote": event.evidence[0].quote if event.evidence else "",
                }
            )


def build_pipeline_metrics(
    *,
    visited_source_count: int,
    raw_document_count: int,
    parsed_document_count: int,
    event_count: int,
    evidence_event_count: int,
    error_count: int,
    filtered_document_count: int = 0,
) -> dict[str, float | int]:
    parse_yield = parsed_document_count / raw_document_count if raw_document_count else 0.0
    event_yield = evidence_event_count / parsed_document_count if parsed_document_count else 0.0
    return {
        "visited_source_count": visited_source_count,
        "raw_document_count": raw_document_count,
        "parsed_document_count": parsed_document_count,
        "event_count": event_count,
        "evidence_event_count": evidence_event_count,
        "error_count": error_count,
        "filtered_document_count": filtered_document_count,
        "parse_yield": round(parse_yield, 4),
        "event_yield_per_parsed_document": round(event_yield, 4),
    }


def build_event_type_counts(events: list[GovernmentEvent]) -> dict[str, int]:
    return dict(Counter(event.event_type.value for event in events if event.evidence))


def build_source_event_counts(events: list[GovernmentEvent]) -> dict[str, int]:
    return dict(Counter(event.source_id for event in events if event.evidence))


def build_markdown(
    run_id: str,
    scores: list[CandidateScore],
    candidate_count: int,
    events: list[GovernmentEvent],
) -> str:
    lines = [
        f"# 上海房地产开发选址信号报告",
        "",
        f"- run_id: `{run_id}`",
        f"- event_count: {len(events)}",
        f"- candidate_count: {candidate_count}",
        "",
        "## Top 区域",
        "",
    ]
    if not scores:
        lines.extend(
            [
                "当前公开证据不足，暂时不能形成房地产开发选址推荐区域。",
                "",
                "建议补充土地供应、项目批复、产业导入、交通建设和公共服务等公开证据后重新评估。",
                "",
            ]
        )
        return "\n".join(lines)
    for idx, score in enumerate(scores, start=1):
        evidence = [event for event in events if event.id in score.evidence_event_ids][:5]
        lines.extend(
            [
                f"### {idx}. {score.area.name}",
                "",
                f"- 分数: {score.opportunity_score}",
                f"- 置信度: {score.confidence}",
                f"- 具体位置描述: {score.area.description}",
                f"- 推荐开发方向: {score.development_direction}",
                f"- 关键理由: {'；'.join(score.key_reasons)}",
                f"- 主要风险: {'；'.join(score.major_risks)}",
                f"- 有效证据数: {score.evidence_count}",
                "",
                "证据链:",
            ]
        )
        for event in evidence:
            quote = event.evidence[0].quote if event.evidence else ""
            lines.append(f"- [{event.title}]({event.source_url})：{quote}")
        lines.append("")
    lines.extend(
        [
            "## 评分口径",
            "",
            "opportunity_score = 0.25 * industrial_import_score + 0.18 * infrastructure_score + "
            "0.12 * public_service_score + 0.15 * land_structure_score + 0.12 * market_entry_score + "
            "0.08 * recency_score + 0.10 * evidence_confidence_score - 0.15 * residential_supply_risk - "
            "0.05 * geo_uncertainty_penalty。",
        ]
    )
    return "\n".join(lines)


def csv_optional(value: object | None) -> object:
    return "" if value is None else value


def render_map(path: Path, events: list[GovernmentEvent], scores: list[Any]) -> None:
    ensure_dir(path.parent)
    try:
        import folium
    except Exception:
        path.write_text(simple_map_html(events, scores), encoding="utf-8")
        return

    located = [event for event in events if event.evidence and event.lat is not None and event.lon is not None]
    center = [31.2304, 121.4737]
    if located:
        center = [sum(event.lat for event in located if event.lat is not None) / len(located), sum(event.lon for event in located if event.lon is not None) / len(located)]
    fmap = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    top_names = {score.area.name for score in scores}
    for score in scores:
        color = "red" if score.area.name in top_names else "blue"
        recommendation = getattr(score, "recommendation", "")
        next_action = getattr(score, "next_action", "")
        popup_lines = [
            f"<b>{html.escape(score.area.name)}</b>",
            f"score={html.escape(str(score.opportunity_score))}",
            f"evidence={html.escape(str(score.evidence_count))}",
        ]
        if recommendation:
            popup_lines.append(f"建议级别：{html.escape(str(recommendation))}")
        if next_action:
            popup_lines.append(f"下一步动作：{html.escape(str(next_action))}")
        if getattr(score, "key_reasons", None):
            popup_lines.append(html.escape(short_text("; ".join(score.key_reasons), 220)))
        folium.Circle(
            location=[score.area.lat, score.area.lon],
            radius=score.area.radius_m,
            color=color,
            fill=True,
            fill_opacity=0.12 if color == "red" else 0.05,
            popup=folium.Popup("<br>".join(popup_lines), max_width=420),
        ).add_to(fmap)

    for event in located:
        color = "green" if "land" in event.event_type.value else "orange" if "infrastructure" in event.event_type.value else "purple"
        quote = event.evidence[0].quote if event.evidence else ""
        popup_html = (
            f"<b>{html.escape(event.title)}</b><br>"
            f"{html.escape(event_type_label(event.event_type.value))}<br>"
            f"<a href=\"{html.escape(event.source_url)}\" target=\"_blank\">source</a><br>"
            f"{html.escape(short_text(quote, 260))}"
        )
        folium.Marker(
            location=[event.lat, event.lon],
            popup=folium.Popup(popup_html, max_width=420),
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(fmap)

    fmap.save(str(path))


def render_visual_summary(
    path: Path,
    *,
    run_id: str,
    title: str,
    events: list[GovernmentEvent],
    scores: list[Any],
    metric_fields: list[tuple[str, str]],
    main_score_field: str,
    visited_source_count: int = 0,
    raw_document_count: int = 0,
    parsed_document_count: int = 0,
    error_count: int = 0,
) -> None:
    ensure_dir(path.parent)
    evidence_events = [event for event in events if event.evidence]
    type_counts = Counter(event_type_label(event.event_type.value) for event in evidence_events)
    source_counts = Counter(event.source_id for event in evidence_events)
    month_counts = Counter(event.event_date[:7] for event in evidence_events if event.event_date)
    top_scores = list(scores)
    action_counts = Counter(
        leading_label(str(getattr(score, "next_action", "")))
        for score in top_scores
        if getattr(score, "next_action", "")
    )
    html_text = "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title>",
            "<style>",
            "body{margin:0;font-family:Arial,'Microsoft YaHei',sans-serif;background:#f6f7f9;color:#1f2933;}",
            "main{max-width:1280px;margin:0 auto;padding:24px;}",
            "h1{font-size:24px;margin:0 0 6px;}h2{font-size:18px;margin:0 0 14px;}h3{font-size:15px;margin:0 0 10px;}",
            ".brand{font-size:13px;font-weight:700;color:#175cd3;text-transform:uppercase;letter-spacing:0;margin-bottom:6px;}",
            ".sub{color:#667085;font-size:13px;margin-bottom:22px;}.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:16px;align-items:start;}",
            ".panel{background:#fff;border:1px solid #d9dee7;border-radius:8px;padding:16px;box-shadow:0 1px 2px rgba(16,24,40,.04);}",
            ".overview{margin-bottom:16px;}.stats{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-bottom:14px;}",
            ".stat{background:#f8fafc;border:1px solid #e4e7ec;border-radius:8px;padding:10px;}.stat strong{display:block;font-size:20px;}.stat span{font-size:12px;color:#667085;}",
            ".flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;}.flow-item{border-left:3px solid #2563eb;background:#f9fafb;padding:8px;font-size:12px;min-height:72px;}.flow-item strong{display:block;margin-bottom:4px;}",
            ".area{border-top:1px solid #eef1f5;padding:14px 0;}.area:first-of-type{border-top:0;padding-top:0;}",
            ".scoreline{display:flex;justify-content:space-between;gap:12px;font-size:13px;margin-bottom:8px;}.score{font-weight:700;}",
            ".bar{height:10px;background:#e7edf3;border-radius:999px;overflow:hidden;}.fill{height:100%;background:#2563eb;border-radius:999px;}",
            ".metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 14px;margin-top:12px;}.metric{font-size:12px;color:#344054;}",
            ".metric .bar{height:7px;margin-top:4px;}.metric .fill{background:#12b76a;}.risk .fill{background:#f97316;}",
            ".reason,.risk-note,.action-note{font-size:12px;line-height:1.55;margin-top:8px;color:#344054;}.risk-note{color:#9a3412;}.action-note{background:#f0f9ff;border-left:3px solid #0284c7;padding:8px;color:#0c4a6e;}",
            ".bars{display:grid;gap:10px;}.row{display:grid;grid-template-columns:120px 1fr 36px;gap:8px;align-items:center;font-size:12px;}",
            ".row .fill{background:#475467;}.timeline{display:flex;gap:6px;align-items:flex-end;height:160px;border-bottom:1px solid #d0d5dd;padding-top:10px;}",
            ".month{flex:1;min-width:18px;background:#0ea5e9;border-radius:4px 4px 0 0;position:relative;}.month span{position:absolute;bottom:-22px;left:50%;transform:translateX(-50%) rotate(-35deg);font-size:10px;color:#667085;white-space:nowrap;}",
            ".empty{color:#667085;font-size:14px;padding:20px 0;}@media(max-width:900px){.grid{grid-template-columns:1fr}.metrics,.stats,.flow{grid-template-columns:1fr}}",
            "</style>",
            "</head>",
            "<body><main>",
            f'<div class="brand">{html.escape(PRODUCT_DISPLAY_NAME)}</div>',
            f"<h1>{html.escape(title)}</h1>",
            f'<div class="sub">{html.escape(PRODUCT_TAGLINE)} · run_id: {html.escape(run_id)} · 有证据事件 {len(evidence_events)} 条 · 展示 Top {len(top_scores)} 区域</div>',
            '<section class="panel overview">',
            "<h2>Agent 工作流与数据管线</h2>",
            build_pipeline_overview(
                visited_source_count=visited_source_count,
                raw_document_count=raw_document_count,
                parsed_document_count=parsed_document_count,
                event_count=len(events),
                evidence_event_count=len(evidence_events),
                error_count=error_count,
            ),
            "</section>",
            '<div class="grid">',
            '<section class="panel">',
            "<h2>Top 区域评分拆解</h2>",
            build_area_visual_blocks(top_scores, metric_fields, main_score_field),
            "</section>",
            '<aside class="panel">',
            "<h2>证据结构</h2>",
            build_next_action_section(action_counts),
            "<h3>事件类型分布</h3>",
            build_count_bars(type_counts),
            "<h3 style=\"margin-top:22px\">数据源贡献</h3>",
            build_count_bars(source_counts),
            "<h3 style=\"margin-top:22px\">月份趋势</h3>",
            build_timeline(month_counts),
            "</aside>",
            "</div>",
            "</main></body></html>",
        ]
    )
    path.write_text(html_text, encoding="utf-8")


def build_pipeline_overview(
    *,
    visited_source_count: int,
    raw_document_count: int,
    parsed_document_count: int,
    event_count: int,
    evidence_event_count: int,
    error_count: int,
) -> str:
    stats = [
        ("数据源", visited_source_count),
        ("原始文档", raw_document_count),
        ("已解析文档", parsed_document_count),
        ("抽取事件", event_count),
        ("有证据事件", evidence_event_count),
        ("异常/访问限制", error_count),
    ]
    stat_html = "".join(
        f'<div class="stat"><strong>{value}</strong><span>{html.escape(label)}</span></div>'
        for label, value in stats
    )
    flow_html = "".join(
        f'<div class="flow-item"><strong>{html.escape(item["agent"])}</strong>{html.escape(item["role"])}</div>'
        for item in landscout_agent_workflow_payload()
    )
    return f'<div class="stats">{stat_html}</div><div class="flow">{flow_html}</div>'


def build_area_visual_blocks(
    scores: list[Any],
    metric_fields: list[tuple[str, str]],
    main_score_field: str,
) -> str:
    if not scores:
        return '<div class="empty">当前没有有证据支撑的候选区域。</div>'
    blocks: list[str] = []
    for idx, score in enumerate(scores, start=1):
        main_value = score_value(score, main_score_field)
        area_name = getattr(getattr(score, "area", None), "name", f"区域 {idx}")
        recommendation = getattr(score, "recommendation", "")
        blocks.extend(
            [
                '<div class="area">',
                f'<div class="scoreline"><strong>{idx}. {html.escape(area_name)}</strong><span class="score">{main_value:.2f}</span></div>',
                html_bar(main_value),
            ]
        )
        if recommendation:
            blocks.append(f'<div class="sub" style="margin:8px 0 0">建议级别：{html.escape(str(recommendation))}</div>')
        reasons = [str(item) for item in (getattr(score, "key_reasons", []) or []) if item]
        risks = [str(item) for item in (getattr(score, "major_risks", []) or []) if item]
        next_action = getattr(score, "next_action", "")
        if next_action:
            blocks.append(f'<div class="action-note"><strong>下一步动作</strong>：{html.escape(str(next_action))}</div>')
        if reasons:
            blocks.append(f'<div class="reason"><strong>理由</strong>：{html.escape(short_text("；".join(reasons[:2]), 220))}</div>')
        if risks:
            blocks.append(f'<div class="risk-note"><strong>风险</strong>：{html.escape(short_text("；".join(risks[:2]), 220))}</div>')
        blocks.append('<div class="metrics">')
        for field, label in metric_fields:
            value = score_value(score, field)
            risk_class = " risk" if "risk" in field or "pressure" in field or "penalty" in field else ""
            blocks.append(
                f'<div class="metric{risk_class}"><span>{html.escape(label)}：{value:.2f}</span>{html_bar(value)}</div>'
            )
        blocks.extend(["</div>", "</div>"])
    return "\n".join(blocks)


def build_count_bars(counts: Counter[str]) -> str:
    if not counts:
        return '<div class="empty">暂无有证据事件。</div>'
    max_count = max(counts.values()) or 1
    rows = []
    for label, count in counts.most_common():
        width = count / max_count * 100
        rows.append(
            f'<div class="row"><span>{html.escape(label)}</span><div class="bar"><div class="fill" style="width:{width:.1f}%"></div></div><strong>{count}</strong></div>'
        )
    return '<div class="bars">' + "\n".join(rows) + "</div>"


def build_next_action_section(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return '<h3>下一步动作分布</h3>' + build_count_bars(counts)


def build_timeline(counts: Counter[str]) -> str:
    if not counts:
        return '<div class="empty">暂无事件日期。</div>'
    max_count = max(counts.values()) or 1
    months = sorted(counts)
    bars = []
    for month in months:
        height = max(8, counts[month] / max_count * 140)
        bars.append(f'<div class="month" style="height:{height:.1f}px" title="{html.escape(month)}: {counts[month]}"><span>{html.escape(month)}</span></div>')
    return '<div class="timeline">' + "\n".join(bars) + "</div>"


def html_bar(value: float) -> str:
    width = max(0.0, min(float(value), 100.0))
    return f'<div class="bar"><div class="fill" style="width:{width:.1f}%"></div></div>'


def score_value(score: Any, field: str) -> float:
    value = getattr(score, field, 0) or 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def simple_map_html(events: list[GovernmentEvent], scores: list[Any]) -> str:
    rows = "".join(
        f"<li>{html.escape(score.area.name)} score={score.opportunity_score} evidence={score.evidence_count} "
        f"action={html.escape(str(getattr(score, 'next_action', '')))}</li>"
        for score in scores
    )
    points = "".join(
        f"<li>{html.escape(event.title)} ({event.lat}, {event.lon})</li>"
        for event in events
        if event.evidence and event.lat is not None and event.lon is not None
    )
    return f"<html><body><h1>Opportunity Map Fallback</h1><h2>Top Areas</h2><ul>{rows}</ul><h2>Events</h2><ul>{points}</ul></body></html>"


def event_type_label(value: str) -> str:
    return EVENT_TYPE_LABELS.get(value, value)


def build_log(
    run_id: str,
    raw_documents: list[RawDocument],
    parsed_documents: list[ParsedDocument],
    events: list[GovernmentEvent],
    scores: list[CandidateScore],
    errors: list[FetchError] | list[dict[str, Any]],
    visited_sources: list[str],
) -> str:
    lines = [
        f"run_id={run_id}",
        f"visited_sources={len(visited_sources)} {', '.join(visited_sources)}",
        f"raw_documents={len(raw_documents)}",
        f"parsed_documents={len(parsed_documents)}",
        f"events={len(events)}",
        f"candidate_scores={len(scores)}",
        "",
        "errors:",
    ]
    triage_skip_count = document_triage_skip_count(errors)
    visible_errors = actionable_errors(errors)
    for error in visible_errors:
        if isinstance(error, FetchError):
            lines.append(f"- {error.source_id} {error.url}: {error.reason} status={error.status_code}")
        else:
            lines.append(f"- {error}")
    if triage_skip_count:
        lines.extend(["", "filtered_documents:", f"- document_triage skipped {triage_skip_count} documents"])
    return "\n".join(lines)
