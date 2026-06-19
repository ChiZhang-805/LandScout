from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.branding import PRODUCT_DISPLAY_NAME, PRODUCT_TAGLINE
from app.core.config import PROJECT_ROOT, settings
from app.core.utils import actionable_errors, document_triage_skip_count
from app.pipeline.orchestrator import LandScoutAgentState
from app.sources.registry import SourceConfig, SourceRegistry


COMMON_ATTACHMENT_TYPES = [".pdf", ".xlsx", ".xls", ".docx", ".doc", ".csv", ".json"]
BRAND_LOGO_FILENAME = "landscout_agent_icon.png"
DEFAULT_CUSTOM_KEYWORDS = [
    "住宅",
    "居住",
    "用地",
    "地块",
    "出让",
    "规划",
    "控规",
    "重大项目",
    "产业",
    "招商",
    "轨道交通",
    "学校",
    "医院",
]

OUTPUT_FILE_LABELS = {
    "recommendation_md": "推荐报告",
    "opportunity_map_html": "空间地图",
    "visual_summary_html": "可视化摘要",
    "events_csv": "事件 CSV",
    "signals_json": "信号 JSON",
    "evidence_pack_json": "证据包",
    "pipeline_log": "流水线日志",
    "investment_memo_md": "投拓备忘录",
    "monitoring_queries_json": "监测查询",
    "embedding_index_json": "Embedding 索引",
    "batch_requests_jsonl": "Batch 请求",
    "quality_review_json": "质量复核",
    "crawler_hints_json": "爬虫建议",
}

OUTPUT_FILE_GROUPS = {
    "recommendation_md": "核心报告",
    "opportunity_map_html": "核心报告",
    "visual_summary_html": "核心报告",
    "events_csv": "数据证据",
    "signals_json": "数据证据",
    "evidence_pack_json": "数据证据",
    "pipeline_log": "数据证据",
    "investment_memo_md": "AI 辅助",
    "monitoring_queries_json": "AI 辅助",
    "embedding_index_json": "AI 辅助",
    "batch_requests_jsonl": "AI 辅助",
    "quality_review_json": "AI 辅助",
    "crawler_hints_json": "AI 辅助",
}


ALLOWED_OUTPUT_FILENAMES = {
    "recommendation.md",
    "opportunity_map.html",
    "visual_summary.html",
    "events.csv",
    "signals.json",
    "evidence_pack.json",
    "pipeline.log",
    "investment_memo.md",
    "monitoring_queries.json",
    "embedding_index.json",
    "batch_requests.jsonl",
    "quality_review.json",
    "crawler_hints.json",
}


class WebRunRequest(BaseModel):
    city: str = "shanghai"
    live: bool = False
    days: int = Field(default=540, ge=1, le=3650)
    top_k: int = Field(default=8, ge=1, le=30)
    source_limit: int = Field(default=12, ge=1, le=100)
    use_builtin_sources: bool = True
    custom_sources_text: str = ""
    openai_api_key: str = Field(default="", max_length=300)
    amap_key: str = ""


def source_options(registry: SourceRegistry) -> list[dict[str, Any]]:
    return [
        {
            "id": source.id,
            "name": source.name,
            "priority": source.priority,
            "access_mode": source.access_mode,
            "urls": source.urls,
        }
        for source in registry.sources
    ]


def build_runtime_registry(
    base_registry: SourceRegistry,
    *,
    source_limit: int,
    use_builtin_sources: bool,
    custom_sources_text: str,
) -> SourceRegistry:
    custom_sources = parse_custom_sources_text(custom_sources_text)
    builtin_sources = base_registry.select(source_limit) if use_builtin_sources else []
    if not builtin_sources and not custom_sources:
        raise ValueError("至少需要选择内置数据源，或配置一个自定义数据源。")
    if not builtin_sources:
        return SourceRegistry(custom_sources)
    return SourceRegistry(builtin_sources).merged(custom_sources)


def parse_custom_sources_text(value: str) -> list[SourceConfig]:
    text = (value or "").strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        return parse_custom_sources_json(text)
    sources: list[SourceConfig] = []
    for idx, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sources.append(parse_custom_source_line(line, idx))
    return sources


def parse_custom_sources_json(text: str) -> list[SourceConfig]:
    payload = json.loads(text)
    if isinstance(payload, dict):
        items = payload.get("sources", [payload])
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("自定义源 JSON 必须是对象、对象数组，或包含 sources 数组的对象。")
    return [source_from_mapping(item, idx) for idx, item in enumerate(items, start=1)]


def source_from_mapping(item: Any, idx: int) -> SourceConfig:
    if not isinstance(item, dict):
        raise ValueError("每个自定义源必须是对象。")
    raw_urls = item.get("base_urls") or item.get("urls") or item.get("url")
    if isinstance(raw_urls, str):
        urls = split_multi_value(raw_urls)
    elif isinstance(raw_urls, list):
        urls = [str(url).strip() for url in raw_urls if str(url).strip()]
    else:
        urls = []
    if not urls:
        raise ValueError("自定义源缺少 url/base_urls。")
    name = str(item.get("name") or source_name_from_url(urls[0])).strip()
    keywords = normalize_keywords(item.get("keywords") or item.get("keyword") or [])
    access_mode = item.get("access_mode") or "http_then_playwright"
    source_id = str(item.get("id") or custom_source_id(name, urls, idx))
    return SourceConfig(
        id=source_id,
        name=name,
        base_urls=urls,
        access_mode=access_mode,
        priority=int(item.get("priority") or 95 + idx),
        max_pages=max(int(item.get("max_pages") or 2), len(urls)),
        delay=float(item.get("delay") or 1.5),
        keywords=keywords or list(DEFAULT_CUSTOM_KEYWORDS),
        attachment_types=normalize_keywords(item.get("attachment_types") or COMMON_ATTACHMENT_TYPES),
        official=bool(item.get("official", True)),
        notes=str(item.get("notes") or "Custom source configured from the LandScout Agent web UI."),
    )


def parse_custom_source_line(line: str, idx: int) -> SourceConfig:
    parts = [part.strip() for part in re.split(r"\s*[|\t]\s*", line) if part.strip()]
    url_index = next((pos for pos, part in enumerate(parts) if part.startswith(("http://", "https://"))), -1)
    if url_index < 0:
        if line.startswith(("http://", "https://")):
            parts = [line]
            url_index = 0
        else:
            raise ValueError(f"第 {idx} 行自定义源缺少 URL。")
    name = parts[0] if url_index > 0 else source_name_from_url(parts[url_index])
    urls = split_multi_value(parts[url_index])
    keywords = split_multi_value(parts[url_index + 1]) if url_index + 1 < len(parts) else []
    return SourceConfig(
        id=custom_source_id(name, urls, idx),
        name=name,
        base_urls=urls,
        access_mode="http_then_playwright",
        priority=95 + idx,
        max_pages=max(2, len(urls)),
        delay=1.5,
        keywords=keywords or list(DEFAULT_CUSTOM_KEYWORDS),
        attachment_types=COMMON_ATTACHMENT_TYPES,
        official=True,
        notes="Custom source configured from the LandScout Agent web UI.",
    )


def split_multi_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[,，;；]\s*", str(value)) if part.strip()]


def normalize_keywords(value: Any) -> list[str]:
    return list(dict.fromkeys(split_multi_value(value)))


def source_name_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc or url


def custom_source_id(name: str, urls: list[str], idx: int) -> str:
    seed = "|".join([name, *urls, str(idx)])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    parsed_host = urllib.parse.urlparse(urls[0]).netloc.lower() if urls else "custom"
    stem = re.sub(r"[^0-9a-z]+", "_", parsed_host).strip("_") or "source"
    return f"custom_{idx:02d}_{stem[:28]}_{digest}"


def state_to_web_response(state: LandScoutAgentState, *, top_k: int | None = None) -> dict[str, Any]:
    top_scores = [score for score in state.residential_scores if score.evidence_count > 0]
    if top_k is not None:
        top_scores = top_scores[:top_k]
    visible_errors = actionable_errors(state.errors)
    return {
        "run_id": state.run_id,
        "visited_sources": state.visited_sources,
        "raw_document_count": len(state.raw_documents),
        "parsed_document_count": len(state.parsed_documents),
        "event_count": len(state.events),
        "candidate_count": len(state.residential_scores),
        "error_count": len(visible_errors),
        "filtered_document_count": document_triage_skip_count(state.errors),
        "api_enrichment_enabled": state.api_enrichment_enabled,
        "top_areas": [
            {
                "name": score.area.name,
                "description": score.area.description,
                "lat": score.area.lat,
                "lon": score.area.lon,
                "radius_m": score.area.radius_m,
                "source": score.area.source,
                "score": score.residential_development_score,
                "recommendation": score.recommendation,
                "confidence": score.confidence,
                "evidence_count": score.evidence_count,
                "next_action": score.next_action,
                "suggested_product": score.suggested_product,
                "key_reasons": score.key_reasons,
                "major_risks": score.major_risks,
            }
            for score in top_scores
        ],
        "errors": visible_errors[:20],
        "files": output_file_links(state),
        "city": "shanghai",
    }


def output_file_links(state: LandScoutAgentState) -> list[dict[str, str]]:
    if not state.outputs:
        return []
    links: list[dict[str, str]] = []
    for key, path_value in state.outputs.model_dump().items():
        if not path_value:
            continue
        filename = Path(path_value).name
        if filename not in ALLOWED_OUTPUT_FILENAMES:
            continue
        links.append(
            {
                "key": key,
                "label": OUTPUT_FILE_LABELS.get(key, key),
                "group": OUTPUT_FILE_GROUPS.get(key, "其他文件"),
                "filename": filename,
                "path": path_value,
                "url": f"/runs/{state.run_id}/files/{filename}",
            }
        )
    return links


def build_dashboard_html(registry: SourceRegistry) -> str:
    sources_json = json.dumps(source_options(registry), ensure_ascii=False)
    source_limit_max = max(1, len(registry.sources))
    source_limit_default = min(12, source_limit_max)
    app_name = html.escape(PRODUCT_DISPLAY_NAME)
    tagline = html.escape(PRODUCT_TAGLINE)
    key_state = "已配置 OpenAI Key" if settings.openai_api_key else "未配置 OpenAI Key"
    amap_state = "已配置后端 AMap Key" if settings.amap_key else "未配置后端 AMap Key"
    logo_html = (
        '<img class="brand-logo" src="/assets/landscout-agent-icon.png" alt="" aria-hidden="true">'
        if brand_logo_path()
        else ""
    )
    html_text = DASHBOARD_TEMPLATE
    html_text = html_text.replace("__APP_NAME__", app_name)
    html_text = html_text.replace("__TAGLINE__", tagline)
    html_text = html_text.replace("__KEY_STATE__", html.escape(key_state))
    html_text = html_text.replace("__AMAP_STATE__", html.escape(amap_state))
    html_text = html_text.replace("__LOGO_HTML__", logo_html)
    html_text = html_text.replace("__SOURCES_JSON__", sources_json.replace("</", "<\\/"))
    html_text = html_text.replace("__SOURCE_LIMIT_MAX__", str(source_limit_max))
    html_text = html_text.replace("__SOURCE_LIMIT_DEFAULT__", str(source_limit_default))
    return html_text


def brand_logo_path() -> Path | None:
    candidates = [
        PROJECT_ROOT / "asset" / BRAND_LOGO_FILENAME,
        PROJECT_ROOT / BRAND_LOGO_FILENAME,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__APP_NAME__</title>
  <style>
    :root{--bg:#eef2f6;--panel:#fff;--line:#d7dee8;--text:#172033;--muted:#667085;--blue:#1d4ed8;--green:#078669;--amber:#b45309;--red:#b42318;--teal:#0f766e;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,"Microsoft YaHei",sans-serif;letter-spacing:0}
    button,input,select,textarea{font:inherit;letter-spacing:0}
    .topbar{min-height:72px;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:14px 28px;border-bottom:1px solid var(--line);background:#fff}
    .brand-block{display:flex;align-items:center;gap:12px;min-width:0}.brand-logo{width:44px;height:44px;border-radius:10px;object-fit:cover;border:1px solid #e4e7ec;box-shadow:0 1px 3px rgba(16,24,40,.12);flex:0 0 auto}.brand-copy{min-width:0}
    .brand{font-weight:700;font-size:18px;color:#111827}.tag{font-size:12px;color:var(--muted);margin-top:4px}.key{font-size:12px;color:var(--muted);white-space:nowrap;text-align:right;line-height:1.6}
    .shell{display:grid;grid-template-columns:420px minmax(0,1fr);grid-template-rows:104px minmax(360px,auto) var(--dashboard-focus-row,auto) minmax(118px,auto) minmax(118px,auto) minmax(118px,auto);column-gap:16px;row-gap:12px;max-width:1680px;margin:0 auto;padding:16px;align-items:stretch}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:0 1px 2px rgba(16,24,40,.04)}.controls,.results{display:contents}
    .controls>.panel:nth-child(1){grid-column:1;grid-row:1 / span 2}.controls>.panel:nth-child(2){grid-column:1;grid-row:3;min-height:0;display:flex;flex-direction:column}.controls>.panel:nth-child(3){grid-column:1;grid-row:4 / span 3}
    .results>.status-card{grid-column:2;grid-row:1}.results>.map-panel{grid-column:2;grid-row:2 / span 2}.results>#summaryPanel{grid-column:2;grid-row:4}.results>#areasPanel{grid-column:2;grid-row:5}.results>#filesPanel{grid-column:2;grid-row:6}
    h1,h2,h3{margin:0}h2{font-size:15px;margin-bottom:12px;display:flex;align-items:center;gap:8px}h3{font-size:14px}.field{display:grid;gap:6px;margin-bottom:10px}
    label,.field-title{font-size:12px;color:#344054;font-weight:700;display:flex;align-items:center;gap:6px}.inline{display:grid;grid-template-columns:1fr 1fr;gap:10px}.toggle{display:flex;align-items:center;gap:8px;font-size:13px;color:#344054}
    .ui-icon{width:16px;height:16px;stroke:currentColor;stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round;display:inline-block;vertical-align:-3px;flex:0 0 auto}.ui-icon.small{width:14px;height:14px}.ui-icon-lg{width:18px;height:18px}.icon-blue{color:var(--blue)}.icon-muted{color:var(--muted)}.icon-green{color:var(--green)}.icon-amber{color:var(--amber)}
    input,select,textarea{width:100%;border:1px solid #cbd5e1;border-radius:6px;background:#fff;color:var(--text);padding:8px 10px;min-height:36px;font-size:14px}
    .secret-field{position:relative}.secret-field input{padding-right:42px}.secret-toggle{position:absolute;right:7px;top:50%;transform:translateY(-50%);width:30px;height:30px;border:0;border-radius:6px;background:transparent;color:#475467;display:grid;place-items:center;cursor:pointer}.secret-toggle:hover{background:#eef2f6;color:#1d2939}.secret-toggle:focus-visible{outline:2px solid #84adff;outline-offset:1px}
    textarea{min-height:116px;resize:vertical;line-height:1.45}.controls>.panel:nth-child(2) textarea{height:283px;min-height:0;resize:vertical}.run{width:100%;border:0;border-radius:6px;background:var(--blue);color:#fff;font-weight:700;min-height:40px;cursor:pointer;font-size:15px;display:flex;align-items:center;justify-content:center;gap:8px}.run:disabled{opacity:.62;cursor:default}
    .secondary{border:1px solid #cbd5e1;background:#fff;color:#344054;border-radius:6px;min-height:34px;padding:7px 10px;cursor:pointer}.source-list{display:grid;gap:6px;max-height:188px;overflow:auto}
    .controls>.panel:last-child{min-height:0;display:flex;flex-direction:column}.controls>.panel:last-child .source-list{flex:1;min-height:188px;max-height:none;align-content:start}
    .source-item{display:grid;grid-template-columns:28px minmax(0,1fr);gap:8px;align-items:start;border-top:1px solid #eef2f6;padding-top:7px;font-size:12px}.source-item:first-child{border-top:0;padding-top:0}
    .rank{display:grid;place-items:center;width:24px;height:24px;border-radius:50%;background:#eef4ff;color:var(--blue);font-weight:700}.source-name{font-weight:700;overflow-wrap:anywhere;display:flex;align-items:center;gap:6px}.source-meta{color:var(--muted);font-size:11px;overflow-wrap:anywhere}
    .status-card{display:flex;flex-direction:column;justify-content:center;gap:10px;padding:12px 20px;background:linear-gradient(180deg,#fff,#f8fbff)}
    .status{display:flex;align-items:center;gap:10px;font-size:14px;font-weight:600;color:#25314a}.dot{width:10px;height:10px;border-radius:50%;background:#98a2b3;box-shadow:0 0 0 5px #eef2f7}.dot.running{background:var(--amber);box-shadow:0 0 0 5px #fff3e0}.dot.done{background:var(--green);box-shadow:0 0 0 5px #e7f7f2}.dot.error{background:var(--red);box-shadow:0 0 0 5px #fee4e2}
    .progress-block{display:grid;gap:8px}.progress-track{height:12px;background:#e3eaf2;border:1px solid #d8e1ec;border-radius:999px;overflow:hidden;position:relative;box-shadow:inset 0 1px 2px rgba(16,24,40,.08)}.progress-fill{height:100%;width:0%;border-radius:999px;background:#98a2b3;transition:width .45s ease,background .2s ease}.progress-fill.running{background:linear-gradient(90deg,var(--blue),var(--teal));position:relative}.progress-fill.running:after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.46),transparent);animation:progress-shine 1.5s linear infinite}.progress-fill.done{background:var(--green)}.progress-fill.error{background:var(--red)}.progress-meta{display:flex;justify-content:space-between;gap:16px;font-size:12px;color:var(--muted);line-height:1.45}.progress-meta span:last-child{font-weight:700;color:#475467}@keyframes progress-shine{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(108px,1fr));gap:8px}.metric{border:1px solid #e4e7ec;border-radius:8px;background:#f8fafc;padding:10px}.metric strong{display:flex;align-items:center;gap:6px;font-size:20px}.metric span{font-size:12px;color:var(--muted)}
    .map-panel{padding:0;overflow:hidden;display:flex;flex-direction:column;min-height:0}.map-head{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid #e4e7ec;flex:0 0 auto}.map-title{font-size:15px;font-weight:700;line-height:1.2;display:flex;align-items:center;gap:8px}.map-sub{font-size:12px;color:var(--muted);margin-top:5px}.map-sub:empty{display:none}.map-tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}.map-tools input{width:190px;min-height:32px;font-size:12px}.map-tools button{min-height:32px;border:1px solid #cbd5e1;border-radius:6px;background:#fff;color:#344054;padding:6px 9px;cursor:pointer;font-size:12px;display:flex;align-items:center;gap:6px}
    #opportunityMap{height:auto;min-height:0;flex:1;position:relative;background:#dfe7ef;overflow:hidden}.map-placeholder{position:absolute;inset:0;background:linear-gradient(90deg,rgba(255,255,255,.35) 1px,transparent 1px),linear-gradient(rgba(255,255,255,.35) 1px,transparent 1px),#dfe7ef;background-size:54px 54px;color:#334155}.map-placeholder:before{content:"上海坐标示意图";position:absolute;left:18px;top:16px;font-size:13px;font-weight:700;color:#334155}.map-placeholder:after{content:"输入高德 JS API Key 后显示真实地图";position:absolute;left:18px;top:38px;font-size:12px;color:#667085}.coord-dot{position:absolute;border:2px solid rgba(29,78,216,.82);background:rgba(29,78,216,.20);border-radius:50%;transform:translate(-50%,-50%);display:grid;place-items:center;color:#0f172a;font-size:12px;font-weight:700}.coord-label{position:absolute;transform:translate(12px,-50%);background:rgba(255,255,255,.92);border:1px solid #d7dee8;border-radius:6px;padding:5px 7px;font-size:12px;white-space:nowrap;box-shadow:0 1px 2px rgba(16,24,40,.08)}.legend{position:absolute;right:12px;bottom:12px;background:rgba(255,255,255,.94);border:1px solid #d7dee8;border-radius:6px;padding:8px 10px;font-size:12px;color:#344054}
    #areasPanel,#filesPanel{min-height:118px}.areas{display:grid;gap:10px}.area{border-top:1px solid #eef2f6;padding-top:12px}.area:first-child{border-top:0;padding-top:0}.area-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.area-title{font-size:15px;font-weight:700;display:flex;align-items:center;gap:6px}.badge{font-size:12px;color:#0f5132;background:#ecfdf3;border:1px solid #abefc6;border-radius:999px;padding:3px 8px;white-space:nowrap}
    .scoreline{display:grid;grid-template-columns:110px 1fr 58px;gap:8px;align-items:center;margin:10px 0}.bar{height:9px;background:#e7edf3;border-radius:999px;overflow:hidden}.fill{height:100%;background:var(--blue);border-radius:999px}.small{font-size:12px;color:var(--muted);line-height:1.55}.reason{font-size:12px;line-height:1.55;color:#344054;margin-top:8px}.risk{color:#7c2d12}.file-groups{display:grid;gap:14px}.file-group-title{font-size:12px;font-weight:700;color:#475467;margin-bottom:8px}.files{display:grid;grid-template-columns:repeat(auto-fit,minmax(138px,1fr));gap:8px}.file{min-height:38px;font-size:12px;text-decoration:none;color:#1849a9;border:1px solid #b2ccff;background:#eff4ff;border-radius:6px;padding:7px 10px;display:flex;align-items:center;gap:6px;justify-content:flex-start}
    .errors{display:grid;gap:6px}.error-row{font-size:12px;color:#7a271a;background:#fff7ed;border:1px solid #fed7aa;border-radius:6px;padding:8px;overflow-wrap:anywhere;display:flex;align-items:flex-start;gap:6px}.empty{font-size:13px;color:var(--muted);padding:18px 0}
    @media(max-width:980px){.shell{grid-template-columns:1fr;grid-template-rows:auto}.controls,.results{display:grid;gap:12px}.controls>.panel,.results>.panel{grid-column:auto!important;grid-row:auto!important}.controls>.panel:last-child .source-list{max-height:260px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.topbar{height:auto;align-items:flex-start;padding:14px 16px;flex-direction:column}.key{white-space:normal}}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand-block">__LOGO_HTML__<div class="brand-copy"><div class="brand">__APP_NAME__</div><div class="tag">__TAGLINE__</div></div></div>
    <div class="key">__KEY_STATE__<br>__AMAP_STATE__</div>
  </header>
  <main class="shell">
    <section class="controls">
      <div class="panel">
        <h2><span data-icon="sliders"></span>运行参数</h2>
        <div class="field">
          <label for="city"><span data-icon="building"></span>城市</label>
          <select id="city">
            <option value="shanghai" selected>上海</option>
          </select>
        </div>
        <div class="field">
          <label for="mode"><span data-icon="radar"></span>数据模式</label>
          <select id="mode">
            <option value="fixture">演示数据</option>
            <option value="live">公开网站抓取</option>
          </select>
        </div>
        <div class="inline">
          <div class="field"><label for="sourceLimit"><span data-icon="database"></span>内置源数量</label><input id="sourceLimit" type="number" min="1" max="__SOURCE_LIMIT_MAX__" step="1" value="__SOURCE_LIMIT_DEFAULT__"></div>
          <div class="field"><label for="days"><span data-icon="calendar"></span>回看天数</label><input id="days" type="number" min="1" max="3650" value="540"></div>
        </div>
        <div class="inline">
          <div class="field"><label for="topK"><span data-icon="ranking"></span>Top K</label><input id="topK" type="number" min="1" max="30" value="8"></div>
          <div class="field">
            <label for="useBuiltin"><span data-icon="layers"></span>内置源</label>
            <select id="useBuiltin">
              <option value="true" selected>使用内置源</option>
              <option value="false">不使用内置源</option>
            </select>
          </div>
        </div>
        <div class="field">
          <label for="openaiKey"><span data-icon="key"></span>OpenAI API Key</label>
          <div class="secret-field">
            <input id="openaiKey" type="password" autocomplete="off" placeholder="请填写 OpenAI API Key">
            <button class="secret-toggle" type="button" data-secret-toggle="openaiKey" aria-label="显示内容" aria-pressed="false"><span data-icon="eye"></span></button>
          </div>
        </div>
        <div class="field">
          <label for="amapKey"><span data-icon="key"></span>高德地图 API Key</label>
          <div class="secret-field">
            <input id="amapKey" type="password" autocomplete="off" placeholder="请填写高德地图 API Key">
            <button class="secret-toggle" type="button" data-secret-toggle="amapKey" aria-label="显示内容" aria-pressed="false"><span data-icon="eye"></span></button>
          </div>
        </div>
        <div class="field">
          <label for="amapSecurityCode"><span data-icon="shield"></span>高德安全密钥</label>
          <div class="secret-field">
            <input id="amapSecurityCode" type="password" autocomplete="off" placeholder="请填写高德安全密钥">
            <button class="secret-toggle" type="button" data-secret-toggle="amapSecurityCode" aria-label="显示内容" aria-pressed="false"><span data-icon="eye"></span></button>
          </div>
        </div>
        <button id="runBtn" class="run"><span data-icon="search"></span>搜索并分析</button>
      </div>
      <div class="panel">
        <h2><span data-icon="edit"></span>自定义源</h2>
        <textarea id="customSources" spellcheck="false" placeholder="浦东规划 | https://www.pudong.gov.cn/ | 住宅,地块,规划&#10;临港公示 | https://www.lingang.gov.cn/ | 产业,招商,公示"></textarea>
      </div>
      <div class="panel">
        <h2><span data-icon="database"></span>本次内置源预览</h2>
        <div id="sourceList" class="source-list"></div>
      </div>
    </section>
    <section class="results">
      <div class="panel status-card">
        <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">等待搜索</span></div>
        <div class="progress-block">
          <div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
          <div class="progress-meta"><span id="progressStep">等待用户启动搜索</span><span id="progressPercent">0%</span></div>
        </div>
      </div>
      <div class="panel map-panel">
        <div class="map-head">
          <div>
            <div class="map-title"><span data-icon="map"></span>机会地图</div>
            <div id="mapSub" class="map-sub"></div>
          </div>
          <div class="map-tools">
            <button id="refreshMapBtn" type="button"><span data-icon="refresh"></span>刷新地图</button>
          </div>
        </div>
        <div id="opportunityMap"><div class="map-placeholder"></div></div>
      </div>
      <div id="summaryPanel" class="panel">
        <h2><span data-icon="chart"></span>运行结果</h2>
        <div class="empty">尚未运行。</div>
      </div>
      <div id="areasPanel" class="panel">
        <h2><span data-icon="target"></span>候选区域</h2>
        <div class="empty">尚未生成候选区域。</div>
      </div>
      <div id="filesPanel" class="panel">
        <h2><span data-icon="file"></span>输出文件</h2>
        <div class="empty">尚未生成输出文件。</div>
      </div>
    </section>
  </main>
  <script>
    if("scrollRestoration" in history){
      history.scrollRestoration = "manual";
    }
    const BUILTIN_SOURCES = __SOURCES_JSON__;
    const sourceLimit = document.getElementById("sourceLimit");
    const useBuiltin = document.getElementById("useBuiltin");
    const sourceList = document.getElementById("sourceList");
    const customSourcesInput = document.getElementById("customSources");
    const runBtn = document.getElementById("runBtn");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const progressFill = document.getElementById("progressFill");
    const progressStep = document.getElementById("progressStep");
    const progressPercent = document.getElementById("progressPercent");
    const summaryPanel = document.getElementById("summaryPanel");
    const areasPanel = document.getElementById("areasPanel");
    const filesPanel = document.getElementById("filesPanel");
    const mapContainer = document.getElementById("opportunityMap");
    const shell = document.querySelector(".shell");
    const mapPanel = document.querySelector(".map-panel");
    const mapSub = document.getElementById("mapSub");
    const refreshMapBtn = document.getElementById("refreshMapBtn");
    const openaiKeyInput = document.getElementById("openaiKey");
    const amapKeyInput = document.getElementById("amapKey");
    const amapSecurityCodeInput = document.getElementById("amapSecurityCode");
    let lastAreas = [];
    let amapInstance = null;
    let amapScriptKey = "";
    let amapLoading = null;
    let progressTimer = null;
    let progressValue = 0;
    let customSourcesUserAdjusted = false;
    const ICONS = {
      alert: '<path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.7 2.4 17.4A2 2 0 0 0 4.1 20h15.8a2 2 0 0 0 1.7-2.6L13.7 3.7a2 2 0 0 0-3.4 0Z"/>',
      building: '<path d="M4 21V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v16"/><path d="M16 8h2a2 2 0 0 1 2 2v11"/><path d="M8 7h4"/><path d="M8 11h4"/><path d="M8 15h4"/><path d="M9 21v-3h2v3"/>',
      calendar: '<path d="M8 2v4"/><path d="M16 2v4"/><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M3 10h18"/>',
      chart: '<path d="M3 3v18h18"/><path d="m7 16 4-4 3 3 5-7"/>',
      database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
      edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
      eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
      eyeOff: '<path d="M3 3l18 18"/><path d="M10.6 10.6A3 3 0 0 0 13.4 13.4"/><path d="M9.9 5.2A9.7 9.7 0 0 1 12 5c6.5 0 10 7 10 7a17.8 17.8 0 0 1-3.2 4.2"/><path d="M6.1 6.1C3.5 7.9 2 12 2 12s3.5 7 10 7a9.9 9.9 0 0 0 5-1.3"/>',
      file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h5"/>',
      key: '<circle cx="7.5" cy="15.5" r="3.5"/><path d="m10 13 8-8"/><path d="m15 5 4 4"/><path d="m13 7 2 2"/>',
      layers: '<path d="m12 2 9 5-9 5-9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
      map: '<path d="M9 18 3 21V6l6-3 6 3 6-3v15l-6 3Z"/><path d="M9 3v15"/><path d="M15 6v15"/>',
      pin: '<path d="M12 21s7-4.4 7-11a7 7 0 1 0-14 0c0 6.6 7 11 7 11Z"/><circle cx="12" cy="10" r="2.5"/>',
      radar: '<path d="M12 12 19 5"/><circle cx="12" cy="12" r="2"/><path d="M4.9 19.1a10 10 0 1 1 14.2 0"/><path d="M7.8 16.2a6 6 0 1 1 8.4 0"/>',
      ranking: '<path d="M5 19V9"/><path d="M12 19V5"/><path d="M19 19v-7"/><path d="M3 19h18"/>',
      refresh: '<path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/><path d="M3 21v-5h5"/><path d="M3 12A9 9 0 0 1 18.5 5.8L21 8"/><path d="M21 3v5h-5"/>',
      search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
      shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-5"/>',
      sliders: '<path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/><path d="M2 14h4"/><path d="M10 8h4"/><path d="M18 16h4"/>',
      target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    };

    function escapeHtml(value){
      return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]));
    }
    function icon(name, extraClass = ""){
      const body = ICONS[name] || "";
      return `<svg class="ui-icon ${extraClass}" viewBox="0 0 24 24" aria-hidden="true">${body}</svg>`;
    }
    function hydrateStaticIcons(){
      document.querySelectorAll("[data-icon]").forEach(node => {
        node.outerHTML = icon(node.getAttribute("data-icon") || "");
      });
    }
    function setupSecretToggles(){
      document.querySelectorAll("[data-secret-toggle]").forEach(button => {
        const input = document.getElementById(button.getAttribute("data-secret-toggle") || "");
        if(!input){
          return;
        }
        button.addEventListener("click", () => {
          const shouldHide = input.type === "text";
          input.type = shouldHide ? "password" : "text";
          button.setAttribute("aria-label", shouldHide ? "显示内容" : "隐藏内容");
          button.setAttribute("aria-pressed", String(!shouldHide));
          button.innerHTML = icon(shouldHide ? "eye" : "eyeOff");
        });
      });
    }
    function setStatus(state, text){
      statusDot.className = "dot " + state;
      statusText.textContent = text;
    }
    function setProgress(value, step, state = ""){
      progressValue = Math.max(0, Math.min(100, Number(value || 0)));
      progressFill.style.width = `${progressValue}%`;
      progressFill.className = `progress-fill ${state}`.trim();
      progressStep.textContent = step;
      progressPercent.textContent = `${Math.round(progressValue)}%`;
    }
    function startProgress(payload){
      stopProgressTimer();
      const live = Boolean(payload.live);
      const sourceCount = Math.max(1, Number(payload.source_limit || 1));
      const stagePlan = live
        ? [
            [10, "准备公开网站源"],
            [28, `抓取 ${sourceCount} 个数据源`],
            [48, "解析网页与附件"],
            [68, "抽取政策和人口流入信号"],
            [84, "生成候选区域与地图"],
            [94, "汇总报告与输出文件"],
          ]
        : [
            [18, "加载演示数据"],
            [42, "解析样例政策与地块信号"],
            [70, "生成候选区域与地图"],
            [92, "汇总报告与输出文件"],
          ];
      let stageIndex = 0;
      setProgress(stagePlan[0][0], stagePlan[0][1], "running");
      progressTimer = window.setInterval(() => {
        const [target, step] = stagePlan[Math.min(stageIndex, stagePlan.length - 1)];
        if(progressValue >= target - 1 && stageIndex < stagePlan.length - 1){
          stageIndex += 1;
        }
        const [nextTarget, nextStep] = stagePlan[Math.min(stageIndex, stagePlan.length - 1)];
        const increment = live ? Math.max(0.35, (nextTarget - progressValue) * 0.035) : Math.max(1.2, (nextTarget - progressValue) * 0.08);
        setProgress(Math.min(nextTarget, progressValue + increment), nextStep, "running");
      }, 700);
    }
    function stopProgressTimer(){
      if(progressTimer){
        window.clearInterval(progressTimer);
        progressTimer = null;
      }
    }
    function alignMapToViewportBottom(){
      if(!mapPanel){
        return;
      }
      mapPanel.style.height = "";
    }
    function alignCustomSourcesEditor(){
      if(!customSourcesInput || !shell || customSourcesUserAdjusted){
        return;
      }
      const panel = customSourcesInput.closest(".panel");
      if(!panel){
        return;
      }
      if(window.matchMedia("(max-width: 980px)").matches){
        shell.style.removeProperty("--dashboard-focus-row");
        customSourcesInput.style.height = "";
        customSourcesInput.style.minHeight = "";
        return;
      }
      shell.style.removeProperty("--dashboard-focus-row");
      customSourcesInput.style.height = "";
      customSourcesInput.style.minHeight = "";
      const panelRect = panel.getBoundingClientRect();
      const editorRect = customSourcesInput.getBoundingClientRect();
      const panelChromeHeight = Math.max(0, panelRect.height - editorRect.height);
      const bottomBreathingRoom = 8;
      const minimumEditorHeight = 8;
      const panelTopAtInitialScroll = panelRect.top + window.scrollY;
      const requestedFocusRowHeight = Math.floor(window.innerHeight - panelTopAtInitialScroll - bottomBreathingRoom);
      const focusRowHeight = Math.max(panelChromeHeight + minimumEditorHeight, requestedFocusRowHeight);
      shell.style.setProperty("--dashboard-focus-row", `${focusRowHeight}px`);
      const editorHeight = Math.max(minimumEditorHeight, focusRowHeight - panelChromeHeight);
      customSourcesInput.style.minHeight = `${editorHeight}px`;
      customSourcesInput.style.height = `${editorHeight}px`;
    }
    function alignDashboardLayout(){
      alignMapToViewportBottom();
      alignCustomSourcesEditor();
    }
    function shouldUseBuiltinSources(){
      return useBuiltin.value === "true";
    }
    function clampSourceLimit(){
      const max = Math.max(1, Number(sourceLimit.max || BUILTIN_SOURCES.length || 1));
      const min = Math.max(1, Number(sourceLimit.min || 1));
      const raw = Number(sourceLimit.value || min);
      const numeric = Number.isFinite(raw) ? Math.trunc(raw) : min;
      const clamped = Math.max(min, Math.min(max, numeric));
      if(sourceLimit.value !== String(clamped)){
        sourceLimit.value = String(clamped);
      }
      return clamped;
    }
    function renderSources(){
      const count = clampSourceLimit();
      const selected = shouldUseBuiltinSources() ? BUILTIN_SOURCES.slice(0, count) : [];
      if(!selected.length){
        sourceList.innerHTML = '<div class="empty">未启用内置源。</div>';
        return;
      }
      sourceList.innerHTML = selected.map((source, idx) => `
        <div class="source-item">
          <div class="rank">${idx + 1}</div>
          <div>
            <div class="source-name">${icon("database","small icon-blue")}${escapeHtml(source.name)}</div>
            <div class="source-meta">${escapeHtml(source.id)} · ${escapeHtml(source.access_mode)}</div>
            <div class="source-meta">${escapeHtml((source.urls || []).join(" "))}</div>
          </div>
        </div>
      `).join("");
    }
    function metricsHtml(data){
      const items = [
        ["database", "数据源", (data.visited_sources || []).length],
        ["file", "原始文档", data.raw_document_count],
        ["layers", "解析文档", data.parsed_document_count],
        ["radar", "事件", data.event_count],
        ["target", "候选区", data.candidate_count],
        ["layers", "已过滤", data.filtered_document_count || 0],
        ["alert", "异常", data.error_count],
      ];
      return `<div class="metrics">${items.map(([iconName,label,value]) => `<div class="metric"><strong>${icon(iconName, "small icon-blue")}${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("")}</div>`;
    }
    function renderFileGroups(files){
      if(!files.length){
        return '<div class="empty">未生成输出文件。</div>';
      }
      const groups = [];
      files.forEach(file => {
        const groupName = file.group || "其他文件";
        let group = groups.find(item => item.name === groupName);
        if(!group){
          group = { name: groupName, files: [] };
          groups.push(group);
        }
        group.files.push(file);
      });
      return `<div class="file-groups">${groups.map(group => `
        <section class="file-group">
          <div class="file-group-title">${escapeHtml(group.name)}</div>
          <div class="files">${group.files.map(file => `<a class="file" href="${escapeHtml(file.url)}" target="_blank" rel="noopener">${icon("file","small")}${escapeHtml(file.label)}</a>`).join("")}</div>
        </section>
      `).join("")}</div>`;
    }
    function renderResult(data){
      lastAreas = data.top_areas || [];
      summaryPanel.innerHTML = `
        <h2>${icon("chart")}运行结果</h2>
        ${metricsHtml(data)}
        <div class="small" style="margin-top:10px">run_id: ${escapeHtml(data.run_id)} · visited_sources: ${escapeHtml((data.visited_sources || []).join(", "))}</div>
      `;
      const areas = data.top_areas || [];
      areasPanel.innerHTML = `<h2>${icon("target")}候选区域</h2>` + (areas.length ? `<div class="areas">${areas.map((area, idx) => {
        const score = Math.max(0, Math.min(100, Number(area.score || 0)));
        return `
          <div class="area">
            <div class="area-head">
              <div><div class="area-title">${icon("pin","small icon-blue")}${idx + 1}. ${escapeHtml(area.name)}</div><div class="small">${escapeHtml(area.description)}</div></div>
              <div class="badge">${escapeHtml(area.recommendation)}</div>
            </div>
            <div class="scoreline"><span class="small">住宅开发分</span><div class="bar"><div class="fill" style="width:${score}%"></div></div><strong>${score.toFixed(2)}</strong></div>
            <div class="reason">${escapeHtml(area.next_action)}</div>
            <div class="reason">${escapeHtml((area.key_reasons || []).slice(0,2).join("；"))}</div>
            <div class="reason risk">${escapeHtml((area.major_risks || []).slice(0,2).join("；"))}</div>
          </div>`;
      }).join("")}</div>` : '<div class="empty">当前没有有证据支撑的候选区域。</div>');
      const files = data.files || [];
      filesPanel.innerHTML = `<h2>${icon("file")}输出文件</h2>` + renderFileGroups(files);
      renderOpportunityMap(lastAreas);
    }
    function locatedAreas(areas){
      return (areas || []).filter(area => Number.isFinite(Number(area.lat)) && Number.isFinite(Number(area.lon)));
    }
    function renderOpportunityMap(areas){
      const located = locatedAreas(areas);
      if(!located.length){
        mapSub.textContent = "没有可绘制坐标的候选区域。";
        mapContainer.innerHTML = '<div class="map-placeholder"><div class="legend">暂无坐标点</div></div>';
        return;
      }
      const amapKey = amapKeyInput.value.trim();
      if(!amapKey){
        renderCoordinateFallback(located);
        return;
      }
      loadAmap(amapKey)
        .then(() => renderAmap(located))
        .catch(() => renderCoordinateFallback(located, "高德地图加载失败，已切换为坐标示意图。"));
    }
    function loadAmap(key){
      if(window.AMap && amapScriptKey === key){
        return Promise.resolve();
      }
      if(amapLoading && amapScriptKey === key){
        return amapLoading;
      }
      const securityCode = amapSecurityCodeInput.value.trim();
      if(securityCode){
        window._AMapSecurityConfig = { securityJsCode: securityCode };
      }
      amapScriptKey = key;
      amapLoading = new Promise((resolve, reject) => {
        const existing = document.getElementById("amap-js-sdk");
        if(existing){ existing.remove(); }
        const script = document.createElement("script");
        script.id = "amap-js-sdk";
        script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(key)}&plugin=AMap.Scale,AMap.ToolBar`;
        script.async = true;
        script.onload = () => window.AMap ? resolve() : reject(new Error("AMap unavailable"));
        script.onerror = reject;
        document.head.appendChild(script);
      });
      return amapLoading;
    }
    function renderAmap(areas){
      if(amapInstance){
        amapInstance.destroy();
        amapInstance = null;
      }
      mapContainer.innerHTML = "";
      amapInstance = new AMap.Map("opportunityMap", {
        center: [121.4737, 31.2304],
        zoom: 10,
        viewMode: "2D",
        mapStyle: "amap://styles/normal",
      });
      amapInstance.addControl(new AMap.Scale());
      amapInstance.addControl(new AMap.ToolBar({ position: { right: "12px", top: "12px" } }));
      const overlays = [];
      areas.forEach((area, idx) => {
        const center = [Number(area.lon), Number(area.lat)];
        const radius = Math.max(800, Math.min(12000, Number(area.radius_m || 5000)));
        const circle = new AMap.Circle({
          center,
          radius,
          strokeColor: "#1d4ed8",
          strokeWeight: 2,
          strokeOpacity: 0.88,
          fillColor: "#2563eb",
          fillOpacity: 0.20,
          zIndex: 20,
        });
        const marker = new AMap.Marker({
          position: center,
          anchor: "center",
          content: `<div style="min-width:26px;height:26px;border-radius:50%;background:#1d4ed8;color:white;display:grid;place-items:center;font-weight:700;border:2px solid white;box-shadow:0 2px 6px rgba(15,23,42,.28)">${idx + 1}</div>`,
          zIndex: 30,
        });
        const info = new AMap.InfoWindow({
          content: `<div style="font-size:13px;line-height:1.55"><strong>${escapeHtml(area.name)}</strong><br>住宅开发分 ${escapeHtml(Number(area.score || 0).toFixed(2))}<br>坐标 ${escapeHtml(Number(area.lat).toFixed(6))}, ${escapeHtml(Number(area.lon).toFixed(6))}<br>${escapeHtml(area.recommendation || "")}</div>`,
          offset: new AMap.Pixel(0, -18),
        });
        circle.on("click", () => info.open(amapInstance, center));
        marker.on("click", () => info.open(amapInstance, center));
        amapInstance.add([circle, marker]);
        overlays.push(circle, marker);
      });
      if(overlays.length){
        amapInstance.setFitView(overlays, false, [64, 64, 64, 64], 14);
      }
      mapSub.textContent = `已绘制 ${areas.length} 个候选区域半透明覆盖圆。`;
    }
    function renderCoordinateFallback(areas, note){
      if(amapInstance){
        amapInstance.destroy();
        amapInstance = null;
      }
      const bounds = { minLon: 120.75, maxLon: 122.15, minLat: 30.62, maxLat: 31.88 };
      const points = areas.map((area, idx) => {
        const lon = Number(area.lon);
        const lat = Number(area.lat);
        const x = Math.max(4, Math.min(96, ((lon - bounds.minLon) / (bounds.maxLon - bounds.minLon)) * 100));
        const y = Math.max(5, Math.min(95, ((bounds.maxLat - lat) / (bounds.maxLat - bounds.minLat)) * 100));
        const size = Math.max(42, Math.min(150, Number(area.radius_m || 5000) / 65));
        return `<div class="coord-dot" title="${escapeHtml(area.name)}" style="left:${x}%;top:${y}%;width:${size}px;height:${size}px">${idx + 1}</div><div class="coord-label" style="left:${x}%;top:${y}%">${escapeHtml(area.name)} · ${escapeHtml(lat.toFixed(5))}, ${escapeHtml(lon.toFixed(5))}</div>`;
      }).join("");
      mapContainer.innerHTML = `<div class="map-placeholder">${points}<div class="legend">${escapeHtml(note || "坐标为候选区中心点；圆半径来自评分候选区。")}</div></div>`;
      mapSub.textContent = `已绘制 ${areas.length} 个候选区域坐标覆盖圆。`;
    }
    async function readApiJson(response){
      const status = `${response.status} ${response.statusText || ""}`.trim();
      const text = await response.text();
      if(!text.trim()){
        throw new Error(`接口返回空响应（HTTP ${status || "unknown"}）。通常是服务器超时、重启或连接被中断。`);
      }
      try{
        return JSON.parse(text);
      }catch(error){
        const snippet = text.replace(/\s+/g, " ").slice(0, 240);
        throw new Error(`接口返回的不是有效 JSON（HTTP ${status || "unknown"}）：${snippet || "无可读内容"}`);
      }
    }
    async function runSearch(){
      const payload = {
        live: document.getElementById("mode").value === "live",
        city: document.getElementById("city").value,
        days: Number(document.getElementById("days").value || 540),
        top_k: Number(document.getElementById("topK").value || 8),
        source_limit: clampSourceLimit(),
        use_builtin_sources: shouldUseBuiltinSources(),
        custom_sources_text: customSourcesInput.value,
        openai_api_key: openaiKeyInput.value.trim(),
        amap_key: amapKeyInput.value.trim(),
      };
      runBtn.disabled = true;
      setStatus("running", "正在抓取、解析、分析");
      startProgress(payload);
      try{
        const response = await fetch("/api/recommend-residential", {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body:JSON.stringify(payload),
        });
        const data = await readApiJson(response);
        if(!response.ok){ throw new Error(data.detail || "运行失败"); }
        renderResult(data);
        setStatus("done", "完成");
        stopProgressTimer();
        setProgress(100, "分析完成，可以查看候选区域和输出文件", "done");
      }catch(error){
        setStatus("error", error.message || String(error));
        stopProgressTimer();
        setProgress(Math.max(progressValue, 12), "运行失败，请查看错误信息或降低源数量重试", "error");
      }finally{
        runBtn.disabled = false;
      }
    }
    sourceLimit.addEventListener("input", renderSources);
    useBuiltin.addEventListener("change", renderSources);
    runBtn.addEventListener("click", runSearch);
    refreshMapBtn.addEventListener("click", () => renderOpportunityMap(lastAreas));
    customSourcesInput.addEventListener("pointerdown", (event) => {
      const rect = customSourcesInput.getBoundingClientRect();
      if(event.clientX >= rect.right - 28 && event.clientY >= rect.bottom - 28){
        customSourcesUserAdjusted = true;
        if(shell){
          shell.style.removeProperty("--dashboard-focus-row");
        }
      }
    });
    window.addEventListener("resize", () => window.requestAnimationFrame(alignDashboardLayout));
    hydrateStaticIcons();
    setupSecretToggles();
    window.scrollTo(0, 0);
    alignDashboardLayout();
    renderSources();
  </script>
</body>
</html>
"""
