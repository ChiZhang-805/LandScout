from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.core.utils import short_text
from app.llm.openai_client import build_openai_client
from app.sources.registry import SourceConfig


DEFAULT_DISCOVERY_QUERIES = [
    "上海 控制性详细规划 公示 住宅 用地 官方",
    "上海 区政府 规划土地公示 商品住宅 地块",
    "上海 重大项目 开工 招商 签约 产业园 官方",
    "上海 轨道交通 新建工程 规划 公示 官方",
    "上海 新城 产业项目 公示 住宅需求 官方",
    "site:gov.cn 上海 住宅用地 拟出让 清单",
    "site:sh.gov.cn 上海 详细规划 实施深化 住宅用地",
]

SOURCE_SCOUT_KEYWORDS = [
    "住宅",
    "居住",
    "用地",
    "地块",
    "出让",
    "控详",
    "详细规划",
    "规划公示",
    "重大项目",
    "产业",
    "招商",
    "签约",
    "轨道交通",
    "新城",
    "园区",
    "公共服务",
]

BLOCKED_HOST_FRAGMENTS = (
    "baidu.",
    "bing.",
    "google.",
    "sogou.",
    "so.com",
    "duckduckgo.",
    "zhihu.",
    "weibo.",
    "douyin.",
    "bilibili.",
    "toutiao.",
    "163.com",
    "qq.com",
)

TRUSTED_NON_GOV_SUFFIXES = (
    "shanghaiinvest.com",
    "shcpe.cn",
)


class SearchResult(BaseModel):
    query: str
    title: str
    url: str
    snippet: str = ""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceDiscoveryCandidate(BaseModel):
    id: str
    name: str
    host: str
    base_urls: list[str]
    score: float
    official: bool
    query: str
    sample_title: str
    sample_snippet: str = ""
    matched_keywords: list[str] = Field(default_factory=list)
    reason: str = ""
    signal_categories: list[str] = Field(default_factory=list)
    scout_confidence: float = 0.0
    recommended_access_mode: str = "http_then_playwright"
    crawl_notes: str = ""
    monitoring_queries: list[str] = Field(default_factory=list)
    suggested_keywords: list[str] = Field(default_factory=list)

    def to_source_config(self, *, priority: int) -> SourceConfig:
        access_mode = self.recommended_access_mode
        if access_mode not in {"http", "http_then_playwright", "playwright_with_network_discovery"}:
            access_mode = "http_then_playwright"
        return SourceConfig(
            id=self.id,
            name=self.name,
            base_urls=self.base_urls,
            access_mode=access_mode,
            priority=priority,
            max_pages=max(2, len(self.base_urls)),
            delay=1.5,
            keywords=list(
                dict.fromkeys(
                    [
                        *SOURCE_SCOUT_KEYWORDS,
                        *self.matched_keywords,
                        *self.signal_categories,
                        *self.suggested_keywords,
                    ]
                )
            ),
            attachment_types=[".pdf", ".xlsx", ".xls", ".docx", ".doc", ".csv", ".json"],
            official=self.official,
            notes=f"Discovered by Source Scout. {self.reason} {self.crawl_notes}".strip(),
        )


class SourceCandidateAssessment(StrictModel):
    candidate_id: str
    keep: bool
    confidence: float = Field(ge=0, le=1)
    signal_categories: list[str]
    recommended_access_mode: str
    suggested_keywords: list[str]
    monitoring_queries: list[str]
    crawl_notes: str
    priority_reason: str


class SourceCandidateAssessmentList(StrictModel):
    assessments: list[SourceCandidateAssessment]


class DiscoveryRun(BaseModel):
    queries: list[str]
    candidates: list[SourceDiscoveryCandidate]
    errors: list[str] = Field(default_factory=list)


@dataclass
class WebSearchClient:
    client: httpx.Client | None = None

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        close_client = self.client is None
        client = self.client or httpx.Client(
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            timeout=settings.request_timeout_seconds,
        )
        try:
            response = client.get("https://duckduckgo.com/html/", params={"q": query})
            response.raise_for_status()
            return parse_duckduckgo_html(response.text, query=query, limit=limit)
        finally:
            if close_client:
                client.close()


class SourceScoutAgent:
    def __init__(self, search_client: WebSearchClient | None = None) -> None:
        self.search_client = search_client or WebSearchClient()

    def discover(
        self,
        *,
        max_sources: int = 5,
        query_limit: int = 6,
        include_non_official: bool = False,
        existing_urls: set[str] | None = None,
        existing_hosts: set[str] | None = None,
        use_llm_queries: bool = True,
    ) -> DiscoveryRun:
        queries = plan_discovery_queries(use_llm=use_llm_queries)[:query_limit]
        errors: list[str] = []
        results: list[SearchResult] = []
        for query in queries:
            try:
                results.extend(self.search_client.search(query, limit=10))
            except Exception as exc:
                errors.append(f"search failed for {query!r}: {exc}")
        candidates = rank_search_results(
            results,
            max_sources=max_sources,
            include_non_official=include_non_official,
            existing_urls=existing_urls or set(),
            existing_hosts=existing_hosts or set(),
        )
        if use_llm_queries and settings.openai_api_key and candidates:
            try:
                candidates = evaluate_candidates_with_llm(candidates, max_sources=max_sources)
            except Exception as exc:
                errors.append(f"source candidate assessment failed: {exc}")
        return DiscoveryRun(queries=queries, candidates=candidates, errors=errors)


def plan_discovery_queries(*, use_llm: bool = True) -> list[str]:
    if use_llm and settings.openai_api_key:
        planned = plan_queries_with_llm()
        if planned:
            return planned
    return list(DEFAULT_DISCOVERY_QUERIES)


def plan_queries_with_llm() -> list[str]:
    try:
        client = build_openai_client()
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You plan web search queries for LandScout Agent. "
                        "Return only search queries that can discover public Shanghai official or quasi-official sources. "
                        "Focus on early residential demand signals before population inflow: planning notices, land supply, "
                        "major projects, industrial investment, transport, schools, hospitals, new towns, and district announcements."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Generate 8 concise Chinese web search queries for Shanghai residential development source discovery. "
                        "Prefer official government, district, park, investment promotion, planning, transport, housing, and land-market sources."
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "SourceScoutQueryPlan",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 8,
                            }
                        },
                        "required": ["queries"],
                    },
                    "strict": True,
                }
            },
        )
        payload = json.loads(response.output_text)
    except Exception:
        return []
    queries = [clean_query(item) for item in payload.get("queries", []) if clean_query(item)]
    merged = [*queries, *DEFAULT_DISCOVERY_QUERIES]
    return list(dict.fromkeys(merged))


def evaluate_candidates_with_llm(
    candidates: list[SourceDiscoveryCandidate],
    *,
    max_sources: int,
) -> list[SourceDiscoveryCandidate]:
    client = build_openai_client()
    payload = [
        {
            "candidate_id": candidate.id,
            "name": candidate.name,
            "host": candidate.host,
            "urls": candidate.base_urls,
            "official": candidate.official,
            "query": candidate.query,
            "title": candidate.sample_title,
            "snippet": candidate.sample_snippet,
            "matched_keywords": candidate.matched_keywords,
            "heuristic_score": candidate.score,
        }
        for candidate in candidates
    ]
    response = client.responses.create(
        model=settings.openai_fast_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are Source Scout V2 for LandScout Agent. Evaluate candidate public data sources for Shanghai "
                    "residential-development early signals. Keep sources likely to contain official or quasi-official "
                    "planning, land, industrial investment, transport, school, hospital, public-service, or district project news. "
                    "Suggest crawl strategy and monitoring queries. Do not keep media/social/search pages."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "SourceCandidateAssessmentList",
                "schema": SourceCandidateAssessmentList.model_json_schema(),
                "strict": True,
            }
        },
    )
    assessments = SourceCandidateAssessmentList.model_validate(json.loads(response.output_text)).assessments
    by_id = {assessment.candidate_id: assessment for assessment in assessments}
    enriched: list[SourceDiscoveryCandidate] = []
    for candidate in candidates:
        assessment = by_id.get(candidate.id)
        if not assessment:
            enriched.append(candidate)
            continue
        if not assessment.keep and assessment.confidence >= 0.5:
            continue
        enriched.append(
            candidate.model_copy(
                update={
                    "score": round(candidate.score + assessment.confidence * 12, 2),
                    "signal_categories": assessment.signal_categories,
                    "scout_confidence": assessment.confidence,
                    "recommended_access_mode": assessment.recommended_access_mode,
                    "crawl_notes": assessment.crawl_notes,
                    "monitoring_queries": assessment.monitoring_queries,
                    "suggested_keywords": assessment.suggested_keywords,
                    "reason": f"{candidate.reason}; LLM: {assessment.priority_reason}",
                }
            )
        )
    enriched.sort(key=lambda item: item.score, reverse=True)
    return enriched[:max_sources]


def rank_search_results(
    results: list[SearchResult],
    *,
    max_sources: int,
    include_non_official: bool,
    existing_urls: set[str],
    existing_hosts: set[str],
) -> list[SourceDiscoveryCandidate]:
    grouped: dict[str, list[tuple[float, SearchResult, list[str], bool, str]]] = defaultdict(list)
    seen_urls: set[str] = set()
    for result in results:
        url = normalize_public_url(result.url)
        if not url or url in seen_urls or url in existing_urls:
            continue
        seen_urls.add(url)
        host = host_from_url(url)
        if not host or host in existing_hosts or is_blocked_host(host):
            continue
        official = is_official_host(host)
        trusted = official or is_trusted_non_gov_host(host)
        if not trusted and not include_non_official:
            continue
        score, matched, reason = score_result(result, url, official=official, trusted=trusted)
        if score < 22:
            continue
        grouped[host].append((score, result.model_copy(update={"url": url}), matched, official, reason))

    candidates: list[SourceDiscoveryCandidate] = []
    for host, items in grouped.items():
        items.sort(key=lambda item: item[0], reverse=True)
        best_score, best_result, matched, official, reason = items[0]
        urls = [item[1].url for item in items[:3]]
        candidates.append(
            SourceDiscoveryCandidate(
                id=dynamic_source_id(host),
                name=source_name_from_result(best_result, host),
                host=host,
                base_urls=urls,
                score=round(best_score + min(len(items), 3) * 2, 2),
                official=official,
                query=best_result.query,
                sample_title=best_result.title,
                sample_snippet=best_result.snippet,
                matched_keywords=matched,
                reason=reason,
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_sources]


def parse_duckduckgo_html(html: str, *, query: str, limit: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for container in soup.select(".result"):
        link = container.select_one("a.result__a") or container.find("a", href=True)
        if not link:
            continue
        href = str(link.get("href", ""))
        url = unwrap_duckduckgo_url(href)
        if not url:
            continue
        title = link.get_text(" ", strip=True)
        snippet_node = container.select_one(".result__snippet")
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        results.append(SearchResult(query=query, title=title, url=url, snippet=snippet))
        if len(results) >= limit:
            break
    return results


def unwrap_duckduckgo_url(value: str) -> str:
    if not value:
        return ""
    parsed = urllib.parse.urlparse(value)
    if parsed.netloc.endswith("duckduckgo.com") or parsed.path.startswith("/l/"):
        qs = urllib.parse.parse_qs(parsed.query)
        redirected = qs.get("uddg", [""])[0]
        if redirected:
            return urllib.parse.unquote(redirected)
    if value.startswith("//"):
        return "https:" + value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return ""


def normalize_public_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    path = parsed.path or "/"
    query = parsed.query
    normalized = urllib.parse.urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", query, ""))
    return normalized


def host_from_url(value: str) -> str:
    return urllib.parse.urlparse(value).netloc.split(":", 1)[0].lower()


def is_blocked_host(host: str) -> bool:
    return any(fragment in host for fragment in BLOCKED_HOST_FRAGMENTS)


def is_official_host(host: str) -> bool:
    return host.endswith(".gov.cn") or host == "gov.cn"


def is_trusted_non_gov_host(host: str) -> bool:
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in TRUSTED_NON_GOV_SUFFIXES)


def score_result(result: SearchResult, url: str, *, official: bool, trusted: bool) -> tuple[float, list[str], str]:
    text = f"{result.title} {result.snippet} {url}"
    matched = [keyword for keyword in SOURCE_SCOUT_KEYWORDS if keyword in text]
    score = 0.0
    if official:
        score += 30
    elif trusted:
        score += 18
    score += min(len(matched) * 6, 42)
    if re.search(r"20(2[4-9]|3[0-9])", text):
        score += 8
    if any(token in text for token in ("公示", "公告", "批复", "清单", "规划", "项目")):
        score += 8
    if any(url.lower().endswith(suffix) for suffix in (".pdf", ".xls", ".xlsx", ".doc", ".docx")):
        score += 4
    reason = "official host" if official else "trusted non-government source"
    if matched:
        reason += "; matched " + ", ".join(matched[:6])
    return score, matched, reason


def source_name_from_result(result: SearchResult, host: str) -> str:
    title = re.sub(r"\s+", " ", result.title).strip(" -_|")
    return short_text(title or host, 48)


def dynamic_source_id(host: str) -> str:
    digest = hashlib.sha1(host.encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"[^0-9a-z]+", "_", host.lower()).strip("_")[:32]
    return f"dynamic_{stem}_{digest}"


def clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
