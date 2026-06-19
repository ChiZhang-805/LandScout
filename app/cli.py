from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.core.config import effective_openai_api_key, settings
from app.core.utils import leading_label, short_text
from app.llm.batch import submit_batch_request_file
from app.llm.client import MissingLLMKey
from app.llm.openai_client import OpenAINonRecoverableError
from app.pipeline.orchestrator import LandScoutAgent


console = Console()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    for command_name in ("landscout-demo", "shanghai-demo"):
        demo = sub.add_parser(command_name, help="Run the LandScout Agent Shanghai end-to-end demo.")
        demo.add_argument("--live", action="store_true", help="Fetch live official public sources and require OPENAI_API_KEY.")
        demo.add_argument("--days", type=positive_int, default=540)
        demo.add_argument("--top-k", type=positive_int, default=8)

    residential = sub.add_parser(
        "recommend-residential",
        help="Fetch/analyze Shanghai public signals and recommend residential development areas.",
    )
    residential.add_argument("--live", action="store_true", help="Fetch live official public sources and require OPENAI_API_KEY.")
    residential.add_argument("--days", type=positive_int, default=540)
    residential.add_argument("--top-k", type=positive_int, default=8)
    residential.add_argument("--source-limit", type=positive_int, default=12)
    residential.add_argument("--discover-sources", action="store_true", help="Use Source Scout web search to add dynamic sources for this run.")
    residential.add_argument("--dynamic-source-limit", type=positive_int, default=5)
    residential.add_argument("--discovery-query-limit", type=positive_int, default=6)
    residential.add_argument("--include-non-official-sources", action="store_true")

    discover_web = sub.add_parser("discover-web-sources", help="Search the web for additional public sources.")
    discover_web.add_argument("--max-sources", type=positive_int, default=8)
    discover_web.add_argument("--query-limit", type=positive_int, default=6)
    discover_web.add_argument("--include-non-official-sources", action="store_true")

    discover = sub.add_parser("discover-source", help="Discover public links/XHR for a source.")
    discover.add_argument("--source-id", required=True)

    fetch = sub.add_parser("fetch-source", help="Fetch one configured public source.")
    fetch.add_argument("--source-id", required=True)
    fetch.add_argument("--pages", type=positive_int, default=2)

    score = sub.add_parser("score", help="Score an existing run.")
    score.add_argument("--run-id", required=True)
    score.add_argument("--days", type=positive_int, default=540)

    render = sub.add_parser("render", help="Render an existing run.")
    render.add_argument("--run-id", required=True)
    render.add_argument("--top-k", type=positive_int, default=8)

    submit_batch = sub.add_parser("submit-batch", help="Submit a generated OpenAI Batch JSONL file.")
    submit_batch.add_argument("--batch-file", required=True)
    submit_batch.add_argument("--completion-window", default="24h")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    landscout_agent = LandScoutAgent()
    try:
        if args.command in {"landscout-demo", "shanghai-demo"}:
            state = landscout_agent.run_landscout_demo(live=args.live, days=args.days, top_k=args.top_k)
            print_state_summary(state, limit=args.top_k)
            return 0
        if args.command == "recommend-residential":
            state = landscout_agent.recommend_residential(
                live=args.live,
                days=args.days,
                top_k=args.top_k,
                source_limit=args.source_limit,
                discover_sources=args.discover_sources,
                dynamic_source_limit=args.dynamic_source_limit,
                discovery_query_limit=args.discovery_query_limit,
                include_non_official_sources=args.include_non_official_sources,
            )
            print_residential_summary(state, limit=args.top_k)
            return 0
        if args.command == "discover-web-sources":
            discovery = landscout_agent.discover_web_sources(
                max_sources=args.max_sources,
                query_limit=args.query_limit,
                include_non_official=args.include_non_official_sources,
            )
            console.print("[bold]Source Scout queries[/bold]")
            for query in discovery.queries:
                console.print(f"- {query}")
            console.print("[bold]Discovered candidate sources[/bold]")
            for candidate in discovery.candidates:
                console.print(f"- {candidate.id} score={candidate.score} official={candidate.official} host={candidate.host}")
                console.print(f"  name: {candidate.name}")
                for url in candidate.base_urls:
                    console.print(f"  url: {url}")
            for error in discovery.errors:
                console.print(f"[red]ERROR[/red] {error}")
            return 0 if discovery.candidates else 2
        if args.command == "discover-source":
            result = landscout_agent.discover_source(source_id=args.source_id)
            console.print(f"run_id: {result.run_id}")
            console.print(f"visited_sources: {', '.join(result.visited_sources)}")
            for url in result.discovered_urls[:50]:
                console.print(url)
            for error in result.errors:
                console.print(f"[red]ERROR[/red] {error.source_id} {error.url}: {error.reason}")
            return 0 if not result.errors else 2
        if args.command == "fetch-source":
            state = landscout_agent.fetch_source(source_id=args.source_id, pages=args.pages)
            console.print(f"run_id: {state.run_id}")
            console.print(f"raw_documents: {len(state.raw_documents)}")
            console.print(f"parsed_documents: {len(state.parsed_documents)}")
            for error in state.errors:
                console.print(f"[red]ERROR[/red] {error}")
            return 0 if state.raw_documents else 2
        if args.command == "score":
            state = landscout_agent.score_run(run_id=args.run_id, days=args.days)
            if state.residential_scores:
                print_residential_scores(state)
            else:
                print_scores(state)
            return 0
        if args.command == "render":
            state = landscout_agent.render_run(run_id=args.run_id, top_k=args.top_k)
            print_outputs(state.outputs)
            return 0
        if args.command == "submit-batch":
            if not effective_openai_api_key():
                raise MissingLLMKey("OPENAI_API_KEY is required to submit an OpenAI Batch job.")
            batch_id = submit_batch_request_file(Path(args.batch_file), completion_window=args.completion_window)
            console.print(f"batch_id: {batch_id}")
            return 0
    except (MissingLLMKey, OpenAINonRecoverableError) as exc:
        console.print(f"[red]Live demo failed:[/red] {exc}")
        return 3
    except Exception as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        return 1
    return 1


def print_state_summary(state, limit: int = 3) -> None:  # type: ignore[no-untyped-def]
    console.print(f"[bold]{settings.app_name}[/bold]")
    console.print(f"run_id: {state.run_id}")
    console.print(f"visited_sources: {', '.join(state.visited_sources)}")
    console.print(f"events: {len(state.events)}")
    print_scores(state, limit=limit)
    print_outputs(state.outputs)


def print_residential_summary(state, limit: int = 3) -> None:  # type: ignore[no-untyped-def]
    console.print(f"[bold]{settings.app_name}[/bold]")
    console.print(f"run_id: {state.run_id}")
    console.print(f"visited_sources: {', '.join(state.visited_sources)}")
    console.print(f"events: {len(state.events)}")
    print_residential_scores(state, limit=limit)
    print_outputs(state.outputs)


def print_scores(state, limit: int = 3) -> None:  # type: ignore[no-untyped-def]
    table = Table(title=f"Top {limit} Candidate Areas")
    table.add_column("区域名")
    table.add_column("分数", justify="right")
    table.add_column("置信度", justify="right")
    table.add_column("具体位置描述")
    table.add_column("推荐开发方向")
    table.add_column("关键理由")
    table.add_column("主要风险")
    scores = [score for score in state.scores if score.evidence_count > 0][:limit]
    for score in scores:
        table.add_row(
            score.area.name,
            f"{score.opportunity_score:.2f}",
            f"{score.confidence:.2f}",
            compact_cell(score.area.description),
            compact_cell(score.development_direction),
            compact_cell("；".join(score.key_reasons)),
            compact_cell("；".join(score.major_risks)),
        )
    console.print(table)
    if not scores:
        console.print("暂无有证据支撑的候选区域；请查看 pipeline.log 和 evidence_pack.json。")


def print_residential_scores(state, limit: int = 3) -> None:  # type: ignore[no-untyped-def]
    table = Table(title=f"Top {limit} Residential Development Areas")
    table.add_column("区域名")
    table.add_column("住宅开发分", justify="right")
    table.add_column("建议级别")
    table.add_column("人口流入", justify="right")
    table.add_column("抢先窗口", justify="right")
    table.add_column("置信度", justify="right")
    table.add_column("动作")
    scores = [score for score in state.residential_scores if score.evidence_count > 0][:limit]
    for score in scores:
        table.add_row(
            score.area.name,
            f"{score.residential_development_score:.2f}",
            score.recommendation,
            f"{score.future_population_inflow_score:.2f}",
            f"{score.land_grab_window_score:.2f}",
            f"{score.confidence:.2f}",
            leading_label(score.next_action),
        )
    console.print(table)
    if not scores:
        console.print("暂无有证据支撑的住宅推荐区域；请查看 pipeline.log 和 evidence_pack.json。")
        return
    console.print("[bold]区域行动摘要[/bold]")
    for idx, score in enumerate(scores, start=1):
        console.print(
            f"{idx}. {score.area.name}: {compact_cell(score.next_action, 56)} "
            f"| 产品: {compact_cell(score.suggested_product, 34)} "
            f"| 理由: {compact_cell('；'.join(score.key_reasons), 48)} "
            f"| 风险: {compact_cell('；'.join(score.major_risks), 48)}"
        )


def print_outputs(outputs) -> None:  # type: ignore[no-untyped-def]
    if not outputs:
        return
    output_items = [(label, value) for label, value in outputs.model_dump().items() if value]
    if not output_items:
        return
    console.print("输出文件路径:")
    for label, value in output_items:
        console.print(f"- {label}: {Path(value)}")


def compact_cell(value: str, limit: int = 34) -> str:
    return short_text(value, limit)


if __name__ == "__main__":
    sys.exit(main())
