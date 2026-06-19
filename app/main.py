from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from app.core.config import settings, use_request_openai_api_key
from app.core.utils import read_json
from app.llm.client import MissingLLMKey
from app.llm.openai_client import OpenAINonRecoverableError
from app.pipeline.orchestrator import LandScoutAgent, validate_run_id
from app.sources.registry import load_shanghai_registry
from app.web import (
    ALLOWED_OUTPUT_FILENAMES,
    WebRunRequest,
    brand_logo_path,
    build_dashboard_html,
    build_runtime_registry,
    state_to_web_response,
)
from app.web_tasks import web_task_manager


app = FastAPI(title=settings.app_name)
registry = load_shanghai_registry()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return build_dashboard_html(registry)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/assets/landscout-agent-icon.png")
def brand_logo() -> FileResponse:
    path = brand_logo_path()
    if not path:
        raise HTTPException(status_code=404, detail="Brand logo not found.")
    return FileResponse(path)


@app.post("/api/recommend-residential")
def recommend_residential(request: WebRunRequest) -> dict:
    if request.city != "shanghai":
        raise HTTPException(status_code=400, detail="当前版本只支持上海；其他城市会在后续扩展。")
    try:
        return web_task_manager.submit(lambda: _run_recommendation(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/recommend-residential/tasks/{task_id}")
def recommendation_task(task_id: str) -> dict:
    try:
        return web_task_manager.snapshot(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found or expired.") from exc


def _run_recommendation(request: WebRunRequest) -> dict:
    try:
        with use_request_openai_api_key(request.openai_api_key):
            if request.city != "shanghai":
                raise ValueError("当前版本只支持上海；其他城市会在后续扩展。")
            if request.live:
                builtin_source_limit = min(request.source_limit, len(registry.sources))
                runtime_registry = build_runtime_registry(
                    registry,
                    source_limit=builtin_source_limit,
                    use_builtin_sources=request.use_builtin_sources,
                    custom_sources_text=request.custom_sources_text,
                )
                state = LandScoutAgent(registry=runtime_registry).recommend_residential(
                    live=True,
                    days=request.days,
                    top_k=request.top_k,
                    source_limit=len(runtime_registry.sources),
                    discover_sources=False,
                    amap_key=request.amap_key.strip() or None,
                )
            else:
                state = LandScoutAgent().recommend_residential(
                    live=False,
                    days=request.days,
                    top_k=request.top_k,
                    amap_key=request.amap_key.strip() or None,
                )
    except (MissingLLMKey, OpenAINonRecoverableError) as exc:
        raise RuntimeError(str(exc)) from exc
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    return state_to_web_response(state, top_k=request.top_k)


@app.get("/latest/signals")
def latest_signals() -> dict:
    path = settings.outputs_dir / "shanghai" / "latest" / "signals.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No latest signals.json found.")
    return read_json(path)


@app.get("/latest/files")
def latest_files() -> dict[str, list[str]]:
    latest_dir = settings.outputs_dir / "shanghai" / "latest"
    if not latest_dir.exists():
        return {"files": []}
    return {"files": [str(Path(p).resolve()) for p in latest_dir.iterdir() if p.is_file()]}


@app.get("/runs/{run_id}/files")
def run_files(run_id: str) -> dict[str, list[str]]:
    try:
        validate_run_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_dir = settings.outputs_dir / "shanghai" / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run output directory not found.")
    return {"files": [item.name for item in run_dir.iterdir() if item.is_file() and item.name in ALLOWED_OUTPUT_FILENAMES]}


@app.get("/runs/{run_id}/files/{filename}")
def run_file(run_id: str, filename: str) -> FileResponse:
    try:
        validate_run_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if filename not in ALLOWED_OUTPUT_FILENAMES or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Output file not found.")
    path = settings.outputs_dir / "shanghai" / run_id / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(path)
