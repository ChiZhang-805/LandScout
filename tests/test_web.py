import json
import time

from fastapi.testclient import TestClient

from app import main
from app.core.config import effective_openai_api_key, settings
from app.main import app
from app.pipeline.orchestrator import LandScoutAgentState
from app.sources.registry import SourceConfig, SourceRegistry
from app.web import WebRunRequest, build_runtime_registry, parse_custom_sources_text, state_to_web_response
from app.web_tasks import WebTaskManager


def wait_for_task(client: TestClient, task_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/recommend-residential/tasks/{task_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Task {task_id} did not finish within {timeout} seconds")


def test_parse_custom_sources_text_accepts_line_format():
    sources = parse_custom_sources_text(
        "浦东规划 | https://www.pudong.gov.cn/ | 住宅,地块,规划\n"
        "https://www.lingang.gov.cn/"
    )

    assert len(sources) == 2
    assert sources[0].name == "浦东规划"
    assert str(sources[0].base_urls[0]) == "https://www.pudong.gov.cn/"
    assert sources[0].keywords[:3] == ["住宅", "地块", "规划"]
    assert sources[0].access_mode == "http_then_playwright"
    assert sources[1].name == "www.lingang.gov.cn"


def test_parse_custom_sources_text_accepts_json_config():
    sources = parse_custom_sources_text(
        """
        {
          "sources": [
            {
              "name": "测试源",
              "base_urls": ["https://example.gov.cn/list.html"],
              "keywords": ["住宅", "招商"],
              "max_pages": 3
            }
          ]
        }
        """
    )

    assert len(sources) == 1
    assert sources[0].name == "测试源"
    assert sources[0].max_pages == 3
    assert sources[0].keywords == ["住宅", "招商"]


def test_build_runtime_registry_merges_builtin_and_custom_sources():
    base = SourceRegistry(
        [
            SourceConfig(id="builtin_1", name="Builtin 1", base_urls=["https://one.gov.cn/"], priority=10),
            SourceConfig(id="builtin_2", name="Builtin 2", base_urls=["https://two.gov.cn/"], priority=20),
        ]
    )

    runtime = build_runtime_registry(
        base,
        source_limit=1,
        use_builtin_sources=True,
        custom_sources_text="Custom | https://custom.gov.cn/ | 住宅",
    )

    assert [source.id for source in runtime.sources][0] == "builtin_1"
    assert any(source.name == "Custom" for source in runtime.sources)
    assert len(runtime.sources) == 2


def test_web_dashboard_renders_without_auto_running_pipeline():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "搜索并分析" in response.text
    assert "尚未运行" in response.text
    assert "/api/recommend-residential" in response.text
    assert 'id="city"' in response.text
    assert 'id="openaiKey"' in response.text
    assert 'id="amapKey"' in response.text
    assert 'id="opportunityMap"' in response.text
    assert 'class="brand-logo"' in response.text
    assert '<input id="openaiKey" type="password"' in response.text
    assert '<input id="amapKey" type="password"' in response.text
    assert '<input id="amapSecurityCode" type="password"' in response.text
    assert 'data-secret-toggle="openaiKey"' in response.text
    assert 'data-secret-toggle="amapKey"' in response.text
    assert 'data-secret-toggle="amapSecurityCode"' in response.text
    assert 'const TASK_STORAGE_KEY = "landscout.currentTaskId.v2";' in response.text
    assert "MAX_TRANSIENT_POLL_FAILURES" in response.text
    assert "function resetExpiredTask(taskId)" in response.text
    assert 'function pollTask(taskId)' in response.text
    assert "function setupSecretToggles()" in response.text
    assert "异常/访问限制" not in response.text
    source_limit_max = len(main.registry.sources)
    source_limit_default = min(12, source_limit_max)
    assert (
        f'id="sourceLimit" type="number" min="1" max="{source_limit_max}" '
        f'step="1" value="{source_limit_default}"'
    ) in response.text


def test_brand_logo_asset_is_served_when_present():
    client = TestClient(app)

    response = client.get("/assets/landscout-agent-icon.png")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 1000


def test_web_recommend_residential_fixture_run_honors_top_k():
    client = TestClient(app)

    response = client.post(
        "/api/recommend-residential",
        json={"live": False, "days": 540, "top_k": 2, "source_limit": 12},
    )

    assert response.status_code == 200
    task = response.json()
    assert task["task_id"]
    task = wait_for_task(client, task["task_id"])
    assert task["status"] == "succeeded"
    payload = task["result"]
    assert payload["run_id"]
    assert payload["city"] == "shanghai"
    assert payload["event_count"] >= 20
    assert len(payload["top_areas"]) == 2
    assert payload["top_areas"][0]["lat"] is not None
    assert payload["top_areas"][0]["lon"] is not None
    assert payload["top_areas"][0]["radius_m"] > 0
    assert any(file["filename"] == "recommendation.md" for file in payload["files"])
    assert any(file["filename"] == "recommendation.md" and file["group"] for file in payload["files"])

    file_response = client.get(f"/runs/{payload['run_id']}/files/recommendation.md")
    assert file_response.status_code == 200
    assert "上海住宅开发机会推荐报告" in file_response.text


def test_web_rejects_unsupported_city():
    client = TestClient(app)

    response = client.post(
        "/api/recommend-residential",
        json={"city": "hangzhou", "live": False, "days": 540, "top_k": 2, "source_limit": 12},
    )

    assert response.status_code == 400
    assert "只支持上海" in response.text


def test_web_task_reports_background_failure(monkeypatch):
    class FakeAgent:
        def __init__(self, registry=None):  # type: ignore[no-untyped-def]
            pass

        def recommend_residential(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("synthetic task failure")

    monkeypatch.setattr(main, "LandScoutAgent", FakeAgent)
    client = TestClient(app)

    response = client.post(
        "/api/recommend-residential",
        json={"live": False, "days": 540, "top_k": 2, "source_limit": 12},
    )

    assert response.status_code == 200
    task = wait_for_task(client, response.json()["task_id"])
    assert task["status"] == "failed"
    assert task["result"] is None
    assert "synthetic task failure" in task["error"]


def test_web_task_manager_validates_task_ids():
    manager = WebTaskManager(max_workers=1)

    try:
        manager.snapshot("../bad")
    except ValueError as exc:
        assert "Invalid task_id format" in str(exc)
    else:
        raise AssertionError("Invalid task id was accepted")


def test_web_task_manager_prunes_completed_records():
    manager = WebTaskManager(max_workers=1, max_records=2)
    task_ids = [manager.submit(lambda idx=idx: {"idx": idx})["task_id"] for idx in range(3)]

    for task_id in task_ids:
        try:
            wait_for_task_snapshot(manager, task_id)
        except KeyError:
            pass

    remaining = [task_id for task_id in task_ids if task_snapshot_exists(manager, task_id)]
    assert len(remaining) <= 2
    assert task_ids[-1] in remaining


def test_web_task_manager_restores_completed_record_from_disk(tmp_path):
    manager = WebTaskManager(max_workers=1, tasks_dir=tmp_path)
    task_id = manager.submit(lambda: {"ok": True})["task_id"]
    wait_for_task_snapshot(manager, task_id)

    restored = WebTaskManager(max_workers=1, tasks_dir=tmp_path)

    payload = restored.snapshot(task_id)
    assert payload["status"] == "succeeded"
    assert payload["result"] == {"ok": True}


def test_web_task_manager_marks_persisted_running_record_interrupted(tmp_path):
    task_id = "0" * 32
    (tmp_path / f"{task_id}.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "status": "running",
                "message": "后台正在抓取、解析和分析",
                "created_at": "2026-06-19T00:00:00+00:00",
                "updated_at": "2026-06-19T00:00:00+00:00",
                "result": None,
                "error": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = WebTaskManager(max_workers=1, tasks_dir=tmp_path)

    payload = manager.snapshot(task_id)

    assert payload["status"] == "failed"
    assert "重启" in payload["message"]


def wait_for_task_snapshot(manager: WebTaskManager, task_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = manager.snapshot(task_id)
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"Task {task_id} did not finish within {timeout} seconds")


def task_snapshot_exists(manager: WebTaskManager, task_id: str) -> bool:
    try:
        manager.snapshot(task_id)
    except KeyError:
        return False
    return True


def test_web_request_passes_amap_key_to_pipeline(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, registry=None):  # type: ignore[no-untyped-def]
            pass

        def recommend_residential(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return LandScoutAgentState(run_id="run")

    monkeypatch.setattr(main, "LandScoutAgent", FakeAgent)

    response = main._run_recommendation(
        WebRunRequest(city="shanghai", live=False, days=540, top_k=2, source_limit=12, amap_key="amap-test")
    )

    assert response["run_id"] == "run"
    assert captured["amap_key"] == "amap-test"


def test_web_request_uses_request_scoped_openai_key(monkeypatch):
    captured = {}
    monkeypatch.setattr(settings, "openai_api_key", "")

    class FakeAgent:
        def __init__(self, registry=None):  # type: ignore[no-untyped-def]
            pass

        def recommend_residential(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["effective_key_during_request"] = effective_openai_api_key()
            return LandScoutAgentState(run_id="run")

    monkeypatch.setattr(main, "LandScoutAgent", FakeAgent)

    response = main._run_recommendation(
        WebRunRequest(
            city="shanghai",
            live=False,
            days=540,
            top_k=2,
            source_limit=12,
            openai_api_key="request-openai-key",
        )
    )

    assert response["run_id"] == "run"
    assert captured["effective_key_during_request"] == "request-openai-key"
    assert effective_openai_api_key() == ""


def test_web_live_source_limit_is_capped_to_registry_size(monkeypatch):
    captured = {}
    small_registry = SourceRegistry(
        [
            SourceConfig(id="builtin_1", name="Builtin 1", base_urls=["https://one.gov.cn/"], priority=10),
            SourceConfig(id="builtin_2", name="Builtin 2", base_urls=["https://two.gov.cn/"], priority=20),
        ]
    )

    class FakeAgent:
        def __init__(self, registry=None):  # type: ignore[no-untyped-def]
            captured["registry_size"] = len(registry.sources)

        def recommend_residential(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return LandScoutAgentState(run_id="run")

    monkeypatch.setattr(main, "registry", small_registry)
    monkeypatch.setattr(main, "LandScoutAgent", FakeAgent)

    response = main._run_recommendation(
        WebRunRequest(city="shanghai", live=True, days=540, top_k=2, source_limit=5)
    )

    assert response["run_id"] == "run"
    assert captured["registry_size"] == 2
    assert captured["source_limit"] == 2


def test_web_response_separates_triage_skips_from_errors():
    state = LandScoutAgentState(
        run_id="run",
        errors=[
            {"source_id": "doc", "url": "fixture://skip", "reason": "document triage skipped: no relevant signal keywords found", "stage": "document_triage"},
            {"source_id": "doc", "url": "fixture://fail", "reason": "extract failed: synthetic"},
        ],
    )

    payload = state_to_web_response(state)

    assert payload["error_count"] == 1
    assert payload["filtered_document_count"] == 1
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["reason"] == "extract failed: synthetic"
