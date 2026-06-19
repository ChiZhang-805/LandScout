from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from threading import Lock
from typing import Any, Callable
import re
import uuid

from app.core.config import settings


TaskWorker = Callable[[], dict[str, Any]]


@dataclass
class TaskRecord:
    task_id: str
    status: str
    message: str
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_task_id() -> str:
    return uuid.uuid4().hex


def validate_task_id(task_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise ValueError(f"Invalid task_id format: {task_id}")


class WebTaskManager:
    def __init__(self, *, max_workers: int = 1, max_records: int = 100, tasks_dir: Path | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="landscout-web-task")
        self._lock = Lock()
        self._tasks: dict[str, TaskRecord] = {}
        self._max_records = max_records
        self._tasks_dir = tasks_dir

    def submit(self, worker: TaskWorker) -> dict[str, Any]:
        task_id = make_task_id()
        now = utc_now_iso()
        record = TaskRecord(
            task_id=task_id,
            status="queued",
            message="任务已创建，等待后台执行",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._tasks[task_id] = record
            self._write_task_locked(record)
            self._prune_completed_locked()
        self._executor.submit(self._run, task_id, worker)
        return self.snapshot(task_id)

    def snapshot(self, task_id: str) -> dict[str, Any]:
        validate_task_id(task_id)
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                record = self._load_task_locked(task_id)
                if not record:
                    raise KeyError(task_id)
                if record.status in {"queued", "running"}:
                    record.status = "failed"
                    record.message = "服务器进程已重启，后台任务已中断"
                    record.error = "Render 或本地服务进程重启后，内存中的后台任务无法继续；请重新运行。"
                    record.updated_at = utc_now_iso()
                    self._write_task_locked(record)
            return task_to_dict(record)

    def _update(self, task_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._tasks[task_id]
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = utc_now_iso()
            self._write_task_locked(record)
            self._prune_completed_locked()

    def _prune_completed_locked(self) -> None:
        overflow = len(self._tasks) - self._max_records
        if overflow <= 0:
            return
        completed = [task for task in self._tasks.values() if task.status in {"succeeded", "failed"}]
        completed.sort(key=lambda task: task.updated_at)
        for task in completed[:overflow]:
            self._tasks.pop(task.task_id, None)
            self._delete_task_file_locked(task.task_id)

    def _task_path(self, task_id: str) -> Path:
        validate_task_id(task_id)
        tasks_dir = self._tasks_dir or settings.data_dir / "web_tasks"
        return tasks_dir / f"{task_id}.json"

    def _write_task_locked(self, record: TaskRecord) -> None:
        try:
            path = self._task_path(record.task_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(task_to_dict(record), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except OSError:
            return

    def _load_task_locked(self, task_id: str) -> TaskRecord | None:
        try:
            payload = json.loads(self._task_path(task_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return TaskRecord(
                task_id=str(payload["task_id"]),
                status=str(payload["status"]),
                message=str(payload["message"]),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                result=payload.get("result"),
                error=payload.get("error"),
            )
        except KeyError:
            return None

    def _delete_task_file_locked(self, task_id: str) -> None:
        try:
            self._task_path(task_id).unlink(missing_ok=True)
        except OSError:
            return

    def _run(self, task_id: str, worker: TaskWorker) -> None:
        self._update(task_id, status="running", message="后台正在抓取、解析和分析")
        try:
            result = worker()
        except Exception as exc:
            self._update(
                task_id,
                status="failed",
                message="任务运行失败",
                error=str(exc) or exc.__class__.__name__,
            )
            return
        self._update(
            task_id,
            status="succeeded",
            message="任务完成",
            result=result,
            error=None,
        )


def task_to_dict(record: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "status": record.status,
        "message": record.message,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "result": record.result,
        "error": record.error,
    }


web_task_manager = WebTaskManager()
