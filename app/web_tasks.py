from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Callable
import re
import uuid


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
    def __init__(self, *, max_workers: int = 2, max_records: int = 100) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="landscout-web-task")
        self._lock = Lock()
        self._tasks: dict[str, TaskRecord] = {}
        self._max_records = max_records

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
            self._prune_completed_locked()
        self._executor.submit(self._run, task_id, worker)
        return self.snapshot(task_id)

    def snapshot(self, task_id: str) -> dict[str, Any]:
        validate_task_id(task_id)
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                raise KeyError(task_id)
            return {
                "task_id": record.task_id,
                "status": record.status,
                "message": record.message,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "result": record.result,
                "error": record.error,
            }

    def _update(self, task_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._tasks[task_id]
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = utc_now_iso()
            self._prune_completed_locked()

    def _prune_completed_locked(self) -> None:
        overflow = len(self._tasks) - self._max_records
        if overflow <= 0:
            return
        completed = [task for task in self._tasks.values() if task.status in {"succeeded", "failed"}]
        completed.sort(key=lambda task: task.updated_at)
        for task in completed[:overflow]:
            self._tasks.pop(task.task_id, None)

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


web_task_manager = WebTaskManager()
