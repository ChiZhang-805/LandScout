from __future__ import annotations

import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def isolate_runtime_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "outputs_dir", tmp_path / "outputs")
