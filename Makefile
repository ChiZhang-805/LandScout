.PHONY: install test landscout-demo shanghai-demo

install:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest

landscout-demo:
	python -m app.cli landscout-demo --live --days 540 --top-k 8

shanghai-demo:
	python -m app.cli shanghai-demo --live --days 540 --top-k 8
