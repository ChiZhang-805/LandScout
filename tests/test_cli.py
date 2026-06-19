from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from app import cli
from app.scoring.candidates import CandidateArea
from app.scoring.scorer import CandidateScore


def test_print_scores_hides_unevidenced_candidates(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, force_terminal=False, width=120))
    state = SimpleNamespace(
        scores=[
            CandidateScore(
                area=CandidateArea(
                    id="area1",
                    name="无证据区域",
                    lat=31.0,
                    lon=121.0,
                    description="测试区域",
                ),
                industrial_import_score=0,
                infrastructure_score=0,
                public_service_score=0,
                land_structure_score=0,
                residential_supply_risk=0,
                market_entry_score=0,
                recency_score=0,
                evidence_confidence_score=0,
                geo_uncertainty_penalty=100,
                opportunity_score=0,
                confidence=0,
                evidence_count=0,
            )
        ]
    )

    cli.print_scores(state, limit=3)

    output = stream.getvalue()
    assert "无证据区域" not in output
    assert "暂无有证据支撑的候选区域" in output
