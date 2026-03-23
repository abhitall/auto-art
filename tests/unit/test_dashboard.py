"""Unit tests for auto_art.core.dashboard."""
import tempfile
from pathlib import Path

from auto_art.core.dashboard import DashboardGenerator
from auto_art.core.orchestrator import OrchestratorReport


def test_generate_empty_report():
    gen = DashboardGenerator()
    r = OrchestratorReport(phases=[], gate_results={}, summary={})
    html = gen.generate(r)
    assert "<!DOCTYPE html>" in html
    assert "<html" in html and "</html>" in html
    assert "Auto-ART Evaluation Dashboard" in html


def test_generate_with_phases():
    gen = DashboardGenerator()
    r = OrchestratorReport(
        phases=[
            {"name": "Evasion Attacks", "passed": True, "duration": 1.0, "summary": "ok", "results": []},
        ],
        summary={"total_phases": 1, "passed_phases": 1},
    )
    html = gen.generate(r)
    assert "Evasion Attacks" in html
    assert "Evaluation Phases" in html


def test_generate_with_gates():
    gen = DashboardGenerator()
    r = OrchestratorReport(
        gate_results={
            "evasion_max_attack_success_rate": {
                "threshold": 0.05,
                "actual": 0.01,
                "passed": True,
            },
        },
    )
    html = gen.generate(r)
    assert "evasion_max_attack_success_rate" in html
    assert "Gate Results" in html


def test_save_to_file():
    gen = DashboardGenerator()
    r = OrchestratorReport()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dash.html"
        gen.save(r, str(out))
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in text
