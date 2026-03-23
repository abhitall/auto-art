"""Unit tests for auto_art.core.warmstart."""
import json
import tempfile
import time
from pathlib import Path

import pytest

from auto_art.core.orchestrator import OrchestratorReport
from auto_art.core.warmstart import EvaluationCache, ModelDiffAnalyzer


def test_evaluation_cache_save_load():
    with tempfile.TemporaryDirectory() as tmp:
        c = EvaluationCache(cache_dir=tmp)
        payload = {"fgsm": {"success_rate": 0.2}}
        c.save("abc" * 10, payload)
        loaded = c.load("abc" * 10)
        assert loaded == payload


def test_evaluation_cache_staleness():
    with tempfile.TemporaryDirectory() as tmp:
        c = EvaluationCache(cache_dir=tmp)
        h = "deadbeef"
        path = c._path_for(h)
        path.write_text(
            json.dumps({"timestamp": time.time() - 7200, "results": {}}),
            encoding="utf-8",
        )
        assert c.is_stale(h, max_age_hours=1.0) is True
        path.write_text(json.dumps({"timestamp": time.time(), "results": {}}), encoding="utf-8")
        assert c.is_stale(h, max_age_hours=24.0) is False


def test_model_hash():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"model-bytes-v1")
        f.flush()
        path = f.name
    try:
        c = EvaluationCache()
        h1 = c.get_model_hash(path)
        h2 = c.get_model_hash(path)
        assert h1 == h2
        assert len(h1) == 64
    finally:
        Path(path).unlink(missing_ok=True)


def test_model_diff_analyzer():
    a = OrchestratorReport(
        target={"model_path": "m1.pt"},
        phases=[
            {
                "name": "Evasion Attacks",
                "results": [{"attack": "fgsm", "success_rate": 0.5}],
            },
        ],
    )
    b = OrchestratorReport(
        target={"model_path": "m2.pt"},
        phases=[
            {
                "name": "Evasion Attacks",
                "results": [{"attack": "fgsm", "success_rate": 0.2}],
            },
        ],
    )
    rep = ModelDiffAnalyzer().compare(a, b)
    assert "fgsm" in rep.improved_attacks
    assert rep.regressed_attacks == []
