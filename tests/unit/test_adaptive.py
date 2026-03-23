"""Unit tests for auto_art.core.adaptive."""
import json
import tempfile
from pathlib import Path

import pytest

from auto_art.core.adaptive import (
    ATTACK_TIERS,
    AdaptiveAttackSelector,
    AttackMemory,
)


def test_attack_memory_save_load():
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "mem.json")
        m1 = AttackMemory(memory_path=p)
        m1.record_result("fgsm", success_rate=0.8, cost_seconds=5.0, model_arch="resnet")
        m1.save()

        m2 = AttackMemory(memory_path=p)
        assert "fgsm" in m2.records
        assert m2.records["fgsm"].total_runs == 1
        assert m2.records["fgsm"].avg_success_rate == pytest.approx(0.8)
        raw = json.loads(Path(p).read_text())
        assert raw["fgsm"]["model_arch_success"]["resnet"] == 0.8


def test_adaptive_selector_basic():
    with tempfile.TemporaryDirectory() as tmp:
        mem = AttackMemory(memory_path=str(Path(tmp) / "sel.json"))
        sel = AdaptiveAttackSelector(memory=mem, budget_seconds=1e6)
        out = sel.select_attacks(model_arch="m1", max_attacks=20)
    assert len(out) >= 1
    tier_order = list(ATTACK_TIERS.keys())
    seen_tiers = [x["tier"] for x in out]
    assert seen_tiers[0] == tier_order[0]
    assert all("name" in x and "priority" in x for x in out)


def test_adaptive_selector_budget():
    with tempfile.TemporaryDirectory() as tmp:
        mem = AttackMemory(memory_path=str(Path(tmp) / "bud.json"))
        # Only tier-1 fgsm (cost 10) fits under 15; bim (20) is excluded from candidates.
        sel = AdaptiveAttackSelector(memory=mem, budget_seconds=15.0)
        out = sel.select_attacks(max_attacks=50)
    total_cost = sum(x["estimated_cost"] for x in out)
    assert total_cost <= 15.0
    assert len(out) >= 1


def test_adaptive_selector_escalation():
    sel = AdaptiveAttackSelector(escalate_if_below=0.2)
    assert sel.should_escalate([]) is True
    assert sel.should_escalate([{"success_rate": 0.0}]) is True
    assert sel.should_escalate([{"success_rate": 0.5}]) is False


def test_attack_memory_priority():
    with tempfile.TemporaryDirectory() as tmp:
        m = AttackMemory(memory_path=str(Path(tmp) / "p.json"))
        assert m.get_priority_score("unknown_atk") == pytest.approx(0.5)
        m.record_result("pgd", success_rate=0.9, cost_seconds=0.0, model_arch="vit")
        hi = m.get_priority_score("pgd", model_arch="vit")
        m.record_result("slow", success_rate=0.9, cost_seconds=3600.0, model_arch="vit")
        penalized = m.get_priority_score("slow", model_arch="vit")
        assert hi > penalized
