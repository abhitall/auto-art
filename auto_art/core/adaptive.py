"""
Adaptive attack selection engine.

Inspired by AutoRedTeamer (NeurIPS 2025), this module implements
memory-guided attack selection that starts with cheap screening attacks
and escalates to expensive ones only when the model survives.

Strategy:
1. Run fast gradient attacks (FGSM/BIM) as quick screen
2. If model survives, escalate to iterative (PGD/AutoAttack)
3. If still robust, try black-box (Square/HopSkipJump/ZOO)
4. Track success history per model architecture family
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ATTACK_TIERS = {
    "tier_1_fast": [
        {"name": "fgsm", "cost": 1},
        {"name": "bim", "cost": 2},
    ],
    "tier_2_iterative": [
        {"name": "pgd", "cost": 5},
        {"name": "auto_pgd", "cost": 8},
        {"name": "autoattack", "cost": 10},
        {"name": "carlini_wagner_l2", "cost": 12},
    ],
    "tier_3_blackbox": [
        {"name": "square_attack", "cost": 7},
        {"name": "hopskipjump", "cost": 15},
        {"name": "zoo", "cost": 20},
        {"name": "geoda", "cost": 12},
    ],
    "tier_4_advanced": [
        {"name": "elastic_net", "cost": 15},
        {"name": "jsma", "cost": 18},
        {"name": "brendel_bethge", "cost": 20},
        {"name": "shadow_attack", "cost": 25},
        {"name": "composite", "cost": 22},
    ],
}


@dataclass
class AttackRecord:
    """Historical record of an attack's effectiveness."""
    attack_name: str
    total_runs: int = 0
    total_successes: int = 0
    avg_success_rate: float = 0.0
    avg_cost_seconds: float = 0.0
    model_arch_success: Dict[str, float] = field(default_factory=dict)


class AttackMemory:
    """Persistent memory of attack effectiveness per model family.

    Stores historical success rates and costs to guide future
    attack selection via Bayesian-inspired prioritization.
    """

    def __init__(self, memory_path: Optional[str] = None):
        self.memory_path = memory_path or os.path.join(
            str(Path.home()), ".auto_art", "attack_memory.json"
        )
        self.records: Dict[str, AttackRecord] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    data = json.load(f)
                for name, rec in data.items():
                    self.records[name] = AttackRecord(**rec)
            except Exception as e:
                logger.warning(f"Failed to load attack memory: {e}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        data = {}
        for name, rec in self.records.items():
            data[name] = {
                "attack_name": rec.attack_name,
                "total_runs": rec.total_runs,
                "total_successes": rec.total_successes,
                "avg_success_rate": rec.avg_success_rate,
                "avg_cost_seconds": rec.avg_cost_seconds,
                "model_arch_success": rec.model_arch_success,
            }
        with open(self.memory_path, "w") as f:
            json.dump(data, f, indent=2)

    def record_result(
        self,
        attack_name: str,
        success_rate: float,
        cost_seconds: float,
        model_arch: str = "unknown",
    ) -> None:
        if attack_name not in self.records:
            self.records[attack_name] = AttackRecord(attack_name=attack_name)
        rec = self.records[attack_name]
        rec.total_runs += 1
        if success_rate > 0.5:
            rec.total_successes += 1
        n = rec.total_runs
        rec.avg_success_rate = (rec.avg_success_rate * (n - 1) + success_rate) / n
        rec.avg_cost_seconds = (rec.avg_cost_seconds * (n - 1) + cost_seconds) / n
        rec.model_arch_success[model_arch] = success_rate

    def get_priority_score(self, attack_name: str, model_arch: str = "unknown") -> float:
        """Higher score = more likely to succeed efficiently."""
        if attack_name not in self.records:
            return 0.5
        rec = self.records[attack_name]
        arch_rate = rec.model_arch_success.get(model_arch, rec.avg_success_rate)
        cost_penalty = min(rec.avg_cost_seconds / 60.0, 1.0) * 0.2
        return arch_rate - cost_penalty


class AdaptiveAttackSelector:
    """Selects attacks based on model type, budget, and historical performance.

    Implements tiered escalation: cheap screening first, expensive deep
    evaluation only when model survives initial probes.
    """

    def __init__(
        self,
        memory: Optional[AttackMemory] = None,
        budget_seconds: float = 3600.0,
        success_threshold: float = 0.5,
        escalate_if_below: float = 0.1,
    ):
        self.memory = memory or AttackMemory()
        self.budget_seconds = budget_seconds
        self.success_threshold = success_threshold
        self.escalate_if_below = escalate_if_below

    def select_attacks(
        self,
        model_arch: str = "unknown",
        available_attacks: Optional[List[str]] = None,
        max_attacks: int = 10,
    ) -> List[Dict[str, Any]]:
        """Select an ordered list of attacks based on adaptive strategy.

        Returns list of attack configs sorted by priority (highest first).
        """
        selected: List[Dict[str, Any]] = []
        budget_remaining = self.budget_seconds

        for tier_name, tier_attacks in ATTACK_TIERS.items():
            if len(selected) >= max_attacks:
                break

            tier_candidates = []
            for atk in tier_attacks:
                name = atk["name"]
                if available_attacks and name not in available_attacks:
                    continue
                priority = self.memory.get_priority_score(name, model_arch)
                estimated_cost = atk["cost"] * 10
                if estimated_cost <= budget_remaining:
                    tier_candidates.append({
                        "name": name,
                        "tier": tier_name,
                        "priority": priority,
                        "estimated_cost": estimated_cost,
                    })

            tier_candidates.sort(key=lambda x: x["priority"], reverse=True)
            for candidate in tier_candidates:
                if len(selected) >= max_attacks:
                    break
                selected.append(candidate)
                budget_remaining -= candidate["estimated_cost"]

        return selected

    def should_escalate(self, current_results: List[Dict[str, Any]]) -> bool:
        """Determine if we should escalate to more expensive attacks."""
        if not current_results:
            return True
        max_success = max(
            (r.get("success_rate", 0.0) for r in current_results
             if isinstance(r.get("success_rate"), (int, float))),
            default=0.0,
        )
        return max_success < self.escalate_if_below

    def update_memory(
        self,
        attack_name: str,
        success_rate: float,
        cost_seconds: float,
        model_arch: str = "unknown",
    ) -> None:
        self.memory.record_result(attack_name, success_rate, cost_seconds, model_arch)
        self.memory.save()
