"""
Multi-norm robustness evaluation engine.

Based on MultiRobustBench (ICML 2023): evaluates adversarial robustness
across multiple Lp norms and attack types simultaneously.

Computes:
- Competitiveness Ratio (CR): performance vs optimal per-attack defense
- Stability Constant (SC): accuracy fluctuation across attack strengths
- Worst-case robustness across all norms
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormEvalResult:
    """Result for a single norm/attack configuration."""
    norm: str
    epsilon: float
    attack_name: str
    success_rate: float = 0.0
    robust_accuracy: float = 0.0
    perturbation_size: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class MultiNormReport:
    """Comprehensive multi-norm evaluation report."""
    results: list[NormEvalResult] = field(default_factory=list)
    competitiveness_ratio: float = 0.0
    stability_constant: float = 0.0
    worst_case_robust_accuracy: float = 1.0
    average_robust_accuracy: float = 0.0
    per_norm_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    total_duration: float = 0.0
    num_configurations: int = 0


class MultiNormEvaluator:
    """Evaluates robustness across multiple Lp norms and epsilon values.

    Inspired by MultiRobustBench, this engine tests models against diverse
    attack types at various strengths to provide a comprehensive robustness
    profile rather than single-point evaluation.
    """

    NORM_ATTACK_MAP = {
        "linf": ["fgsm", "pgd", "auto_pgd"],
        "l2": ["carlini_wagner_l2", "deepfool"],
        "l1": ["elastic_net"],
        "l0": ["jsma", "pixel_attack"],
    }

    DEFAULT_EPSILONS = {
        "linf": [0.01, 0.03, 0.05, 0.1, 0.3],
        "l2": [0.1, 0.3, 0.5, 1.0, 2.0],
        "l1": [1.0, 3.0, 5.0, 10.0],
        "l0": [1, 3, 5, 10],
    }

    def __init__(
        self,
        norms: Optional[list[str]] = None,
        epsilons: Optional[dict[str, list[float]]] = None,
        batch_size: int = 32,
    ):
        self.norms = norms or ["linf", "l2"]
        self.epsilons = epsilons or self.DEFAULT_EPSILONS
        self.batch_size = batch_size

    def evaluate(
        self,
        classifier: Any,
        x: np.ndarray,
        y: np.ndarray,
        attack_generator: Optional[Any] = None,
        metadata: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> MultiNormReport:
        """Run multi-norm evaluation across all configured norms and epsilons."""
        start_time = time.time()
        report = MultiNormReport()
        all_robust_accs: list[float] = []

        for norm in self.norms:
            eps_list = self.epsilons.get(norm, [0.1])
            attacks_for_norm = self.NORM_ATTACK_MAP.get(norm, ["pgd"])
            norm_accs: list[float] = []

            for eps_val in eps_list:
                for atk_name in attacks_for_norm:
                    result = self._run_single_eval(
                        classifier, x, y, norm, eps_val, atk_name,
                        attack_generator, metadata, model,
                    )
                    report.results.append(result)
                    if result.error is None:
                        norm_accs.append(result.robust_accuracy)
                        all_robust_accs.append(result.robust_accuracy)

            if norm_accs:
                report.per_norm_summary[norm] = {
                    "mean_robust_accuracy": float(np.mean(norm_accs)),
                    "worst_case": float(np.min(norm_accs)),
                    "best_case": float(np.max(norm_accs)),
                    "std": float(np.std(norm_accs)),
                }

        if all_robust_accs:
            report.average_robust_accuracy = float(np.mean(all_robust_accs))
            report.worst_case_robust_accuracy = float(np.min(all_robust_accs))
            report.stability_constant = float(np.std(all_robust_accs))
            optimal = max(all_robust_accs)
            report.competitiveness_ratio = (
                report.average_robust_accuracy / optimal if optimal > 0 else 0.0
            )

        report.num_configurations = len(report.results)
        report.total_duration = time.time() - start_time
        return report

    def _run_single_eval(
        self, classifier: Any, x: np.ndarray, y: np.ndarray,
        norm: str, epsilon: float, attack_name: str,
        attack_generator: Optional[Any], metadata: Optional[Any],
        model: Optional[Any],
    ) -> NormEvalResult:
        """Execute a single attack configuration."""
        start = time.time()
        try:
            if attack_generator and model and metadata:
                from auto_art.core.interfaces import AttackConfig
                cfg = AttackConfig(
                    attack_type=attack_name, epsilon=epsilon,
                    norm=norm.replace("l", ""), batch_size=self.batch_size,
                )
                attack_instance = attack_generator.create_attack(model, metadata, cfg)
                adv_x = attack_generator.apply_attack(attack_instance, x, y)
            else:
                adv_x = self._fallback_attack(classifier, x, y, norm, epsilon)

            clean_preds = np.argmax(classifier.predict(x), axis=1)
            adv_preds = np.argmax(classifier.predict(adv_x), axis=1)
            true_labels = np.argmax(y, axis=1) if y.ndim > 1 else y

            success_rate = float(np.mean(clean_preds != adv_preds))
            robust_acc = float(np.mean(adv_preds == true_labels))
            pert_size = float(np.mean(np.abs(adv_x - x)))

            return NormEvalResult(
                norm=norm, epsilon=epsilon, attack_name=attack_name,
                success_rate=success_rate, robust_accuracy=robust_acc,
                perturbation_size=pert_size, duration=time.time() - start,
            )
        except Exception as e:
            logger.warning(f"Multi-norm eval failed for {attack_name} norm={norm} eps={epsilon}: {e}")
            return NormEvalResult(
                norm=norm, epsilon=epsilon, attack_name=attack_name,
                duration=time.time() - start, error=str(e),
            )

    @staticmethod
    def _fallback_attack(
        classifier: Any, x: np.ndarray, y: np.ndarray,
        norm: str, epsilon: float,
    ) -> np.ndarray:
        """Fallback PGD attack when no attack generator is available."""
        from art.attacks.evasion import ProjectedGradientDescent
        norm_val = np.inf if norm == "linf" else int(norm.replace("l", ""))
        pgd = ProjectedGradientDescent(
            estimator=classifier, eps=epsilon,
            eps_step=epsilon / 4, max_iter=40, norm=norm_val,
        )
        return pgd.generate(x=x, y=y)
