"""
Gradient masking detection module.

Detects when models produce misleading gradients that cause gradient-based
attacks to fail while remaining vulnerable to gradient-free attacks.
Based on ICLR 2025 findings on TRADES overestimation and instability.

Detection strategy:
1. Compare white-box (gradient-based) vs black-box (gradient-free) attack success
2. Compute FOSC score — if high, gradients are unreliable
3. Gaussian noise injection to verify gradient paths
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MaskingReport:
    """Result from gradient masking detection."""
    is_masked: bool = False
    confidence: float = 0.0
    fosc_score: float = 0.0
    wb_success_rate: float = 0.0
    bb_success_rate: float = 0.0
    discrepancy: float = 0.0
    noise_injection_delta: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


class GradientMaskingDetector:
    """Detects gradient masking in adversarial robustness evaluations.
    
    Gradient masking occurs when a model's gradients become misleading,
    causing gradient-based attacks to underestimate the true vulnerability.
    This creates false positives in robustness evaluations.
    """

    def __init__(
        self,
        fosc_threshold: float = 0.1,
        discrepancy_threshold: float = 0.15,
        noise_sigma: float = 0.01,
        num_noise_samples: int = 10,
    ):
        self.fosc_threshold = fosc_threshold
        self.discrepancy_threshold = discrepancy_threshold
        self.noise_sigma = noise_sigma
        self.num_noise_samples = num_noise_samples

    def detect(
        self,
        classifier: Any,
        x: np.ndarray,
        y: np.ndarray,
        wb_results: Optional[dict[str, Any]] = None,
        bb_results: Optional[dict[str, Any]] = None,
    ) -> MaskingReport:
        """Run gradient masking detection.
        
        Args:
            classifier: ART classifier with gradient access.
            x: Clean input samples.
            y: True labels (one-hot encoded).
            wb_results: Dict with 'success_rate' from white-box attacks.
            bb_results: Dict with 'success_rate' from black-box attacks.
        
        Returns:
            MaskingReport with detection results.
        """
        report = MaskingReport()
        signals = []

        # Signal 1: WB vs BB discrepancy
        if wb_results and bb_results:
            wb_sr = wb_results.get("success_rate", 0.0)
            bb_sr = bb_results.get("success_rate", 0.0)
            report.wb_success_rate = wb_sr
            report.bb_success_rate = bb_sr
            report.discrepancy = bb_sr - wb_sr
            if report.discrepancy > self.discrepancy_threshold:
                signals.append("wb_bb_discrepancy")

        # Signal 2: FOSC score
        try:
            fosc = self._compute_fosc(classifier, x, y)
            report.fosc_score = fosc
            if fosc > self.fosc_threshold:
                signals.append("high_fosc")
        except Exception as e:
            logger.warning(f"FOSC computation failed: {e}")
            report.details["fosc_error"] = str(e)

        # Signal 3: Noise injection sensitivity
        try:
            delta = self._noise_injection_test(classifier, x, y)
            report.noise_injection_delta = delta
            if delta > 0.1:
                signals.append("noise_sensitive")
        except Exception as e:
            logger.warning(f"Noise injection test failed: {e}")
            report.details["noise_error"] = str(e)

        report.is_masked = len(signals) >= 2
        report.confidence = min(len(signals) / 3.0, 1.0)
        report.details["signals"] = signals

        if report.is_masked:
            report.recommendation = (
                "Gradient masking detected. Do NOT trust gradient-based attack results alone. "
                "Use black-box attacks (Square, HopSkipJump) and transfer attacks for reliable evaluation. "
                "Consider Gaussian input noise injection during training (per ICLR 2025 guidance)."
            )
        else:
            report.recommendation = "No gradient masking detected. Gradient-based evaluation appears reliable."

        return report

    def _compute_fosc(self, classifier: Any, x: np.ndarray, y: np.ndarray) -> float:
        """Compute First-Order Stationarity Condition score.
        
        High FOSC indicates that the loss landscape has not been properly
        optimized by the attack, suggesting gradient masking.
        """
        try:
            grads = classifier.loss_gradient(x, y)
        except Exception:
            if hasattr(classifier, 'class_gradient'):
                grads = classifier.class_gradient(x, label=np.argmax(y, axis=1))
                if grads.ndim > x.ndim:
                    grads = grads.reshape(x.shape)
            else:
                raise
        grad_norms = np.sqrt(np.sum(grads ** 2, axis=tuple(range(1, grads.ndim))))
        return float(np.mean(grad_norms))

    def _noise_injection_test(
        self, classifier: Any, x: np.ndarray, y: np.ndarray,
    ) -> float:
        """Test if small Gaussian noise significantly changes gradients.
        
        If gradients are highly sensitive to tiny noise, the gradient
        landscape is likely unreliable (shattered gradients).
        """
        try:
            clean_grads = classifier.loss_gradient(x[:min(20, len(x))], y[:min(20, len(y))])
        except Exception:
            return 0.0

        deltas = []
        x_subset = x[:min(20, len(x))]
        y_subset = y[:min(20, len(y))]
        for _ in range(self.num_noise_samples):
            noise = np.random.normal(0, self.noise_sigma, x_subset.shape).astype(x_subset.dtype)
            x_noisy = np.clip(x_subset + noise, 0.0, 1.0)
            try:
                noisy_grads = classifier.loss_gradient(x_noisy, y_subset)
                cos_sim = np.sum(clean_grads * noisy_grads) / (
                    np.linalg.norm(clean_grads) * np.linalg.norm(noisy_grads) + 1e-10
                )
                deltas.append(1.0 - float(cos_sim))
            except Exception:
                continue

        return float(np.mean(deltas)) if deltas else 0.0

    def detect_from_attack_results(
        self,
        attack_results: list[dict[str, Any]],
    ) -> MaskingReport:
        """Quick detection from already-collected attack results.
        
        Compares gradient-based attacks (FGSM, PGD, C&W, BIM, AutoPGD)
        with gradient-free attacks (Square, HopSkipJump, SimBA, ZOO, GeoDA).
        """
        wb_attacks = {"fgsm", "pgd", "bim", "auto_pgd", "carlini_wagner_l2",
                      "deepfool", "autoattack", "elastic_net", "jsma"}
        bb_attacks = {"square_attack", "hopskipjump", "simba", "zoo", "geoda",
                      "boundary_attack", "pixel_attack"}

        wb_rates = []
        bb_rates = []
        for r in attack_results:
            name = r.get("attack", "").lower()
            sr = r.get("success_rate")
            if sr is None or r.get("status") != "completed":
                continue
            if name in wb_attacks:
                wb_rates.append(sr)
            elif name in bb_attacks:
                bb_rates.append(sr)

        report = MaskingReport()
        if wb_rates:
            report.wb_success_rate = float(np.mean(wb_rates))
        if bb_rates:
            report.bb_success_rate = float(np.mean(bb_rates))
        report.discrepancy = report.bb_success_rate - report.wb_success_rate
        report.is_masked = report.discrepancy > self.discrepancy_threshold
        report.confidence = min(abs(report.discrepancy) / 0.3, 1.0) if report.discrepancy > 0 else 0.0

        if report.is_masked:
            report.recommendation = (
                f"Gradient masking likely: black-box attacks succeed {report.bb_success_rate:.1%} "
                f"vs white-box {report.wb_success_rate:.1%}. "
                "Use adaptive attacks and gradient-free methods for reliable evaluation."
            )
        else:
            report.recommendation = "No significant gradient masking detected from attack result comparison."

        return report
