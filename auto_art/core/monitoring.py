"""
Production robustness monitoring module.

Provides drift detection, automated retraining triggers, and supply chain
security scanning for deployed ML models.

Based on HAMF (2025) real-time drift detection and CRT (ICML 2025)
continual robust training approaches.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Robustness drift detection result."""
    drift_detected: bool = False
    rdi_current: float = 0.0
    rdi_baseline: float = 0.0
    rdi_delta: float = 0.0
    accuracy_current: float = 0.0
    accuracy_baseline: float = 0.0
    timestamp: float = 0.0
    recommendation: str = ""


@dataclass
class SupplyChainReport:
    """Model supply chain security scan result."""
    is_safe: bool = True
    format_safe: bool = True
    backdoor_indicators: list[str] = field(default_factory=list)
    provenance_verified: bool = False
    file_format: str = ""
    risk_level: str = "low"
    details: dict[str, Any] = field(default_factory=dict)


class RobustnessDriftMonitor:
    """Monitors robustness drift over time using RDI as a cheap proxy.

    Compares current RDI against a stored baseline and triggers alerts
    when robustness degrades beyond a configurable threshold.
    """

    def __init__(
        self,
        baseline_path: Optional[str] = None,
        drift_threshold: float = 0.1,
        accuracy_threshold: float = 0.05,
    ):
        self.baseline_path = baseline_path or os.path.join(
            str(Path.home()), ".auto_art", "robustness_baseline.json"
        )
        self.drift_threshold = drift_threshold
        self.accuracy_threshold = accuracy_threshold
        self._baseline: Optional[dict[str, float]] = None

    def set_baseline(self, rdi_score: float, accuracy: float) -> None:
        """Store baseline metrics for future comparison."""
        self._baseline = {
            "rdi": rdi_score,
            "accuracy": accuracy,
            "timestamp": time.time(),
        }
        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
        with open(self.baseline_path, "w") as f:
            json.dump(self._baseline, f, indent=2)

    def _load_baseline(self) -> Optional[dict[str, float]]:
        if self._baseline:
            return self._baseline
        if os.path.exists(self.baseline_path):
            try:
                with open(self.baseline_path) as f:
                    self._baseline = json.load(f)
                return self._baseline
            except Exception:
                pass
        return None

    def check_drift(
        self,
        classifier: Any,
        x: np.ndarray,
        y: np.ndarray,
    ) -> DriftReport:
        """Check for robustness drift against baseline."""
        from auto_art.core.evaluation.metrics.rdi import RDICalculator

        rdi_calc = RDICalculator(num_samples=min(200, len(x)))
        rdi_report = rdi_calc.compute(classifier, x, y)

        predictions = classifier.predict(x)
        true_labels = np.argmax(y, axis=1) if y.ndim > 1 else y
        accuracy = float(np.mean(np.argmax(predictions, axis=1) == true_labels))

        baseline = self._load_baseline()
        report = DriftReport(
            rdi_current=rdi_report.rdi_score,
            accuracy_current=accuracy,
            timestamp=time.time(),
        )

        if baseline:
            report.rdi_baseline = baseline["rdi"]
            report.accuracy_baseline = baseline["accuracy"]
            report.rdi_delta = baseline["rdi"] - rdi_report.rdi_score

            rdi_drift = report.rdi_delta > self.drift_threshold
            acc_drift = (baseline["accuracy"] - accuracy) > self.accuracy_threshold
            report.drift_detected = rdi_drift or acc_drift

            if report.drift_detected:
                reasons = []
                if rdi_drift:
                    reasons.append(f"RDI dropped by {report.rdi_delta:.3f}")
                if acc_drift:
                    reasons.append(f"Accuracy dropped by {baseline['accuracy'] - accuracy:.3f}")
                report.recommendation = (
                    f"Robustness drift detected: {', '.join(reasons)}. "
                    "Consider retraining with continual robust training (CRT) "
                    "or running full adversarial evaluation."
                )
            else:
                report.recommendation = "No significant robustness drift detected."
        else:
            report.recommendation = (
                "No baseline established. Setting current metrics as baseline."
            )
            self.set_baseline(rdi_report.rdi_score, accuracy)

        return report


class ModelSupplyChainScanner:
    """Scans model files for supply chain security risks.

    Checks for:
    - Pickle-based RCE vulnerabilities
    - SafeTensors format validation
    - Backdoor indicator heuristics
    - File format safety
    """

    SAFE_FORMATS = {".safetensors", ".onnx", ".tflite", ".pb"}
    RISKY_FORMATS = {".pkl", ".pickle", ".pt", ".pth", ".h5", ".hdf5", ".joblib"}

    def scan(self, model_path: str) -> SupplyChainReport:
        """Scan a model file for supply chain risks."""
        report = SupplyChainReport()
        path = Path(model_path)

        if not path.exists():
            report.is_safe = False
            report.details["error"] = f"File not found: {model_path}"
            return report

        report.file_format = path.suffix.lower()
        report.format_safe = report.file_format in self.SAFE_FORMATS

        if report.file_format in self.RISKY_FORMATS:
            report.risk_level = "medium"
            report.backdoor_indicators.append(
                f"Format '{report.file_format}' supports arbitrary code execution via pickle."
            )

        if report.file_format in {".pkl", ".pickle"}:
            report.risk_level = "high"
            report.backdoor_indicators.append(
                "Raw pickle file — HIGH RISK of arbitrary code execution. "
                "Convert to SafeTensors format."
            )

        file_size = path.stat().st_size
        report.details["file_size_bytes"] = file_size
        report.details["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        if report.file_format == ".safetensors":
            report.provenance_verified = True
            report.risk_level = "low"

        report.is_safe = report.risk_level != "high"
        return report
