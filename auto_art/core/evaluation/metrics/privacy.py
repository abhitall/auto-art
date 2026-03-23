"""
Privacy metrics wrappers from ART.

Covers: Pointwise Differential Training Privacy, SHAPr Membership Privacy Risk.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrivacyReport:
    """Privacy metric results."""
    pdtp_scores: Optional[np.ndarray] = None
    pdtp_mean: float = 0.0
    shapr_scores: Optional[np.ndarray] = None
    shapr_mean: float = 0.0
    high_risk_fraction: float = 0.0
    error: Optional[str] = None


class PrivacyMetricsCalculator:
    """Computes privacy-related metrics using ART's built-in functions."""

    def __init__(self, risk_threshold: float = 0.7):
        self.risk_threshold = risk_threshold

    def compute_pdtp(
        self,
        classifier: Any,
        x: np.ndarray,
        y: np.ndarray,
        num_iter: int = 10,
    ) -> dict[str, Any]:
        """Compute Pointwise Differential Training Privacy scores."""
        try:
            from art.metrics import pdtp
            scores = pdtp(classifier, x, y, num_iter=num_iter)
            return {
                "pdtp_scores": scores,
                "pdtp_mean": float(np.mean(scores)),
                "pdtp_max": float(np.max(scores)),
                "high_risk_count": int(np.sum(scores > self.risk_threshold)),
            }
        except ImportError:
            logger.warning("ART pdtp metric not available.")
            return {"error": "pdtp not available"}
        except Exception as e:
            logger.error(f"PDTP computation failed: {e}")
            return {"error": str(e)}

    def compute_shapr(
        self,
        classifier: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Compute SHAPr Membership Privacy Risk scores."""
        try:
            from art.metrics import SHAPr
            scores = SHAPr(classifier, x_train, y_train, x_test, y_test)
            return {
                "shapr_scores": scores,
                "shapr_mean": float(np.mean(scores)),
                "shapr_max": float(np.max(scores)),
                "high_risk_fraction": float(np.mean(scores > self.risk_threshold)),
            }
        except ImportError:
            logger.warning("ART SHAPr metric not available.")
            return {"error": "SHAPr not available"}
        except Exception as e:
            logger.error(f"SHAPr computation failed: {e}")
            return {"error": str(e)}

    def compute_all(
        self,
        classifier: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> PrivacyReport:
        """Compute all privacy metrics."""
        report = PrivacyReport()
        pdtp_result = self.compute_pdtp(classifier, x_train, y_train)
        if "error" not in pdtp_result:
            report.pdtp_scores = pdtp_result.get("pdtp_scores")
            report.pdtp_mean = pdtp_result.get("pdtp_mean", 0.0)

        shapr_result = self.compute_shapr(classifier, x_train, y_train, x_test, y_test)
        if "error" not in shapr_result:
            report.shapr_scores = shapr_result.get("shapr_scores")
            report.shapr_mean = shapr_result.get("shapr_mean", 0.0)
            report.high_risk_fraction = shapr_result.get("high_risk_fraction", 0.0)

        return report
