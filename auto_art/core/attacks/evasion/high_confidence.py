"""
High Confidence Low Uncertainty (HCLU) attack wrapper.

Reference: ART - HighConfidenceLowUncertainty
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import HighConfidenceLowUncertainty as ARTHCLU
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class HighConfidenceLowUncertaintyWrapper:
    def __init__(self, estimator: Any, conf_threshold: float = 0.75,
                 unc_increase: float = 100.0, min_val: float = 0.0,
                 max_val: float = 1.0, verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART HighConfidenceLowUncertainty not available.")
        self.art_attack = ARTHCLU(
            classifier=estimator, conf_threshold=conf_threshold,
            unc_increase=unc_increase, min_val=min_val, max_val=max_val,
            verbose=verbose,
        )
        self.attack_params = {"conf_threshold": conf_threshold, "unc_increase": unc_increase}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
