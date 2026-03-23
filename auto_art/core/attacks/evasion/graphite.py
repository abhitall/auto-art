"""
GRAPHITE attack wrappers for physical-world adversarial robustness.

Reference: ART - GRAPHITEBlackbox, GRAPHITEWhiteboxPyTorch
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import GRAPHITEBlackbox as ARTGraphiteBlackbox
    ART_GRAPHITE_BB_AVAILABLE = True
except ImportError:
    ART_GRAPHITE_BB_AVAILABLE = False

try:
    from art.attacks.evasion import GRAPHITEWhiteboxPyTorch as ARTGraphiteWB
    ART_GRAPHITE_WB_AVAILABLE = True
except ImportError:
    ART_GRAPHITE_WB_AVAILABLE = False


class GRAPHITEBlackboxWrapper:
    def __init__(self, estimator: Any, noise_budget: float = 0.1,
                 num_xforms: int = 100, max_iter: int = 200,
                 batch_size: int = 32, verbose: bool = True, **kwargs):
        if not ART_GRAPHITE_BB_AVAILABLE:
            raise ImportError("ART GRAPHITEBlackbox not available.")
        self.art_attack = ARTGraphiteBlackbox(
            classifier=estimator, noise_budget=noise_budget,
            num_xforms=num_xforms, max_iter=max_iter,
            batch_size=batch_size, verbose=verbose, **kwargs,
        )
        self.attack_params = {"noise_budget": noise_budget, "num_xforms": num_xforms,
                              "max_iter": max_iter}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)


class GRAPHITEWhiteboxWrapper:
    def __init__(self, estimator: Any, noise_budget: float = 0.1,
                 num_xforms: int = 100, max_iter: int = 200,
                 batch_size: int = 32, verbose: bool = True, **kwargs):
        if not ART_GRAPHITE_WB_AVAILABLE:
            raise ImportError("ART GRAPHITEWhiteboxPyTorch not available.")
        self.art_attack = ARTGraphiteWB(
            estimator=estimator, noise_budget=noise_budget,
            num_xforms=num_xforms, max_iter=max_iter,
            batch_size=batch_size, verbose=verbose, **kwargs,
        )
        self.attack_params = {"noise_budget": noise_budget, "num_xforms": num_xforms,
                              "max_iter": max_iter}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
