"""
Laser Attack wrapper — physical laser perturbation attack.

Reference: Duan et al., "Adversarial Laser Beam", 2021
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import LaserAttack as ARTLaserAttack
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class LaserAttackWrapper:
    def __init__(self, estimator: Any, iterations: int = 10, laser_generator: Optional[Any] = None,
                 image_generator: Optional[Any] = None, random_initializations: int = 1,
                 optimisation_algorithm: Optional[Any] = None, verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART LaserAttack not available.")
        kwargs: dict = {"estimator": estimator, "iterations": iterations,
                        "random_initializations": random_initializations, "verbose": verbose}
        if laser_generator is not None:
            kwargs["laser_generator"] = laser_generator
        if image_generator is not None:
            kwargs["image_generator"] = image_generator
        if optimisation_algorithm is not None:
            kwargs["optimisation_algorithm"] = optimisation_algorithm
        self.art_attack = ARTLaserAttack(**kwargs)
        self.attack_params = {"iterations": iterations,
                              "random_initializations": random_initializations}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
