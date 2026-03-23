"""
Decision Tree Attack evasion attack wrapper.

Wraps ART's DecisionTreeAttack into auto-art's wrapper pattern.
Exploits the structure of decision-tree-based classifiers to craft
adversarial examples with minimal perturbation.

Reference: Papernot et al., "Adversarial examples for malware
detection", ESORICS 2017
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import DecisionTreeAttack as ARTDecisionTreeAttack
    ART_DT_AVAILABLE = True
except ImportError:
    ART_DT_AVAILABLE = False


class DecisionTreeAttackWrapper:
    """Wrapper for ART's Decision Tree Attack (Papernot et al., 2017).

    Crafts adversarial examples by traversing decision-tree splits
    and applying the minimum offset to cross leaf boundaries.
    """

    def __init__(
        self,
        classifier: Any,
        offset: float = 0.001,
        verbose: bool = True,
    ):
        if not ART_DT_AVAILABLE:
            raise ImportError("ART DecisionTreeAttack not available. Install adversarial-robustness-toolbox.")

        self.art_attack = ARTDecisionTreeAttack(
            classifier=classifier,
            offset=offset,
            verbose=verbose,
        )
        self.attack_params = {
            "offset": offset,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
