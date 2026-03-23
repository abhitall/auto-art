"""
ART Rounding Postprocessor defence wrapper.

Wraps ART's Rounded postprocessor defence into the auto-art DefenceStrategy
interface. Rounding output probabilities reduces the precision of confidence
information available to adversaries.

Reference: ART Defences Wiki - Postprocessor section
"""

from typing import Any, Dict, Optional
import logging

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.postprocessor import Rounded as ARTRounded
    ART_ROUNDING_POSTPROCESSOR_AVAILABLE = True
except ImportError:
    ART_ROUNDING_POSTPROCESSOR_AVAILABLE = False


class RoundingDefence(DefenceStrategy):
    """Rounding postprocessor defence.

    Rounds model output probabilities to a fixed number of decimal places,
    reducing the precision of confidence information available to adversaries
    attempting model extraction or membership inference attacks.
    """

    def __init__(self, decimals: int = 4):
        super().__init__(defence_name="Rounding")
        self.decimals = decimals
        self._postprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_ROUNDING_POSTPROCESSOR_AVAILABLE:
            raise ImportError(
                "ART Rounded postprocessor defence not available. "
                "Install adversarial-robustness-toolbox."
            )
        self._postprocessor = ARTRounded(decimals=self.decimals)
        if hasattr(art_estimator, 'set_params'):
            current = getattr(
                art_estimator, 'postprocessing_defences', None
            )
            defences = list(current) if current else []
            defences.append(self._postprocessor)
            art_estimator.set_params(postprocessing_defences=defences)
        return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {"decimals": self.decimals}
