"""
White-box attribute inference attack wrappers.

Wraps ART's white-box attribute inference attacks that leverage
internal model parameters for inferring sensitive attributes:
- AttributeInferenceWhiteBoxDecisionTree
- AttributeInferenceWhiteBoxLifestyleDecisionTree

Reference: ART documentation on attribute inference attacks.
"""

from typing import Any, Optional
import numpy as np

try:
    from art.attacks.inference.attribute_inference import (
        AttributeInferenceWhiteBoxDecisionTree as ARTAttrInfWBDT,
    )
    ATTR_INF_WB_DT_AVAILABLE = True
except ImportError:
    ATTR_INF_WB_DT_AVAILABLE = False

try:
    from art.attacks.inference.attribute_inference import (
        AttributeInferenceWhiteBoxLifestyleDecisionTree as ARTAttrInfWBLifestyleDT,
    )
    ATTR_INF_WB_LIFESTYLE_DT_AVAILABLE = True
except ImportError:
    ATTR_INF_WB_LIFESTYLE_DT_AVAILABLE = False


class AttributeInferenceWhiteBoxDTWrapper:
    """Wrapper for ART's AttributeInferenceWhiteBoxDecisionTree.

    Uses a decision tree trained on the target model's internal
    representations to infer the value of a sensitive attribute
    from non-sensitive features and model predictions.
    """

    def __init__(
        self,
        estimator: Any,
        attack_feature_index: int,
        **kwargs,
    ):
        if not ATTR_INF_WB_DT_AVAILABLE:
            raise ImportError(
                "ART AttributeInferenceWhiteBoxDecisionTree not available. "
                "Ensure adversarial-robustness-toolbox is installed."
            )
        self.estimator = estimator
        self.attack_feature_index = attack_feature_index
        self.art_attack = ARTAttrInfWBDT(
            estimator=estimator,
            attack_feature=attack_feature_index,
            **kwargs,
        )
        self.attack_params = {
            "attack_feature_index": attack_feature_index,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def fit(
        self,
        x_train: np.ndarray,
        **kwargs,
    ) -> None:
        self.art_attack.fit(x=x_train, **kwargs)

    def infer(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.infer(x=x, y=y, **kwargs)


class AttributeInferenceWhiteBoxLifestyleDTWrapper:
    """Wrapper for ART's AttributeInferenceWhiteBoxLifestyleDecisionTree.

    A lifestyle-aware variant of white-box attribute inference that
    incorporates demographic and behavioral features into the decision
    tree attack model for improved inference accuracy.
    """

    def __init__(
        self,
        estimator: Any,
        attack_feature_index: int,
        **kwargs,
    ):
        if not ATTR_INF_WB_LIFESTYLE_DT_AVAILABLE:
            raise ImportError(
                "ART AttributeInferenceWhiteBoxLifestyleDecisionTree not available. "
                "Ensure adversarial-robustness-toolbox is installed."
            )
        self.estimator = estimator
        self.attack_feature_index = attack_feature_index
        self.art_attack = ARTAttrInfWBLifestyleDT(
            estimator=estimator,
            attack_feature=attack_feature_index,
            **kwargs,
        )
        self.attack_params = {
            "attack_feature_index": attack_feature_index,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def fit(
        self,
        x_train: np.ndarray,
        **kwargs,
    ) -> None:
        self.art_attack.fit(x=x_train, **kwargs)

    def infer(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.infer(x=x, y=y, **kwargs)
