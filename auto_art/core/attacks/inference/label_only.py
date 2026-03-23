"""
Label-only membership inference attack wrappers.

Wraps ART's label-only membership inference attacks that operate
without access to model confidence scores:
- LabelOnlyDecisionBoundary: Boundary distance-based membership inference
- LabelOnlyGapAttack: Gap-based membership inference

Reference: Choquette-Choo et al., "Label-Only Membership Inference Attacks", ICML 2020.
"""

from typing import Any, Optional
import numpy as np

try:
    from art.attacks.inference.membership_inference import (
        LabelOnlyDecisionBoundary as ARTLabelOnlyDecisionBoundary,
    )
    from art.estimators.classification import ClassifierMixin
    LABEL_ONLY_BOUNDARY_AVAILABLE = True
except ImportError:
    LABEL_ONLY_BOUNDARY_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass

try:
    from art.attacks.inference.membership_inference import (
        LabelOnlyGapAttack as ARTLabelOnlyGapAttack,
    )
    LABEL_ONLY_GAP_AVAILABLE = True
except ImportError:
    LABEL_ONLY_GAP_AVAILABLE = False
    try:
        from art.attacks.inference.membership_inference import (
            MembershipInferenceBlackBox as ARTMembershipInferenceBBFallback,
        )
        LABEL_ONLY_GAP_FALLBACK_AVAILABLE = True
    except ImportError:
        LABEL_ONLY_GAP_FALLBACK_AVAILABLE = False


class LabelOnlyBoundaryDistanceWrapper:
    """Wrapper for ART's LabelOnlyDecisionBoundary attack (Choquette-Choo et al., 2020).

    Infers training set membership by measuring the distance from each sample
    to the model's decision boundary. Members tend to be farther from the
    boundary than non-members.
    """

    def __init__(
        self,
        estimator: Any,
        distance_threshold_tau: float = 0.5,
        **kwargs,
    ):
        if not LABEL_ONLY_BOUNDARY_AVAILABLE:
            raise ImportError(
                "ART LabelOnlyDecisionBoundary not available. "
                "Ensure adversarial-robustness-toolbox is installed."
            )
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.estimator = estimator
        self.distance_threshold_tau = distance_threshold_tau
        self.art_attack = ARTLabelOnlyDecisionBoundary(
            estimator=estimator,
            distance_threshold_tau=distance_threshold_tau,
            **kwargs,
        )
        self.attack_params = {
            "distance_threshold_tau": distance_threshold_tau,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def calibrate_distance_threshold(
        self,
        x_member: np.ndarray,
        y_member: np.ndarray,
        x_nonmember: np.ndarray,
        y_nonmember: np.ndarray,
        **kwargs,
    ) -> None:
        self.art_attack.calibrate_distance_threshold(
            x=np.concatenate([x_member, x_nonmember], axis=0),
            y=np.concatenate([y_member, y_nonmember], axis=0),
            is_member=np.concatenate([
                np.ones(len(x_member), dtype=int),
                np.zeros(len(x_nonmember), dtype=int),
            ]),
            **kwargs,
        )

    def infer(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.infer(x=x, y=y, **kwargs)


class LabelOnlyGapAttackWrapper:
    """Wrapper for label-only gap-based membership inference.

    If ART provides LabelOnlyGapAttack, uses that directly. Otherwise
    falls back to MembershipInferenceBlackBox with input_type='loss'
    as a proxy for gap-based inference.
    """

    def __init__(
        self,
        estimator: Any,
        distance_threshold_tau: float = 0.5,
        **kwargs,
    ):
        self.estimator = estimator
        self.distance_threshold_tau = distance_threshold_tau
        self._using_fallback = False

        if LABEL_ONLY_GAP_AVAILABLE:
            self.art_attack = ARTLabelOnlyGapAttack(
                estimator=estimator,
                distance_threshold_tau=distance_threshold_tau,
                **kwargs,
            )
        elif LABEL_ONLY_GAP_FALLBACK_AVAILABLE:
            self._using_fallback = True
            self.art_attack = ARTMembershipInferenceBBFallback(
                classifier=estimator,
                input_type="loss",
                **kwargs,
            )
        else:
            raise ImportError(
                "Neither LabelOnlyGapAttack nor MembershipInferenceBlackBox "
                "fallback is available from ART."
            )

        self.attack_params = {
            "distance_threshold_tau": distance_threshold_tau,
            "using_fallback": self._using_fallback,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def fit(
        self,
        x_member: np.ndarray,
        y_member: np.ndarray,
        x_nonmember: np.ndarray,
        y_nonmember: np.ndarray,
        **kwargs,
    ) -> None:
        if self._using_fallback:
            x_combined = np.concatenate([x_member, x_nonmember], axis=0)
            y_combined = np.concatenate([y_member, y_nonmember], axis=0)
            is_member = np.concatenate([
                np.ones(len(x_member), dtype=int),
                np.zeros(len(x_nonmember), dtype=int),
            ])
            self.art_attack.fit(
                x=x_combined, y=y_combined,
                is_member_labels=is_member,
                **kwargs,
            )
        elif hasattr(self.art_attack, 'calibrate_distance_threshold'):
            self.art_attack.calibrate_distance_threshold(
                x=np.concatenate([x_member, x_nonmember], axis=0),
                y=np.concatenate([y_member, y_nonmember], axis=0),
                is_member=np.concatenate([
                    np.ones(len(x_member), dtype=int),
                    np.zeros(len(x_nonmember), dtype=int),
                ]),
                **kwargs,
            )

    def infer(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.infer(x=x, y=y, **kwargs)
