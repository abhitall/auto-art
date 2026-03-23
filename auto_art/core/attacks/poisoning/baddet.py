"""
BadDet poisoning attack wrappers for object detection (Chan et al., 2022).

Wraps ART's BadDet family of backdoor attacks targeting object detectors:
- BadDetOGA: Object Generation Attack
- BadDetRMA: Regional Misclassification Attack
- BadDetGMA: Global Misclassification Attack
- BadDetODA: Object Disappearance Attack

Reference: Chan et al., "BadDet: Backdoor Attacks on Object Detection", ECCV 2022.
"""

from typing import Any, Optional, Tuple
import numpy as np

try:
    from art.attacks.poisoning.backdoor_attack import (
        PoisoningAttackBackdoor,
    )
    ART_BADDET_AVAILABLE = True
except ImportError:
    ART_BADDET_AVAILABLE = False

try:
    from art.attacks.poisoning import BadDetOGA as ARTBadDetOGA
    BADDET_OGA_AVAILABLE = True
except ImportError:
    BADDET_OGA_AVAILABLE = False

try:
    from art.attacks.poisoning import BadDetRMA as ARTBadDetRMA
    BADDET_RMA_AVAILABLE = True
except ImportError:
    BADDET_RMA_AVAILABLE = False

try:
    from art.attacks.poisoning import BadDetGMA as ARTBadDetGMA
    BADDET_GMA_AVAILABLE = True
except ImportError:
    BADDET_GMA_AVAILABLE = False

try:
    from art.attacks.poisoning import BadDetODA as ARTBadDetODA
    BADDET_ODA_AVAILABLE = True
except ImportError:
    BADDET_ODA_AVAILABLE = False


class BadDetOGAWrapper:
    """Wrapper for ART's BadDet Object Generation Attack (Chan et al., 2022).

    Poisons object detection training data by inserting trigger patterns that
    cause the detector to generate phantom objects at inference time.
    """

    def __init__(
        self,
        estimator: Any,
        target_class: int = 0,
        poisoning_rate: float = 0.1,
        trigger_size: int = 30,
        **kwargs,
    ):
        if not BADDET_OGA_AVAILABLE:
            raise ImportError(
                "ART BadDetOGA not available. Ensure adversarial-robustness-toolbox "
                "is installed with object detection support."
            )
        self.estimator = estimator
        self.target_class = target_class
        self.poisoning_rate = poisoning_rate
        self.trigger_size = trigger_size
        self.art_attack = ARTBadDetOGA(
            estimator=estimator,
            target_class=target_class,
            trigger_size=trigger_size,
            **kwargs,
        )
        self.attack_params = {
            "target_class": target_class,
            "poisoning_rate": poisoning_rate,
            "trigger_size": trigger_size,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(
        self,
        x_clean: np.ndarray,
        y_clean: np.ndarray,
        indices_to_poison: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = x_clean.shape[0]
        if indices_to_poison is None:
            num_poison = max(1, int(num_samples * self.poisoning_rate))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            indices_to_poison = all_indices[:num_poison]

        x_poisoned = x_clean.copy()
        y_poisoned = y_clean.copy()
        poisoned_data = self.art_attack.poison(
            x=x_clean[indices_to_poison],
            y=y_clean[indices_to_poison],
        )
        if isinstance(poisoned_data, tuple):
            x_poisoned[indices_to_poison] = poisoned_data[0]
            y_poisoned[indices_to_poison] = poisoned_data[1]
        else:
            x_poisoned[indices_to_poison] = poisoned_data

        return x_poisoned, y_poisoned


class BadDetRMAWrapper:
    """Wrapper for ART's BadDet Regional Misclassification Attack (Chan et al., 2022).

    Poisons object detection training data so that objects in a specific region
    are misclassified at inference time when a trigger is present.
    """

    def __init__(
        self,
        estimator: Any,
        target_class: int = 0,
        poisoning_rate: float = 0.1,
        trigger_size: int = 30,
        **kwargs,
    ):
        if not BADDET_RMA_AVAILABLE:
            raise ImportError(
                "ART BadDetRMA not available. Ensure adversarial-robustness-toolbox "
                "is installed with object detection support."
            )
        self.estimator = estimator
        self.target_class = target_class
        self.poisoning_rate = poisoning_rate
        self.trigger_size = trigger_size
        self.art_attack = ARTBadDetRMA(
            estimator=estimator,
            target_class=target_class,
            trigger_size=trigger_size,
            **kwargs,
        )
        self.attack_params = {
            "target_class": target_class,
            "poisoning_rate": poisoning_rate,
            "trigger_size": trigger_size,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(
        self,
        x_clean: np.ndarray,
        y_clean: np.ndarray,
        indices_to_poison: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = x_clean.shape[0]
        if indices_to_poison is None:
            num_poison = max(1, int(num_samples * self.poisoning_rate))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            indices_to_poison = all_indices[:num_poison]

        x_poisoned = x_clean.copy()
        y_poisoned = y_clean.copy()
        poisoned_data = self.art_attack.poison(
            x=x_clean[indices_to_poison],
            y=y_clean[indices_to_poison],
        )
        if isinstance(poisoned_data, tuple):
            x_poisoned[indices_to_poison] = poisoned_data[0]
            y_poisoned[indices_to_poison] = poisoned_data[1]
        else:
            x_poisoned[indices_to_poison] = poisoned_data

        return x_poisoned, y_poisoned


class BadDetGMAWrapper:
    """Wrapper for ART's BadDet Global Misclassification Attack (Chan et al., 2022).

    Poisons object detection training data so that all detected objects are
    misclassified when a trigger pattern is present in the input.
    """

    def __init__(
        self,
        estimator: Any,
        target_class: int = 0,
        poisoning_rate: float = 0.1,
        trigger_size: int = 30,
        **kwargs,
    ):
        if not BADDET_GMA_AVAILABLE:
            raise ImportError(
                "ART BadDetGMA not available. Ensure adversarial-robustness-toolbox "
                "is installed with object detection support."
            )
        self.estimator = estimator
        self.target_class = target_class
        self.poisoning_rate = poisoning_rate
        self.trigger_size = trigger_size
        self.art_attack = ARTBadDetGMA(
            estimator=estimator,
            target_class=target_class,
            trigger_size=trigger_size,
            **kwargs,
        )
        self.attack_params = {
            "target_class": target_class,
            "poisoning_rate": poisoning_rate,
            "trigger_size": trigger_size,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(
        self,
        x_clean: np.ndarray,
        y_clean: np.ndarray,
        indices_to_poison: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = x_clean.shape[0]
        if indices_to_poison is None:
            num_poison = max(1, int(num_samples * self.poisoning_rate))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            indices_to_poison = all_indices[:num_poison]

        x_poisoned = x_clean.copy()
        y_poisoned = y_clean.copy()
        poisoned_data = self.art_attack.poison(
            x=x_clean[indices_to_poison],
            y=y_clean[indices_to_poison],
        )
        if isinstance(poisoned_data, tuple):
            x_poisoned[indices_to_poison] = poisoned_data[0]
            y_poisoned[indices_to_poison] = poisoned_data[1]
        else:
            x_poisoned[indices_to_poison] = poisoned_data

        return x_poisoned, y_poisoned


class BadDetODAWrapper:
    """Wrapper for ART's BadDet Object Disappearance Attack (Chan et al., 2022).

    Poisons object detection training data so that the detector fails to
    detect certain objects when a trigger pattern is present.
    """

    def __init__(
        self,
        estimator: Any,
        target_class: int = 0,
        poisoning_rate: float = 0.1,
        trigger_size: int = 30,
        **kwargs,
    ):
        if not BADDET_ODA_AVAILABLE:
            raise ImportError(
                "ART BadDetODA not available. Ensure adversarial-robustness-toolbox "
                "is installed with object detection support."
            )
        self.estimator = estimator
        self.target_class = target_class
        self.poisoning_rate = poisoning_rate
        self.trigger_size = trigger_size
        self.art_attack = ARTBadDetODA(
            estimator=estimator,
            target_class=target_class,
            trigger_size=trigger_size,
            **kwargs,
        )
        self.attack_params = {
            "target_class": target_class,
            "poisoning_rate": poisoning_rate,
            "trigger_size": trigger_size,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(
        self,
        x_clean: np.ndarray,
        y_clean: np.ndarray,
        indices_to_poison: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = x_clean.shape[0]
        if indices_to_poison is None:
            num_poison = max(1, int(num_samples * self.poisoning_rate))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            indices_to_poison = all_indices[:num_poison]

        x_poisoned = x_clean.copy()
        y_poisoned = y_clean.copy()
        poisoned_data = self.art_attack.poison(
            x=x_clean[indices_to_poison],
            y=y_clean[indices_to_poison],
        )
        if isinstance(poisoned_data, tuple):
            x_poisoned[indices_to_poison] = poisoned_data[0]
            y_poisoned[indices_to_poison] = poisoned_data[1]
        else:
            x_poisoned[indices_to_poison] = poisoned_data

        return x_poisoned, y_poisoned
