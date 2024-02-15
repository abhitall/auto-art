"""
Inference attack strategies.
"""

from typing import Optional, Tuple, Any
import numpy as np

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
from art.attacks.inference.model_inversion import ModelInversion

from .base import AttackStrategy

class MembershipInferenceBlackBoxAttack(AttackStrategy):
    """Membership inference black-box attack."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        num_shadow_models: int = 5
    ):
        """Initialize attack."""
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.num_shadow_models = num_shadow_models
        self._attack = MembershipInferenceBlackBox(
            estimator=None,
            attack_model_type='nn',
            input_type='prediction'
        )
    
    def execute(
        self,
        classifier: Any,
        x: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Execute attack."""
        self._attack.estimator = classifier
        membership_preds = self._attack.infer(x, y)
        success_rate = np.mean(membership_preds)
        return membership_preds, success_rate

class AttributeInferenceBlackBoxAttack(AttackStrategy):
    """Attribute inference black-box attack."""
    
    def __init__(
        self,
        attack_feature: int = 0,
        sensitive_features: Optional[list] = None,
        confidence_threshold: float = 0.5
    ):
        """Initialize attack."""
        super().__init__()
        self.attack_feature = attack_feature
        self.sensitive_features = sensitive_features or []
        self.confidence_threshold = confidence_threshold
        self._attack = AttributeInferenceBlackBox(
            estimator=None,
            attack_feature=attack_feature,
            attack_model_type='nn'
        )
    
    def execute(
        self,
        classifier: Any,
        x: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Execute attack."""
        self._attack.estimator = classifier
        inferred_attrs = self._attack.infer(x, y)
        success_rate = np.mean(inferred_attrs > self.confidence_threshold)
        return inferred_attrs, success_rate

class AttributeInferenceWhiteBoxAttack(AttackStrategy):
    """Attribute inference white-box attack."""
    
    def __init__(
        self,
        attack_feature: int = 0,
        sensitive_features: Optional[list] = None,
        confidence_threshold: float = 0.5
    ):
        """Initialize attack."""
        super().__init__()
        self.attack_feature = attack_feature
        self.sensitive_features = sensitive_features or []
        self.confidence_threshold = confidence_threshold
        self._attack = AttributeInferenceBlackBox(
            estimator=None,
            attack_feature=attack_feature,
            attack_model_type='nn'
        )
    
    def execute(
        self,
        classifier: Any,
        x: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Execute attack."""
        self._attack.estimator = classifier
        inferred_attrs = self._attack.infer(x, y)
        success_rate = np.mean(inferred_attrs > self.confidence_threshold)
        return inferred_attrs, success_rate

class ModelInversionAttack(AttackStrategy):
    """Model inversion attack."""
    
    def __init__(
        self,
        target_class: int = 0,
        learning_rate: float = 0.01,
        max_iter: int = 100
    ):
        """Initialize attack."""
        super().__init__()
        self.target_class = target_class
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._attack = ModelInversion(
            estimator=None,
            max_iter=max_iter,
            learning_rate=learning_rate
        )
    
    def execute(
        self,
        classifier: Any,
        x: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Execute attack."""
        self._attack.estimator = classifier
        reconstructed = self._attack.infer(x)
        # Calculate reconstruction success rate based on classifier predictions
        predictions = classifier.predict(reconstructed)
        success_rate = np.mean(np.argmax(predictions, axis=1) == self.target_class)
        return reconstructed, success_rate 