"""
Wrapper for ART's MembershipInferenceBlackBox attack.
"""
from typing import Any, Optional, Dict, Union
import numpy as np

try:
    from art.attacks.inference.membership_inference import MembershipInferenceBlackBox as ARTMembershipInferenceBB
    from art.estimators.classification import ClassifierMixin
    # Scikit-learn is a common choice for the attack model in MembershipInferenceBlackBox
    from sklearn.base import ClassifierMixin as SKLearnClassifierBase # For type hint
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    # Define dummy types for type hinting if ART or sklearn is not available
    class ClassifierMixin: pass
    class SKLearnClassifierBase: pass
    class ARTMembershipInferenceBB: pass


class MembershipInferenceBlackBoxWrapper:
    """
    A wrapper for ART's MembershipInferenceBlackBox attack.
    This attack trains a model to predict if a sample was in the training set of a target model.
    """
    def __init__(self,
                 target_model_estimator: ClassifierMixin,
                 attack_model_type: str = 'rf',
                 attack_model_instance: Optional[SKLearnClassifierBase] = None,
                 input_type: str = 'prediction',
                 **kwargs: Any
                 ):
        """
        Initializes the MembershipInferenceBlackBoxWrapper.
        Args:
            target_model_estimator: The ART classifier whose training membership is to be inferred.
            attack_model_type: Type of sklearn model to use for the attack ('rf', 'gb', 'lr', 'nn').
                               Ignored if `attack_model_instance` is provided.
            attack_model_instance: A pre-initialized scikit-learn binary classifier. If None, one is created.
            input_type: Specifies what information to use from the target model's output:
                        'prediction' (output probabilities), 'loss' (not typically used BB), or 'logits'.
            **kwargs: Additional parameters for ART's MembershipInferenceBlackBox
                      (e.g., `membership_attack_model_params` if `attack_model_type` is 'nn').
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. MembershipInferenceBlackBox cannot be used.")

        if not isinstance(target_model_estimator, ClassifierMixin): # type: ignore
            raise TypeError("target_model_estimator must be an ART ClassifierMixin.")

        self.target_model = target_model_estimator
        self.attack_model_type_str = attack_model_type
        self.user_provided_attack_model = attack_model_instance
        self.input_type = input_type
        self.kwargs_init = kwargs
        self.art_attack_instance: Optional[ARTMembershipInferenceBB] = None # Initialized lazily before fit/infer

    def _initialize_attack_instance_if_needed(self):
        """Helper to initialize the ART attack instance if not already done."""
        if self.art_attack_instance is None:
            # This logic might move to __init__ if attack_model can always be defined there.
            # ART's MembershipInferenceBlackBox __init__ takes the target_model and attack_model.
            current_attack_model = self.user_provided_attack_model
            if current_attack_model is None and self.attack_model_type_str != 'nn': # 'nn' is handled by ART if model=None
                # Dynamically import sklearn classifiers only when needed
                if self.attack_model_type_str == 'rf':
                    from sklearn.ensemble import RandomForestClassifier
                    current_attack_model = RandomForestClassifier(n_estimators=100, random_state=42) # Add defaults
                elif self.attack_model_type_str == 'gb':
                    from sklearn.ensemble import GradientBoostingClassifier
                    current_attack_model = GradientBoostingClassifier(random_state=42)
                elif self.attack_model_type_str == 'lr':
                    from sklearn.linear_model import LogisticRegression
                    current_attack_model = LogisticRegression(solver='liblinear', random_state=42)
                else: # Should not happen if 'nn' is the only other main supported type by ART for auto-creation
                    raise ValueError(f"Unsupported internal attack_model_type: {self.attack_model_type_str} for auto-creation without 'nn'.")

            # If attack_model_type is 'nn' and current_attack_model is still None, ART will create its default MLP.
            # If current_attack_model is provided, attack_model_type is ignored by ART.
            self.art_attack_instance = ARTMembershipInferenceBB( # type: ignore
                classifier=self.target_model,
                attack_model=current_attack_model,
                attack_model_type=self.attack_model_type_str if current_attack_model is None else None,
                input_type=self.input_type,
                **(self.kwargs_init)
            )

    def fit(self,
            x_member_train: np.ndarray, y_member_train: np.ndarray,
            x_nonmember_train: np.ndarray, y_nonmember_train: np.ndarray,
            **kwargs_fit) -> None:
        """
        Trains the membership inference attack model.
        Args:
            x_member_train: Input samples known to be members of the target model's training set.
            y_member_train: Corresponding labels for x_member_train.
            x_nonmember_train: Input samples known not to be members.
            y_nonmember_train: Corresponding labels for x_nonmember_train.
            **kwargs_fit: Additional arguments for ART MembershipInferenceBlackBox's `fit` method.
        """
        if not ART_AVAILABLE: raise ImportError("ART not available for MembershipInferenceBlackBox.fit.")
        self._initialize_attack_instance_if_needed()
        if self.art_attack_instance is None: raise RuntimeError("ART Attack instance failed to initialize.")

        x_combined = np.concatenate((x_member_train, x_nonmember_train), axis=0)
        y_combined = np.concatenate((y_member_train, y_nonmember_train), axis=0)

        is_member_labels = np.concatenate((np.ones(len(x_member_train), dtype=int),
                                           np.zeros(len(x_nonmember_train), dtype=int)), axis=0)

        # ART's `fit` expects `x`, `y` (features and true labels of combined set)
        # and `is_member_labels` (binary labels indicating membership for the attack model).
        # `test_x`, `test_y` are optional for evaluating the attack model during fit.
        self.art_attack_instance.fit(x=x_combined, y=y_combined,
                                     is_member_labels=is_member_labels,
                                     test_x=None, test_y=None, # Not providing test set for attack model here
                                     **kwargs_fit)

    def infer(self,
              x_infer_on: np.ndarray, y_infer_on: np.ndarray,
              **kwargs_infer) -> np.ndarray:
        """
        Performs membership inference on the provided data.
        The attack model must be trained using `fit` before calling `infer`.
        Args:
            x_infer_on: Input samples to infer membership for.
            y_infer_on: Corresponding labels for x_infer_on.
            **kwargs_infer: Additional arguments for ART MembershipInferenceBlackBox's `infer` method.
        Returns:
            An array of binary predictions (0 for non-member, 1 for member).
        """
        if not ART_AVAILABLE: raise ImportError("ART not available for MembershipInferenceBlackBox.infer.")
        if self.art_attack_instance is None:
            raise RuntimeError("Attack model not initialized/trained. Call `fit` first.")

        # ART's `infer` method expects `x` and `y` (the data to infer membership for).
        return self.art_attack_instance.infer(x=x_infer_on, y=y_infer_on, **kwargs_infer)
