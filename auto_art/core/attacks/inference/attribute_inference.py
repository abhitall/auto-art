"""
Wrapper for ART's AttributeInferenceBlackBox attack.
"""
from typing import Any, Optional, List, Dict, Union
import numpy as np

try:
    from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox as ARTAttributeInferenceBB
    from art.estimators.estimator import BaseEstimator as ARTBaseEstimator
    from sklearn.base import ClassifierMixin as SKLearnClassifierBase # For default attack_model type hint
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ARTBaseEstimator: pass
    class SKLearnClassifierBase: pass
    class ARTAttributeInferenceBB: pass


class AttributeInferenceBlackBoxWrapper:
    """
    A wrapper for ART's AttributeInferenceBlackBox attack.
    Infers a specified attribute of a dataset, using a model trained on that dataset.
    """
    def __init__(self,
                 target_model_estimator: ARTBaseEstimator,
                 attack_feature_index: Union[int, slice], # Index/slice of attribute to infer IN THE ORIGINAL DATASET
                 attack_model_type: str = 'rf',
                 attack_model_instance: Optional[SKLearnClassifierBase] = None,
                 non_attack_feature_indices: Optional[List[int]] = None, # Indices of features NOT to be used by attack model (optional)
                 **kwargs: Any
                 ):
        """
        Initializes the AttributeInferenceBlackBoxWrapper.
        Args:
            target_model_estimator: ART estimator trained on data including the attribute to infer.
            attack_feature_index: Index or slice of the column in the original dataset that represents the attribute to infer.
            attack_model_type: Type of sklearn model for the inference attack ('rf', 'gb', 'lr', 'nn').
                               Ignored if `attack_model_instance` is provided.
            attack_model_instance: Pre-initialized scikit-learn classifier for inference. If None, one is created.
            non_attack_feature_indices: Optional list of indices of features to exclude from being input to the attack model.
                                       If None, all features apart from the attack_feature are considered.
            **kwargs: Additional parameters for ART's AttributeInferenceBlackBox (e.g., `scale_attack_model_input`).
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not installed. AttributeInferenceBlackBox cannot be used.")

        if not isinstance(target_model_estimator, ARTBaseEstimator): # type: ignore
            raise TypeError("target_model_estimator must be an ART BaseEstimator derivative.")

        self.target_model = target_model_estimator
        self.attack_feature = attack_feature_index # This is the crucial index for ART's attack

        # Logic to create the attack_model for ART, which will predict the `attack_feature`
        current_attack_model_for_art = attack_model_instance
        if current_attack_model_for_art is None and attack_model_type != 'nn':
            if attack_model_type == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                current_attack_model_for_art = RandomForestClassifier(n_estimators=100, random_state=42)
            elif attack_model_type == 'gb':
                from sklearn.ensemble import GradientBoostingClassifier
                current_attack_model_for_art = GradientBoostingClassifier(random_state=42)
            elif attack_model_type == 'lr':
                from sklearn.linear_model import LogisticRegression
                current_attack_model_for_art = LogisticRegression(solver='liblinear', random_state=42)
            else:
                raise ValueError(f"Unsupported attack_model_type for auto-creation: {attack_model_type}")

        # If attack_model_type is 'nn' and current_attack_model_for_art is None, ART creates its default MLP.
        # If current_attack_model_for_art is provided, attack_model_type is passed as None to ART.
        art_attack_model_type_arg = attack_model_type if current_attack_model_for_art is None else None

        self.art_attack_instance = ARTAttributeInferenceBB( # type: ignore
            estimator=self.target_model,
            attack_model=current_attack_model_for_art,
            attack_model_type=art_attack_model_type_arg,
            attack_feature=self.attack_feature,
            non_attack_feature_indices=non_attack_feature_indices,
            **kwargs
        )

    def fit(self,
            x_train_full: np.ndarray, # Full training data including the attribute to infer and other features
            y_train_labels: Optional[np.ndarray] = None, # Labels for x_train_full (for the target_model)
                                                         # Required if attack uses target model's predictions.
            **kwargs_fit) -> None:
        """
        Trains the attribute inference attack model.
        Args:
            x_train_full: The full dataset (numpy array) that was used to train the target_model.
                          This data must contain the attribute to be inferred (at `attack_feature_index`)
                          and other features.
            y_train_labels: Optional. The true labels for `x_train_full` (used by the target_model).
                            These are needed if the attribute inference attack model uses the
                            target_model's predictions as part of its input.
            **kwargs_fit: Additional arguments for ART AttributeInferenceBlackBox's `fit` method.
        """
        if not ART_AVAILABLE: raise ImportError("ART not available for AttributeInferenceBlackBox.fit.")
        if self.art_attack_instance is None: raise RuntimeError("ART Attack instance not initialized.")

        # ART's `fit` method takes `x` (the full data) and `y` (labels for `x` for the target model).
        # The `attack_feature` specified in __init__ tells ART which column in `x` is the attribute to learn to predict.
        self.art_attack_instance.fit(x=x_train_full, y=y_train_labels, **kwargs_fit)


    def infer(self,
              x_query_full: np.ndarray, # Full data records for which to infer the attribute.
                                        # Must contain all features target_model might use,
                                        # and other known attributes if attack model uses them.
              y_query_labels: Optional[np.ndarray] = None, # Labels for x_query_full (for target_model)
                                                           # if attack model uses target_model's predictions.
              **kwargs_infer) -> np.ndarray:
        """
        Performs attribute inference on the query data.
        Args:
            x_query_full: Dataset (numpy array) for which the attribute (specified by `attack_feature_index`)
                          is to be inferred. It should have the same columns as `x_train_full` used in `fit`.
            y_query_labels: Optional. Labels for `x_query_full` (for the target_model). Needed if the
                            attribute inference attack model uses target_model's predictions.
            **kwargs_infer: Additional arguments for ART AttributeInferenceBlackBox's `infer` method.
        Returns:
            An array of predictions for the inferred attribute for each sample in x_query_full.
        """
        if not ART_AVAILABLE: raise ImportError("ART not available for AttributeInferenceBlackBox.infer.")
        if self.art_attack_instance is None:
            raise RuntimeError("Attack model not initialized. Call `fit` first if training is required by the ART attack, "
                               "or ensure __init__ created the attack instance.")

        # ART's `infer` method takes `x` (full query data) and `y` (labels for `x` for the target model).
        return self.art_attack_instance.infer(x=x_query_full, y=y_query_labels, **kwargs_infer)
