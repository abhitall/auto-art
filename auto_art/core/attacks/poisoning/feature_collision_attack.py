"""
Wrapper for ART's FeatureCollisionAttack.
"""
from typing import Any, Callable, Optional, Tuple, Union, List
import numpy as np

try:
    from art.attacks.poisoning import FeatureCollisionAttack as ARTFeatureCollisionAttack
    from art.estimators.classification import ClassifierNeuralNetwork
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierNeuralNetwork: pass
    class ARTFeatureCollisionAttack: pass

class FeatureCollisionAttackWrapper:
    """
    A wrapper for the ART FeatureCollisionAttack.
    This attack generates poisoning samples that cause feature collisions
    at a specified layer of a neural network.
    """
    def __init__(self,
                 target_classifier: ClassifierNeuralNetwork,
                 target_feature_layer: Union[int, str],
                 target_image: np.ndarray,
                 target_label_for_collision: int,
                 max_iter: int = 100,
                 learning_rate: float = 0.1, # ART default is 0.1
                 batch_size: int = 32,    # ART default is 128
                 verbose: bool = True,
                 poisoning_rate: float = 0.1,
                 **kwargs):
        """
        Initializes the FeatureCollisionAttackWrapper.
        Args:
            target_classifier: The ART neural network classifier to target.
            target_feature_layer: The layer index or name where feature collision should occur.
            target_image: The image whose features will be the collision target. Shape should match classifier input (no batch).
            target_label_for_collision: The label that target_classifier should predict for target_image after poisoning.
            max_iter: Max iterations for optimizing a batch of poison samples.
            learning_rate: Learning rate for the optimization.
            batch_size: Batch size for poison generation.
            verbose: Show progress.
            poisoning_rate: Default poisoning rate for the generate method.
            **kwargs: Additional parameters for ART's FeatureCollisionAttack (e.g., 'optimizer', 'decay_coeff').
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. FeatureCollisionAttack cannot be used.")

        if not isinstance(target_classifier, ClassifierNeuralNetwork): # type: ignore
            raise TypeError("target_classifier must be an ART ClassifierNeuralNetwork.")
        if not isinstance(target_image, np.ndarray):
            raise TypeError("target_image must be a numpy array.")

        current_target_image = target_image
        # ART's FeatureCollisionAttack expects x_target_feature to have a batch dimension.
        if len(target_image.shape) == len(target_classifier.input_shape): # type: ignore
            current_target_image = np.expand_dims(target_image, axis=0)
        elif len(target_image.shape) != len(target_classifier.input_shape) + 1: # type: ignore
            raise ValueError(f"target_image shape {target_image.shape} is not compatible with classifier input shape {target_classifier.input_shape}") # type: ignore

        self.art_attack_instance = ARTFeatureCollisionAttack( # type: ignore
            classifier=target_classifier,
            target=target_label_for_collision,
            feature_layer=target_feature_layer,
            x_target_feature=current_target_image,
            max_iter=max_iter,
            learning_rate=learning_rate, # Parameter name in ART is 'learning_rate' in recent versions
            batch_size=batch_size,
            verbose=verbose,
            **kwargs # Pass other params like 'optimizer', 'decay_coeff'
        )
        self.poisoning_rate = poisoning_rate

    def generate(self, x_clean_victim: np.ndarray, y_clean_victim: np.ndarray,
                 indices_to_poison: Optional[np.ndarray] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a dataset with feature-colliding poisoned samples.
        Args:
            x_clean_victim: Clean input samples that will be modified to become poisons.
            y_clean_victim: Clean labels for x_clean_victim (class indices). These labels are maintained.
            indices_to_poison: Optional. Indices of samples in x_clean_victim to select as base for poisons.
                               If None, chosen randomly based on poisoning_rate.
            **kwargs: Additional arguments for ART FeatureCollisionAttack's `poison` method.
        Returns:
            A tuple (x_poisoned_dataset, y_poisoned_dataset). Labels y remain "clean".
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for FeatureCollisionAttack.generate.")

        if not isinstance(x_clean_victim, np.ndarray) or not isinstance(y_clean_victim, np.ndarray):
            raise TypeError("x_clean_victim and y_clean_victim must be numpy arrays.")
        if x_clean_victim.shape[0] != y_clean_victim.shape[0]:
            raise ValueError("x_clean_victim and y_clean_victim must have the same number of samples.")

        num_samples = x_clean_victim.shape[0]

        if indices_to_poison is None:
            num_poison_float = num_samples * self.poisoning_rate
            num_poison = int(num_poison_float)
            if num_poison == 0:
                if self.poisoning_rate > 0 and num_samples > 0: num_poison = 1
                else: return x_clean_victim.copy(), y_clean_victim.copy()

            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            actual_indices_to_poison = all_indices[:num_poison]
        else:
            if not isinstance(indices_to_poison, np.ndarray) or indices_to_poison.ndim != 1:
                raise ValueError("indices_to_poison must be a 1D numpy array of indices.")
            actual_indices_to_poison = indices_to_poison
            if np.any(actual_indices_to_poison >= num_samples) or np.any(actual_indices_to_poison < 0):
                raise ValueError("Invalid values in indices_to_poison (out of bounds).")
            num_poison = len(actual_indices_to_poison)
            if num_poison == 0: return x_clean_victim.copy(), y_clean_victim.copy()

        x_subset_to_poison_base = x_clean_victim[actual_indices_to_poison]
        # y_subset_labels_base = y_clean_victim[actual_indices_to_poison] # Original labels of base images

        # ART FeatureCollisionAttack.poison takes x (base images) and optional y (base labels).
        # If y is None, it might behave differently or use a default strategy.
        # The attack aims to make these x's, when perturbed, cause features similar to x_target_feature,
        # and make the model (re-trained on these poisons) predict `target_label_for_collision` for `x_target_feature`.
        # The labels of the poisons themselves remain their original clean labels.
        poisoned_x_subset = self.art_attack_instance.poison(
            x=x_subset_to_poison_base,
            y=None, # y is optional and can be used to guide the optimization.
                    # For simplicity, relying on target_image and target_label_for_collision from init.
            **kwargs
        )

        x_poisoned_dataset = x_clean_victim.copy()
        y_poisoned_dataset = y_clean_victim.copy()

        x_poisoned_dataset[actual_indices_to_poison] = poisoned_x_subset

        return x_poisoned_dataset, y_poisoned_dataset
