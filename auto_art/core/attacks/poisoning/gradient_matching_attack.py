"""
Wrapper for ART's GradientMatchingAttack.
"""
from typing import Any, Optional, Tuple, List
import numpy as np

try:
    from art.attacks.poisoning import GradientMatchingAttack as ARTGradientMatchingAttack
    from art.estimators.classification import ClassifierNeuralNetwork
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierNeuralNetwork: pass # Dummy for type hint
    class ARTGradientMatchingAttack: pass # Dummy for type hint

class GradientMatchingAttackWrapper:
    """
    A wrapper for the ART GradientMatchingAttack.
    This attack generates poisoning points by matching the gradients of the poisoned batch
    to the gradients of a target batch.
    """
    def __init__(self,
                 target_classifier: ClassifierNeuralNetwork, # ART attack needs a classifier
                 learning_rate: float = 0.1,
                 max_iter: int = 100,
                 lambda_hyper: float = 0.1, # Hyperparameter lambda in ART is lambda_val
                 batch_size: int = 32,
                 verbose: bool = True,
                 poisoning_rate: float = 0.1, # Default rate if num_poisons_to_generate is None
                 **kwargs):
        """
        Initializes the GradientMatchingAttackWrapper.
        Args:
            target_classifier: The ART neural network classifier to target.
            learning_rate: Learning rate for the optimization.
            max_iter: Maximum number of iterations for optimizing poison samples.
            lambda_hyper: The hyperparameter lambda for balancing gradient matching and poison usability.
            batch_size: Batch size for poison generation.
            verbose: Show progress.
            poisoning_rate: Default poisoning rate for the generate method.
            **kwargs: Additional parameters for ART's GradientMatchingAttack (e.g., 'optimizer').
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. GradientMatchingAttack cannot be used.")

        if not isinstance(target_classifier, ClassifierNeuralNetwork): # type: ignore
            raise TypeError("target_classifier must be an ART ClassifierNeuralNetwork.")

        self.art_attack_instance = ARTGradientMatchingAttack( # type: ignore
            classifier=target_classifier,
            learning_rate=learning_rate,
            max_iter=max_iter,
            lambda_val=lambda_hyper, # ART's param name
            batch_size=batch_size,
            verbose=verbose,
            **kwargs
        )
        self.poisoning_rate = poisoning_rate
        # self.target_classifier = target_classifier # Stored in self.art_attack_instance.classifier

    def generate(self,
                 x_poison_pool: np.ndarray, y_poison_pool: np.ndarray,
                 x_target_batch: np.ndarray, y_target_batch: np.ndarray,
                 num_poisons_to_generate: Optional[int] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a specified number of poison data points from a pool.
        Args:
            x_poison_pool: Clean samples to be modified into poisons.
            y_poison_pool: Corresponding labels for x_poison_pool (these will be the labels of the poisons).
            x_target_batch: A batch of clean target samples whose gradients will be matched.
            y_target_batch: Corresponding labels for x_target_batch.
            num_poisons_to_generate: Number of poison samples to generate from x_poison_pool.
                                     If None, defaults to int(poisoning_rate * x_poison_pool.shape[0]).
        Returns:
            A tuple (x_poisons, y_poisons_labels) containing the generated poison samples and their labels.
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for GradientMatchingAttack.generate.")

        if not all(isinstance(arr, np.ndarray) for arr in [x_poison_pool, y_poison_pool, x_target_batch, y_target_batch]):
            raise TypeError("All data inputs (x_poison_pool, y_poison_pool, x_target_batch, y_target_batch) must be numpy arrays.")

        actual_num_poisons: int
        if num_poisons_to_generate is None:
            num_p_float = self.poisoning_rate * x_poison_pool.shape[0]
            actual_num_poisons = int(num_p_float)
            if actual_num_poisons == 0 and self.poisoning_rate > 0 and x_poison_pool.shape[0] > 0:
                actual_num_poisons = 1 # Ensure at least one if rate > 0 and pool is not empty
        else:
            actual_num_poisons = num_poisons_to_generate

        if actual_num_poisons == 0: # No poisons to generate
            # Return empty arrays with compatible dimension structure if possible
            empty_x_shape = (0,) + x_poison_pool.shape[1:] if len(x_poison_pool.shape) > 1 else (0,)
            empty_y_shape = (0,) + y_poison_pool.shape[1:] if len(y_poison_pool.shape) > 1 else (0,)
            return np.empty(empty_x_shape, dtype=x_poison_pool.dtype), np.empty(empty_y_shape, dtype=y_poison_pool.dtype)


        if actual_num_poisons > x_poison_pool.shape[0]:
            raise ValueError(f"num_poisons_to_generate ({actual_num_poisons}) cannot exceed the size of x_poison_pool ({x_poison_pool.shape[0]}).")

        # Select a subset from the pool to turn into poisons
        # Ensure indices are unique and within bounds
        if x_poison_pool.shape[0] > 0 :
            indices_to_select = np.random.choice(x_poison_pool.shape[0], actual_num_poisons, replace=False)
            x_to_perturb_source = x_poison_pool[indices_to_select]
            y_poisons_labels_source = y_poison_pool[indices_to_select]
        else: # x_poison_pool is empty but actual_num_poisons > 0 (should be caught by previous check, but defensive)
            empty_x_shape = (0,) + x_poison_pool.shape[1:] if len(x_poison_pool.shape) > 1 else (0,)
            empty_y_shape = (0,) + y_poison_pool.shape[1:] if len(y_poison_pool.shape) > 1 else (0,)
            return np.empty(empty_x_shape, dtype=x_poison_pool.dtype), np.empty(empty_y_shape, dtype=y_poison_pool.dtype)


        # ART's GradientMatchingAttack.poison expects:
        # x_source, y_source (the batch to be perturbed into poisons)
        # x_target, y_target (the batch whose gradients are matched)
        # It returns (poisoned_x_source, y_source_labels_of_poisons)

        # The attack's internal batch_size parameter in __init__ should handle batching of x_to_perturb_source.
        # x_target_batch and y_target_batch are expected to be single batches by ART's poison method.
        x_generated_poisons, y_final_labels_for_poisons = self.art_attack_instance.poison(
            x_source=x_to_perturb_source,
            y_source=y_poisons_labels_source,
            x_target=x_target_batch,
            y_target=y_target_batch
        )

        return x_generated_poisons, y_final_labels_for_poisons
