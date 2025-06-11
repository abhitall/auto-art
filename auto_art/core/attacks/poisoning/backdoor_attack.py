"""
Wrapper for ART's PoisoningAttackBackdoor.
"""
from typing import Any, Callable, Optional, Tuple, Union, List # Added List
import numpy as np

try:
    from art.attacks.poisoning import PoisoningAttackBackdoor as ARTBackdoorAttack
    # PoisoningAttackBackdoor does not require an estimator for its __init__ or poison method.
    # It's a data transformation.
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ARTBackdoorAttack: pass # Dummy for type hint

class BackdoorAttackWrapper:
    """
    A wrapper for the ART PoisoningAttackBackdoor.
    This attack creates poisoned data that, when used to train a model,
    will result in the model having a backdoor for specific trigger inputs.
    The core ART attack modifies data based on a perturbation function.
    """
    def __init__(self,
                 backdoor_trigger_fn: Callable[[np.ndarray], np.ndarray],
                 target_class_idx: Union[int, List[int]],
                 poisoning_rate: float = 0.1,
                 **kwargs):
        """
        Initializes the BackdoorAttackWrapper.

        Args:
            backdoor_trigger_fn: A function that takes an array x and returns an array with the backdoor trigger.
                                 Example: lambda x: np.clip(x + pattern, 0, 1)
            target_class_idx: The target class index (or list of indices if the trigger should map to multiple targets,
                              though typically one target per backdoor type).
            poisoning_rate: The fraction of the training data to poison (0.0 to 1.0).
            **kwargs: Additional parameters for ART's PoisoningAttackBackdoor if any future versions have them.
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. PoisoningAttackBackdoor cannot be used.")

        if not callable(backdoor_trigger_fn):
            raise TypeError("backdoor_trigger_fn must be a callable function.")

        self.art_attack_instance = ARTBackdoorAttack(
            perturbation=backdoor_trigger_fn,
            **kwargs
        )
        self.poisoning_rate = poisoning_rate
        self.target_class_idx = target_class_idx
        # self.backdoor_trigger_fn = backdoor_trigger_fn # Not needed to store separately, it's in self.art_attack_instance.perturbation

    def generate(self, x_clean: np.ndarray, y_clean: np.ndarray,
                 indices_to_poison: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a dataset with poisoned samples.

        Args:
            x_clean: Clean input samples (numpy array).
            y_clean: Clean labels for the samples (numpy array, expected to be class indices).
            indices_to_poison: Optional. A 1D array of indices indicating which samples in x_clean to poison.
                               If None, indices will be chosen randomly based on poisoning_rate.

        Returns:
            A tuple (x_poisoned_dataset, y_poisoned_dataset) representing the full dataset
            with specified samples poisoned.
        """
        if not ART_AVAILABLE: # Should be caught by __init__ but defense in depth
            raise ImportError("ART not available for BackdoorAttack.generate.")

        if not isinstance(x_clean, np.ndarray) or not isinstance(y_clean, np.ndarray):
            raise TypeError("x_clean and y_clean must be numpy arrays.")
        if x_clean.shape[0] != y_clean.shape[0]:
            raise ValueError("x_clean and y_clean must have the same number of samples.")

        num_samples = x_clean.shape[0]

        if indices_to_poison is None:
            num_poison_float = num_samples * self.poisoning_rate
            num_poison = int(num_poison_float)
            # Ensure at least one sample is poisoned if rate is small but non-zero, and there's data
            if num_poison == 0 and self.poisoning_rate > 0 and num_samples > 0:
                num_poison = 1
            elif num_poison == 0: # No poisoning needed if rate is 0 or no samples
                return x_clean.copy(), y_clean.copy()

            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            actual_indices_to_poison = all_indices[:num_poison]
        else:
            if not isinstance(indices_to_poison, np.ndarray) or indices_to_poison.ndim != 1:
                raise ValueError("indices_to_poison must be a 1D numpy array of indices.")
            actual_indices_to_poison = indices_to_poison
            if np.any(actual_indices_to_poison >= num_samples) or np.any(actual_indices_to_poison < 0):
                raise ValueError("Invalid values in indices_to_poison (out of bounds).")
            num_poison = len(actual_indices_to_poison) # Update num_poison based on provided indices
            if num_poison == 0:
                return x_clean.copy(), y_clean.copy()


        final_target_class: int
        if isinstance(self.target_class_idx, list):
            if not self.target_class_idx: raise ValueError("target_class_idx list cannot be empty.")
            final_target_class = self.target_class_idx[0]
        elif isinstance(self.target_class_idx, (int, np.integer)): # Allow numpy integers too
            final_target_class = int(self.target_class_idx)
        else:
            raise TypeError(f"target_class_idx must be an int or a list of ints, got {type(self.target_class_idx)}.")

        y_target_for_poisoned_samples = np.full(shape=(num_poison,), fill_value=final_target_class, dtype=y_clean.dtype)

        x_poisoned_dataset = x_clean.copy()
        y_poisoned_dataset = y_clean.copy()

        # The perturbation function is stored in the ART attack instance
        # It's applied to the subset of data that needs to be poisoned.
        x_subset_to_poison = x_clean[actual_indices_to_poison]
        x_triggered_subset = self.art_attack_instance.perturbation(x_subset_to_poison)

        x_poisoned_dataset[actual_indices_to_poison] = x_triggered_subset
        y_poisoned_dataset[actual_indices_to_poison] = y_target_for_poisoned_samples

        return x_poisoned_dataset, y_poisoned_dataset
