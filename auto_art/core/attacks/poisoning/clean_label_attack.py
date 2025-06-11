"""
Wrapper for ART's CleanLabelAttack (Clean Label Backdoor).
"""
from typing import Any, Callable, Optional, Tuple, Union, List
import numpy as np
import sys # For potential warning print

try:
    from art.attacks.poisoning import CleanLabelAttack as ARTCleanLabelAttack
    from art.estimators.classification import ClassifierMixin
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierMixin: pass
    class ARTCleanLabelAttack: pass # Dummy for type hint

class CleanLabelAttackWrapper:
    """
    A wrapper for the ART CleanLabelAttack.
    This attack generates poisoned data that appears correctly labeled but
    causes a model trained on it to misclassify specific inputs with a trigger.
    """
    def __init__(self,
                 target_classifier: ClassifierMixin,
                 backdoor_trigger_fn: Callable[[np.ndarray], np.ndarray],
                 target_class_idx: Union[int, List[int]],
                 poisoning_rate: float = 0.1,
                 max_iter_attack: int = 100,
                 max_iter_perturb: int = 10,
                 perturb_eps: float = 0.3,
                 learning_rate_attack: float = 0.1,
                 learning_rate_perturb: float = 0.01,
                 batch_size: int = 32,
                 verbose: bool = True,
                 **kwargs):
        """
        Initializes the CleanLabelAttackWrapper.

        Args:
            target_classifier: The ART classifier to be targeted by the backdoor.
            backdoor_trigger_fn: Function to apply the backdoor trigger to an input.
            target_class_idx: Target class index (or list of indices) for the backdoor. ART's CleanLabelAttack takes a single target class.
            poisoning_rate: Fraction of data to poison.
            max_iter_attack: Max iterations for the overall attack optimization.
            max_iter_perturb: Max iterations for refining each poison perturbation.
            perturb_eps: Max L-inf norm for the perturbation added to make a sample poisonous.
            learning_rate_attack: Learning rate for the attack optimization.
            learning_rate_perturb: Learning rate for perturbation refinement.
            batch_size: Batch size for processing.
            verbose: Show progress.
            **kwargs: Additional parameters for ART's CleanLabelAttack.
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. CleanLabelAttack cannot be used.")

        if not isinstance(target_classifier, ClassifierMixin): # type: ignore
            raise TypeError("target_classifier must be an ART ClassifierMixin.")
        if not callable(backdoor_trigger_fn):
            raise TypeError("backdoor_trigger_fn must be a callable function.")

        # ART CleanLabelAttack expects a single integer for target class
        actual_target_idx: int
        if isinstance(target_class_idx, list):
            if not target_class_idx: raise ValueError("target_class_idx list cannot be empty if provided as list.")
            actual_target_idx = target_class_idx[0]
            if len(target_class_idx) > 1:
                # Using sys.stderr for warnings if appropriate in this environment
                print(f"Warning: CleanLabelAttack typically uses a single target class. Using first from list: {actual_target_idx}", file=sys.stderr)
        elif isinstance(target_class_idx, (int, np.integer)):
            actual_target_idx = int(target_class_idx)
        else:
            raise TypeError(f"target_class_idx must be an int or list of ints, got {type(target_class_idx)}")


        self.art_attack_instance = ARTCleanLabelAttack( # type: ignore
            classifier=target_classifier,
            target=actual_target_idx,
            perturbation=backdoor_trigger_fn,
            eps=perturb_eps,
            max_iter_attack=max_iter_attack,
            max_iter_perturb=max_iter_perturb,
            learning_rate_attack=learning_rate_attack,
            learning_rate_perturb=learning_rate_perturb,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs
        )
        self.poisoning_rate = poisoning_rate

    def generate(self, x_clean: np.ndarray, y_clean: np.ndarray,
                 indices_to_poison: Optional[np.ndarray] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a dataset with clean-label poisoned samples.
        Args:
            x_clean: Clean input samples (numpy array).
            y_clean: Clean labels (numpy array, class indices).
            indices_to_poison: Optional. Indices of samples in x_clean to poison.
                               If None, chosen randomly based on poisoning_rate.
            **kwargs: Additional arguments for ART CleanLabelAttack's `poison` method.
        Returns:
            A tuple (x_poisoned_dataset, y_poisoned_dataset). Labels y remain unchanged.
        """
        if not ART_AVAILABLE: # Should be caught by __init__
            raise ImportError("ART not available for CleanLabelAttack.generate.")

        if not isinstance(x_clean, np.ndarray) or not isinstance(y_clean, np.ndarray):
            raise TypeError("x_clean and y_clean must be numpy arrays.")
        if x_clean.shape[0] != y_clean.shape[0]:
            raise ValueError("x_clean and y_clean must have the same number of samples.")

        num_samples = x_clean.shape[0]

        if indices_to_poison is None:
            num_poison_float = num_samples * self.poisoning_rate
            num_poison = int(num_poison_float)
            if num_poison == 0:
                if self.poisoning_rate > 0 and num_samples > 0: num_poison = 1
                else: return x_clean.copy(), y_clean.copy()

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
            if num_poison == 0: return x_clean.copy(), y_clean.copy()

        x_subset_to_poison = x_clean[actual_indices_to_poison]
        y_subset_labels = y_clean[actual_indices_to_poison]

        # ART's CleanLabelAttack.poison returns only the modified x samples (the poisons).
        # Labels are implicitly the same as y_subset_labels.
        poisoned_x_subset = self.art_attack_instance.poison(
            x=x_subset_to_poison,
            y=y_subset_labels,
            **kwargs
        )

        x_poisoned_dataset = x_clean.copy()
        y_poisoned_dataset = y_clean.copy() # Labels do not change

        x_poisoned_dataset[actual_indices_to_poison] = poisoned_x_subset

        return x_poisoned_dataset, y_poisoned_dataset
