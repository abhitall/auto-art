"""
Wrapper for ART's AutoAttack.
"""
from typing import Any, Dict, Optional, Union
import numpy as np # For np.inf and array checks

# Try to import ART components, but don't fail at module load time
# This allows the module to be imported even if ART is not installed yet,
# though an error will occur when trying to use the class.
try:
    from art.attacks.evasion import AutoAttack as ARTAutoAttack
    from art.estimators.classification import ClassifierMixin
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    # Define dummy types for type hinting if ART is not available
    # This helps maintain code structure and type hints even if ART is missing.
    class ClassifierMixin: pass
    class ARTAutoAttack: pass


class AutoAttackWrapper:
    """
    A wrapper for the Adversarial Robustness Toolbox (ART) AutoAttack.
    """
    def __init__(self,
                 estimator: ClassifierMixin,
                 norm: Union[float, str] = 'inf',
                 eps: float = 0.3,
                 eps_step: float = 0.1, # Note: AutoAttack primarily uses eps, eps_step might be for specific sub-attacks or ignored.
                 batch_size: int = 32,
                 targeted: bool = False, # AutoAttack's targeted mode is complex, often set by providing `y` to generate
                 verbose: bool = True,
                 **kwargs: Any):
        """
        Initializes the AutoAttack wrapper.
        Args:
            estimator: An ART-compatible classifier estimator.
            norm: The norm of the adversarial perturbation. Accepts 'inf', 'l1', 'l2' (strings) or np.inf, 1, 2 (numbers).
            eps: Maximum perturbation.
            eps_step: Step size for attacks like PGD if used by AutoAttack's components.
            batch_size: Size of the batch on which adversarial samples are generated.
            targeted: If the attack aims for specific target classes. AutoAttack determines this largely by 'y' in generate.
            verbose: Show progress bars during attack.
            **kwargs: Additional parameters for ART AutoAttack (e.g., 'version', 'attacks').
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. AutoAttack cannot be used.")

        if not isinstance(estimator, ClassifierMixin):
            raise TypeError(f"Estimator must be an ART ClassifierMixin, got {type(estimator)}")

        art_norm_val: Union[float, int]
        if isinstance(norm, str):
            norm_low = norm.lower()
            if norm_low == 'inf': art_norm_val = np.inf
            elif norm_low == 'l2': art_norm_val = 2
            elif norm_low == 'l1': art_norm_val = 1
            else: raise ValueError(f"Unsupported norm string: {norm}. Use 'inf', 'l2', or 'l1'.")
        elif isinstance(norm, (int, float)):
            art_norm_val = norm
        else:
            raise TypeError(f"Norm must be float, int, or str, got {type(norm)}")

        self.attack = ARTAutoAttack(
            estimator=estimator,
            norm=art_norm_val, # Pass the processed norm value
            eps=eps,
            eps_step=eps_step,
            batch_size=batch_size,
            targeted=targeted, # This flag in ART's AutoAttack __init__ can be subtle.
                               # Effective targeting often depends on `y` in `generate`.
            verbose=verbose,
            **kwargs
        )
        self._targeted_at_init = targeted # Store initial targeted flag for reference

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generates adversarial examples using AutoAttack.

        Args:
            x: Input samples (numpy array).
            y: Target labels for targeted attacks (if self.targeted was True at init and y is provided).
               If None, AutoAttack performs untargeted attacks or uses its default target strategy.
            **kwargs: Additional arguments for ART AutoAttack's generate method.

        Returns:
            Adversarial examples (numpy array).
        """
        if not ART_AVAILABLE: # Should have been caught in __init__ but good for safety
            raise ImportError("ART not available for AutoAttack.generate.")

        # AutoAttack's `generate` method uses `y` as the target labels.
        # If `y` is None, it's untargeted.
        # If `y` is provided, it's targeted towards those `y`.
        # The `targeted` flag in __init__ might influence default behavior of sub-attacks if `y` is None.
        return self.attack.generate(x=x, y=y, **kwargs)
