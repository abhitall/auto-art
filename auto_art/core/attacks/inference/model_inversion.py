"""
Wrapper for ART's MIFace (Model Inversion) attack.
"""
from typing import Any, Optional, Dict, Union, List # Added Union and List
import numpy as np

try:
    from art.attacks.inference.model_inversion import MIFace as ARTMIFace
    from art.estimators.classification import ClassifierMixin
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierMixin: pass
    class ARTMIFace: pass


class MIFaceWrapper:
    """
    A wrapper for ART's MIFace (Model Inversion Face) attack.
    This attack reconstructs features (e.g., average faces) for classes of a target model.
    """
    def __init__(self,
                 target_classifier: ClassifierMixin,
                 max_iter: int = 10000,
                 batch_size: int = 1, # MIFace typically processes one class at a time for reconstruction
                 learning_rate: float = 0.1,
                 lambda_tv: float = 0.1,
                 lambda_l2: float = 0.001,
                 verbose: bool = True,
                 **kwargs: Any
                 ):
        """
        Initializes the MIFaceWrapper.
        Args:
            target_classifier: The ART classifier from which features will be reconstructed.
            max_iter: Maximum number of iterations for the optimization per class.
            batch_size: Batch size. For MIFace, this often relates to how many class representations
                        are generated in parallel if y_target has multiple classes, but the optimization
                        for each class's representation is often individual. ART default is 1.
            learning_rate: Learning rate for the optimization.
            lambda_tv: Coefficient for the total variation regularization term.
            lambda_l2: Coefficient for the L2 regularization term.
            verbose: Show progress.
            **kwargs: Additional parameters for ART's MIFace (e.g., 'optimizer', 'scheduler').
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. MIFace cannot be used.")

        if not isinstance(target_classifier, ClassifierMixin): # type: ignore
            raise TypeError("target_classifier must be an ART ClassifierMixin.")

        self.art_attack_instance = ARTMIFace( # type: ignore
            classifier=target_classifier,
            max_iter=max_iter,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lambda_tv=lambda_tv,
            lambda_l2=lambda_l2,
            verbose=verbose,
            **kwargs
        )

    def infer(self,
              y_target_classes: Optional[Union[np.ndarray, List[int]]] = None,
              **kwargs_infer) -> np.ndarray:
        """
        Performs model inversion to reconstruct class representations.

        Args:
            y_target_classes: Optional. A numpy array or list of target class indices
                              (e.g., [0, 1, 2] or np.array([0,1,2])) for which to reconstruct features.
                              If None, ART's MIFace attempts to reconstruct features for all classes
                              known to the target_classifier (0 to nb_classes-1).
            **kwargs_infer: Additional arguments for ART MIFace's `infer` method (e.g., `init`, `mask`).

        Returns:
            An array of reconstructed features/images, typically of shape
            (num_target_classes, C, H, W) or (num_target_classes, features).
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for MIFace.infer.")

        if self.art_attack_instance is None:
            raise RuntimeError("ART MIFace instance not initialized (should not happen).")

        y_to_pass_to_art = None
        if y_target_classes is not None:
            if isinstance(y_target_classes, list):
                y_to_pass_to_art = np.array(y_target_classes, dtype=int)
            elif isinstance(y_target_classes, np.ndarray):
                y_to_pass_to_art = y_target_classes.astype(int)
            else:
                raise TypeError("y_target_classes must be a list or numpy array of integers.")

        # ART's MIFace `infer` method takes `x=None` (as it generates the data) and `y` (target labels).
        # If `y` is None, it iterates through all classes of the classifier.
        return self.art_attack_instance.infer(x=None, y=y_to_pass_to_art, **kwargs_infer)
