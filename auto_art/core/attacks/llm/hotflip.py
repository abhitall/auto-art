"""
Wrapper for ART's HotFlip attack for text.
"""
from typing import Any, Optional, Dict
import numpy as np

try:
    from art.attacks.evasion import HotFlip as ARTHotFlip
    from art.estimators.classification import ClassifierMixin
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierMixin: pass
    class ARTHotFlip: pass


class HotFlipWrapper:
    """
    A wrapper for the ART HotFlip attack, designed for text inputs.
    """
    def __init__(self,
                 text_classifier_estimator: ClassifierMixin,
                 max_iter: int = 100,
                 batch_size: int = 32,
                 verbose: bool = True,
                 **kwargs: Any # e.g., 'vocab_size', 'max_length' if not inferred from estimator
                 ):
        """
        Initializes the HotFlipWrapper.
        Args:
            text_classifier_estimator: An ART classifier that processes text data (sequences of token IDs)
                                       and can provide gradients w.r.t. input token embeddings.
            max_iter: Maximum number of iterations for the attack (number of words to flip).
            batch_size: Batch size for processing.
            verbose: Show progress.
            **kwargs: Additional parameters for ART's HotFlip constructor.
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. HotFlip cannot be used.")

        if not isinstance(text_classifier_estimator, ClassifierMixin): # type: ignore
            raise TypeError("text_classifier_estimator must be an ART ClassifierMixin suitable for text data.")

        self.art_attack_instance = ARTHotFlip( # type: ignore
            classifier=text_classifier_estimator, # ART HotFlip uses 'classifier'
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs
        )

    def generate(self,
              x_text_token_ids: np.ndarray,
              y_target_labels: Optional[np.ndarray] = None,
              **kwargs_generate) -> np.ndarray:
        """
        Generates adversarial text examples using HotFlip.
        Args:
            x_text_token_ids: Input text samples as sequences of numerical token IDs (numpy array).
                              Shape typically (batch_size, sequence_length).
            y_target_labels: Optional. Target labels for targeted attacks. If None, attack is untargeted.
            **kwargs_generate: Additional arguments for ART HotFlip's `generate` method.
        Returns:
            Adversarial text examples (numpy array of token IDs with some tokens flipped).
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for HotFlip.generate.")

        if self.art_attack_instance is None:
            raise RuntimeError("ART HotFlip instance not initialized.")

        return self.art_attack_instance.generate(x=x_text_token_ids, y=y_target_labels, **kwargs_generate)
