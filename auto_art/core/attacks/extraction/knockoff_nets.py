"""
Wrapper for ART's KnockoffNets attack.
"""
from typing import Any, Optional, Tuple, List, Union, TYPE_CHECKING
import numpy as np

try:
    from art.attacks.extraction import KnockoffNets as ARTKnockoffNets
    from art.estimators.classification import ClassifierMixin
    from art.utils import to_categorical
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierMixin: pass
    class ARTKnockoffNets: pass
    def to_categorical(labels, nb_classes=None): pass

if TYPE_CHECKING:
    from art.estimators.classification.pytorch import PyTorchClassifier as HintPyTorchClassifier
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier as HintTFClassifier
    from art.estimators.classification.keras import KerasClassifier as HintKerasClassifier
    ThiefEstimatorType = Union[HintPyTorchClassifier, HintTFClassifier, HintKerasClassifier, ClassifierMixin]
else:
    ThiefEstimatorType = Any


class KnockoffNetsWrapper:
    """
    A wrapper for the ART KnockoffNets model extraction attack.
    """
    def __init__(self,
                 victim_classifier: ClassifierMixin,
                 batch_size_query: int = 64,
                 nb_epochs_thief: int = 10,
                 nb_stolen_samples: int = 1000,
                 use_probabilities: bool = True,
                 verbose: bool = True,
                 **kwargs: Any
                 ):
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. KnockoffNets cannot be used.")

        if not isinstance(victim_classifier, ClassifierMixin): # type: ignore
            raise TypeError("victim_classifier must be an ART ClassifierMixin.")

        self.victim_classifier = victim_classifier
        self.batch_size_query = batch_size_query
        self.nb_epochs_thief = nb_epochs_thief
        self.nb_stolen_samples = nb_stolen_samples
        self.use_probabilities = use_probabilities
        self.verbose = verbose
        self.kwargs = kwargs
        self.art_attack_instance: Optional[ARTKnockoffNets] = None

    def extract(self,
                thief_estimator_instance: ThiefEstimatorType,
                x_query_data: np.ndarray,
                thief_optimizer: Optional[Any] = None,
                y_query_reference: Optional[np.ndarray] = None,
                **kwargs) -> ClassifierMixin:
        """
        Performs model extraction by training the thief_estimator.
        Args:
            thief_estimator_instance: An un-trained ART classifier instance for the thief model.
            x_query_data: Data used to query the victim model. Labels are obtained from the victim.
            thief_optimizer: Optimizer for training the thief_estimator.
            y_query_reference: Optional. True labels for x_query_data. Not directly used by ART's
                               KnockoffNets.extract, but provided for API consistency if needed elsewhere.
            **kwargs: Additional arguments for ART KnockoffNets' `extract` method.
        Returns:
            The trained thief_estimator.
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for KnockoffNets.extract.")

        if not isinstance(thief_estimator_instance, ClassifierMixin): # type: ignore
            raise TypeError(f"thief_estimator_instance must be an ART ClassifierMixin, got {type(thief_estimator_instance)}")
        if not isinstance(x_query_data, np.ndarray):
            raise TypeError("x_query_data must be a numpy array.")

        self.art_attack_instance = ARTKnockoffNets( # type: ignore
            classifier_victim=self.victim_classifier,
            classifier_thief=thief_estimator_instance,
            batch_size_query=self.batch_size_query,
            nb_epochs=self.nb_epochs_thief,
            nb_stolen_samples=self.nb_stolen_samples,
            use_probability=self.use_probabilities,
            verbose=self.verbose,
            **(self.kwargs)
        )

        # ART's KnockoffNets `extract` method expects `x` (query data).
        # The `y` argument in `extract` is optional and if provided, it's used as reference labels
        # for the stolen data points, but the primary labeling comes from querying the victim.
        # If `y` is None, ART might generate random labels or handle it internally.
        # For consistency and to ensure ART has what it might need (e.g., for class number inference),
        # we provide a dummy `y` if `y_query_reference` is None.
        current_y_for_art = y_query_reference
        if current_y_for_art is None:
            nb_classes = getattr(self.victim_classifier, 'nb_classes', None)
            if nb_classes is None and hasattr(thief_estimator_instance, 'nb_classes'):
                 nb_classes = thief_estimator_instance.nb_classes
            if nb_classes is None:
                if self.victim_classifier.output_shape and len(self.victim_classifier.output_shape) == 1:
                     nb_classes = self.victim_classifier.output_shape[0]
            if nb_classes is None or nb_classes == 0:
                raise ValueError("Number of classes (nb_classes) could not be determined for KnockoffNets setup.")

            # Use number of samples from x_query_data that will actually be stolen.
            num_y_samples = min(x_query_data.shape[0], self.nb_stolen_samples)
            dummy_y_labels_indices = np.random.randint(0, nb_classes, num_y_samples)
            current_y_for_art = to_categorical(dummy_y_labels_indices, nb_classes=nb_classes) if nb_classes > 1 else dummy_y_labels_indices.reshape(-1,1).astype(np.float32)

        # Ensure x_query_data and current_y_for_art have same number of samples if current_y_for_art was derived
        # from self.nb_stolen_samples. ART's extract will sample from x_query_data up to nb_stolen_samples.
        # The y passed to extract should correspond to the x passed.
        # It's safer to pass the full x_query_data and its corresponding full y_query_reference (or dummy).
        # ART will then internally handle the sampling up to nb_stolen_samples.
        if current_y_for_art.shape[0] != x_query_data.shape[0]:
            # This case occurs if dummy y was created based on nb_stolen_samples, but full x_query_data is passed.
            # Recreate dummy y to match full x_query_data length for ART's internal sampling consistency.
            num_y_samples = x_query_data.shape[0]
            nb_classes = getattr(self.victim_classifier, 'nb_classes', getattr(thief_estimator_instance, 'nb_classes', 2)) # Default to 2 if unknown
            dummy_y_labels_indices = np.random.randint(0, nb_classes, num_y_samples)
            current_y_for_art = to_categorical(dummy_y_labels_indices, nb_classes=nb_classes) if nb_classes > 1 else dummy_y_labels_indices.reshape(-1,1).astype(np.float32)


        extracted_thief_model = self.art_attack_instance.extract(
            x=x_query_data,
            y=current_y_for_art,
            thief_optimizer=thief_optimizer,
            **kwargs
        )
        return extracted_thief_model
