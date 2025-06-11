"""
Wrapper for ART's CopycatCNN attack.
"""
from typing import Any, Optional, Tuple, List, Union, TYPE_CHECKING
import numpy as np

try:
    from art.attacks.extraction import CopycatCNN as ARTCopycatCNN
    from art.estimators.classification import ClassifierMixin
    from art.utils import to_categorical # For dummy_y_train
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierMixin: pass # Dummy
    class ARTCopycatCNN: pass # Dummy
    def to_categorical(labels, nb_classes=None): pass # Dummy

# For type hinting the copycat_estimator, which can be various ART classifier types.
if TYPE_CHECKING:
    from art.estimators.classification.pytorch import PyTorchClassifier as HintPyTorchClassifier
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier as HintTFClassifier # TF v2
    from art.estimators.classification.keras import KerasClassifier as HintKerasClassifier
    # Add other relevant ART classifier types if needed for more precise hinting
    CopycatEstimatorType = Union[HintPyTorchClassifier, HintTFClassifier, HintKerasClassifier, ClassifierMixin]
else:
    CopycatEstimatorType = Any


class CopycatCNNWrapper:
    """
    A wrapper for the ART CopycatCNN model extraction attack.
    """
    def __init__(self,
                 victim_classifier: ClassifierMixin,
                 batch_size_query: int = 128,
                 nb_epochs_copycat: int = 10,
                 nb_stolen_samples: int = 1000,
                 use_probabilities: bool = True,
                 verbose: bool = True,
                 **kwargs: Any # For other ARTCopycatCNN init params e.g. query_batch_size
                 ):
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. CopycatCNN cannot be used.")

        if not isinstance(victim_classifier, ClassifierMixin): # type: ignore
            raise TypeError("victim_classifier must be an ART ClassifierMixin.")

        self.victim_classifier = victim_classifier
        self.batch_size_query = batch_size_query
        self.nb_epochs_copycat = nb_epochs_copycat
        self.nb_stolen_samples = nb_stolen_samples
        self.use_probabilities = use_probabilities
        self.verbose = verbose
        self.kwargs = kwargs
        self.art_attack_instance: Optional[ARTCopycatCNN] = None


    def extract(self,
                copycat_estimator_instance: CopycatEstimatorType,
                thief_optimizer: Optional[Any] = None,
                x_reference_victim_data: Optional[np.ndarray] = None,
                **kwargs) -> ClassifierMixin: # Returns the trained copycat estimator
        """
        Performs model extraction by training the copycat_estimator.
        Args:
            copycat_estimator_instance: An un-trained ART classifier instance for the copycat model.
            thief_optimizer: Optimizer for training the copycat_estimator (e.g., for PyTorch).
            x_reference_victim_data: Optional. Sample data from victim's domain (e.g., one batch).
                                     Used by ART to infer input data characteristics (min/max values, shape)
                                     for generating query samples. If None, random data in [0,1] might be used by ART.
            **kwargs: Additional arguments for ART CopycatCNN's `extract` method.
        Returns:
            The trained copycat_estimator (thief model).
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for CopycatCNN.extract.")

        if not isinstance(copycat_estimator_instance, ClassifierMixin): # type: ignore
            raise TypeError(f"copycat_estimator_instance must be an ART ClassifierMixin instance, got {type(copycat_estimator_instance)}")

        # Instantiate ARTCopycatCNN here
        self.art_attack_instance = ARTCopycatCNN( # type: ignore
            classifier_victim=self.victim_classifier,
            classifier_thief=copycat_estimator_instance,
            batch_size_query=self.batch_size_query,
            nb_epochs=self.nb_epochs_copycat,
            nb_stolen_samples=self.nb_stolen_samples,
            use_probability=self.use_probabilities,
            verbose=self.verbose,
            **(self.kwargs)
        )

        current_x_ref = x_reference_victim_data
        if current_x_ref is None:
            if self.victim_classifier.input_shape is None:
                raise ValueError("x_reference_victim_data must be provided if victim_classifier.input_shape is not available.")

            # Create dummy x_reference_victim_data based on input_shape
            # Use a small batch for reference, e.g., batch_size_query or just a few samples.
            ref_batch_size = min(self.batch_size_query, 32) if self.batch_size_query > 0 else 32
            dummy_shape = (ref_batch_size,) + self.victim_classifier.input_shape
            current_x_ref = np.zeros(dummy_shape, dtype=np.float32)
            if hasattr(self.victim_classifier, 'clip_values') and self.victim_classifier.clip_values is not None:
                min_val, max_val = self.victim_classifier.clip_values
                current_x_ref = np.random.uniform(min_val, max_val, size=dummy_shape).astype(np.float32)

        # Dummy y_train for class count inference by ART, not for actual labels.
        nb_classes = getattr(self.victim_classifier, 'nb_classes', None)
        if nb_classes is None and hasattr(copycat_estimator_instance, 'nb_classes'):
             nb_classes = copycat_estimator_instance.nb_classes
        if nb_classes is None: # Try to infer from victim output shape if possible
            if self.victim_classifier.output_shape and len(self.victim_classifier.output_shape) == 1:
                 nb_classes = self.victim_classifier.output_shape[0]

        if nb_classes is None or nb_classes == 0: # Check for 0 too
            raise ValueError("Number of classes (nb_classes) could not be determined for CopycatCNN setup.")

        # Create dummy_y_train: ART's extract method needs y, but it's not used for labeling stolen data.
        # It's used internally, possibly to infer the number of classes if the thief model isn't fully defined.
        # Shape should match the number of samples in current_x_ref or nb_stolen_samples.
        # Let's use a small number of samples for y, consistent with current_x_ref if possible.
        num_dummy_y_samples = current_x_ref.shape[0]
        dummy_y_labels_indices = np.random.randint(0, nb_classes, num_dummy_y_samples)
        dummy_y_ref = to_categorical(dummy_y_labels_indices, nb_classes=nb_classes) if nb_classes > 1 else dummy_y_labels_indices.reshape(-1,1).astype(np.float32)


        # The 'thief_optimizer' is passed directly to ART's extract method.
        # If it's None, ART's CopycatCNN will attempt to create a default one for PyTorch/Keras.
        extracted_thief_model = self.art_attack_instance.extract(
            x=current_x_ref,
            y=dummy_y_ref,
            thief_optimizer=thief_optimizer,
            **kwargs
        )
        return extracted_thief_model
