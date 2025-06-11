"""
Wrapper for ART's FunctionallyEquivalentExtraction (FEE) attack.
"""
from typing import Any, Optional, List, Union, TYPE_CHECKING
import numpy as np

try:
    from art.attacks.extraction import FunctionallyEquivalentExtraction as ARTFEE
    from art.estimators.classification import ClassifierMixin
    # The thief model created by FEE is typically an ART MLPClassifier
    from art.estimators.classification.mlp import MLPClassifier as ARTMLPClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierMixin: pass
    class ARTFEE: pass
    class ARTMLPClassifier: pass # Dummy for type hint

if TYPE_CHECKING:
    # Type hint for the returned thief model, which is usually an MLPClassifier from ART
    ThiefModelType = ARTMLPClassifier
else:
    ThiefModelType = Any


class FunctionallyEquivalentExtractionWrapper:
    """
    A wrapper for the ART FunctionallyEquivalentExtraction (FEE) attack.
    This attack attempts to find a functionally equivalent model to the victim,
    typically by training a Multi-Layer Perceptron (MLP) as the thief model.
    """
    def __init__(self,
                 victim_classifier: ClassifierMixin,
                 num_neurons: Optional[List[int]] = None, # Structure of thief MLP, e.g. [128, 64]
                                                        # If None, ART FEE might use a default.
                 activation: str = 'relu',
                 verbose: bool = True,
                 **kwargs: Any # Other params for ARTFEE init e.g. 'loss_fn' for thief
                 ):
        """
        Initializes the FunctionallyEquivalentExtractionWrapper.
        Args:
            victim_classifier: The ART classifier to be stolen.
            num_neurons: List of integers for hidden layer neurons of the thief MLP.
                         If None, ART's default structure ([128]) might be used.
            activation: Activation function for the thief MLP's hidden layers.
            verbose: Show progress during the attack.
            **kwargs: Additional parameters for ART's FunctionallyEquivalentExtraction.
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. FunctionallyEquivalentExtraction cannot be used.")

        if not isinstance(victim_classifier, ClassifierMixin): # type: ignore
            raise TypeError("victim_classifier must be an ART ClassifierMixin.")

        # Store params for instantiating ARTFEE in the extract method, as it defines the thief structure.
        self.victim_classifier = victim_classifier
        self.num_neurons_thief = num_neurons if num_neurons is not None else [128] # Default if None
        self.activation_thief = activation
        self.verbose = verbose
        self.kwargs_init = kwargs
        self.art_attack_instance: Optional[ARTFEE] = None


    def extract(self,
                x_reference_data: np.ndarray,
                y_reference_data: np.ndarray,
                max_iter_thief: int = 1000, # Iterations to train the thief MLP
                batch_size_thief: int = 32, # Batch size for training thief
                **kwargs_extract) -> ThiefModelType:
        """
        Performs the model extraction attack by training an MLP thief model.
        Args:
            x_reference_data: Data used to query the victim and train the extracted model.
            y_reference_data: Labels corresponding to x_reference_data, used for training the thief.
            max_iter_thief: Number of training iterations for the extracted MLP model.
            batch_size_thief: Batch size for training the extracted MLP model.
            **kwargs_extract: Additional arguments for ART FEE's `extract` method.
        Returns:
            The extracted model (an ART MLPClassifier instance defined and trained by FEE).
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for FunctionallyEquivalentExtraction.extract.")

        if not isinstance(x_reference_data, np.ndarray) or not isinstance(y_reference_data, np.ndarray):
            raise TypeError("x_reference_data and y_reference_data must be numpy arrays.")

        # Instantiate ARTFEE here. It internally defines and trains an MLP thief.
        self.art_attack_instance = ARTFEE( # type: ignore
            classifier=self.victim_classifier,
            num_neurons=self.num_neurons_thief,
            activation=self.activation_thief,
            verbose=self.verbose, # Verbosity for the attack process
            **(self.kwargs_init)
        )

        extracted_thief_model = self.art_attack_instance.extract(
            x=x_reference_data,
            y=y_reference_data,
            max_iter=max_iter_thief, # Parameter name in ART FEE.extract is 'max_iter'
            batch_size=batch_size_thief, # Parameter name in ART FEE.extract is 'batch_size'
            **kwargs_extract # Pass through other extract-specific kwargs
        )
        # The returned model is the trained thief model (an MLPClassifier)
        return extracted_thief_model
