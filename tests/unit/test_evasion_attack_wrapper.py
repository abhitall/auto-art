import pytest
from unittest.mock import MagicMock, patch
from auto_art.core.attacks.evasion.fast_gradient_method import FastGradientMethodWrapper
# from auto_art.core.attacks.attack_generator import AttackGenerator # Not directly used in these unit tests now
from art.attacks.evasion import FastGradientMethod as ArtFastGradientMethod
from art.estimators.classification import PyTorchClassifier # Example ART estimator
import numpy as np
import torch

# Minimal PyTorch model and ART classifier for testing
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def art_classifier():
    model = DummyModel()
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(10,),
        nb_classes=2,
        optimizer=torch.optim.Adam(model.parameters())
    )
    return classifier

@pytest.fixture
def attack_params_fgsm():
    return {
        "eps": 0.1,
        "eps_step": 0.01,
        "batch_size": 32,
        "minimal": False,
        "summary_writer": False
    }

# Test basic instantiation of the wrapper
def test_fgsm_wrapper_instantiation(art_classifier, attack_params_fgsm):
    wrapper = FastGradientMethodWrapper(art_classifier, attack_params_fgsm)
    assert wrapper.attack_params == attack_params_fgsm
    assert isinstance(wrapper.attack, ArtFastGradientMethod)
    assert wrapper.attack.estimator == art_classifier
    assert wrapper.attack.eps == attack_params_fgsm["eps"]

# Test the generate method
def test_fgsm_wrapper_generate(art_classifier, attack_params_fgsm):
    wrapper = FastGradientMethodWrapper(art_classifier, attack_params_fgsm)

    # Mock the underlying ART attack's generate method
    wrapper.attack.generate = MagicMock(return_value=np.random.rand(5, 10))

    x_test = np.random.rand(5, 10).astype(np.float32)
    x_adv = wrapper.generate(x_test)

    wrapper.attack.generate.assert_called_once_with(x=x_test, y=None) # y=None is default
    assert isinstance(x_adv, np.ndarray)
    assert x_adv.shape == (5, 10)

# Test with y (targeted attack)
def test_fgsm_wrapper_generate_with_y(art_classifier, attack_params_fgsm):
    # Modify params for targeted if necessary
    attack_params_fgsm["targeted"] = True # Ensure targeted is set
    wrapper = FastGradientMethodWrapper(art_classifier, attack_params_fgsm)

    wrapper.attack.generate = MagicMock(return_value=np.random.rand(5, 10))

    x_test = np.random.rand(5, 10).astype(np.float32)
    y_target = np.random.randint(0, 2, size=5)

    x_adv = wrapper.generate(x_test, y=y_target)

    wrapper.attack.generate.assert_called_once()
    called_args, called_kwargs = wrapper.attack.generate.call_args
    assert np.array_equal(called_args[0], x_test) # x
    assert np.array_equal(called_kwargs.get('y'), y_target) # y

    assert isinstance(x_adv, np.ndarray)
    assert x_adv.shape == (5, 10)

def test_fgsm_wrapper_invalid_params(art_classifier):
    with pytest.raises(TypeError): # Example: if required param 'eps' is missing from ART's perspective
        # Assuming FastGradientMethodWrapper passes params directly and ART's FGSM needs 'eps'
        FastGradientMethodWrapper(art_classifier, {"eps_step": 0.01, "minimal": False, "summary_writer": False})

    with pytest.raises(ValueError): # Example: if 'eps' is negative (ART's FGSM validation)
        FastGradientMethodWrapper(art_classifier, {"eps": -0.1, "eps_step": 0.01, "minimal": False, "summary_writer": False})

# Note: The actual exceptions (TypeError, ValueError) and conditions depend on
# ART's FastGradientMethod's validation, as the wrapper passes params through.
# The wrapper itself might not add much validation beyond what ART provides.
# The ValueError for negative eps is a common check in ART attacks.
# The TypeError for missing 'eps' depends on ART's specific __init__ for FastGradientMethod.
# If ART's FGSM has a default for 'eps', TypeError might not be raised.
# For this test, we assume 'eps' is mandatory for ART's FGSM for demonstration.
# If the wrapper has its own validation, this test would target that.
# Based on current FastGradientMethodWrapper, it passes params directly.
