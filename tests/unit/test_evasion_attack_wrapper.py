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
    _, called_kwargs = wrapper.attack.generate.call_args
    assert np.array_equal(called_kwargs.get('x'), x_test)
    assert np.array_equal(called_kwargs.get('y'), y_target)

    assert isinstance(x_adv, np.ndarray)
    assert x_adv.shape == (5, 10)

def test_fgsm_wrapper_invalid_params(art_classifier):
    # ART's FGSM has a default for eps, so missing eps does not raise TypeError.
    # However, negative eps raises ValueError.
    with pytest.raises(ValueError):
        FastGradientMethodWrapper(art_classifier, {"eps": -0.1, "eps_step": 0.01, "minimal": False, "summary_writer": False})

    # Non-classifier estimator should raise TypeError from our wrapper
    with pytest.raises(TypeError):
        FastGradientMethodWrapper("not_a_classifier", {"eps": 0.1})
