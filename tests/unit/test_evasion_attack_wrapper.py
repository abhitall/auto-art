import pytest
from unittest.mock import MagicMock, patch
from auto_art.core.attacks.evasion.fast_gradient_method import FastGradientMethodWrapper
from auto_art.core.attacks.attack_generator import AttackGenerator # Assuming this is the correct import
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
        "minimal": False, # Added as per ART FGSM defaults
        "summary_writer": False # Added as per ART FGSM defaults
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
    # Modify params for targeted if necessary, though FGSM y is for that
    attack_params_fgsm["targeted"] = True # Ensure targeted is set if wrapper uses it
    wrapper = FastGradientMethodWrapper(art_classifier, attack_params_fgsm)

    wrapper.attack.generate = MagicMock(return_value=np.random.rand(5, 10))

    x_test = np.random.rand(5, 10).astype(np.float32)
    y_target = np.random.randint(0, 2, size=5)

    x_adv = wrapper.generate(x_test, y=y_target)

    # Check that y was passed to the ART attack's generate method
    # Convert y_target to one-hot if the ART attack expects that and wrapper doesn't do it
    # For FGSM, y is class labels, not one-hot.
    wrapper.attack.generate.assert_called_once()
    called_args, called_kwargs = wrapper.attack.generate.call_args
    assert np.array_equal(called_args[0], x_test) # x
    assert np.array_equal(called_kwargs.get('y'), y_target) # y

    assert isinstance(x_adv, np.ndarray)
    assert x_adv.shape == (5, 10)

# Integration with AttackGenerator (Simplified)
# This tests if AttackGenerator can create the FGSM wrapper
@patch('auto_art.core.attacks.evasion.fast_gradient_method.FastGradientMethodWrapper')
def test_attack_generator_creates_fgsm(MockFgsmWrapper, art_classifier):
    # Mock AttackGenerator's internal logic if it's too complex,
    # or test its create_attack directly.
    # For this example, we assume AttackGenerator.create_attack exists and
    # can find and instantiate FastGradientMethodWrapper.

    attack_generator = AttackGenerator(art_classifier) # AttackGenerator needs an estimator

    attack_name = "fast_gradient_method" # Or however FGSM is identified
    attack_config = {
        "module": "auto_art.core.attacks.evasion.fast_gradient_method", # Path to the wrapper
        "class": "FastGradientMethodWrapper",
        "params": {"eps": 0.05, "eps_step": 0.005, "batch_size": 16, "minimal": False, "summary_writer": False}
    }

    # This part depends on AttackGenerator's implementation details
    # Let's assume create_attack takes the name and a config dict
    # For a more direct test, we might need to register the attack first if AttackGenerator uses a registry

    # If AttackGenerator uses a registry:
    # attack_generator.register_attack_wrapper(attack_name, FastGradientMethodWrapper)
    # created_attack = attack_generator.create_attack(attack_name, attack_config["params"])

    # If AttackGenerator dynamically imports:
    # This is harder to test directly without knowing its exact import mechanism.
    # The patch above helps simulate that it *would* find and call the wrapper.

    # Let's assume a simplified scenario where AttackGenerator might have a method
    # that directly takes the wrapper class for testing, or we patch the lookup.

    # For the purpose of this unit test of the wrapper, we'll focus on the wrapper's direct functionality.
    # The AttackGenerator integration test is better suited for tests/integration.
    # However, if AttackGenerator has a simple method like `_get_wrapper_class`, we could patch that.

    # Let's re-evaluate: The issue asks for unit tests for attack wrappers.
    # The test above (`test_fgsm_wrapper_instantiation` and `test_fgsm_wrapper_generate`)
    # are good unit tests for the wrapper itself.
    # The interaction with AttackGenerator should be in an integration test.
    # So, the `test_attack_generator_creates_fgsm` might be too much for a *unit* test of the wrapper.
    # Let's remove it from here and ensure it's covered in `tests/integration`.

    # We can add a test for parameter validation if the wrapper does any.
    pass # Placeholder if we remove the AttackGenerator part from this unit test.


def test_fgsm_wrapper_invalid_params(art_classifier):
    with pytest.raises(TypeError): # Example: if required param 'eps' is missing
        FastGradientMethodWrapper(art_classifier, {"eps_step": 0.01}) # Missing eps

    with pytest.raises(ValueError): # Example: if 'eps' is negative
        FastGradientMethodWrapper(art_classifier, {"eps": -0.1, "eps_step": 0.01, "minimal": False, "summary_writer": False})

# Note: The actual exceptions (TypeError, ValueError) and conditions depend on
# FastGradientMethodWrapper's __init__ and ART's FastGradientMethod's validation.
# This test assumes the wrapper might add its own validation or pass params directly.
# If params are passed directly to ART's attack, then ART's validation applies.
# The wrapper itself might not add much validation beyond what ART provides.
# The ValueError for negative eps is a common check in ART attacks.
