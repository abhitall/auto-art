import pytest
import numpy as np
import torch
import torch.nn as nn

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod as ArtFastGradientMethod

from auto_art.core.attacks.attack_generator import AttackGenerator
from auto_art.core.interfaces import AttackConfig
from auto_art.core.base import ModelMetadata # For creating metadata

# --- PyTorch Model for Estimator ---
class SimpleTorchModel(nn.Module): # Renamed for clarity
    def __init__(self, input_features=20, num_classes=5):
        super().__init__()
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.linear(x) # Raw logits

@pytest.fixture
def raw_pytorch_model(): # Fixture for the raw model
    return SimpleTorchModel(input_features=20, num_classes=5)

@pytest.fixture
def model_metadata_for_simple_torch(): # Fixture for metadata
    return ModelMetadata(
        model_type='classification',
        framework='pytorch',
        input_shape=(20,), # No batch dimension
        output_shape=(5,),
        input_type='tabular',
        output_type='logits',
        layer_info=[] # Simplified for this test
    )

@pytest.fixture
def attack_generator(): # Corrected instantiation
    return AttackGenerator()

@pytest.fixture
def sample_test_data():
    x_test = np.random.rand(10, 20).astype(np.float32)
    y_test_indices = np.random.randint(0, 5, 10)
    # For ART attacks, y is often class indices for classification
    return x_test, y_test_indices

def test_attack_generator_with_fgsm(
    attack_generator: AttackGenerator,
    raw_pytorch_model: SimpleTorchModel,
    model_metadata_for_simple_torch: ModelMetadata,
    sample_test_data
):
    x_test, y_test = sample_test_data

    # Create AttackConfig for FGSM
    fgsm_attack_config = AttackConfig(
        attack_type="fgsm", # Ensure this matches a key in AttackGenerator.supported_attacks
        epsilon=0.1,
        batch_size=5,
        additional_params={ # ART FGSM specific params not in base AttackConfig
            "eps_step": 0.02, # FGSM doesn't use eps_step, but PGD does. ART FGSM ignores it.
            "minimal": False,
            "summary_writer": False
        }
    )

    # Create the attack using AttackGenerator
    # AttackGenerator's _create_classification_attack will create an ART estimator internally
    attack_instance = attack_generator.create_attack(
        model=raw_pytorch_model,
        metadata=model_metadata_for_simple_torch,
        config=fgsm_attack_config
    )

    assert isinstance(attack_instance, ArtFastGradientMethod)
    assert attack_instance.eps == fgsm_attack_config.epsilon
    assert attack_instance.batch_size == fgsm_attack_config.batch_size
    # The estimator is created inside AttackGenerator, so we can't directly compare with an external one here.
    # We can check its type if needed, but that's testing ClassifierFactory through AttackGenerator.
    assert isinstance(attack_instance.estimator, PyTorchClassifier)
    assert attack_instance.estimator.input_shape == model_metadata_for_simple_torch.input_shape
    assert attack_instance.estimator.nb_classes == model_metadata_for_simple_torch.output_shape[0]


    # Generate adversarial examples using the ART attack instance
    # Note: The 'y' for ART's generate can be true labels for untargeted attacks
    # to ensure misclassification, or target labels for targeted attacks.
    x_adv = attack_instance.generate(x=x_test, y=y_test)

    assert x_adv is not None
    assert x_adv.shape == x_test.shape
    assert not np.allclose(x_adv, x_test) # Check values changed

    # Check if values are within clip_values (default is (0,1) for PyTorchClassifier if not specified)
    # The internal PyTorchClassifier in AttackGenerator is created with default clip_values (0,1)
    # if not otherwise specified via additional_params for ClassifierFactory.
    # For this test, we assume default clip_values.
    assert np.all(x_adv >= 0.0 - 1e-6)
    assert np.all(x_adv <= 1.0 + 1e-6)


def test_attack_generator_unknown_attack(attack_generator, raw_pytorch_model, model_metadata_for_simple_torch):
    unknown_attack_config = AttackConfig(attack_type="non_existent_super_attack")

    with pytest.raises(ValueError, match="Unsupported attack type: non_existent_super_attack or category not determined."):
        attack_generator.create_attack(
            model=raw_pytorch_model,
            metadata=model_metadata_for_simple_torch,
            config=unknown_attack_config
        )

def test_attack_generator_estimator_mismatch():
    """Test for scenarios where estimator features might mismatch attack requirements."""
    pytest.skip("Skipping estimator mismatch test: complex to set up generically.")

# TODO: Add integration tests for other attack categories (poisoning, extraction, inference, llm)
#       showing how AttackGenerator creates their respective wrappers or ART attacks.
#       This will involve mocking the execution methods of those wrappers/attacks.
#
# Example structure for a wrapper-based attack (e.g., a hypothetical 'CustomWrapperAttack'):
# @patch('auto_art.core.attacks.some_module.ActualWrapperClass') # Patch the actual wrapper
# def test_attack_generator_with_custom_wrapper_attack(
#     MockActualWrapper, attack_generator, art_estimator_for_wrapper, model_metadata_for_wrapper
# ):
#     mock_wrapper_instance = MockActualWrapper.return_value # This is what create_attack should return
#
#     attack_config = AttackConfig(
#         attack_type="custom_wrapper_attack_name", # Registered in AttackGenerator
#         additional_params={"wrapper_param_1": True}
#     )
#     # Assume 'art_estimator_for_wrapper' and 'model_metadata_for_wrapper' are appropriate fixtures
#     created_instance = attack_generator.create_attack(
#         model=art_estimator_for_wrapper, # Or raw model depending on wrapper type
#         metadata=model_metadata_for_wrapper,
#         config=attack_config
#     )
#     MockActualWrapper.assert_called_once_with(
#         estimator=art_estimator_for_wrapper, # Or other model param
#         wrapper_param_1=True
#     )
#     assert created_instance is mock_wrapper_instance
#
#     # If this wrapper has its own 'execute' or 'generate' method:
#     # mock_wrapper_instance.execute.return_value = ...
#     # result = created_instance.execute(...)
#     # mock_wrapper_instance.execute.assert_called_once()
#     # assert result == ...
