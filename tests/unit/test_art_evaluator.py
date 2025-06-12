import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch # For dummy model

from auto_art.core.evaluation.art_evaluator import ARTEvaluator, EvasionAttackStrategy, EvaluationMetrics
from auto_art.core.evaluation.config.evaluation_config import EvaluationConfig, ModelType, Framework, EvaluationResult
from auto_art.core.base import ModelMetadata
from auto_art.core.testing.data_generator import TestData

# Minimal PyTorch model for testing
class DummyTorchModel(torch.nn.Module):
    def __init__(self, in_features=10, num_classes=2):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_classes)
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)

@pytest.fixture
def mock_model_obj():
    return DummyTorchModel(in_features=784, num_classes=2)

@pytest.fixture
def mock_evaluation_config():
    return EvaluationConfig(
        model_type=ModelType.CLASSIFICATION,
        framework=Framework.PYTORCH,
        input_shape=(784,), # Example: flattened 28x28 images, no batch
        nb_classes=2,
        attack_params={"epsilon": 0.03} # For evaluate_robustness_from_path tests
    )

@pytest.fixture
def mock_art_estimator():
    estimator = MagicMock()
    estimator.predict.return_value = np.array([[0.2, 0.8], [0.7, 0.3]]) # Example predictions
    estimator.input_shape = (784,)
    estimator.nb_classes = 2
    return estimator

@pytest.fixture
def mock_test_data_np():
    inputs = np.random.rand(2, 784).astype(np.float32)
    labels = np.array([1, 0]).astype(np.int64) # Class indices
    return inputs, labels

@pytest.fixture
def mock_evasion_attack_strategy():
    strategy = MagicMock(spec=EvasionAttackStrategy)
    strategy.attack_name = "MockedEvasionAttack"
    # execute returns: attack_name, adversarial_examples, success_rate
    adv_examples = np.random.rand(2, 784).astype(np.float32)
    strategy.execute.return_value = (strategy.attack_name, adv_examples, 0.75)
    return strategy

@pytest.fixture
def mock_defence_strategy():
    defence = MagicMock(spec=DefenceStrategy)
    defence.name = "MockedDefence"
    defence.get_params.return_value = {"param1": "value1"}
    # apply should return a defended estimator
    mock_defended_estimator = MagicMock()
    mock_defended_estimator.predict.return_value = np.array([[0.3, 0.7],[0.6,0.4]])
    defence.apply.return_value = mock_defended_estimator
    return defence

@pytest.fixture
@patch('auto_art.core.evaluation.metrics.calculator.MetricsCalculator')
def art_evaluator_instance(MockMetricsCalculator, mock_model_obj, mock_evaluation_config, mock_art_estimator):
    # Patch the art_estimator property to return our mock_art_estimator
    with patch.object(ARTEvaluator, 'art_estimator', new_callable=pytest.PropertyMock) as mock_art_estimator_prop:
        mock_art_estimator_prop.return_value = mock_art_estimator

        # Mock MetricsCalculator instance used by ARTEvaluator
        mock_mc_instance = MockMetricsCalculator.return_value
        mock_mc_instance.calculate_basic_metrics.return_value = {'accuracy': 0.9}
        mock_mc_instance.calculate_robustness_metrics.return_value = {'empirical_robustness': 0.6}
        mock_mc_instance.calculate_security_score.return_value = 75.0
        mock_mc_instance.calculate_wasserstein_distance.return_value = 0.1

        evaluator = ARTEvaluator(
            model_obj=mock_model_obj,
            config=mock_evaluation_config
        )
        # Replace the ARTEvaluator's _metrics_calculator with our controlled mock instance
        evaluator._metrics_calculator = mock_mc_instance
        return evaluator

# Test for evaluate_model method
def test_evaluate_model(art_evaluator_instance, mock_test_data_np, mock_evasion_attack_strategy, mock_defence_strategy, mock_art_estimator):
    test_data_x, test_data_y = mock_test_data_np

    # Call the method under test
    result = art_evaluator_instance.evaluate_model(
        test_data=test_data_x,
        test_labels=test_data_y,
        attacks=[mock_evasion_attack_strategy],
        defences=[mock_defence_strategy]
    )

    assert result.success is True
    assert "metrics" in result.metrics_data
    assert "attacks" in result.metrics_data
    assert "defences" in result.metrics_data

    # Check if attack strategy was executed
    mock_evasion_attack_strategy.execute.assert_called_once_with(mock_art_estimator, test_data_x, test_data_y)
    assert "MockedEvasionAttack" in result.metrics_data["attacks"]
    assert result.metrics_data["attacks"]["MockedEvasionAttack"]["success_rate"] == 0.75

    # Check if defence strategy was applied and evaluated
    mock_defence_strategy.apply.assert_called_once_with(mock_art_estimator)
    assert "MockedDefence" in result.metrics_data["defences"]
    assert "accuracy_after_defence" in result.metrics_data["defences"]["MockedDefence"]

    # Check if metrics calculator methods were called for original and defended model
    # Original model basic metrics
    art_evaluator_instance._metrics_calculator.calculate_basic_metrics.assert_any_call(mock_art_estimator, test_data_x, test_data_y)
    # Defended model basic metrics
    defended_estimator = mock_defence_strategy.apply.return_value
    art_evaluator_instance._metrics_calculator.calculate_basic_metrics.assert_any_call(defended_estimator, test_data_x, test_data_y)

    art_evaluator_instance._metrics_calculator.calculate_robustness_metrics.assert_called_once()
    art_evaluator_instance._metrics_calculator.calculate_security_score.assert_called_once()


# Test for generate_report method
def test_generate_report(art_evaluator_instance):
    mock_eval_result = EvaluationResult(
        success=True,
        metrics_data={
            'metrics': {'accuracy': 0.9, 'security_score': 75.0, 'empirical_robustness': 0.6},
            'attacks': {"MockedEvasionAttack": {'success_rate': 0.75, 'perturbation_size': 0.05}},
            'defences': {"MockedDefence": {'accuracy_after_defence': 0.85, "params": {"p1":1}}}
        },
        execution_time=12.34
    )
    report_str = art_evaluator_instance.generate_report(mock_eval_result)

    assert isinstance(report_str, str)
    assert "Adversarial Robustness Evaluation Report" in report_str
    assert "Execution Time: 12.34 seconds" in report_str
    assert "Overall Security Score: 75.00 / 100.0" in report_str
    assert "MockedEvasionAttack" in report_str
    assert "MockedDefence" in report_str
    assert "Accuracy After Defence: 0.8500" in report_str


# Test for evaluate_robustness_from_path (high-level, mocking collaborators)
@patch('auto_art.implementations.models.factory.ModelFactory')
@patch('auto_art.core.evaluation.art_evaluator.analyze_model_architecture_utility')
@patch('auto_art.core.evaluation.art_evaluator.TestDataGenerator')
@patch('auto_art.core.evaluation.art_evaluator.AttackGenerator')
def test_evaluate_robustness_from_path(
    MockAttackGenerator, MockTestDataGenerator, MockAnalyzeArch, MockModelFactory,
    art_evaluator_instance, mock_evaluation_config # Use the instance for its config
):
    mock_model_loader = MockModelFactory.create_model.return_value
    mock_model_obj_loaded = DummyTorchModel() # A fresh dummy model
    mock_model_loader.load_model.return_value = (mock_model_obj_loaded, mock_evaluation_config.framework.value)

    mock_model_metadata = ModelMetadata(
        model_type=mock_evaluation_config.model_type.value,
        framework=mock_evaluation_config.framework.value,
        input_shape=(784,), output_shape=(2,),
        input_type="tabular", output_type="classification"
    )
    MockAnalyzeArch.return_value = mock_model_metadata

    mock_tdg_instance = MockTestDataGenerator.return_value
    mock_test_data_obj = TestData(inputs=np.random.rand(2, 784), expected_outputs=np.array([0,1]))
    mock_tdg_instance.generate_test_data.return_value = mock_test_data_obj
    mock_tdg_instance.generate_expected_outputs.return_value = mock_test_data_obj.expected_outputs

    mock_attack_gen_instance = MockAttackGenerator.return_value
    mock_attack_gen_instance.supported_attacks = {mock_evaluation_config.model_type.value: ["fgsm"]} # Ensure it has some attacks

    # Mock the _evaluate_single_attack_for_robustness method which is called in loop
    # This avoids needing to mock AttackGenerator.create_attack and apply_attack deeply
    mock_eval_metrics = EvaluationMetrics(0.9,0.2,0.78,0.05,0.1,mock_evaluation_config.model_type.value,"fgsm")
    with patch.object(art_evaluator_instance, '_evaluate_single_attack_for_robustness', return_value=mock_eval_metrics) as mock_eval_single:
        results = art_evaluator_instance.evaluate_robustness_from_path(
            model_path="dummy/path.pt",
            framework=mock_evaluation_config.framework.value,
            num_samples=10
        )

    MockModelFactory.create_model.assert_called_once_with(mock_evaluation_config.framework.value)
    mock_model_loader.load_model.assert_called_once_with("dummy/path.pt")
    MockAnalyzeArch.assert_called_once_with(mock_model_obj_loaded, mock_evaluation_config.framework.value)
    MockTestDataGenerator.assert_called_once() # Check it was instantiated
    mock_tdg_instance.generate_test_data.assert_called_once_with(mock_model_metadata, 10)
    mock_tdg_instance.generate_expected_outputs.assert_called_once_with(mock_model_obj_loaded, mock_test_data_obj)
    MockAttackGenerator.assert_called_once() # Instantiated

    assert "model_metadata" in results
    assert "attack_results" in results
    assert "fgsm" in results["attack_results"]
    assert results["attack_results"]["fgsm"]["clean_accuracy"] == 0.9
    mock_eval_single.assert_called_once() # Ensure the mocked attack evaluation loop ran

# TODO: Add tests for error conditions in evaluate_robustness_from_path (e.g., model load fail)
# TODO: Add tests for observer notifications
# TODO: Test art_estimator property logic more directly
# TODO: Test _calculate_accuracy_for_robustness, _calculate_perturbation_size_for_robustness, etc.
#       if they have complex logic not covered by the end-to-end tests.
#       Currently, _calculate_accuracy_for_robustness has significant branching.
#       _evaluate_single_attack_for_robustness also has important logic.
