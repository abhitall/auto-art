import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from auto_art.core.evaluation.metrics.calculator import MetricsCalculator


@pytest.fixture
def metrics_calculator():
    return MetricsCalculator()


@pytest.fixture
def mock_classifier():
    """Mock ART classifier for testing."""
    classifier = MagicMock()
    # Predictions: 2 samples, 3 classes
    classifier.predict.return_value = np.array([
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1]
    ])
    classifier.input_shape = (10,)
    classifier.nb_classes = 3
    return classifier


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    data = np.random.rand(2, 10).astype(np.float32)
    # One-hot encoded labels matching the predictions
    labels = np.array([
        [0, 1, 0],  # class 1
        [1, 0, 0]   # class 0
    ])
    return data, labels


def test_calculate_basic_metrics(metrics_calculator, mock_classifier, sample_data):
    data, labels = sample_data
    metrics = metrics_calculator.calculate_basic_metrics(mock_classifier, data, labels)

    assert 'accuracy' in metrics
    assert 'average_confidence' in metrics
    assert isinstance(metrics['accuracy'], float)
    assert isinstance(metrics['average_confidence'], float)
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['average_confidence'] <= 1.0


def test_calculate_basic_metrics_accuracy(metrics_calculator):
    """Test accuracy is computed correctly from predictions and labels."""
    classifier = MagicMock()
    classifier.predict.return_value = np.array([
        [0.9, 0.1],  # predicts class 0
        [0.3, 0.7],  # predicts class 1
        [0.8, 0.2],  # predicts class 0
        [0.4, 0.6],  # predicts class 1
    ])
    data = np.random.rand(4, 5).astype(np.float32)
    labels = np.array([
        [1, 0],  # true class 0
        [0, 1],  # true class 1
        [1, 0],  # true class 0
        [1, 0],  # true class 0 (prediction wrong)
    ])

    metrics = metrics_calculator.calculate_basic_metrics(classifier, data, labels)
    assert metrics['accuracy'] == pytest.approx(0.75)


def test_calculate_security_score(metrics_calculator):
    base_accuracy = 0.9
    attack_results = {
        'fgsm': {'success_rate': 0.3},
        'pgd': {'success_rate': 0.5}
    }
    robustness_metrics = {
        'empirical_robustness': 0.6
    }

    score = metrics_calculator.calculate_security_score(
        base_accuracy, attack_results, robustness_metrics
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_calculate_security_score_no_attacks(metrics_calculator):
    score = metrics_calculator.calculate_security_score(0.9, {}, {})
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_calculate_security_score_perfect(metrics_calculator):
    score = metrics_calculator.calculate_security_score(
        1.0,
        {'fgsm': {'success_rate': 0.0}},
        {'empirical_robustness': 1.0}
    )
    assert score == pytest.approx(100.0)


def test_calculate_wasserstein_distance(metrics_calculator):
    data1 = np.random.rand(10, 5).astype(np.float32)
    data2 = np.random.rand(10, 5).astype(np.float32)

    distance = metrics_calculator.calculate_wasserstein_distance(data1, data2)
    # May return None if scipy is not available
    if distance is not None:
        assert isinstance(distance, float)
        assert distance >= 0.0


def test_calculate_wasserstein_distance_same_data(metrics_calculator):
    data = np.random.rand(10, 5).astype(np.float32)
    distance = metrics_calculator.calculate_wasserstein_distance(data, data)
    if distance is not None:
        assert distance == pytest.approx(0.0, abs=1e-6)


def test_calculate_wasserstein_distance_empty(metrics_calculator):
    result = metrics_calculator.calculate_wasserstein_distance(
        np.array([]).reshape(0, 5), np.array([]).reshape(0, 5)
    )
    assert result is None


def test_calculate_wasserstein_distance_invalid_input(metrics_calculator):
    result = metrics_calculator.calculate_wasserstein_distance("not_array", "not_array")
    assert result is None


def test_metrics_calculator_init():
    calc = MetricsCalculator(cache_size=64)
    assert calc.cache_size == 64
