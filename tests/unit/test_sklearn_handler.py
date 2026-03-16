import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from auto_art.implementations.models.sklearn import SklearnModel
from auto_art.core.base import ModelMetadata


@pytest.fixture
def simple_sklearn_model_and_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def simple_sklearn_model(simple_sklearn_model_and_data):
    return simple_sklearn_model_and_data[0]


@pytest.fixture
def sklearn_handler():
    return SklearnModel()


def test_handler_creation(sklearn_handler):
    assert sklearn_handler is not None


def test_analyze_architecture(sklearn_handler, simple_sklearn_model):
    metadata = sklearn_handler.analyze_architecture(simple_sklearn_model, 'sklearn')
    assert isinstance(metadata, ModelMetadata)
    assert metadata.framework == 'sklearn'
    assert isinstance(metadata.layer_info, list)


def test_analyze_architecture_shapes(sklearn_handler, simple_sklearn_model):
    metadata = sklearn_handler.analyze_architecture(simple_sklearn_model, 'sklearn')
    assert metadata.input_shape is not None
    assert metadata.output_shape is not None


def test_validate_model(sklearn_handler, simple_sklearn_model):
    assert sklearn_handler.validate_model(simple_sklearn_model) is True


def test_validate_invalid_model(sklearn_handler):
    assert sklearn_handler.validate_model({}) is False


def test_get_model_predictions(sklearn_handler, simple_sklearn_model_and_data):
    model, X, _ = simple_sklearn_model_and_data
    X_sample = X[:5]
    predictions = sklearn_handler.get_model_predictions(model, X_sample)
    assert predictions is not None
    assert len(predictions) == 5


def test_preprocess_input(sklearn_handler):
    data = np.random.rand(5, 10).astype(np.float64)
    result = sklearn_handler.preprocess_input(data)
    assert isinstance(result, np.ndarray)
