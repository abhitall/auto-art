import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from unittest.mock import MagicMock
import numpy as np
from auto_art.implementations.models.sklearn import SklearnModel
from auto_art.core.interfaces import ModelType

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
def sklearn_handler(simple_sklearn_model):
    # Pass the model instance and the specific ModelType
    return SklearnModel(simple_sklearn_model, model_type=ModelType.SKLEARN)

def test_sklearn_handler_creation(sklearn_handler, simple_sklearn_model):
    assert sklearn_handler.model is simple_sklearn_model
    assert sklearn_handler.model_type == ModelType.SKLEARN

def test_get_model_type(sklearn_handler):
    assert sklearn_handler.get_model_type() == ModelType.SKLEARN

def test_get_framework(sklearn_handler):
    assert sklearn_handler.get_framework() == "sklearn"

def test_get_input_shape(sklearn_handler, simple_sklearn_model_and_data):
    _, X, _ = simple_sklearn_model_and_data
    # For Sklearn, input shape is often (n_samples, n_features)
    # The handler might infer n_features or expect it to be set.
    # If it's inferred from model attributes like `n_features_in_`
    shape = sklearn_handler.get_input_shape()
    assert shape == (None, simple_sklearn_model.n_features_in_)

def test_get_output_shape(sklearn_handler, simple_sklearn_model):
    # For classifiers, output shape is often (n_samples,) or (n_samples, n_classes) for probabilities
    # This depends on how the handler defines it.
    # Let's assume it tries to get n_classes if available.
    # LogisticRegression stores classes_ in an array.
    num_classes = len(simple_sklearn_model.classes_)
    shape = sklearn_handler.get_output_shape()
    if num_classes == 2: # Binary classification might be (None,) or (None, 1) by some conventions
        assert shape == (None,) or shape == (None, 1) or shape == (None, 2) # For predict_proba
    else: # Multiclass
        assert shape == (None, num_classes)

def test_get_layer_info(sklearn_handler):
    # Sklearn models don't have "layers" in the same way deep learning models do.
    # This method might return an empty list or basic model info.
    layer_info = sklearn_handler.get_layer_info()
    assert isinstance(layer_info, list)
    # For Sklearn, it's likely to be empty or contain high-level info
    # For this test, let's assume it's empty if no specific structure is extracted.
    assert len(layer_info) == 0 or isinstance(layer_info[0], dict)


def test_predict(sklearn_handler, simple_sklearn_model_and_data):
    model, X_test, _ = simple_sklearn_model_and_data
    # Take a subset for prediction test
    X_sample = X_test[:5]

    # Sklearn's predict method
    predictions = sklearn_handler.predict(X_sample)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == X_sample.shape[0]
    # For LogisticRegression, predict() returns class labels

def test_predict_proba_if_available(sklearn_handler, simple_sklearn_model_and_data):
    model, X_test, _ = simple_sklearn_model_and_data
    X_sample = X_test[:5]

    if hasattr(model, "predict_proba"):
        probabilities = sklearn_handler.predict_proba(X_sample)
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == X_sample.shape[0]
        assert probabilities.shape[1] == len(model.classes_)
    else:
        # If the model doesn't have predict_proba, the handler might raise an error or return None
        with pytest.raises(AttributeError): # Assuming the handler would try to call it and fail
            sklearn_handler.predict_proba(X_sample)


def test_predict_with_mock_model(simple_sklearn_model):
    # Mock the model's predict method
    simple_sklearn_model.predict = MagicMock(return_value=np.array([0, 1, 0]))
    # Ensure model_type is passed
    handler = SklearnModel(simple_sklearn_model, model_type=ModelType.SKLEARN)

    dummy_input = np.random.rand(3, simple_sklearn_model.n_features_in_)
    predictions = handler.predict(dummy_input)

    simple_sklearn_model.predict.assert_called_once()
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (3,)
