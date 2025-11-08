import pytest
from unittest.mock import MagicMock
from auto_art.core.analysis.model_analyzer import ModelAnalyzer
from auto_art.core.base import ModelMetadata
from auto_art.core.evaluation.config.evaluation_config import ModelType, Framework

# Mock models and handlers for testing
class MockModel:
    pass

class MockHandler:
    def __init__(self, framework, model_type="classification"):
        self.framework = framework
        self.model_type = model_type

    def get_model_type(self):
        return self.model_type

    def get_framework(self):
        return self.framework

    def get_input_shape(self):
        return (None, 28, 28, 1) if self.framework != "sklearn" else (None, 784)

    def get_output_shape(self):
        return (None, 10)

    def get_layer_info(self):
        return [{"name": "conv2d", "type": "Conv2D"}] if self.framework != "sklearn" else []

@pytest.fixture
def model_analyzer():
    return ModelAnalyzer()

@pytest.fixture
def mock_pytorch_handler():
    return MockHandler("pytorch", "classification")

@pytest.fixture
def mock_tensorflow_handler():
    return MockHandler("tensorflow", "classification")

@pytest.fixture
def mock_keras_handler():
    return MockHandler("keras", "classification")

@pytest.fixture
def mock_sklearn_handler():
    return MockHandler("sklearn", "classification")

def test_analyze_pytorch_model(model_analyzer, mock_pytorch_handler):
    model = MockModel()
    metadata = model_analyzer.analyze(model, mock_pytorch_handler)
    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == "classification"
    assert metadata.framework == "pytorch"
    assert metadata.input_shape == (None, 28, 28, 1)
    assert metadata.output_shape == (None, 10)
    assert len(metadata.layer_info) > 0

def test_analyze_tensorflow_model(model_analyzer, mock_tensorflow_handler):
    model = MockModel()
    metadata = model_analyzer.analyze(model, mock_tensorflow_handler)
    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == "classification"
    assert metadata.framework == "tensorflow"
    assert metadata.input_shape == (None, 28, 28, 1)
    assert metadata.output_shape == (None, 10)
    assert len(metadata.layer_info) > 0

def test_analyze_keras_model(model_analyzer, mock_keras_handler):
    model = MockModel()
    metadata = model_analyzer.analyze(model, mock_keras_handler)
    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == "classification"
    assert metadata.framework == "keras"
    assert metadata.input_shape == (None, 28, 28, 1)
    assert metadata.output_shape == (None, 10)
    assert len(metadata.layer_info) > 0

def test_analyze_sklearn_model(model_analyzer, mock_sklearn_handler):
    model = MockModel()
    metadata = model_analyzer.analyze(model, mock_sklearn_handler)
    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == "classification"
    assert metadata.framework == "sklearn"
    assert metadata.input_shape == (None, 784)
    assert metadata.output_shape == (None, 10)
    assert len(metadata.layer_info) == 0

def test_analyze_with_missing_handler_methods(model_analyzer):
    model = MockModel()
    # Create a handler that is missing some methods
    handler_with_missing_methods = MagicMock()
    handler_with_missing_methods.get_model_type.return_value = "classification"
    handler_with_missing_methods.get_framework.return_value = "pytorch"
    # Simulate missing get_input_shape
    del handler_with_missing_methods.get_input_shape

    # Depending on implementation, this might raise an error or return metadata with None/defaults
    # For this example, let's assume it logs a warning and proceeds with available info
    # If it's supposed to raise an error, the test should be `with pytest.raises(AttributeError):`
    metadata = model_analyzer.analyze(model, handler_with_missing_methods)
    assert metadata.input_shape is None # Or whatever default value is assigned
