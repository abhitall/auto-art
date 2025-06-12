import pytest
from auto_art.implementations.models.factory import ModelFactory
from auto_art.implementations.models.pytorch import PyTorchModel
from auto_art.implementations.models.tensorflow import TensorFlowModel
from auto_art.implementations.models.keras import KerasModel
from auto_art.implementations.models.sklearn import SklearnModel
from auto_art.core.interfaces import ModelType

# Mock models for testing
class DummyPyTorchModel:
    def __init__(self):
        self.model = "PyTorchModel"

class DummyTensorFlowModel:
    def __init__(self):
        self.model = "TensorFlowModel"

class DummyKerasModel:
    def __init__(self):
        self.model = "KerasModel"

class DummySklearnModel:
    def __init__(self):
        self.model = "SklearnModel"

@pytest.fixture
def model_factory():
    return ModelFactory()

def test_create_pytorch_model(model_factory):
    model_handler = model_factory.create_model_handler(DummyPyTorchModel(), ModelType.PYTORCH)
    assert isinstance(model_handler, PyTorchModel)

def test_create_tensorflow_model(model_factory):
    model_handler = model_factory.create_model_handler(DummyTensorFlowModel(), ModelType.TENSORFLOW)
    assert isinstance(model_handler, TensorFlowModel)

def test_create_keras_model(model_factory):
    model_handler = model_factory.create_model_handler(DummyKerasModel(), ModelType.KERAS)
    assert isinstance(model_handler, KerasModel)

def test_create_sklearn_model(model_factory):
    model_handler = model_factory.create_model_handler(DummySklearnModel(), ModelType.SKLEARN)
    assert isinstance(model_handler, SklearnModel)

def test_create_invalid_model_type(model_factory):
    with pytest.raises(ValueError):
        model_factory.create_model_handler(DummyPyTorchModel(), "invalid_type")

def test_create_unsupported_model_type(model_factory):
    with pytest.raises(NotImplementedError):
        model_factory.create_model_handler(DummyPyTorchModel(), ModelType.MXNET) # Assuming MXNET is not implemented
