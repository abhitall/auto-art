import pytest
from auto_art.implementations.models.factory import ModelFactory
from auto_art.implementations.models.pytorch import PyTorchModel
from auto_art.implementations.models.tensorflow import TensorFlowModel
from auto_art.implementations.models.keras import KerasModel
from auto_art.implementations.models.sklearn import SklearnModel

try:
    import tensorflow as _tf
    _tf_available = True
except ImportError:
    _tf_available = False


@pytest.fixture
def model_factory():
    return ModelFactory


def test_create_pytorch_model(model_factory):
    handler = model_factory.create_model('pytorch')
    assert isinstance(handler, PyTorchModel)

@pytest.mark.skipif(not _tf_available, reason="TensorFlow not installed")
def test_create_tensorflow_model(model_factory):
    handler = model_factory.create_model('tensorflow')
    assert isinstance(handler, TensorFlowModel)

@pytest.mark.skipif(not _tf_available, reason="TensorFlow not installed")
def test_create_keras_model(model_factory):
    handler = model_factory.create_model('keras')
    assert isinstance(handler, KerasModel)

def test_create_sklearn_model(model_factory):
    handler = model_factory.create_model('sklearn')
    assert isinstance(handler, SklearnModel)

def test_create_model_case_insensitive(model_factory):
    handler = model_factory.create_model('PyTorch')
    assert isinstance(handler, PyTorchModel)

def test_create_tf_keras_alias(model_factory):
    handler = model_factory.create_model('tf.keras')
    assert isinstance(handler, (KerasModel, TensorFlowModel))

def test_create_invalid_framework(model_factory):
    with pytest.raises(ValueError, match="Unsupported framework"):
        model_factory.create_model("nonexistent_framework")

def test_get_supported_frameworks(model_factory):
    frameworks = model_factory.get_supported_frameworks()
    assert 'pytorch' in frameworks
    assert 'tensorflow' in frameworks
    assert 'keras' in frameworks
    assert 'sklearn' in frameworks

def test_get_implementation(model_factory):
    impl = model_factory.get_implementation('pytorch')
    assert impl is PyTorchModel

def test_get_implementation_missing(model_factory):
    impl = model_factory.get_implementation('nonexistent')
    assert impl is None
