import pytest
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from auto_art.implementations.models.tensorflow import TensorFlowModel
from auto_art.core.base import ModelMetadata

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")


def create_simple_tf_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, kernel_size=5, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


@pytest.fixture
def simple_tf_model():
    return create_simple_tf_model()


@pytest.fixture
def tf_handler():
    return TensorFlowModel()


def test_handler_creation(tf_handler):
    assert tf_handler is not None
    assert '.h5' in tf_handler.supported_extensions or '.keras' in tf_handler.supported_extensions


def test_analyze_architecture(tf_handler, simple_tf_model):
    metadata = tf_handler.analyze_architecture(simple_tf_model, 'tensorflow')
    assert isinstance(metadata, ModelMetadata)
    assert metadata.framework == 'tensorflow'
    assert len(metadata.layer_info) > 0


def test_analyze_architecture_shapes(tf_handler, simple_tf_model):
    metadata = tf_handler.analyze_architecture(simple_tf_model, 'tensorflow')
    assert metadata.input_shape is not None
    assert metadata.output_shape is not None


def test_validate_model(tf_handler, simple_tf_model):
    assert tf_handler.validate_model(simple_tf_model) is True


def test_validate_invalid_model(tf_handler):
    assert tf_handler.validate_model({}) is False


def test_get_model_predictions(tf_handler, simple_tf_model):
    dummy_input = np.random.rand(2, 28, 28, 1).astype(np.float32)
    predictions = tf_handler.get_model_predictions(simple_tf_model, dummy_input)
    assert predictions is not None
    assert len(predictions) == 2
