import pytest
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from auto_art.implementations.models.keras import KerasModel
from auto_art.core.base import ModelMetadata

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow/Keras not available")


def create_simple_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, kernel_size=5, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


@pytest.fixture
def simple_keras_model():
    return create_simple_keras_model()


@pytest.fixture
def keras_handler():
    return KerasModel()


def test_handler_creation(keras_handler):
    assert keras_handler is not None


def test_analyze_architecture(keras_handler, simple_keras_model):
    metadata = keras_handler.analyze_architecture(simple_keras_model, 'keras')
    assert isinstance(metadata, ModelMetadata)
    assert metadata.framework == 'keras'
    assert len(metadata.layer_info) > 0


def test_analyze_architecture_shapes(keras_handler, simple_keras_model):
    metadata = keras_handler.analyze_architecture(simple_keras_model, 'keras')
    assert metadata.input_shape is not None
    assert metadata.output_shape is not None


def test_validate_model(keras_handler, simple_keras_model):
    assert keras_handler.validate_model(simple_keras_model) is True


def test_validate_invalid_model(keras_handler):
    assert keras_handler.validate_model({}) is False


def test_get_model_predictions(keras_handler, simple_keras_model):
    dummy_input = np.random.rand(2, 28, 28, 1).astype(np.float32)
    predictions = keras_handler.get_model_predictions(simple_keras_model, dummy_input)
    assert predictions is not None
    assert len(predictions) == 2
