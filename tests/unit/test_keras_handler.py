import pytest
import tensorflow as tf # Keras is often part of TensorFlow
from unittest.mock import MagicMock
from auto_art.implementations.models.keras import KerasModel
from auto_art.core.interfaces import ModelType

# A simple Keras model for testing
def create_simple_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, kernel_size=5, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(20, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

@pytest.fixture
def simple_keras_model():
    return create_simple_keras_model()

@pytest.fixture
def keras_handler(simple_keras_model):
    # Pass the model instance and the specific ModelType
    return KerasModel(simple_keras_model, model_type=ModelType.KERAS)


def test_keras_handler_creation(keras_handler, simple_keras_model):
    assert keras_handler.model is simple_keras_model
    assert keras_handler.model_type == ModelType.KERAS

def test_get_model_type(keras_handler):
    assert keras_handler.get_model_type() == ModelType.KERAS

def test_get_framework(keras_handler):
    # Keras can run on different backends, but "keras" itself is a valid framework identifier here
    assert keras_handler.get_framework() == "keras"

def test_get_input_shape(keras_handler, simple_keras_model):
    # Keras models (especially Sequential/Functional) have input_shape attribute
    # This might be on the model itself or on its first layer depending on how it's built/accessed
    shape = keras_handler.get_input_shape()
    assert shape == (None, 28, 28, 1) # (Batch, Height, Width, Channels)

def test_get_output_shape(keras_handler, simple_keras_model):
    # Keras models have output_shape attribute
    shape = keras_handler.get_output_shape()
    assert shape == (None, 10) # (Batch, Num_classes)

def test_get_layer_info(keras_handler):
    layer_info = keras_handler.get_layer_info()
    assert isinstance(layer_info, list)
    # Check if some expected layers are present by type (simplified check)
    has_conv = any(item.get("type") == "Conv2D" for item in layer_info)
    has_dense = any(item.get("type") == "Dense" for item in layer_info)
    assert has_conv
    assert has_dense

def test_predict_with_mock(simple_keras_model):
    # Mock the model's predict method
    simple_keras_model.predict = MagicMock(return_value=tf.random.normal((1, 10)))
    # Ensure the model_type is passed correctly
    handler = KerasModel(simple_keras_model, model_type=ModelType.KERAS)

    dummy_input = tf.random.normal((1, 28, 28, 1))
    predictions = handler.predict(dummy_input)

    simple_keras_model.predict.assert_called_once()
    # Keras predict typically returns NumPy arrays if the input is NumPy or tf.Tensor
    # If the underlying predict returns tf.Tensor, that's also fine.
    # Let's be flexible or align with expected behavior of KerasModel wrapper.
    # Assuming KerasModel.predict might return tf.Tensor if input is tf.Tensor.
    assert isinstance(predictions, tf.Tensor) or isinstance(predictions, type(dummy_input.numpy()))
    assert predictions.shape == (1, 10)
