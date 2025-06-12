import pytest
import tensorflow as tf
from unittest.mock import MagicMock
from auto_art.implementations.models.tensorflow import TensorFlowModel
from auto_art.core.interfaces import ModelType

# A simple TensorFlow model for testing (using Keras API for simplicity)
def create_simple_tf_model():
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
def simple_tf_model():
    return create_simple_tf_model()

@pytest.fixture
def tensorflow_handler(simple_tf_model):
    return TensorFlowModel(simple_tf_model, model_type=ModelType.TENSORFLOW) # Added model_type

def test_tensorflow_handler_creation(tensorflow_handler, simple_tf_model):
    assert tensorflow_handler.model is simple_tf_model
    assert tensorflow_handler.model_type == ModelType.TENSORFLOW # Check model_type

def test_get_model_type(tensorflow_handler):
    assert tensorflow_handler.get_model_type() == ModelType.TENSORFLOW

def test_get_framework(tensorflow_handler):
    assert tensorflow_handler.get_framework() == "tensorflow"

def test_get_input_shape(tensorflow_handler, simple_tf_model):
    # TensorFlow models usually have input_shape attribute
    shape = tensorflow_handler.get_input_shape()
    assert shape == (None, 28, 28, 1) # (Batch, Height, Width, Channels)

def test_get_output_shape(tensorflow_handler, simple_tf_model):
    # TensorFlow models usually have output_shape attribute
    shape = tensorflow_handler.get_output_shape()
    assert shape == (None, 10) # (Batch, Num_classes)

def test_get_layer_info(tensorflow_handler):
    layer_info = tensorflow_handler.get_layer_info()
    assert isinstance(layer_info, list)
    # Check if some expected layers are present by type (simplified check)
    # This depends heavily on how get_layer_info is implemented for TF/Keras models
    has_conv = any(item.get("type") == "Conv2D" for item in layer_info)
    has_dense = any(item.get("type") == "Dense" for item in layer_info)
    assert has_conv
    assert has_dense

def test_predict_with_mock(simple_tf_model):
    # Mock the model's call method or predict method for this specific test
    # If it's a Keras model, simple_tf_model.predict might be easier to mock
    simple_tf_model.predict = MagicMock(return_value=tf.random.normal((1, 10)))
    handler = TensorFlowModel(simple_tf_model, model_type=ModelType.TENSORFLOW) # Added model_type

    # For TensorFlow, a common input format is a NumPy array or tf.Tensor
    dummy_input = tf.random.normal((1, 28, 28, 1))
    predictions = handler.predict(dummy_input)

    simple_tf_model.predict.assert_called_once()
    assert isinstance(predictions, tf.Tensor) # Or np.ndarray if predict is configured to return that
    assert predictions.shape == (1, 10)
