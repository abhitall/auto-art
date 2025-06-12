import pytest
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import numpy as np # Added import

from auto_art.implementations.models.factory import ModelFactory
from auto_art.core.analysis.model_analyzer import ModelAnalyzer
from auto_art.core.interfaces import ModelType, ModelMetadata

# --- PyTorch Model Example ---
class SimplePyTorchConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 26 * 26, 10) # Assuming 28x28 input, (28-3+1) = 26

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        return self.fc1(x)

# --- TensorFlow/Keras Model Example ---
def create_simple_tf_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    return model

# --- Scikit-learn Model Example ---
def create_simple_sklearn_model():
    # Requires fitting to infer attributes like n_features_in_
    X = np.random.rand(10, 5) # 10 samples, 5 features
    y = np.random.randint(0, 2, 10)
    model = LogisticRegression()
    model.fit(X, y)
    return model

@pytest.fixture
def model_factory():
    return ModelFactory()

@pytest.fixture
def model_analyzer():
    return ModelAnalyzer()

# Test PyTorch model factory and analyzer integration
def test_pytorch_factory_analyzer_integration(model_factory, model_analyzer):
    pytorch_model_instance = SimplePyTorchConvNet()

    # 1. Create handler using Factory
    handler = model_factory.create_model_handler(pytorch_model_instance, ModelType.PYTORCH)
    assert handler is not None
    assert handler.get_model_type() == ModelType.PYTORCH
    assert handler.get_framework() == "pytorch"

    # 2. Analyze using ModelAnalyzer
    # The handler itself is passed to analyze, not the raw model for this flow
    metadata = model_analyzer.analyze(pytorch_model_instance, handler)

    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == ModelType.PYTORCH
    assert metadata.framework == "pytorch"
    assert metadata.input_type is not None
    assert metadata.output_type is not None

    assert metadata.input_shape is not None
    assert metadata.output_shape is not None
    if metadata.input_shape != (0,):
        assert len(metadata.input_shape) == 4
        assert metadata.input_shape[1] == 1
    if metadata.output_shape != (0,):
        assert len(metadata.output_shape) == 2
        assert metadata.output_shape[1] == 10

    assert len(metadata.layer_info) > 0
    assert any(li['type'] == 'Conv2d' for li in metadata.layer_info)
    assert any(li['type'] == 'Linear' for li in metadata.layer_info)


# Test TensorFlow/Keras model factory and analyzer integration
def test_tf_keras_factory_analyzer_integration(model_factory, model_analyzer):
    tf_keras_model_instance = create_simple_tf_keras_model()

    handler = model_factory.create_model_handler(tf_keras_model_instance, ModelType.KERAS)
    assert handler is not None
    assert handler.get_model_type() == ModelType.KERAS
    assert handler.get_framework() == "keras"

    metadata = model_analyzer.analyze(tf_keras_model_instance, handler)

    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == ModelType.KERAS
    assert metadata.framework == "keras"
    assert metadata.input_shape == (None, 28, 28, 1)
    assert metadata.output_shape == (None, 10)
    assert len(metadata.layer_info) > 0
    assert any(li['type'] == 'Conv2D' for li in metadata.layer_info)
    assert any(li['type'] == 'Dense' for li in metadata.layer_info)


# Test Scikit-learn model factory and analyzer integration
def test_sklearn_factory_analyzer_integration(model_factory, model_analyzer):
    sklearn_model_instance = create_simple_sklearn_model()

    handler = model_factory.create_model_handler(sklearn_model_instance, ModelType.SKLEARN)
    assert handler is not None
    assert handler.get_model_type() == ModelType.SKLEARN
    assert handler.get_framework() == "sklearn"

    metadata = model_analyzer.analyze(sklearn_model_instance, handler)

    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == ModelType.SKLEARN
    assert metadata.framework == "sklearn"
    assert metadata.input_shape == (None, 5)
    assert metadata.output_shape == (None,) or metadata.output_shape == (None, 1) or metadata.output_shape == (None, 2)
    assert isinstance(metadata.layer_info, list)
    if metadata.layer_info:
        assert "LogisticRegression" in metadata.layer_info[0].get("type", "")


# Test handling of an unsupported model type by the factory
def test_unsupported_type_factory(model_factory):
    dummy_model = {}
    with pytest.raises(NotImplementedError):
        model_factory.create_model_handler(dummy_model, ModelType.MXNET)

    pytorch_model_instance = SimplePyTorchConvNet()
    with pytest.raises(TypeError):
        model_factory.create_model_handler(pytorch_model_instance, ModelType.SKLEARN)
