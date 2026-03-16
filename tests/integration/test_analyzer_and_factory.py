import pytest
import torch
import torch.nn as nn
import numpy as np

from auto_art.implementations.models.factory import ModelFactory
from auto_art.core.analysis.model_analyzer import ModelAnalyzer, analyze_model_architecture
from auto_art.core.base import ModelMetadata


# --- PyTorch Model Example ---
class SimplePyTorchConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 26 * 26, 10)  # Assuming 28x28 input

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        return self.fc1(x)


@pytest.fixture
def model_analyzer():
    return ModelAnalyzer()


# Test PyTorch model factory and analyzer integration
def test_pytorch_factory_analyzer_integration(model_analyzer):
    pytorch_model = SimplePyTorchConvNet()

    # Create handler using Factory
    handler = ModelFactory.create_model('pytorch')
    assert handler is not None

    # Analyze using handler's analyze_architecture method
    metadata = handler.analyze_architecture(pytorch_model, 'pytorch')

    assert isinstance(metadata, ModelMetadata)
    assert metadata.framework == 'pytorch'
    assert metadata.input_shape is not None
    assert metadata.output_shape is not None
    assert len(metadata.layer_info) > 0


# Test using analyze_model_architecture utility function
def test_analyze_model_architecture_pytorch():
    pytorch_model = SimplePyTorchConvNet()
    metadata = analyze_model_architecture(pytorch_model, 'pytorch')

    assert isinstance(metadata, ModelMetadata)
    assert metadata.framework == 'pytorch'
    assert len(metadata.layer_info) > 0
    assert any(li['type'] == 'Conv2d' for li in metadata.layer_info)
    assert any(li['type'] == 'Linear' for li in metadata.layer_info)


# Test Scikit-learn model factory and analyzer integration
def test_sklearn_factory_analyzer_integration():
    from sklearn.linear_model import LogisticRegression

    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    sklearn_model = LogisticRegression()
    sklearn_model.fit(X, y)

    handler = ModelFactory.create_model('sklearn')
    metadata = handler.analyze_architecture(sklearn_model, 'sklearn')

    assert isinstance(metadata, ModelMetadata)
    assert metadata.framework == 'sklearn'
    assert isinstance(metadata.layer_info, list)


# Test handling of an unsupported framework by the factory
def test_unsupported_framework_factory():
    with pytest.raises(ValueError, match="Unsupported framework"):
        ModelFactory.create_model("nonexistent_framework")


# Test analyze_model_architecture fallback on error
def test_analyze_model_architecture_fallback():
    metadata = analyze_model_architecture({}, 'nonexistent')
    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == 'unknown'
    assert metadata.additional_info is not None
    assert 'error' in metadata.additional_info
