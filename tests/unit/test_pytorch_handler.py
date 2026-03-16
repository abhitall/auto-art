import pytest
import torch
import torch.nn as nn
import numpy as np
from auto_art.implementations.models.pytorch import PyTorchModel
from auto_art.core.base import ModelMetadata


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc(x)
        return x


@pytest.fixture
def simple_pytorch_model():
    return SimpleConvNet()


@pytest.fixture
def pytorch_handler():
    return PyTorchModel()


def test_handler_creation(pytorch_handler):
    assert pytorch_handler is not None
    assert '.pt' in pytorch_handler.supported_extensions


def test_analyze_architecture(pytorch_handler, simple_pytorch_model):
    metadata = pytorch_handler.analyze_architecture(simple_pytorch_model, 'pytorch')
    assert isinstance(metadata, ModelMetadata)
    assert metadata.framework == 'pytorch'
    assert len(metadata.layer_info) > 0
    assert any(li['type'] == 'Conv2d' for li in metadata.layer_info)
    assert any(li['type'] == 'Linear' for li in metadata.layer_info)


def test_analyze_architecture_input_output_shapes(pytorch_handler, simple_pytorch_model):
    metadata = pytorch_handler.analyze_architecture(simple_pytorch_model, 'pytorch')
    assert metadata.input_shape is not None
    assert metadata.output_shape is not None


def test_analyze_architecture_additional_info(pytorch_handler, simple_pytorch_model):
    metadata = pytorch_handler.analyze_architecture(simple_pytorch_model, 'pytorch')
    assert metadata.additional_info is not None
    assert 'num_parameters' in metadata.additional_info
    assert metadata.additional_info['num_parameters'] > 0
    assert metadata.additional_info['model_class'] == 'SimpleConvNet'


def test_validate_model(pytorch_handler, simple_pytorch_model):
    assert pytorch_handler.validate_model(simple_pytorch_model) is True


def test_validate_invalid_model(pytorch_handler):
    assert pytorch_handler.validate_model({}) is False


def test_preprocess_numpy_input(pytorch_handler):
    np_data = np.random.rand(1, 1, 28, 28).astype(np.float32)
    result = pytorch_handler.preprocess_input(np_data)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 1, 28, 28)


def test_preprocess_tensor_input(pytorch_handler):
    tensor_data = torch.randn(1, 1, 28, 28)
    result = pytorch_handler.preprocess_input(tensor_data)
    assert isinstance(result, torch.Tensor)
    assert result is tensor_data  # Should be pass-through


def test_postprocess_output(pytorch_handler):
    tensor_output = torch.randn(1, 10)
    result = pytorch_handler.postprocess_output(tensor_output)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 10)


def test_get_model_predictions(pytorch_handler, simple_pytorch_model):
    dummy_input = np.random.rand(1, 1, 28, 28).astype(np.float32)
    predictions = pytorch_handler.get_model_predictions(simple_pytorch_model, dummy_input)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1, 10)


def test_analyze_architecture_invalid_model(pytorch_handler):
    with pytest.raises(ValueError, match="must be a PyTorch nn.Module"):
        pytorch_handler.analyze_architecture({}, 'pytorch')
