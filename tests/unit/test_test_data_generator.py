import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_art.core.testing.data_generator import TestDataGenerator, TestData, ModelMetadata
from auto_art.core.interfaces import ModelType # For ModelMetadata

@pytest.fixture
def test_data_generator():
    return TestDataGenerator()

@pytest.fixture
def sample_model_metadata_image():
    return ModelMetadata(
        model_type=ModelType.PYTORCH, framework="pytorch",
        input_shape=(None, 3, 32, 32), output_shape=(None, 10), # Channels first
        input_type="image", output_type="classification"
    )

@pytest.fixture
def sample_model_metadata_text_tokens():
    return ModelMetadata(
        model_type=ModelType.PYTORCH, framework="pytorch",
        input_shape=(None, 50), output_shape=(None, 10), # Sequence length 50
        input_type="text_token_ids", output_type="classification",
        additional_info={"vocab_size": 1000}
    )

@pytest.fixture
def sample_model_metadata_text_embeddings():
    return ModelMetadata(
        model_type=ModelType.PYTORCH, framework="pytorch",
        input_shape=(None, 50, 128), output_shape=(None, 10), # Seq len 50, embed dim 128
        input_type="text_embeddings", output_type="classification"
    )

@pytest.fixture
def sample_model_metadata_tabular():
    return ModelMetadata(
        model_type=ModelType.SKLEARN, framework="sklearn",
        input_shape=(None, 20), output_shape=(None, 3), # 20 features
        input_type="tabular", output_type="classification"
    )

# --- Synthetic Data Generation Tests ---
def test_generate_synthetic_image_data(test_data_generator, sample_model_metadata_image):
    num_samples = 5
    data = test_data_generator.generate_test_data(sample_model_metadata_image, num_samples=num_samples)
    assert isinstance(data, TestData)
    assert isinstance(data.inputs, np.ndarray)
    assert data.inputs.shape == (num_samples, 3, 32, 32) # Batch, Channels, H, W
    assert data.inputs.dtype == np.float32
    assert data.expected_outputs is None # Synthetic data doesn't generate labels by default here
    assert data.metadata['input_type'] == 'image'

def test_generate_synthetic_text_token_data(test_data_generator, sample_model_metadata_text_tokens):
    num_samples = 7
    data = test_data_generator.generate_test_data(sample_model_metadata_text_tokens, num_samples=num_samples)
    assert isinstance(data, TestData)
    assert isinstance(data.inputs, np.ndarray)
    assert data.inputs.shape == (num_samples, 50) # Batch, SeqLen
    assert data.inputs.dtype == np.int32 # Token IDs are usually integers
    vocab_size = sample_model_metadata_text_tokens.additional_info['vocab_size']
    assert np.all(data.inputs < vocab_size)
    assert np.all(data.inputs >= 0)

def test_generate_synthetic_text_embedding_data(test_data_generator, sample_model_metadata_text_embeddings):
    num_samples = 3
    data = test_data_generator.generate_test_data(sample_model_metadata_text_embeddings, num_samples=num_samples)
    assert isinstance(data, TestData)
    assert isinstance(data.inputs, np.ndarray)
    assert data.inputs.shape == (num_samples, 50, 128) # Batch, SeqLen, EmbedDim
    assert data.inputs.dtype == np.float32

def test_generate_synthetic_tabular_data(test_data_generator, sample_model_metadata_tabular):
    num_samples = 10
    data = test_data_generator.generate_test_data(sample_model_metadata_tabular, num_samples=num_samples)
    assert isinstance(data, TestData)
    assert isinstance(data.inputs, np.ndarray)
    assert data.inputs.shape == (num_samples, 20) # Batch, NumFeatures
    assert data.inputs.dtype == np.float32

def test_generate_unsupported_synthetic_data(test_data_generator):
    metadata = ModelMetadata(input_type="unknown_exotic_type", input_shape=(None, 10))
    with pytest.raises(ValueError, match="Unsupported input_type 'unknown_exotic_type'"):
        test_data_generator.generate_test_data(metadata)

def test_generate_insufficient_shape_info(test_data_generator):
    metadata = ModelMetadata(input_type="tabular", input_shape=None) # No shape
    with pytest.raises(ValueError, match="Cannot generate synthetic data for input_type 'tabular' due to insufficient shape info"):
        test_data_generator.generate_test_data(metadata)

    metadata_non_concrete = ModelMetadata(input_type="image", input_shape=(None, None, 3)) # Non-concrete shape
    with pytest.raises(ValueError, match="Cannot generate image data without valid concrete input_shape"):
        test_data_generator.generate_test_data(metadata_non_concrete)


# --- Data Loading Tests ---
@pytest.fixture
def temp_data_files(tmp_path: Path):
    # Create dummy data files for loading tests
    # .npy file
    npy_file = tmp_path / "test_data.npy"
    np.save(npy_file, np.random.rand(10, 5)) # 10 samples, 5 features

    # .npz file (features and labels)
    npz_file = tmp_path / "test_data.npz"
    np.savez(npz_file, x=np.random.rand(12, 4), y=np.random.randint(0, 2, 12))

    # .csv file
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(np.random.rand(15, 3), columns=['feat1', 'feat2', 'label_col'])
    df['label_col'] = df['label_col'].round().astype(int)
    df.to_csv(csv_file, index=False)

    return {"npy": npy_file, "npz": npz_file, "csv": csv_file}

def test_load_from_npy(test_data_generator, temp_data_files):
    data = test_data_generator.load_data_from_source(temp_data_files["npy"])
    assert isinstance(data, TestData)
    assert data.inputs.shape == (10, 5)
    assert data.inputs.dtype == np.float32 # Should be converted
    assert data.expected_outputs is None
    assert data.metadata['format'] == 'npy'

def test_load_from_npz_with_xy(test_data_generator, temp_data_files):
    data = test_data_generator.load_data_from_source(temp_data_files["npz"])
    assert isinstance(data, TestData)
    assert data.inputs.shape == (12, 4)
    assert data.inputs.dtype == np.float32
    assert data.expected_outputs is not None
    assert data.expected_outputs.shape == (12,)
    # npz might load labels as various int types, ensure it's usable.
    # The generator does not force label dtype, which is fine.
    assert data.metadata['format'] == 'npz'
    assert data.metadata['x_key_used'] == 'x'
    assert data.metadata['y_key_used'] == 'y'


def test_load_from_csv_auto_detect_xy(test_data_generator, temp_data_files):
    data = test_data_generator.load_data_from_source(temp_data_files["csv"])
    assert isinstance(data, TestData)
    assert data.inputs.shape == (15, 2) # feat1, feat2
    assert data.inputs.dtype == np.float32
    assert data.expected_outputs is not None
    assert data.expected_outputs.shape == (15,1) # label_col (reshaped)
    assert data.metadata['format'] == 'csv'
    assert data.metadata['feature_columns_used'] == ['feat1', 'feat2'] # Inferred
    assert data.metadata['label_columns_used'] is None # Auto-detected, not explicitly set

def test_load_from_csv_explicit_cols(test_data_generator, temp_data_files):
    data = test_data_generator.load_data_from_source(
        temp_data_files["csv"],
        feature_columns=['feat1'],
        label_columns='label_col'
    )
    assert data.inputs.shape == (15, 1)
    assert data.expected_outputs.shape == (15,1)
    assert data.metadata['feature_columns_used'] == ['feat1']
    assert data.metadata['label_columns_used'] == 'label_col'

def test_load_from_numpy_tuple(test_data_generator):
    x_np = np.random.rand(8, 3)
    y_np = np.random.randint(0, 3, 8)
    data = test_data_generator.load_data_from_source((x_np, y_np))
    assert np.array_equal(data.inputs, x_np.astype(np.float32))
    assert np.array_equal(data.expected_outputs, y_np)
    assert data.metadata['format'] == 'numpy_tuple'

def test_load_from_numpy_array_inputs_only(test_data_generator):
    x_np = np.random.rand(7, 6)
    data = test_data_generator.load_data_from_source(x_np)
    assert np.array_equal(data.inputs, x_np.astype(np.float32))
    assert data.expected_outputs is None
    assert data.metadata['format'] == 'numpy_array_inputs_only'

def test_load_with_num_samples(test_data_generator, temp_data_files):
    data = test_data_generator.load_data_from_source(temp_data_files["npy"], num_samples=5)
    assert data.inputs.shape[0] == 5
    assert data.metadata['num_samples_loaded'] == 5

def test_load_file_not_found(test_data_generator, tmp_path):
    with pytest.raises(FileNotFoundError):
        test_data_generator.load_data_from_source(tmp_path / "non_existent.npy")

def test_load_unsupported_extension(test_data_generator, tmp_path):
    dummy_file = tmp_path / "data.txt"
    dummy_file.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
        test_data_generator.load_data_from_source(dummy_file)

# --- generate_expected_outputs Tests (requires mock model) ---
@pytest.fixture
def mock_pytorch_model():
    model = MagicMock(spec=torch.nn.Module)
    # Mock parameters to determine device
    mock_param = MagicMock(spec=torch.nn.Parameter)
    mock_param.device = torch.device('cpu')
    model.parameters.return_value = iter([mock_param])
    # Mock forward pass
    model.return_value = torch.randn(5, 2) # Batch 5, 2 classes
    return model

@pytest.fixture
def mock_tf_model():
    if TestDataGenerator.tf is None: pytest.skip("TensorFlow not available")
    model = MagicMock(spec=TestDataGenerator.tf.Module) # or tf.keras.Model
    model.predict.return_value = TestDataGenerator.tf.random.normal((5,2))
    # If it's not a Keras model, it might be called directly
    model.return_value = TestDataGenerator.tf.random.normal((5,2))
    return model

@pytest.fixture
def mock_sklearn_model():
    model = MagicMock() # spec=some sklearn estimator like LogisticRegression
    model.predict.return_value = np.random.randint(0, 2, 5)
    return model

def test_generate_expected_outputs_pytorch(test_data_generator, mock_pytorch_model):
    inputs = np.random.rand(5, 10).astype(np.float32) # Batch 5, 10 features
    test_data_obj = TestData(inputs=inputs)

    outputs = test_data_generator.generate_expected_outputs(mock_pytorch_model, test_data_obj)
    assert isinstance(outputs, np.ndarray)
    assert outputs.shape == (5, 2)
    mock_pytorch_model.assert_called_once() # Check if model's forward was called

@pytest.mark.skipif(TestDataGenerator.tf is None, reason="TensorFlow not available")
def test_generate_expected_outputs_tensorflow(test_data_generator, mock_tf_model):
    inputs = np.random.rand(5, 10).astype(np.float32)
    test_data_obj = TestData(inputs=inputs)

    outputs = test_data_generator.generate_expected_outputs(mock_tf_model, test_data_obj)
    assert isinstance(outputs, np.ndarray)
    assert outputs.shape == (5, 2)
    # Check if model was called (either directly or via .predict)
    if hasattr(mock_tf_model, 'predict') and mock_tf_model.predict.called:
        mock_tf_model.predict.assert_called_once()
    else:
        mock_tf_model.assert_called_once()


def test_generate_expected_outputs_sklearn(test_data_generator, mock_sklearn_model):
    inputs = np.random.rand(5, 10).astype(np.float32)
    test_data_obj = TestData(inputs=inputs)

    outputs = test_data_generator.generate_expected_outputs(mock_sklearn_model, test_data_obj)
    assert isinstance(outputs, np.ndarray)
    assert outputs.shape == (5,) # or (5, num_classes) if predict_proba was used
    mock_sklearn_model.predict.assert_called_once_with(inputs)

def test_generate_expected_outputs_preloaded(test_data_generator, mock_sklearn_model):
    inputs = np.random.rand(5,10)
    preloaded_outputs = np.array([0,1,0,1,0])
    test_data_obj = TestData(inputs=inputs, expected_outputs=preloaded_outputs)

    outputs = test_data_generator.generate_expected_outputs(mock_sklearn_model, test_data_obj)
    assert np.array_equal(outputs, preloaded_outputs)
    mock_sklearn_model.predict.assert_not_called() # Model should not be called if outputs are preloaded
