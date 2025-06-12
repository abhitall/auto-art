import pytest
import json
from unittest.mock import patch

from auto_art.api.app import app as flask_app # Import the Flask app instance
from auto_art.core.evaluation.config.evaluation_config import ModelType, Framework # For valid enum values
from auto_art.core.evaluation.art_evaluator import ARTEvaluator # To mock its methods

@pytest.fixture
def client():
    """Create a Flask test client for the app."""
    with flask_app.test_client() as client:
        yield client

def test_status_endpoint(client):
    """Test the /status endpoint."""
    response = client.get('/status')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "AutoART API is running"
    assert "version" in json_data

def test_evaluate_model_endpoint_success(client):
    """Test the /evaluate_model endpoint for a successful evaluation (mocked)."""
    mock_eval_results = {
        "model_metadata": {"model_type": "classification", "framework": "pytorch"},
        "attack_results": {"fgsm": {"clean_accuracy": 0.9, "adversarial_accuracy": 0.2}},
        "summary": {"average_adversarial_accuracy": 0.2}
    }

    with patch.object(ARTEvaluator, 'evaluate_robustness_from_path', return_value=mock_eval_results) as mock_evaluate:
        payload = {
            "model_path": "dummy/path/model.pt",
            "framework": "pytorch",
            "model_type": "classification",
            "num_samples": 50,
            "device_preference": "cpu"
        }
        response = client.post('/evaluate_model', json=payload)

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["summary"]["average_adversarial_accuracy"] == 0.2
        mock_evaluate.assert_called_once_with(
            model_path="dummy/path/model.pt",
            framework="pytorch",
            num_samples=50
        )

def test_evaluate_model_endpoint_missing_params(client):
    """Test /evaluate_model with missing required parameters."""
    payload = {
        "model_path": "dummy/path/model.pt",
        # "framework": "pytorch", # Missing framework
        "model_type": "classification"
    }
    response = client.post('/evaluate_model', json=payload)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Missing required parameters" in json_data["error"]
    assert "framework" in json_data["error"]

def test_evaluate_model_endpoint_invalid_framework(client):
    """Test /evaluate_model with an invalid framework string."""
    payload = {
        "model_path": "dummy/path/model.pt",
        "framework": "invalid_framework_name",
        "model_type": "classification"
    }
    response = client.post('/evaluate_model', json=payload)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid framework or model_type" in json_data["error"]

def test_evaluate_model_endpoint_invalid_model_type(client):
    """Test /evaluate_model with an invalid model_type string."""
    payload = {
        "model_path": "dummy/path/model.pt",
        "framework": "pytorch",
        "model_type": "invalid_model_type_name"
    }
    response = client.post('/evaluate_model', json=payload)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid framework or model_type" in json_data["error"]

def test_evaluate_model_endpoint_model_not_found(client):
    """Test /evaluate_model when the model file is not found (mocked error)."""
    with patch.object(ARTEvaluator, 'evaluate_robustness_from_path', side_effect=FileNotFoundError("Mocked: Model not found")) as mock_evaluate:
        payload = {
            "model_path": "non_existent/model.pt",
            "framework": "pytorch",
            "model_type": "classification"
        }
        response = client.post('/evaluate_model', json=payload)

        assert response.status_code == 404 # As per current app.py error handling
        json_data = response.get_json()
        assert "Model file not found" in json_data["error"]
        mock_evaluate.assert_called_once()

def test_evaluate_model_endpoint_internal_eval_error(client):
    """Test /evaluate_model when ARTEvaluator returns an error in its results."""
    mock_eval_error_results = {
        "error": "Something went wrong during evaluation steps."
    }
    with patch.object(ARTEvaluator, 'evaluate_robustness_from_path', return_value=mock_eval_error_results) as mock_evaluate:
        payload = {
            "model_path": "dummy/path/model.pt",
            "framework": "pytorch",
            "model_type": "classification"
        }
        response = client.post('/evaluate_model', json=payload)

        assert response.status_code == 500 # As per current app.py error handling
        json_data = response.get_json()
        assert "Evaluation failed" in json_data["error"]
        assert mock_eval_error_results["error"] in json_data["error"]
        mock_evaluate.assert_called_once()

def test_evaluate_model_endpoint_unexpected_exception(client):
    """Test /evaluate_model for generic unexpected exceptions during evaluation."""
    with patch.object(ARTEvaluator, 'evaluate_robustness_from_path', side_effect=Exception("Unexpected mock problem")) as mock_evaluate:
        payload = {
            "model_path": "dummy/path/model.pt",
            "framework": "pytorch",
            "model_type": "classification"
        }
        response = client.post('/evaluate_model', json=payload)

        assert response.status_code == 500
        json_data = response.get_json()
        assert "An unexpected server error occurred" in json_data["error"]
        mock_evaluate.assert_called_once()

def test_evaluate_model_endpoint_not_json(client):
    """Test /evaluate_model with non-JSON payload."""
    response = client.post('/evaluate_model', data="this is not json")
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Request must be JSON" in json_data["error"]

# TODO: Add tests for different successful evaluation outputs if structure varies.
# TODO: Add tests for optional parameters like num_samples, device_preference if they affect ARTEvaluator calls.
#       (Currently, num_samples is passed to evaluate_robustness_from_path, device_preference to EvalConfig)
