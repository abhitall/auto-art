"""
Main Flask application for the AutoART REST API.
"""
from flask import Flask, request, jsonify
import sys

# AutoART core imports
from auto_art.core.evaluation.art_evaluator import ARTEvaluator
from auto_art.core.evaluation.config.evaluation_config import EvaluationConfig, ModelType, Framework
# ConfigManager can be used if there's a global API config file for defaults
# from auto_art.config.manager import ConfigManager

app = Flask(__name__)

# FUTURE: Consider adding global error handlers (e.g., for 404, 500) for a more robust API.

@app.route('/status', methods=['GET'])
def get_status():
    """Returns the status of the API."""
    return jsonify({"status": "AutoART API is running", "version": "0.1.0-alpha"})

@app.route('/evaluate_model', methods=['POST'])
def evaluate_model_endpoint():
    """
    Endpoint to submit a model for evaluation.
    Expects JSON payload with 'model_path', 'framework', and 'model_type'.
    Optional parameters include 'num_samples', 'device_preference', and 'batch_size'.
    Currently, this endpoint uses ARTEvaluator's `evaluate_robustness_from_path` method,
    which runs a default suite of attacks based on the model type and configured
    attack parameters in `EvaluationConfig`. Custom attack selection via API
    (e.g., using a passed 'attack_configs' list) is a future enhancement.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    model_path = data.get('model_path')
    framework_str = data.get('framework') # e.g., 'pytorch', 'tensorflow'
    model_type_str = data.get('model_type') # e.g., 'classification'
    # attack_configs_dicts = data.get('attack_configs') # Acknowledged but not used by current flow.

    # Optional parameters for evaluate_robustness_from_path
    num_samples = data.get('num_samples', 100) # Default to 100 samples
    # Optional parameters for EvaluationConfig
    device_preference = data.get('device_preference') # e.g., 'cpu', 'gpu', 'auto'
    batch_size = data.get('batch_size', 32) # Default batch size
    # Note: attack_configs_dicts from request is currently ignored for this simplified endpoint.
    # Future work could use it to customize attacks.

    required_params = {
        "model_path": model_path,
        "framework": framework_str,
        "model_type": model_type_str,
    }
    missing_params = [key for key, value in required_params.items() if value is None]
    if missing_params:
        return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

    try:
        framework_enum = Framework(framework_str.lower())
        model_type_enum = ModelType(model_type_str.lower())
    except ValueError as e:
        return jsonify({"error": f"Invalid framework or model_type: {str(e)}"}), 400

    try:
        # Create EvaluationConfig
        # For a production API, some of these might come from a global app config
        # or more detailed API request fields.
        eval_config_params = {
            'model_type': model_type_enum,
            'framework': framework_enum,
            'batch_size': batch_size,
            'device_preference': device_preference,
            # Other EvaluationConfig params will use their defaults (input_shape, nb_classes, attack_params etc.)
            # ARTEvaluator's art_estimator property will try to infer input_shape/nb_classes if not set here
            # and if model_metadata (set by evaluate_robustness_from_path) doesn't provide them.
        }
        # Remove None values so that dataclass defaults are used
        eval_config_params = {k: v for k, v in eval_config_params.items() if v is not None}
        eval_config = EvaluationConfig(**eval_config_params)

        # Initialize ARTEvaluator
        # model_obj is None because evaluate_robustness_from_path will load it.
        art_evaluator = ARTEvaluator(model_obj=None, config=eval_config)

        # Run evaluation
        # This is a synchronous call. For long evaluations, an async task queue would be better.
        app.logger.info(f"Starting evaluation for model: {model_path} ({framework_str}/{model_type_str})")
        results = art_evaluator.evaluate_robustness_from_path(
            model_path=model_path,
            framework=framework_str, # evaluate_robustness_from_path takes string
            num_samples=num_samples
        )
        app.logger.info(f"Evaluation completed for model: {model_path}")

        if "error" in results: # Check if the evaluation itself returned an error
            return jsonify({"error": f"Evaluation failed: {results['error']}"}), 500

        return jsonify(results), 200

    except FileNotFoundError as e:
        app.logger.error(f"Model file not found: {model_path} - {str(e)}")
        return jsonify({"error": f"Model file not found: {model_path}"}), 404
    except ValueError as e: # Catch ValueErrors from enum creation or other AutoART logic
        app.logger.error(f"Invalid parameter or configuration error: {str(e)}")
        return jsonify({"error": f"Invalid parameter or configuration: {str(e)}"}), 400
    except ImportError as e: # Catch ART not installed or other import issues if not caught earlier
        app.logger.error(f"Import error during evaluation: {str(e)}")
        return jsonify({"error": f"Internal server error (dependency): {str(e)}"}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during evaluation: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Note: For development, use app.run(). For production, use a WSGI server like Gunicorn.
    # Example: flask run --host=0.0.0.0 --port=5000
    # Or in code:
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True for development only
