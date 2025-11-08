"""
Validation utilities for model and data validation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from ..core.base import BaseModel
from ..core.interfaces import ModelInterface

logger = logging.getLogger(__name__)

def validate_model(model: Any) -> None:
    """
    Validate model structure and required attributes.

    Args:
        model: The model object or model handler to validate.

    Raises:
        ValueError: If model is invalid or missing required attributes.
    """
    if model is None:
        raise ValueError("Model cannot be None")

    # Check if model is a handler (BaseModel subclass) or implements ModelInterface protocol
    # Note: Model handlers inherit from BaseModel, not ModelInterface (which is a Protocol)
    is_base_model = isinstance(model, BaseModel)
    is_model_interface = isinstance(model, ModelInterface)

    if not (is_base_model or is_model_interface):
        # For raw models (PyTorch, TensorFlow, etc.), we check for basic prediction capability
        if not (hasattr(model, 'predict') or hasattr(model, 'forward') or hasattr(model, '__call__')):
            logger.warning(
                "Model does not inherit from BaseModel or implement ModelInterface. "
                "Checking for basic prediction methods (predict/forward/__call__)."
            )

    # For model handlers, check required methods
    if is_base_model:
        required_methods = ['load_model', 'analyze_architecture', 'get_model_predictions']
        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model handler must have {method} method")

    # Raw models don't necessarily have input_shape/output_shape attributes
    # These are determined during analysis

def validate_data(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    check_labels: bool = True
) -> None:
    """Validate data and labels."""
    if data is None:
        raise ValueError("Data cannot be None")
    
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    
    if data.size == 0:
        raise ValueError("Data cannot be empty")
    
    if check_labels and labels is not None:
        if not isinstance(labels, np.ndarray):
            raise ValueError("Labels must be a numpy array")
        
        if labels.size == 0:
            raise ValueError("Labels cannot be empty")
        
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")

def validate_attack_config(config: Dict[str, Any]) -> None:
    """Validate attack configuration."""
    if not isinstance(config, dict):
        raise ValueError("Attack config must be a dictionary")
    
    required_keys = ['name', 'class_', 'params']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Attack config must have {key} key")
    
    if not isinstance(config['name'], str):
        raise ValueError("Attack name must be a string")
    
    if not isinstance(config['params'], dict):
        raise ValueError("Attack parameters must be a dictionary")

def validate_defence_config(config: Dict[str, Any]) -> None:
    """Validate defence configuration."""
    if not isinstance(config, dict):
        raise ValueError("Defence config must be a dictionary")
    
    required_keys = ['name', 'class_', 'params']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Defence config must have {key} key")
    
    if not isinstance(config['name'], str):
        raise ValueError("Defence name must be a string")
    
    if not isinstance(config['params'], dict):
        raise ValueError("Defence parameters must be a dictionary")

def validate_evaluation_results(results: Dict[str, Any]) -> None:
    """Validate evaluation results."""
    if not isinstance(results, dict):
        raise ValueError("Results must be a dictionary")
    
    required_keys = ['evasion_attacks', 'poisoning_attacks', 'defences', 'metrics']
    for key in required_keys:
        if key not in results:
            raise ValueError(f"Results must have {key} key")
    
    # Validate attack results
    for attack_type in ['evasion_attacks', 'poisoning_attacks']:
        if not isinstance(results[attack_type], dict):
            raise ValueError(f"{attack_type} must be a dictionary")
    
    # Validate defence results
    if not isinstance(results['defences'], dict):
        raise ValueError("defences must be a dictionary")
    
    required_defence_keys = ['preprocessor', 'postprocessor']
    for key in required_defence_keys:
        if key not in results['defences']:
            raise ValueError(f"defences must have {key} key")
    
    # Validate metrics
    if not isinstance(results['metrics'], dict):
        raise ValueError("metrics must be a dictionary") 