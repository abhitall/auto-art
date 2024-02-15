"""
Common utility functions for the framework.
"""

import os
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from pathlib import Path
from datetime import datetime

def ensure_dir(directory: Union[str, Path]) -> None:
    """Ensure a directory exists, create it if it doesn't."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 4) -> None:
    """Save data to a JSON file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load data from a JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def get_timestamp() -> str:
    """Get current timestamp in a consistent format."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """Format metric values to strings with specified precision."""
    return {
        key: f"{value:.{precision}f}"
        for key, value in metrics.items()
    }

def calculate_statistics(
    values: List[float]
) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    values_array = np.array(values)
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array))
    }

def validate_file_extension(
    filepath: Union[str, Path],
    allowed_extensions: List[str]
) -> bool:
    """Validate if a file has an allowed extension."""
    filepath = Path(filepath)
    return filepath.suffix.lower() in allowed_extensions

def get_file_size(filepath: Union[str, Path]) -> int:
    """Get file size in bytes."""
    filepath = Path(filepath)
    return filepath.stat().st_size

def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default value if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator

def batch_iterator(
    data: List[Any],
    batch_size: int
) -> Iterator[List[Any]]:
    """Iterate over data in batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '_'
) -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(
    d: Dict[str, Any],
    sep: str = '_'
) -> Dict[str, Any]:
    """Unflatten a dictionary with keys containing separators."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result 