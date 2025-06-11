"""
Configuration manager for the framework.
"""

import json
import os
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path # Ensure Path is imported if used for cache_dir validation

@dataclass
class FrameworkConfig:
    """Framework-wide configuration settings."""
    default_batch_size: int = 32
    default_num_samples: int = 100
    default_epsilon: float = 0.3
    default_eps_step: float = 0.01
    default_max_iter: int = 100
    output_dir: str = 'output'
    log_level: str = 'INFO'
    save_results: bool = True

    # GPU/Device configuration
    use_gpu: bool = True # If default_device is 'auto', this hints whether to try GPU.
    default_device: str = "auto" # Options: "auto", "cpu", "gpu"

    num_workers: int = 4 # For parallel operations like attack evaluation
    timeout: int = 3600  # Timeout for long operations (e.g., single attack eval)
    cache_dir: Optional[str] = None

class ConfigManager:
    """Singleton configuration manager."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[FrameworkConfig] = None
    _valid_log_levels: List[str] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    _valid_devices: List[str] = ['auto', 'cpu', 'gpu']


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None: # Ensure init is only run once effectively
            self._config = FrameworkConfig()
    
    @property
    def config(self) -> FrameworkConfig:
        """Get the current configuration."""
        if self._config is None: # Should not happen if __init__ ran
            self._config = FrameworkConfig()
        return self._config
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create a new config instance to load into, then validate
            new_config_instance = FrameworkConfig() # Start with defaults
            for key, value in config_dict.items():
                if hasattr(new_config_instance, key):
                    setattr(new_config_instance, key, value)
                # else: # Optionally warn about unknown keys in file
                    # print(f"Warning: Unknown key '{key}' in config file ignored.", file=sys.stderr)

            if not self.validate_config(new_config_instance):
                # This path should ideally not be reached if validate_config raises on error
                raise ValueError("Invalid configuration loaded from file (post-validation check).")
            self._config = new_config_instance # Assign validated config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {str(e)}")
        except ValueError as e: # Catch validation errors
            raise ValueError(f"Invalid configuration in file '{config_path}': {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from '{config_path}': {str(e)}")
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to file."""
        try:
            output_dir_path = os.path.dirname(config_path)
            if output_dir_path and not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path, exist_ok=True)

            config_dict = asdict(self.config)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Failed to save config to '{config_path}': {str(e)}")
    
    def update_config(self, **kwargs: Any) -> None:
        """Update configuration with new values."""
        current_config_dict = asdict(self.config)
        new_config_instance = FrameworkConfig(**current_config_dict) # Create a copy to modify

        invalid_keys = []
        for key, value in kwargs.items():
            if hasattr(new_config_instance, key):
                setattr(new_config_instance, key, value)
            else:
                invalid_keys.append(key)
        
        if invalid_keys:
            raise ValueError(f"Unknown configuration keys: {', '.join(invalid_keys)}")
        
        if not self.validate_config(new_config_instance): # Validate the modified copy
             # This path should ideally not be reached if validate_config raises on error
            raise ValueError("Invalid configuration after update (post-validation check).")
        self._config = new_config_instance # Assign validated config
    
    def reset_config(self) -> None:
        """Reset configuration to default values."""
        self._config = FrameworkConfig()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return getattr(self.config, key, default)
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        if not hasattr(self.config, key): # Check on current config instance
            raise ValueError(f"Unknown configuration key: {key}")
        
        # Create a temporary instance to validate the change
        current_config_dict = asdict(self.config)
        temp_config_instance = FrameworkConfig(**current_config_dict)
        setattr(temp_config_instance, key, value) # Apply change to temp instance

        if not self.validate_config(temp_config_instance): # Validate the temp instance
            # Error is raised by validate_config, so no need to raise again here
            # The original self._config remains unchanged
            return
        
        # If validation passes, apply to the actual config
        setattr(self._config, key, value)
    
    def validate_config(self, config_to_validate: FrameworkConfig) -> bool:
        """Validate current configuration. Raises ValueError on failure."""
        cfg = config_to_validate
        try:
            if not (isinstance(cfg.default_batch_size, int) and cfg.default_batch_size > 0):
                raise ValueError("default_batch_size must be a positive integer")
            if not (isinstance(cfg.default_num_samples, int) and cfg.default_num_samples > 0):
                raise ValueError("default_num_samples must be a positive integer")
            if not (isinstance(cfg.default_epsilon, float) and 0 < cfg.default_epsilon <= 1.0):
                raise ValueError("default_epsilon must be a float between 0 (exclusive) and 1 (inclusive)")
            if not (isinstance(cfg.default_eps_step, float) and 0 < cfg.default_eps_step <= 1.0):
                raise ValueError("default_eps_step must be a float between 0 (exclusive) and 1 (inclusive)")
            if not (isinstance(cfg.default_max_iter, int) and cfg.default_max_iter > 0):
                raise ValueError("default_max_iter must be a positive integer")
            if not (isinstance(cfg.num_workers, int) and cfg.num_workers >= 0):
                raise ValueError("num_workers must be a non-negative integer")
            if not (isinstance(cfg.timeout, int) and cfg.timeout > 0):
                raise ValueError("timeout must be a positive integer")
            
            if not isinstance(cfg.log_level, str) or cfg.log_level.upper() not in self._valid_log_levels:
                raise ValueError(f"log_level must be one of {self._valid_log_levels}")
            
            if not isinstance(cfg.default_device, str) or cfg.default_device.lower() not in self._valid_devices:
                raise ValueError(f"default_device must be one of {self._valid_devices}")

            if cfg.cache_dir is not None and not isinstance(cfg.cache_dir, str):
                 raise ValueError("cache_dir must be a string or None")
            
            if not isinstance(cfg.use_gpu, bool):
                raise ValueError("use_gpu must be a boolean")
            if not isinstance(cfg.save_results, bool):
                raise ValueError("save_results must be a boolean")
            if not isinstance(cfg.output_dir, str): # Assuming output_dir is always a string
                raise ValueError("output_dir must be a string")

            return True
        except ValueError: # Re-raise specific ValueErrors from checks
            raise
        except Exception as e:
            raise ValueError(f"Unexpected validation error: {e}")

```
I've refined the `load_config`, `update_config`, and `set_value` methods to better handle validation. They now attempt to validate a temporary copy of the configuration before applying changes to the main `self._config`. `validate_config` is also designed to raise `ValueError` directly on failure, simplifying the calling code. `Path` from `pathlib` was added to imports, though its direct use in `validate_config` for `cache_dir` checks was commented out in the prompt.The temporary file `auto_art/config/manager_py_device_config.py` has been created with the new comprehensive content for `ConfigManager` and `FrameworkConfig`.

Key changes included:
- `FrameworkConfig` now has `default_device`.
- `ConfigManager` has `_valid_devices`.
- `validate_config` now takes the config object as an argument and includes more thorough type and value checks, raising `ValueError` on failure.
- `load_config`, `update_config`, and `set_value` have been refined to validate changes on a temporary copy before applying them to the active configuration object, ensuring atomicity of valid changes.

The next step is to move this temporary file to the correct location, overwriting the original `auto_art/config/manager.py`.
