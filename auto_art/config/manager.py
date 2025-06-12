"""
Configuration manager for the AutoART framework.

This module provides a singleton `ConfigManager` class to load, access,
and manage framework-wide configurations from a JSON file.
"""

import json
import os
from typing import Any, Dict, Optional, List, Type, Union
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import logging # Import logging

# Default configuration file name
DEFAULT_CONFIG_FILENAME = "auto_art_config.json"

# Potential locations for the config file (e.g., user's home, project root)
# This can be expanded. For now, using current directory as primary.
DEFAULT_CONFIG_PATHS = [
    Path(".") / DEFAULT_CONFIG_FILENAME,
    Path.home() / ".auto_art" / DEFAULT_CONFIG_FILENAME,
]

@dataclass
class FrameworkConfig:
    """Dataclass for storing framework-wide configuration settings.

    Attributes:
        default_batch_size: Default batch size for operations like attack generation.
        default_num_samples: Default number of samples to use for evaluations if not specified.
        default_epsilon: Default epsilon value for adversarial attacks (if applicable).
        default_eps_step: Default epsilon step value for iterative adversarial attacks.
        default_max_iter: Default maximum iterations for iterative adversarial attacks.
        output_dir: Default directory for saving outputs (results, logs, etc.).
        log_level: Logging level for the application (e.g., 'INFO', 'DEBUG').
        save_results: Boolean indicating whether to save evaluation results to file by default.
        use_gpu: If True and `default_device` is 'auto', attempts to use GPU if available.
                 If False, CPU will be used.
        default_device: Preferred device for computations: "auto", "cpu", or "gpu".
        num_workers: Default number of workers for parallel processing tasks.
        timeout: Default timeout in seconds for long-running operations.
        cache_dir: Optional path to a directory for caching. If None, caching might be disabled
                   or use a default system temp location.
    """
    default_batch_size: int = 32
    default_num_samples: int = 100
    default_epsilon: float = 0.3
    default_eps_step: float = 0.01
    default_max_iter: int = 100
    output_dir: str = 'output'
    log_level: str = 'INFO'
    save_results: bool = True
    use_gpu: bool = True
    default_device: str = "auto"
    num_workers: int = 4
    timeout: int = 3600
    cache_dir: Optional[str] = None

class ConfigManager:
    """Singleton configuration manager for AutoART.

    This class handles loading configuration settings from a JSON file,
    providing access to these settings, and allowing them to be updated or saved.
    It ensures that only one instance of the configuration exists throughout the application.
    The configuration is loaded from `auto_art_config.json` by default, searched for in
    the current directory and then in `~/.auto_art/`. If no file is found,
    default settings from `FrameworkConfig` are used.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config_data: Optional[FrameworkConfig] = None
    _config_file_path: Optional[Path] = None # Store path of loaded config

    _valid_log_levels: List[str] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    _valid_devices: List[str] = ['auto', 'cpu', 'gpu']

    def __new__(cls: Type['ConfigManager']) -> 'ConfigManager':
        """Ensures only one instance of ConfigManager is created (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # Initialize config when instance is first created
            cls._config_data = FrameworkConfig() # Start with defaults
            cls.load_config() # Attempt to load from default paths
        return cls._instance
    
    def __init__(self):
        """Initializes the ConfigManager.

        Note: Actual initialization of config data happens in `__new__` to ensure
        it's done only once for the singleton instance, typically by calling `load_config`.
        This `__init__` is called every time `ConfigManager()` is invoked but the
        core logic relies on the class-level `_config_data`.
        """
        pass # Initialization logic is in __new__ to ensure it runs once.

    @classmethod
    def load_config(cls: Type['ConfigManager'], file_path: Optional[Union[str, Path]] = None) -> None:
        """Loads configuration from a JSON file.

        If `file_path` is provided, it attempts to load from that path.
        If `file_path` is None, it searches for `auto_art_config.json` in default locations
        (current directory, then `~/.auto_art/`). If no file is found,
        default settings are maintained. If called multiple times, it reloads and
        overwrites the existing configuration.

        Args:
            file_path: Optional path to the configuration file.

        Raises:
            FileNotFoundError: If a specified `file_path` is not found.
            ValueError: If the configuration file contains invalid JSON or data
                        that fails validation.
            RuntimeError: For other unexpected errors during loading.
        """
        path_to_load: Optional[Path] = None
        if file_path:
            path_to_load = Path(file_path)
            if not path_to_load.exists():
                raise FileNotFoundError(f"Specified config file not found: {path_to_load}")
        else:
            for p in DEFAULT_CONFIG_PATHS:
                if p.exists():
                    path_to_load = p
                    break

        current_config = FrameworkConfig() # Start with defaults to load into

        if path_to_load:
            cls._config_file_path = path_to_load # Store path if successfully loaded
            try:
                with open(path_to_load, 'r') as f:
                    config_dict = json.load(f)

                loaded_keys = set()
                for field_info in fields(FrameworkConfig):
                    if field_info.name in config_dict:
                        setattr(current_config, field_info.name, config_dict[field_info.name])
                        loaded_keys.add(field_info.name)

                # Optionally warn about unknown keys
                unknown_keys = set(config_dict.keys()) - loaded_keys
                if unknown_keys:
                    logger = logging.getLogger(__name__) # Get logger instance
                    logger.warning(f"Unknown keys in config file '{path_to_load}' ignored: {', '.join(unknown_keys)}")

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file '{path_to_load}': {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Failed to load config from '{path_to_load}': {str(e)}")

        # Always validate, even if it's just defaults or partially loaded
        try:
            cls.validate_config_static(current_config) # Use static method for validation
            cls._config_data = current_config
        except ValueError as e:
            # If validation fails, reset to ensure a valid default state if this was not the first load
            cls._config_data = FrameworkConfig()
            cls._config_file_path = None # Reset path if load failed validation
            raise ValueError(f"Invalid configuration data (from file or defaults): {e}")


    @property
    def config(self) -> FrameworkConfig:
        """Provides access to the current FrameworkConfig object.
        
        Returns:
            The active FrameworkConfig instance.
        """
        if ConfigManager._config_data is None: # Should be initialized by __new__
            ConfigManager.load_config() # Try to load defaults if somehow missed
        return ConfigManager._config_data # type: ignore

    def save_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Saves the current configuration to a JSON file.

        If `file_path` is provided, saves to that path. Otherwise, saves to the
        path from which the configuration was last loaded, or to the default
        config path in the current directory if never loaded from a specific file.

        Args:
            file_path: Optional path to save the configuration file.

        Raises:
            RuntimeError: If saving fails.
            ValueError: If no path is specified and no default path can be determined.
        """
        path_to_save: Optional[Path] = None
        if file_path:
            path_to_save = Path(file_path)
        elif ConfigManager._config_file_path:
            path_to_save = ConfigManager._config_file_path
        else:
            path_to_save = DEFAULT_CONFIG_PATHS[0] # Default to current directory

        if path_to_save is None: # Should not happen with current logic
            raise ValueError("No file path specified for saving configuration and no default path available.")

        try:
            output_dir_path = path_to_save.parent
            if not output_dir_path.exists():
                output_dir_path.mkdir(parents=True, exist_ok=True)

            config_dict = asdict(self.config)
            with open(path_to_save, 'w') as f:
                json.dump(config_dict, f, indent=4)
            ConfigManager._config_file_path = path_to_save # Update loaded path
        except Exception as e:
            raise RuntimeError(f"Failed to save config to '{path_to_save}': {str(e)}")
    
    def update_config(self, **kwargs: Any) -> None:
        """Updates the current configuration with new values.

        Changes are validated before being applied. If validation fails,
        the configuration remains unchanged and a ValueError is raised.

        Args:
            **kwargs: Key-value pairs for configuration settings to update.

        Raises:
            ValueError: If unknown configuration keys are provided or if
                        the new values fail validation.
        """
        current_config_dict = asdict(self.config)
        temp_config_instance = FrameworkConfig(**current_config_dict) # Create a copy

        unknown_keys = []
        for key, value in kwargs.items():
            if hasattr(temp_config_instance, key):
                setattr(temp_config_instance, key, value)
            else:
                unknown_keys.append(key)
        
        if unknown_keys:
            raise ValueError(f"Unknown configuration keys provided for update: {', '.join(unknown_keys)}")
        
        ConfigManager.validate_config_static(temp_config_instance) # Validate the modified copy
        ConfigManager._config_data = temp_config_instance # Assign validated config
    
    def reset_config(self) -> None:
        """Resets the configuration to default FrameworkConfig values."""
        ConfigManager._config_data = FrameworkConfig()
        ConfigManager._config_file_path = None # Reset as it's default, not from a file
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration setting by its key.

        Args:
            key: The name of the configuration setting.
            default: The value to return if the key is not found.

        Returns:
            The value of the configuration setting, or the default if not found.
        """
        return getattr(self.config, key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """Sets a single configuration value after validation.

        If the new value fails validation, the configuration is not changed,
        and a ValueError is raised by the validation method.

        Args:
            key: The name of the configuration setting to set.
            value: The new value for the setting.

        Raises:
            ValueError: If the key is unknown or the value fails validation.
        """
        if not hasattr(self.config, key):
            raise ValueError(f"Unknown configuration key: {key}")
        
        current_config_dict = asdict(self.config)
        temp_config_instance = FrameworkConfig(**current_config_dict)
        setattr(temp_config_instance, key, value)

        ConfigManager.validate_config_static(temp_config_instance) # This will raise ValueError on failure
        
        # If validation passes, apply to the actual config
        setattr(ConfigManager._config_data, key, value)
    
    @staticmethod
    def validate_config_static(config_to_validate: FrameworkConfig) -> None:
        """Validates a FrameworkConfig object. Raises ValueError on failure.

        Args:
            config_to_validate: The FrameworkConfig instance to validate.

        Raises:
            ValueError: If any configuration setting is invalid.
        """
        cfg = config_to_validate
        # Type checks are largely handled by dataclass, focus on value constraints
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
        if not (isinstance(cfg.timeout, int) and cfg.timeout > 0): # Changed to int as per FrameworkConfig
            raise ValueError("timeout must be a positive integer")

        if not isinstance(cfg.log_level, str) or cfg.log_level.upper() not in ConfigManager._valid_log_levels:
            raise ValueError(f"log_level must be one of {ConfigManager._valid_log_levels}")

        if not isinstance(cfg.default_device, str) or cfg.default_device.lower() not in ConfigManager._valid_devices:
            raise ValueError(f"default_device must be one of {ConfigManager._valid_devices}")

        if cfg.cache_dir is not None:
            if not isinstance(cfg.cache_dir, str):
                raise ValueError("cache_dir must be a string path or None")
            # Optionally, try to create Path(cfg.cache_dir) to validate path format
            try:
                Path(cfg.cache_dir)
            except Exception as e:
                raise ValueError(f"cache_dir path format is invalid: {e}")

        if not isinstance(cfg.use_gpu, bool):
            raise ValueError("use_gpu must be a boolean")
        if not isinstance(cfg.save_results, bool):
            raise ValueError("save_results must be a boolean")
        if not isinstance(cfg.output_dir, str):
            raise ValueError("output_dir must be a string")

    # For direct access if needed, though property is preferred.
    # def get_config_object(self) -> FrameworkConfig:
    #     return self.config
