"""
Configuration manager for the framework.
"""

import json
import os
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict

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
    use_gpu: bool = True
    num_workers: int = 4
    timeout: int = 3600  # 1 hour
    cache_dir: Optional[str] = None

class ConfigManager:
    """Singleton configuration manager."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[FrameworkConfig] = None
    _valid_log_levels: List[str] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = FrameworkConfig()
    
    @property
    def config(self) -> FrameworkConfig:
        """Get the current configuration."""
        return self._config
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration
            for key, value in config_dict.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
            
            # Validate after loading
            if not self.validate_config():
                raise ValueError("Invalid configuration loaded from file")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {str(e)}")
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to file."""
        try:
            config_dict = asdict(self._config)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {str(e)}")
    
    def update_config(self, **kwargs: Any) -> None:
        """Update configuration with new values."""
        invalid_keys = []
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                invalid_keys.append(key)
        
        if invalid_keys:
            raise ValueError(f"Unknown configuration keys: {', '.join(invalid_keys)}")
        
        if not self.validate_config():
            raise ValueError("Invalid configuration after update")
    
    def reset_config(self) -> None:
        """Reset configuration to default values."""
        self._config = FrameworkConfig()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if hasattr(self._config, key):
            return getattr(self._config, key)
        return default
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        if not hasattr(self._config, key):
            raise ValueError(f"Unknown configuration key: {key}")
        
        old_value = getattr(self._config, key)
        setattr(self._config, key, value)
        
        if not self.validate_config():
            # Revert on validation failure
            setattr(self._config, key, old_value)
            raise ValueError(f"Invalid value for configuration key: {key}")
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            # Validate numeric values
            if self._config.default_batch_size <= 0:
                raise ValueError("default_batch_size must be positive")
            if self._config.default_num_samples <= 0:
                raise ValueError("default_num_samples must be positive")
            if not 0 < self._config.default_epsilon <= 1:
                raise ValueError("default_epsilon must be between 0 and 1")
            if not 0 < self._config.default_eps_step <= 1:
                raise ValueError("default_eps_step must be between 0 and 1")
            if self._config.default_max_iter <= 0:
                raise ValueError("default_max_iter must be positive")
            if self._config.num_workers < 0:
                raise ValueError("num_workers must be non-negative")
            if self._config.timeout <= 0:
                raise ValueError("timeout must be positive")
            
            # Validate string values
            if self._config.log_level.upper() not in self._valid_log_levels:
                raise ValueError(f"log_level must be one of {self._valid_log_levels}")
            
            # Validate paths
            if self._config.cache_dir:
                cache_dir = os.path.dirname(self._config.cache_dir)
                if not os.path.exists(cache_dir) and not os.access(cache_dir, os.W_OK):
                    raise ValueError(f"Cannot create or access cache directory: {cache_dir}")
            
            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}") 