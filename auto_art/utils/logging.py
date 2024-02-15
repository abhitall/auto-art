"""
Logging manager for the framework.
"""

import logging
import os
from typing import Optional, Dict
from datetime import datetime
from ..config.manager import ConfigManager

class LogManager:
    """Singleton logging manager."""
    
    _instance: Optional['LogManager'] = None
    _logger: Optional[logging.Logger] = None
    _file_handlers: Dict[str, logging.FileHandler] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._logger is None:
            self._setup_logger()
    
    def __del__(self):
        """Clean up file handlers on deletion."""
        self.close_all_handlers()
    
    def _setup_logger(self) -> None:
        """Set up the logger with configuration."""
        try:
            config = ConfigManager().config
            
            # Create logger
            self._logger = logging.getLogger('auto_robustness')
            self._logger.setLevel(getattr(logging, config.log_level))
            
            # Remove any existing handlers
            self._logger.handlers.clear()
            
            # Create formatters
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            
            # Create handlers
            # File handler
            log_dir = os.path.join(config.output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(log_dir, f'auto_robustness_{timestamp}.log')
            
            file_handler = logging.FileHandler(filepath)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, config.log_level))
            
            # Add handlers to logger
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)
            
            # Store file handler reference
            self._file_handlers[filepath] = file_handler
        except Exception as e:
            raise RuntimeError(f"Failed to set up logger: {str(e)}")
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        if self._logger is None:
            self._setup_logger()
        return self._logger
    
    def set_level(self, level: str) -> None:
        """Set the logging level."""
        if not hasattr(logging, level.upper()):
            raise ValueError(f"Invalid logging level: {level}")
        
        try:
            self._logger.setLevel(getattr(logging, level.upper()))
            for handler in self._logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(getattr(logging, level.upper()))
        except Exception as e:
            raise RuntimeError(f"Failed to set logging level: {str(e)}")
    
    def add_file_handler(self, filepath: str, level: str = 'DEBUG') -> None:
        """Add a new file handler to the logger."""
        if not hasattr(logging, level.upper()):
            raise ValueError(f"Invalid logging level: {level}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create and configure handler
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler = logging.FileHandler(filepath)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(getattr(logging, level.upper()))
            
            # Add handler to logger
            self._logger.addHandler(file_handler)
            
            # Store handler reference
            self._file_handlers[filepath] = file_handler
        except Exception as e:
            raise RuntimeError(f"Failed to add file handler: {str(e)}")
    
    def remove_file_handler(self, filepath: str) -> None:
        """Remove a file handler from the logger."""
        try:
            if filepath in self._file_handlers:
                handler = self._file_handlers[filepath]
                self._logger.removeHandler(handler)
                handler.close()
                del self._file_handlers[filepath]
        except Exception as e:
            raise RuntimeError(f"Failed to remove file handler: {str(e)}")
    
    def close_all_handlers(self) -> None:
        """Close all file handlers."""
        try:
            for filepath, handler in self._file_handlers.items():
                self._logger.removeHandler(handler)
                handler.close()
            self._file_handlers.clear()
        except Exception as e:
            raise RuntimeError(f"Failed to close handlers: {str(e)}")
    
    def get_log_file(self) -> str:
        """Get the path of the current log file."""
        if not self._file_handlers:
            return ""
        return next(iter(self._file_handlers.keys())) 