"""
Unit tests for the ConfigManager.
"""
import pytest
import json
import os
from pathlib import Path
from auto_art.config.manager import ConfigManager, FrameworkConfig # Adjust import if needed based on PYTHONPATH in test env
from dataclasses import asdict # For creating test config dicts

# Test basic singleton and default values
def test_config_manager_singleton(mocker): # Add mocker fixture
    # Reset singleton for test isolation if ConfigManager._instance is accessible
    # This is tricky. For true singleton testing across test functions, state can persist.
    # A common pattern is ConfigManager().reset_config() at start of tests needing defaults.
    # Or, if _instance is accessible: ConfigManager._instance = None before first call in test.
    # For now, we assume ConfigManager behaves as a singleton across calls within a session.
    # To make tests more independent if state was an issue:

    # Patch the _instance to None before each ConfigManager() call in this specific test
    # to simulate re-initialization for the purpose of testing singleton behavior.
    # Note: This directly manipulates the class's internal state for testing, which is
    # sometimes necessary for true singletons but should be used with caution.

    # Force re-creation for cm1
    ConfigManager._instance = None # type: ignore
    cm1 = ConfigManager()

    # Force re-creation for cm2
    ConfigManager._instance = None # type: ignore
    cm2 = ConfigManager()

    # cm1 and cm2 are now different instances because we reset the singleton.
    # This part of the test was conceptually flawed for testing *if it's a singleton*.
    # The original intent was likely to show that ConfigManager() normally returns the same instance.
    # Let's test that standard behavior first.
    ConfigManager._instance = None # type: ignore # Ensure a clean start for the actual test
    cm_first_call = ConfigManager()
    cm_second_call = ConfigManager()
    assert cm_first_call is cm_second_call, "ConfigManager() should return the same instance."

    # Clean up for other tests by resetting the singleton state if it was globally patched.
    # This ensures other tests get a fresh (or consistent) singleton.
    ConfigManager._instance = None # type: ignore


def test_config_manager_default_config():
    ConfigManager._instance = None # type: ignore # Ensure a fresh instance for this test
    cm = ConfigManager()
    # reset_config() is now called by default in __init__ if _config is None,
    # and this test gets a fresh instance, so its _config will be default.

    cfg_defaults = FrameworkConfig() # Expected defaults

    assert cm.config.default_batch_size == cfg_defaults.default_batch_size
    assert cm.config.log_level == cfg_defaults.log_level
    assert cm.config.default_device == cfg_defaults.default_device
    assert cm.config.use_gpu == cfg_defaults.use_gpu
    # Reset for other tests just in case, though each test should manage its state.
    ConfigManager._instance = None # type: ignore


def test_config_manager_update_and_set_value():
    ConfigManager._instance = None # type: ignore
    cm = ConfigManager()
    # cm.reset_config() # Not strictly needed due to fresh instance

    cm.update_config(default_batch_size=64, log_level="DEBUG")
    assert cm.config.default_batch_size == 64
    assert cm.config.log_level == "DEBUG"

    cm.set_value("default_num_samples", 200)
    assert cm.config.default_num_samples == 200
    cm.set_value("use_gpu", False)
    assert cm.config.use_gpu is False

    ConfigManager._instance = None # type: ignore


def test_config_manager_validation():
    ConfigManager._instance = None # type: ignore
    cm = ConfigManager()

    with pytest.raises(ValueError, match="log_level must be one of"):
        cm.set_value("log_level", "INVALID_LEVEL_FOR_TEST") # Made value more unique

    with pytest.raises(ValueError, match="default_device must be one of"):
        cm.set_value("default_device", "tpu_device_test_unique")

    with pytest.raises(ValueError, match="default_batch_size must be a positive integer"):
        cm.set_value("default_batch_size", -10)

    with pytest.raises(ValueError, match="use_gpu must be a boolean"):
        cm.set_value("use_gpu", "not_a_bool_string_unique")

    with pytest.raises(ValueError, match="Unknown configuration key: unknown_key_test_unique"):
        cm.set_value("unknown_key_test_unique", "some_value")

    with pytest.raises(ValueError, match="Unknown configuration keys: another_unknown_key_test_unique"):
        cm.update_config(another_unknown_key_test_unique="test")

    ConfigManager._instance = None # type: ignore


def test_config_manager_save_load(tmp_path: Path): # tmp_path is a pytest fixture
    ConfigManager._instance = None # type: ignore
    cm = ConfigManager()

    test_output_dir = str(tmp_path / "test_output_dir_save_load_unique")
    cm.update_config(output_dir=test_output_dir, default_device="gpu", log_level="WARNING")

    config_file = tmp_path / "test_config_save_load_unique.json"

    cm.save_config(str(config_file))
    assert config_file.exists()

    with open(config_file, 'r') as f:
        saved_data = json.load(f)
    assert saved_data['output_dir'] == test_output_dir
    assert saved_data['default_device'] == "gpu"
    assert saved_data['log_level'] == "WARNING"

    ConfigManager._instance = None # type: ignore # Force re-load from file
    cm_new_instance = ConfigManager()
    # At this point, cm_new_instance will have default FrameworkConfig.
    # We need to call load_config on it.
    assert cm_new_instance.config.output_dir != test_output_dir # Verify it's not the saved one yet

    cm_new_instance.load_config(str(config_file))
    assert cm_new_instance.config.output_dir == test_output_dir
    assert cm_new_instance.config.default_device == "gpu"
    assert cm_new_instance.config.log_level == "WARNING"

    ConfigManager._instance = None # type: ignore


def test_config_manager_load_non_existent_file(tmp_path: Path):
    ConfigManager._instance = None # type: ignore
    cm = ConfigManager()
    with pytest.raises(FileNotFoundError):
        cm.load_config(str(tmp_path / "non_existent_config_file_unique.json"))
    ConfigManager._instance = None # type: ignore


def test_config_manager_load_invalid_json_format(tmp_path: Path):
    ConfigManager._instance = None # type: ignore
    cm = ConfigManager()
    config_file = tmp_path / "invalid_json_format_unique.json"
    with open(config_file, 'w') as f:
        f.write("{'invalid_json_due_to_quotes_unique': True,}")

    with pytest.raises(ValueError, match="Invalid JSON in config file"):
        cm.load_config(str(config_file))
    ConfigManager._instance = None # type: ignore


def test_config_manager_load_config_with_invalid_value(tmp_path: Path):
    ConfigManager._instance = None # type: ignore
    cm = ConfigManager()

    config_file = tmp_path / "invalid_config_value_test_unique.json"
    # Use asdict to get current valid structure then override one value
    # Create a default config to base the invalid one on
    default_valid_config = FrameworkConfig()
    invalid_settings_dict = asdict(default_valid_config)
    invalid_settings_dict['log_level'] = "SUPERBAD_LOG_LEVEL_UNIQUE"

    with open(config_file, 'w') as f:
        json.dump(invalid_settings_dict, f)

    # The error message from validate_config is re-raised by load_config
    with pytest.raises(ValueError, match="log_level must be one of"):
        cm.load_config(str(config_file))
    ConfigManager._instance = None # type: ignore


def test_config_manager_reset_restores_defaults():
    ConfigManager._instance = None # type: ignore
    cm = ConfigManager()
    # Modify from defaults
    cm.update_config(default_batch_size=128, log_level="WARNING", use_gpu=False)
    assert cm.config.default_batch_size == 128
    assert cm.config.log_level == "WARNING"
    assert cm.config.use_gpu is False

    cm.reset_config() # This re-assigns self._config = FrameworkConfig()
    default_cfg = FrameworkConfig()
    assert cm.config.default_batch_size == default_cfg.default_batch_size
    assert cm.config.log_level == default_cfg.log_level
    assert cm.config.use_gpu == default_cfg.use_gpu
    assert cm.config.default_device == default_cfg.default_device
    ConfigManager._instance = None # type: ignore

```
I've made some adjustments to the singleton test to correctly test the singleton behavior and then reset `_instance` for other tests. Also, made test values more unique to avoid accidental matches in error messages if tests were run in a different order or if error messages were similar.
The `mocker` fixture is not actually used in the revised singleton test, as direct manipulation of `_instance` achieves the isolation for that specific test. The `mocker` import can be removed if not used elsewhere. For now, I'll keep it as it was in the prompt. The key for test isolation with this singleton pattern is `ConfigManager._instance = None` at the start of each test function that needs a truly fresh `ConfigManager` state.The file `tests/unit/test_config_manager.py` has been successfully created with the provided pytest test suite.

**Key aspects of the created test file:**
-   Imports `pytest`, `json`, `os`, `Path`, and the necessary components from `auto_art.config.manager`.
-   Includes tests for:
    -   Singleton behavior of `ConfigManager` (with manual resetting of `_instance` for test isolation).
    -   Verification of default configuration values.
    -   Functionality of `update_config` and `set_value`.
    -   Configuration validation, checking for `ValueError` with invalid inputs.
    -   Saving and loading configuration to/from JSON files, using the `tmp_path` pytest fixture for temporary file creation.
    -   Error handling for loading non-existent or malformed JSON files.
    -   Error handling for loading configuration files that contain values failing validation rules.
    -   Functionality of `reset_config` to restore default settings.
-   The tests make use of `ConfigManager._instance = None` to ensure a fresh state for the singleton where necessary, which is a common pattern for testing singletons that don't have a public reset method for their instance.

The subtask is complete. I will now submit the report.
