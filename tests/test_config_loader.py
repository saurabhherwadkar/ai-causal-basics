# Unit tests for the configuration loader module.
"""
Tests for ConfigLoader - Validates YAML loading, merging, and access patterns.
"""

import os  # OS operations for temp file management
import tempfile  # Temporary file creation for test isolation
from pathlib import Path  # Path manipulation for test directories

import pytest  # Testing framework

from src.utils.config_loader import ConfigLoader, ConfigurationError, get_config


class TestConfigLoader:
    """Test suite for the ConfigLoader class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test config files
        self._temp_dir = tempfile.mkdtemp()

        # Write a minimal valid config file for testing
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "application:\n"
            "  name: test-app\n"
            "  version: 1.0.0\n"
            "logging:\n"
            "  level: DEBUG\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "nested:\n"
            "  deep:\n"
            "    value: 42\n",
            encoding="utf-8",
        )

    def teardown_method(self) -> None:
        """Clean up temporary files after each test method."""
        # Remove the temp config file if it exists
        if self._config_path.exists():
            self._config_path.unlink()

        # Remove any local override file if it exists
        local_path = Path(self._temp_dir) / "config.local.yaml"
        if local_path.exists():
            local_path.unlink()

        # Remove the temporary directory
        os.rmdir(self._temp_dir)

    def test_load_valid_config(self) -> None:
        """Test that a valid YAML config file loads successfully."""
        # Create the loader pointing to our temp directory
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Verify the application name was loaded correctly
        assert loader.get("application.name") == "test-app"

    def test_get_nested_value(self) -> None:
        """Test dot-notation access to deeply nested config values."""
        # Load the config
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Access the deeply nested value using dot notation
        result = loader.get("nested.deep.value")

        # Verify the value matches what was written
        assert result == 42

    def test_get_missing_key_returns_default(self) -> None:
        """Test that missing keys return the specified default value."""
        # Load the config
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Access a non-existent key with a default
        result = loader.get("nonexistent.key", default="fallback")

        # Verify the default value is returned
        assert result == "fallback"

    def test_get_missing_key_returns_none_by_default(self) -> None:
        """Test that missing keys return None when no default is specified."""
        # Load the config
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Access a non-existent key without specifying a default
        result = loader.get("nonexistent.key")

        # Verify None is returned
        assert result is None

    def test_get_section(self) -> None:
        """Test retrieving an entire config section as a dictionary."""
        # Load the config
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Get the application section
        section = loader.get_section("application")

        # Verify the section is a dict with expected keys
        assert isinstance(section, dict)
        assert section["name"] == "test-app"
        assert section["version"] == "1.0.0"

    def test_get_section_missing_returns_empty_dict(self) -> None:
        """Test that a missing section returns an empty dictionary."""
        # Load the config
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Get a non-existent section
        section = loader.get_section("nonexistent")

        # Verify empty dict is returned
        assert section == {}

    def test_local_override_merges_correctly(self) -> None:
        """Test that config.local.yaml values override base config."""
        # Write a local override file
        local_path = Path(self._temp_dir) / "config.local.yaml"
        local_path.write_text(
            "application:\n"
            "  name: overridden-app\n"
            "logging:\n"
            "  level: ERROR\n",
            encoding="utf-8",
        )

        # Load config (should merge base + local)
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Verify the override took effect
        assert loader.get("application.name") == "overridden-app"

        # Verify the logging level was overridden
        assert loader.get("logging.level") == "ERROR"

        # Verify non-overridden values are preserved
        assert loader.get("application.version") == "1.0.0"

    def test_missing_base_config_raises_error(self) -> None:
        """Test that a missing base config file raises ConfigurationError."""
        # Create an empty temp directory with no config file
        empty_dir = tempfile.mkdtemp()

        # Expect ConfigurationError when loading from empty directory
        with pytest.raises(ConfigurationError):
            ConfigLoader(config_dir=empty_dir)

        # Clean up the empty directory
        os.rmdir(empty_dir)

    def test_project_root_property(self) -> None:
        """Test that the project_root property returns a valid Path."""
        # Load the config
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Verify project_root is a Path object
        assert isinstance(loader.project_root, Path)

    def test_all_config_returns_copy(self) -> None:
        """Test that all_config returns a copy that won't mutate internal state."""
        # Load the config
        loader = ConfigLoader(config_dir=self._temp_dir)

        # Get the full config
        config_copy = loader.all_config

        # Mutate the copy
        config_copy["application"]["name"] = "mutated"

        # Verify the internal state was not affected
        assert loader.get("application.name") == "test-app"

    def test_invalid_yaml_raises_error(self) -> None:
        """Test that invalid YAML content raises ConfigurationError."""
        # Write invalid YAML content
        self._config_path.write_text(
            "invalid: yaml: content: [unclosed",
            encoding="utf-8",
        )

        # Expect ConfigurationError when parsing invalid YAML
        with pytest.raises(ConfigurationError):
            ConfigLoader(config_dir=self._temp_dir)
