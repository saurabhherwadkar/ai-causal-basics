# Configuration loader module for reading and validating application settings.
"""
Configuration Loader - Reads YAML config files and provides typed access.

Supports environment-specific overrides via config.local.yaml which is
git-ignored and can vary between deployment environments.
"""

import os  # OS-level path and environment variable operations
import yaml  # YAML file parsing library
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Any, Dict, Optional  # Type hints for function signatures


class ConfigurationError(Exception):
    """Custom exception raised when configuration loading or validation fails."""

    pass  # No additional behavior needed beyond base Exception


class ConfigLoader:
    """
    Loads and merges application configuration from YAML files.

    Reads the base config.yaml and optionally overlays config.local.yaml
    for environment-specific overrides. Provides dictionary-style access
    to nested configuration values.
    """

    # Class-level constant for the default config directory relative to project root
    DEFAULT_CONFIG_DIR = "config"

    def __init__(self, config_dir: Optional[str] = None) -> None:
        """
        Initialize the configuration loader.

        Args:
            config_dir: Optional path to the configuration directory.
                        Defaults to 'config/' relative to the project root.
        """
        # Determine the project root directory (two levels up from this file)
        self._project_root = Path(__file__).resolve().parent.parent.parent

        # Set the configuration directory path
        if config_dir is not None:
            self._config_dir = Path(config_dir)  # Use the provided path
        else:
            self._config_dir = self._project_root / self.DEFAULT_CONFIG_DIR  # Use default

        # Initialize the configuration dictionary as empty
        self._config: Dict[str, Any] = {}

        # Load configuration files on initialization
        self._load_configuration()

    def _load_configuration(self) -> None:
        """
        Load base config and apply any local overrides.

        Reads config.yaml as the base configuration, then merges any
        values from config.local.yaml on top (if it exists).

        Raises:
            ConfigurationError: If the base config file cannot be found or parsed.
        """
        # Define path to the base configuration file
        base_config_path = self._config_dir / "config.yaml"

        # Verify the base config file exists before attempting to read
        if not base_config_path.exists():
            raise ConfigurationError(
                f"Base configuration file not found: {base_config_path}"
            )

        # Load the base configuration from YAML file
        self._config = self._read_yaml_file(base_config_path)

        # Define path to the optional local override file
        local_config_path = self._config_dir / "config.local.yaml"

        # Apply local overrides if the file exists
        if local_config_path.exists():
            local_config = self._read_yaml_file(local_config_path)  # Read local file
            self._config = self._deep_merge(self._config, local_config)  # Merge configs

    def _read_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Safely read and parse a YAML file.

        Args:
            file_path: Path to the YAML file to read.

        Returns:
            Dictionary containing the parsed YAML content.

        Raises:
            ConfigurationError: If the file cannot be read or contains invalid YAML.
        """
        try:
            # Open the file with explicit UTF-8 encoding for cross-platform support
            with open(file_path, "r", encoding="utf-8") as config_file:
                # Use safe_load to prevent arbitrary code execution from YAML
                parsed_content = yaml.safe_load(config_file)

            # Handle empty YAML files gracefully
            if parsed_content is None:
                return {}  # Return empty dict for empty files

            # Validate that the parsed content is a dictionary
            if not isinstance(parsed_content, dict):
                raise ConfigurationError(
                    f"Configuration file must contain a YAML mapping: {file_path}"
                )

            return parsed_content  # Return the validated configuration dictionary

        except yaml.YAMLError as yaml_error:
            # Wrap YAML parsing errors with a descriptive message
            raise ConfigurationError(
                f"Failed to parse YAML file {file_path}: {yaml_error}"
            ) from yaml_error

        except OSError as io_error:
            # Wrap file I/O errors with a descriptive message
            raise ConfigurationError(
                f"Failed to read configuration file {file_path}: {io_error}"
            ) from io_error

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge override dictionary into base dictionary.

        Values in override take precedence over values in base.
        Nested dictionaries are merged recursively rather than replaced.

        Args:
            base: The base dictionary to merge into.
            override: The override dictionary whose values take precedence.

        Returns:
            A new dictionary containing the merged result.
        """
        # Create a copy of base to avoid mutating the original
        merged = base.copy()

        # Iterate through each key-value pair in the override dictionary
        for key, override_value in override.items():
            # Check if both base and override have dicts for this key
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(override_value, dict)
            ):
                # Recursively merge nested dictionaries
                merged[key] = self._deep_merge(merged[key], override_value)
            else:
                # For non-dict values, override replaces base
                merged[key] = override_value

        return merged  # Return the fully merged dictionary

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using dot-notation path.

        Args:
            key_path: Dot-separated path to the config value (e.g., 'logging.level').
            default: Value to return if the key path is not found.

        Returns:
            The configuration value at the specified path, or default if not found.
        """
        # Split the dot-notation path into individual keys
        keys = key_path.split(".")

        # Start traversal at the root of the config dictionary
        current_value = self._config

        # Traverse each level of the nested dictionary
        for key in keys:
            # Check if current level is a dict and contains the key
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]  # Move to the next level
            else:
                return default  # Key not found, return the default value

        return current_value  # Return the found configuration value

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Retrieve an entire configuration section as a dictionary.

        Args:
            section_name: The top-level section name to retrieve.

        Returns:
            Dictionary containing the section's configuration, or empty dict.
        """
        # Get the section from config, defaulting to empty dict if missing
        section = self._config.get(section_name, {})

        # Ensure the returned value is always a dictionary
        if not isinstance(section, dict):
            return {}  # Return empty dict if section is not a mapping

        return section  # Return the section dictionary

    @property
    def project_root(self) -> Path:
        """Return the resolved project root directory path."""
        return self._project_root  # Expose project root as read-only property

    @property
    def all_config(self) -> Dict[str, Any]:
        """Return a copy of the entire configuration dictionary."""
        return self._config.copy()  # Return copy to prevent external mutation


# Module-level singleton instance for convenient access
_global_config: Optional[ConfigLoader] = None  # Global config instance placeholder


def get_config(config_dir: Optional[str] = None) -> ConfigLoader:
    """
    Get or create the global configuration loader singleton.

    Args:
        config_dir: Optional path to override the default config directory.

    Returns:
        The global ConfigLoader singleton instance.
    """
    global _global_config  # Access the module-level global variable

    # Create a new instance if one doesn't exist or if a custom dir is specified
    if _global_config is None or config_dir is not None:
        _global_config = ConfigLoader(config_dir)  # Initialize the singleton

    return _global_config  # Return the global config instance
