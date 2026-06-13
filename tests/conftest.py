# Shared test fixtures and configuration for the test suite.
"""
Conftest - Provides shared pytest fixtures used across multiple test modules.

Contains fixtures for temporary config directories, sample data,
and logger state management to ensure test isolation.
"""

import os  # OS operations for temp directories
import tempfile  # Temporary directory creation
from pathlib import Path  # Path manipulation

import pytest  # Testing framework for fixture definitions

from src.utils.logger import LoggerFactory  # Logger state management


@pytest.fixture(autouse=True)
def reset_logger_state():
    """
    Automatically reset logger state before and after each test.

    Ensures no handler leakage between tests that could cause
    duplicate log output or stale handler references.
    """
    # Reset before test runs
    LoggerFactory.reset()

    # Yield control to the test
    yield

    # Reset after test completes
    LoggerFactory.reset()


@pytest.fixture
def temp_config_dir():
    """
    Provide a temporary directory with a minimal valid config file.

    Yields the path to the temporary directory containing config.yaml.
    Cleans up after the test completes.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Write minimal config for test isolation
    config_path = Path(temp_dir) / "config.yaml"
    config_path.write_text(
        "logging:\n"
        "  level: WARNING\n"
        "  file_enabled: false\n"
        "  console_enabled: false\n"
        "application:\n"
        "  name: test\n",
        encoding="utf-8",
    )

    # Yield the directory path to the test
    yield temp_dir

    # Clean up the config file
    if config_path.exists():
        config_path.unlink()

    # Remove the temporary directory
    os.rmdir(temp_dir)
