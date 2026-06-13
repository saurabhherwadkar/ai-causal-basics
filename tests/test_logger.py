# Unit tests for the logging infrastructure module.
"""
Tests for LoggerFactory - Validates logger creation, levels, and handlers.
"""

import logging  # Logging module for level constants
import os  # OS operations for temp file handling
import tempfile  # Temporary file creation
from pathlib import Path  # Path manipulation

import pytest  # Testing framework

from src.utils.logger import LoggerFactory


class TestLoggerFactory:
    """Test suite for the LoggerFactory class."""

    def setup_method(self) -> None:
        """Set up test fixtures and reset logger state before each test."""
        # Reset all loggers to ensure clean state
        LoggerFactory.reset()

        # Create a temporary config directory with logging config
        self._temp_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: DEBUG\n"
            "  format: '%(name)s - %(levelname)s - %(message)s'\n"
            "  date_format: '%Y-%m-%d'\n"
            "  file_enabled: false\n"
            "  console_enabled: true\n"
            "  file_path: 'logs/test.log'\n"
            "  max_file_size_mb: 1\n"
            "  backup_count: 2\n",
            encoding="utf-8",
        )

    def teardown_method(self) -> None:
        """Clean up after each test."""
        # Reset loggers to prevent state leakage between tests
        LoggerFactory.reset()

        # Remove temp config file
        if self._config_path.exists():
            self._config_path.unlink()

        # Remove temp directory
        os.rmdir(self._temp_dir)

    def test_get_logger_returns_logger_instance(self) -> None:
        """Test that get_logger returns a valid logging.Logger instance."""
        # Create a logger with our test config
        test_logger = LoggerFactory.get_logger("test.module", config_dir=self._temp_dir)

        # Verify it's a Logger instance
        assert isinstance(test_logger, logging.Logger)

    def test_get_logger_uses_correct_name(self) -> None:
        """Test that the logger uses the provided name."""
        # Create a named logger
        test_logger = LoggerFactory.get_logger("my.custom.name", config_dir=self._temp_dir)

        # Verify the logger name matches
        assert test_logger.name == "my.custom.name"

    def test_get_logger_sets_correct_level(self) -> None:
        """Test that the logger level matches the config setting."""
        # Create a logger (config sets level to DEBUG)
        test_logger = LoggerFactory.get_logger("test.level", config_dir=self._temp_dir)

        # Verify the logger level is DEBUG
        assert test_logger.level == logging.DEBUG

    def test_get_logger_singleton_behavior(self) -> None:
        """Test that requesting the same logger name returns the same instance."""
        # Create a logger
        logger_first = LoggerFactory.get_logger("singleton.test", config_dir=self._temp_dir)

        # Request the same logger again
        logger_second = LoggerFactory.get_logger("singleton.test", config_dir=self._temp_dir)

        # Verify they are the same object
        assert logger_first is logger_second

    def test_get_logger_different_names_different_instances(self) -> None:
        """Test that different names produce different logger instances."""
        # Create two loggers with different names
        logger_a = LoggerFactory.get_logger("module.a", config_dir=self._temp_dir)
        logger_b = LoggerFactory.get_logger("module.b", config_dir=self._temp_dir)

        # Verify they are different objects
        assert logger_a is not logger_b

    def test_console_handler_attached(self) -> None:
        """Test that a console handler is attached when enabled in config."""
        # Create a logger (console_enabled: true in config)
        test_logger = LoggerFactory.get_logger("console.test", config_dir=self._temp_dir)

        # Check that at least one handler is a StreamHandler
        stream_handlers = [
            h for h in test_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) > 0

    def test_no_propagation(self) -> None:
        """Test that logger propagation is disabled to prevent duplicate output."""
        # Create a logger
        test_logger = LoggerFactory.get_logger("propagation.test", config_dir=self._temp_dir)

        # Verify propagation is disabled
        assert test_logger.propagate is False

    def test_reset_clears_all_loggers(self) -> None:
        """Test that reset() removes all handlers and clears the registry."""
        # Create a logger
        test_logger = LoggerFactory.get_logger("reset.test", config_dir=self._temp_dir)

        # Verify it has handlers
        assert len(test_logger.handlers) > 0

        # Reset all loggers
        LoggerFactory.reset()

        # Verify handlers were removed
        assert len(test_logger.handlers) == 0

    def test_parse_log_level_valid_levels(self) -> None:
        """Test that all valid log level strings are parsed correctly."""
        # Test each valid level string
        assert LoggerFactory._parse_log_level("DEBUG") == logging.DEBUG
        assert LoggerFactory._parse_log_level("INFO") == logging.INFO
        assert LoggerFactory._parse_log_level("WARNING") == logging.WARNING
        assert LoggerFactory._parse_log_level("ERROR") == logging.ERROR
        assert LoggerFactory._parse_log_level("CRITICAL") == logging.CRITICAL

    def test_parse_log_level_case_insensitive(self) -> None:
        """Test that log level parsing is case-insensitive."""
        # Test lowercase input
        assert LoggerFactory._parse_log_level("debug") == logging.DEBUG

        # Test mixed case input
        assert LoggerFactory._parse_log_level("Info") == logging.INFO

    def test_parse_log_level_invalid_defaults_to_info(self) -> None:
        """Test that unrecognized level strings default to INFO."""
        # Test with an invalid level string
        result = LoggerFactory._parse_log_level("INVALID_LEVEL")

        # Should default to INFO
        assert result == logging.INFO
