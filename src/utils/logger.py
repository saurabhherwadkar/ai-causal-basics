# Centralized logging module with configurable levels and output targets.
"""
Logger - Provides structured logging with file rotation and console output.

Log level is configurable via config.yaml, supporting DEBUG, INFO,
WARNING, ERROR, and CRITICAL levels. Supports both file and console output
with automatic log rotation to manage disk space.
"""

import logging  # Python standard library logging framework
import sys  # System-specific parameters for console output configuration
from logging.handlers import RotatingFileHandler  # File handler with size-based rotation
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Optional  # Type hints for optional parameters

from src.utils.config_loader import get_config  # Application configuration access


class LoggerFactory:
    """
    Factory class for creating and managing application loggers.

    Creates loggers with consistent formatting, configurable levels,
    and dual output (file + console) based on application settings.
    Implements the singleton pattern per logger name to avoid duplicate handlers.
    """

    # Class-level registry to track initialized loggers and prevent duplicates
    _initialized_loggers: dict = {}

    @classmethod
    def get_logger(cls, name: str, config_dir: Optional[str] = None) -> logging.Logger:
        """
        Create or retrieve a named logger with configured handlers.

        Args:
            name: The logger name, typically the module's __name__.
            config_dir: Optional config directory path override.

        Returns:
            A configured logging.Logger instance ready for use.
        """
        # Return existing logger if already initialized to prevent duplicate handlers
        if name in cls._initialized_loggers:
            return cls._initialized_loggers[name]  # Return cached logger instance

        # Load logging configuration from the application config file
        config = get_config(config_dir)  # Get the config loader singleton
        log_config = config.get_section("logging")  # Get the logging section

        # Extract logging parameters with sensible defaults
        log_level_str = log_config.get("level", "INFO")  # Default to INFO level
        log_format = log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")  # Timestamp format
        file_enabled = log_config.get("file_enabled", True)  # File logging toggle
        file_path = log_config.get("file_path", "logs/application.log")  # Log file path
        max_size_mb = log_config.get("max_file_size_mb", 10)  # Max file size in megabytes
        backup_count = log_config.get("backup_count", 5)  # Number of backup files to keep
        console_enabled = log_config.get("console_enabled", True)  # Console output toggle

        # Convert the string log level to a logging module constant
        log_level = cls._parse_log_level(log_level_str)

        # Create the logger instance with the specified name
        logger = logging.getLogger(name)

        # Set the logger's threshold level (messages below this are ignored)
        logger.setLevel(log_level)

        # Create a formatter instance for consistent message formatting
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

        # Add file handler if file logging is enabled
        if file_enabled:
            file_handler = cls._create_file_handler(
                file_path, max_size_mb, backup_count, formatter, log_level
            )
            logger.addHandler(file_handler)  # Attach file handler to logger

        # Add console handler if console logging is enabled
        if console_enabled:
            console_handler = cls._create_console_handler(formatter, log_level)
            logger.addHandler(console_handler)  # Attach console handler to logger

        # Prevent log messages from propagating to the root logger
        logger.propagate = False

        # Register this logger as initialized to prevent duplicate setup
        cls._initialized_loggers[name] = logger

        return logger  # Return the fully configured logger

    @classmethod
    def _parse_log_level(cls, level_string: str) -> int:
        """
        Convert a string log level name to its numeric logging constant.

        Args:
            level_string: Log level name (e.g., 'DEBUG', 'INFO', 'WARNING').

        Returns:
            The integer logging level constant.
        """
        # Define mapping from string names to logging constants
        level_mapping = {
            "DEBUG": logging.DEBUG,        # Most verbose, for development diagnostics
            "INFO": logging.INFO,          # General operational messages
            "WARNING": logging.WARNING,    # Potential issues that may need attention
            "ERROR": logging.ERROR,        # Errors that prevent specific operations
            "CRITICAL": logging.CRITICAL,  # Severe errors that may halt the application
        }

        # Normalize the input string to uppercase for case-insensitive matching
        normalized_level = level_string.upper().strip()

        # Return the mapped level, defaulting to INFO if the string is unrecognized
        return level_mapping.get(normalized_level, logging.INFO)

    @classmethod
    def _create_file_handler(
        cls,
        file_path: str,
        max_size_mb: int,
        backup_count: int,
        formatter: logging.Formatter,
        log_level: int,
    ) -> RotatingFileHandler:
        """
        Create a rotating file handler for writing logs to disk.

        Args:
            file_path: Path to the log output file.
            max_size_mb: Maximum file size in megabytes before rotation.
            backup_count: Number of rotated backup files to retain.
            formatter: Formatter instance for message formatting.
            log_level: Minimum log level for this handler.

        Returns:
            A configured RotatingFileHandler instance.
        """
        # Resolve the file path relative to the project structure
        log_file_path = Path(file_path)

        # Create the parent directory if it doesn't exist
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert max size from megabytes to bytes for the handler
        max_bytes = max_size_mb * 1024 * 1024

        # Create the rotating file handler with size limit and backup count
        file_handler = RotatingFileHandler(
            filename=str(log_file_path),  # Path to the log file
            maxBytes=max_bytes,           # Maximum bytes before rotation
            backupCount=backup_count,     # Number of backup files to keep
            encoding="utf-8",            # Use UTF-8 encoding for log files
        )

        # Apply the formatter to the file handler
        file_handler.setFormatter(formatter)

        # Set the minimum log level for file output
        file_handler.setLevel(log_level)

        return file_handler  # Return the configured file handler

    @classmethod
    def _create_console_handler(
        cls, formatter: logging.Formatter, log_level: int
    ) -> logging.StreamHandler:
        """
        Create a stream handler for writing logs to console (stdout).

        Args:
            formatter: Formatter instance for message formatting.
            log_level: Minimum log level for this handler.

        Returns:
            A configured StreamHandler instance writing to stdout.
        """
        # Create a stream handler that writes to standard output
        console_handler = logging.StreamHandler(stream=sys.stdout)

        # Apply the formatter to the console handler
        console_handler.setFormatter(formatter)

        # Set the minimum log level for console output
        console_handler.setLevel(log_level)

        return console_handler  # Return the configured console handler

    @classmethod
    def reset(cls) -> None:
        """
        Reset all initialized loggers by removing their handlers.

        Used primarily in testing to ensure clean logger state between tests.
        """
        # Iterate through all registered loggers and clear their handlers
        for logger_name, logger in cls._initialized_loggers.items():
            logger.handlers.clear()  # Remove all handlers from the logger

        # Clear the registry of initialized loggers
        cls._initialized_loggers.clear()
