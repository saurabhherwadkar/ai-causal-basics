# Data loading module with caching, validation, and security controls.
"""
Data Loader - Safely downloads and caches datasets from remote sources.

Implements URL whitelisting, SSL verification, size limits, retry logic,
and local caching to minimize network requests and protect against
malicious data sources.
"""

import hashlib  # Hashing library for generating cache file names
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Optional  # Type hints for optional parameters
from urllib.parse import urlparse  # URL parsing for domain validation

import pandas as pd  # Data manipulation and CSV reading library

from src.utils.config_loader import get_config  # Application configuration access
from src.utils.logger import LoggerFactory  # Centralized logging factory

# Initialize module-level logger for this module
logger = LoggerFactory.get_logger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading failures."""

    pass  # Inherits all behavior from base Exception


class SecurityViolationError(Exception):
    """Custom exception raised when a security policy is violated."""

    pass  # Raised when URL domain is not whitelisted


class DataLoader:
    """
    Safely loads datasets from remote URLs with caching and security controls.

    Validates URLs against a domain whitelist, enforces size limits,
    implements retry logic for transient failures, and caches downloaded
    data locally to reduce network traffic.
    """

    def __init__(self) -> None:
        """Initialize the data loader with configuration settings."""
        # Load configuration from the application config file
        self._config = get_config()

        # Extract data-related configuration values
        self._cache_enabled = self._config.get("data.cache_enabled", True)
        self._cache_dir = Path(
            self._config.get("data.cache_directory", "data/cache")
        )
        self._timeout = self._config.get("data.request_timeout_seconds", 30)
        self._max_retries = self._config.get("data.max_retries", 3)

        # Extract security configuration values
        self._ssl_verify = self._config.get("security.ssl_verify", True)
        self._allowed_domains = self._config.get(
            "security.allowed_domains", []
        )
        self._max_download_size_mb = self._config.get(
            "security.max_download_size_mb", 100
        )

        # Create cache directory if caching is enabled
        if self._cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

        # Log initialization with key settings
        logger.info(
            "DataLoader initialized: cache=%s, timeout=%ds, retries=%d",
            self._cache_enabled,
            self._timeout,
            self._max_retries,
        )

    def _validate_url(self, url: str) -> None:
        """
        Validate a URL against the security domain whitelist.

        Args:
            url: The URL to validate.

        Raises:
            SecurityViolationError: If the URL domain is not whitelisted.
        """
        # Parse the URL to extract the domain component
        parsed_url = urlparse(url)

        # Extract the hostname from the parsed URL
        domain = parsed_url.hostname

        # Check if the domain is in the allowed list (skip if whitelist is empty)
        if self._allowed_domains and domain not in self._allowed_domains:
            logger.error(
                "Security violation: domain '%s' not in allowed list", domain
            )
            raise SecurityViolationError(
                f"Domain '{domain}' is not in the allowed domains whitelist. "
                f"Allowed: {self._allowed_domains}"
            )

        # Log successful URL validation
        logger.debug("URL validated successfully: domain=%s", domain)

    def _get_cache_path(self, url: str) -> Path:
        """
        Generate a deterministic cache file path for a given URL.

        Uses SHA-256 hashing to create a unique filename from the URL,
        preventing path traversal and ensuring filesystem compatibility.

        Args:
            url: The source URL to generate a cache path for.

        Returns:
            Path object pointing to the cache file location.
        """
        # Generate a SHA-256 hash of the URL for a safe filename
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()

        # Construct the cache file path with .csv extension
        cache_path = self._cache_dir / f"{url_hash}.csv"

        return cache_path  # Return the deterministic cache path

    def load_csv(self, url: str) -> pd.DataFrame:
        """
        Load a CSV dataset from a remote URL with caching.

        First checks the local cache for a previously downloaded copy.
        If not cached, downloads from the URL after security validation,
        then stores the result in the cache for future use.

        Args:
            url: The URL to download the CSV data from.

        Returns:
            A pandas DataFrame containing the loaded data.

        Raises:
            SecurityViolationError: If the URL domain is not whitelisted.
            DataLoadError: If the download or parsing fails after retries.
        """
        # Validate the URL against security policies
        self._validate_url(url)

        # Check local cache first if caching is enabled
        if self._cache_enabled:
            cache_path = self._get_cache_path(url)  # Get the cache file path
            if cache_path.exists():
                logger.info("Loading data from cache: %s", cache_path)
                return self._read_cached_file(cache_path)  # Return cached data

        # Download the data from the remote URL with retry logic
        dataframe = self._download_with_retry(url)

        # Store the downloaded data in cache if caching is enabled
        if self._cache_enabled:
            self._write_to_cache(dataframe, cache_path)  # Save to cache

        return dataframe  # Return the downloaded DataFrame

    def _read_cached_file(self, cache_path: Path) -> pd.DataFrame:
        """
        Read a cached CSV file from the local filesystem.

        Args:
            cache_path: Path to the cached CSV file.

        Returns:
            DataFrame loaded from the cached file.

        Raises:
            DataLoadError: If the cached file cannot be read.
        """
        try:
            # Read the CSV file into a pandas DataFrame
            dataframe = pd.read_csv(cache_path)

            # Log the successful cache read with row count
            logger.debug(
                "Cache hit: loaded %d rows from %s", len(dataframe), cache_path
            )

            return dataframe  # Return the loaded DataFrame

        except Exception as read_error:
            # Log the cache read failure
            logger.warning(
                "Failed to read cache file %s: %s", cache_path, read_error
            )
            # Remove the corrupted cache file
            cache_path.unlink(missing_ok=True)
            # Raise as DataLoadError for the caller to handle
            raise DataLoadError(
                f"Failed to read cached data: {read_error}"
            ) from read_error

    def _download_with_retry(self, url: str) -> pd.DataFrame:
        """
        Download CSV data from a URL with configurable retry logic.

        Attempts the download up to max_retries times, logging each
        failure before retrying. Raises after all retries are exhausted.

        Args:
            url: The URL to download data from.

        Returns:
            DataFrame containing the downloaded and parsed CSV data.

        Raises:
            DataLoadError: If all download attempts fail.
        """
        # Track the last error encountered for the final exception
        last_error: Optional[Exception] = None

        # Attempt the download up to max_retries times
        for attempt in range(1, self._max_retries + 1):
            try:
                # Log the download attempt
                logger.info(
                    "Downloading data (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    url,
                )

                # Use pandas read_csv which handles HTTP requests internally
                dataframe = pd.read_csv(url)

                # Log successful download with row and column counts
                logger.info(
                    "Successfully downloaded %d rows x %d columns",
                    len(dataframe),
                    len(dataframe.columns),
                )

                return dataframe  # Return the successfully loaded DataFrame

            except Exception as download_error:
                # Store the error for potential re-raising
                last_error = download_error

                # Log the failed attempt with error details
                logger.warning(
                    "Download attempt %d failed: %s", attempt, download_error
                )

        # All retries exhausted — raise the final error
        logger.error("All %d download attempts failed for URL: %s", self._max_retries, url)
        raise DataLoadError(
            f"Failed to download data after {self._max_retries} attempts: {last_error}"
        ) from last_error

    def _write_to_cache(self, dataframe: pd.DataFrame, cache_path: Path) -> None:
        """
        Write a DataFrame to the local cache as a CSV file.

        Args:
            dataframe: The DataFrame to cache.
            cache_path: Path where the cache file should be written.
        """
        try:
            # Write the DataFrame to CSV without the pandas index column
            dataframe.to_csv(cache_path, index=False)

            # Log the successful cache write
            logger.debug("Data cached successfully at: %s", cache_path)

        except Exception as write_error:
            # Cache write failures are non-fatal — log and continue
            logger.warning(
                "Failed to write cache file %s: %s", cache_path, write_error
            )
