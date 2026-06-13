# Unit tests for the data loader module.
"""
Tests for DataLoader - Validates URL validation, caching, and security controls.
"""

import os  # OS operations for temp file handling
import tempfile  # Temporary file creation
from pathlib import Path  # Path manipulation
from unittest.mock import patch  # Mocking for network isolation

import pandas as pd  # DataFrame for test assertions
import pytest  # Testing framework

from src.utils.logger import LoggerFactory  # Logger for test setup


class TestDataLoader:
    """Test suite for the DataLoader class."""

    def setup_method(self) -> None:
        """Set up test fixtures with security-configured config."""
        LoggerFactory.reset()
        self._temp_dir = tempfile.mkdtemp()
        self._cache_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: WARNING\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "data:\n"
            f"  cache_enabled: true\n"
            f"  cache_directory: '{self._cache_dir}'\n"
            "  request_timeout_seconds: 5\n"
            "  max_retries: 2\n"
            "security:\n"
            "  ssl_verify: true\n"
            "  allowed_domains:\n"
            "    - raw.githubusercontent.com\n"
            "    - example.com\n"
            "  max_download_size_mb: 10\n",
            encoding="utf-8",
        )
        from src.utils.config_loader import get_config
        get_config(config_dir=self._temp_dir)

    def teardown_method(self) -> None:
        """Clean up temporary files and directories."""
        LoggerFactory.reset()
        # Clean up cache directory contents
        cache_path = Path(self._cache_dir)
        if cache_path.exists():
            for file in cache_path.iterdir():
                file.unlink()
            cache_path.rmdir()
        # Clean up config
        if self._config_path.exists():
            self._config_path.unlink()
        os.rmdir(self._temp_dir)

    def test_validate_url_allowed_domain(self) -> None:
        """Test that allowed domains pass URL validation."""
        from src.utils.data_loader import DataLoader

        # Create loader and validate an allowed URL
        loader = DataLoader()

        # Should not raise for allowed domain
        loader._validate_url("https://raw.githubusercontent.com/test/data.csv")

    def test_validate_url_blocked_domain_raises_error(self) -> None:
        """Test that blocked domains raise SecurityViolationError."""
        from src.utils.data_loader import DataLoader, SecurityViolationError

        # Create loader
        loader = DataLoader()

        # Should raise for non-whitelisted domain
        with pytest.raises(SecurityViolationError):
            loader._validate_url("https://malicious-site.com/data.csv")

    def test_get_cache_path_deterministic(self) -> None:
        """Test that cache paths are deterministic for the same URL."""
        from src.utils.data_loader import DataLoader

        # Create loader
        loader = DataLoader()

        # Get cache path for same URL twice
        path_1 = loader._get_cache_path("https://example.com/data.csv")
        path_2 = loader._get_cache_path("https://example.com/data.csv")

        # Verify paths are identical
        assert path_1 == path_2

    def test_get_cache_path_different_urls_different_paths(self) -> None:
        """Test that different URLs produce different cache paths."""
        from src.utils.data_loader import DataLoader

        # Create loader
        loader = DataLoader()

        # Get cache paths for different URLs
        path_1 = loader._get_cache_path("https://example.com/data1.csv")
        path_2 = loader._get_cache_path("https://example.com/data2.csv")

        # Verify paths are different
        assert path_1 != path_2

    def test_cache_write_and_read(self) -> None:
        """Test that writing to cache and reading back produces same data."""
        from src.utils.data_loader import DataLoader

        # Create loader
        loader = DataLoader()

        # Create test DataFrame
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Write to cache
        cache_path = Path(self._cache_dir) / "test_cache.csv"
        loader._write_to_cache(test_data, cache_path)

        # Verify cache file exists
        assert cache_path.exists()

        # Read from cache
        cached_data = loader._read_cached_file(cache_path)

        # Verify data matches
        assert len(cached_data) == 3
        assert list(cached_data.columns) == ["col1", "col2"]

    @patch("pandas.read_csv")
    def test_load_csv_uses_cache_on_second_call(self, mock_read_csv) -> None:
        """Test that cached data is used on subsequent calls."""
        from src.utils.data_loader import DataLoader

        # Set up mock to return test data
        test_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_read_csv.return_value = test_data

        # Create loader
        loader = DataLoader()

        # First call should download (call read_csv)
        url = "https://example.com/dataset.csv"
        result_1 = loader.load_csv(url)

        # Verify read_csv was called for download
        assert mock_read_csv.called

        # Reset mock
        mock_read_csv.reset_mock()

        # Second call should use cache (not call read_csv for download)
        result_2 = loader.load_csv(url)

        # Read_csv is called for cache read too, so check the URL wasn't passed
        # The cache file path is what would be passed on cache hit
        if mock_read_csv.called:
            # If called, it should be with the cache path, not the URL
            call_args = mock_read_csv.call_args[0][0]
            assert call_args != url

    @patch("pandas.read_csv")
    def test_download_with_retry_all_attempts_fail(self, mock_read_csv) -> None:
        """Test that DataLoadError is raised after all retries fail."""
        from src.utils.data_loader import DataLoader, DataLoadError

        # Set up mock to always raise
        mock_read_csv.side_effect = Exception("Network error")

        # Create loader
        loader = DataLoader()

        # Disable cache to force download attempt
        loader._cache_enabled = False

        # Should raise after max_retries attempts
        with pytest.raises(DataLoadError):
            loader.load_csv("https://example.com/data.csv")

    def test_load_csv_invalid_domain_raises_security_error(self) -> None:
        """Test that load_csv validates URL domain before downloading."""
        from src.utils.data_loader import DataLoader, SecurityViolationError

        # Create loader
        loader = DataLoader()

        # Should raise SecurityViolationError for blocked domain
        with pytest.raises(SecurityViolationError):
            loader.load_csv("https://evil-domain.com/data.csv")
