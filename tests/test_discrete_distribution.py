# Unit tests for the discrete probability distribution module.
"""
Tests for DiscreteDistribution - Validates creation, normalization, and validation.
"""

import os  # OS operations for temp file handling
import tempfile  # Temporary file creation
from pathlib import Path  # Path manipulation

import pytest  # Testing framework

from src.utils.logger import LoggerFactory  # Logger for test setup


class TestDiscreteDistribution:
    """Test suite for the DiscreteDistribution class."""

    def setup_method(self) -> None:
        """Set up test fixtures with a minimal config for logging."""
        # Reset loggers to ensure clean state
        LoggerFactory.reset()

        # Create a temporary config directory
        self._temp_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: WARNING\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "probability:\n"
            "  default_variable_name: X\n"
            "  default_cardinality: 3\n"
            "  default_values: [0.45, 0.30, 0.25]\n"
            "  default_state_names: ['1', '2', '3']\n",
            encoding="utf-8",
        )

        # Set the config directory for imports
        os.environ["CAUSAL_CONFIG_DIR"] = self._temp_dir

        # Import after config is set up to avoid import-time config loading issues
        from src.utils.config_loader import get_config
        # Force reload of config with our test directory
        get_config(config_dir=self._temp_dir)

    def teardown_method(self) -> None:
        """Clean up temporary files."""
        LoggerFactory.reset()
        if self._config_path.exists():
            self._config_path.unlink()
        os.rmdir(self._temp_dir)
        # Remove environment variable
        os.environ.pop("CAUSAL_CONFIG_DIR", None)

    def test_create_valid_distribution(self) -> None:
        """Test creating a distribution with valid parameters."""
        from src.models.discrete_distribution import DiscreteDistribution

        # Create a valid distribution
        dist = DiscreteDistribution(
            variables=["X"],
            cardinality=[3],
            values=[0.45, 0.30, 0.25],
            state_names={"X": ["1", "2", "3"]},
        )

        # Verify the distribution was created
        assert dist is not None
        assert dist.variables == ["X"]

    def test_get_values_returns_correct_probabilities(self) -> None:
        """Test that get_values returns the original probability values."""
        from src.models.discrete_distribution import DiscreteDistribution

        # Create a distribution with known values
        dist = DiscreteDistribution(
            variables=["X"],
            cardinality=[3],
            values=[0.45, 0.30, 0.25],
            state_names={"X": ["1", "2", "3"]},
        )

        # Get the values
        values = dist.get_values()

        # Verify values match (with floating point tolerance)
        assert abs(values[0] - 0.45) < 1e-6
        assert abs(values[1] - 0.30) < 1e-6
        assert abs(values[2] - 0.25) < 1e-6

    def test_normalize_unnormalized_values(self) -> None:
        """Test that normalization converts phi values to probabilities summing to 1."""
        from src.models.discrete_distribution import DiscreteDistribution

        # Create a distribution with unnormalized values
        dist = DiscreteDistribution(
            variables=["X"],
            cardinality=[3],
            values=[45, 30, 25],
            state_names={"X": ["1", "2", "3"]},
        )

        # Normalize the distribution
        dist.normalize()

        # Get the normalized values
        values = dist.get_values()

        # Verify values sum to 1.0
        assert abs(sum(values) - 1.0) < 1e-6

        # Verify individual normalized values
        assert abs(values[0] - 0.45) < 1e-6
        assert abs(values[1] - 0.30) < 1e-6
        assert abs(values[2] - 0.25) < 1e-6

    def test_normalize_returns_self_for_chaining(self) -> None:
        """Test that normalize() returns self for method chaining."""
        from src.models.discrete_distribution import DiscreteDistribution

        # Create a distribution
        dist = DiscreteDistribution(
            variables=["X"],
            cardinality=[3],
            values=[45, 30, 25],
        )

        # Verify normalize returns self
        result = dist.normalize()
        assert result is dist

    def test_empty_variables_raises_error(self) -> None:
        """Test that empty variables list raises DiscreteDistributionError."""
        from src.models.discrete_distribution import (
            DiscreteDistribution,
            DiscreteDistributionError,
        )

        # Attempt to create with empty variables
        with pytest.raises(DiscreteDistributionError):
            DiscreteDistribution(
                variables=[],
                cardinality=[],
                values=[],
            )

    def test_mismatched_cardinality_raises_error(self) -> None:
        """Test that mismatched cardinality and variables lengths raise error."""
        from src.models.discrete_distribution import (
            DiscreteDistribution,
            DiscreteDistributionError,
        )

        # Attempt to create with mismatched lengths
        with pytest.raises(DiscreteDistributionError):
            DiscreteDistribution(
                variables=["X", "Y"],  # 2 variables
                cardinality=[3],        # Only 1 cardinality
                values=[0.5, 0.5, 0.5],
            )

    def test_wrong_values_count_raises_error(self) -> None:
        """Test that incorrect number of values raises error."""
        from src.models.discrete_distribution import (
            DiscreteDistribution,
            DiscreteDistributionError,
        )

        # Attempt to create with wrong number of values (need 3, provide 2)
        with pytest.raises(DiscreteDistributionError):
            DiscreteDistribution(
                variables=["X"],
                cardinality=[3],
                values=[0.5, 0.5],  # Should be 3 values
            )

    def test_negative_values_raises_error(self) -> None:
        """Test that negative probability values raise error."""
        from src.models.discrete_distribution import (
            DiscreteDistribution,
            DiscreteDistributionError,
        )

        # Attempt to create with a negative value
        with pytest.raises(DiscreteDistributionError):
            DiscreteDistribution(
                variables=["X"],
                cardinality=[3],
                values=[0.5, -0.1, 0.6],  # Negative value
            )

    def test_invalid_state_names_count_raises_error(self) -> None:
        """Test that mismatched state names count raises error."""
        from src.models.discrete_distribution import (
            DiscreteDistribution,
            DiscreteDistributionError,
        )

        # Attempt to create with wrong number of state names
        with pytest.raises(DiscreteDistributionError):
            DiscreteDistribution(
                variables=["X"],
                cardinality=[3],
                values=[0.45, 0.30, 0.25],
                state_names={"X": ["a", "b"]},  # Should be 3 state names
            )

    def test_str_representation(self) -> None:
        """Test that __str__ returns a non-empty string representation."""
        from src.models.discrete_distribution import DiscreteDistribution

        # Create a distribution
        dist = DiscreteDistribution(
            variables=["X"],
            cardinality=[3],
            values=[0.45, 0.30, 0.25],
        )

        # Verify string representation is non-empty
        str_repr = str(dist)
        assert len(str_repr) > 0

    def test_factor_property_returns_discrete_factor(self) -> None:
        """Test that the factor property returns a pgmpy DiscreteFactor."""
        from pgmpy.factors.discrete import DiscreteFactor
        from src.models.discrete_distribution import DiscreteDistribution

        # Create a distribution
        dist = DiscreteDistribution(
            variables=["X"],
            cardinality=[3],
            values=[0.45, 0.30, 0.25],
        )

        # Verify the factor property type
        assert isinstance(dist.factor, DiscreteFactor)

    def test_multivariable_distribution(self) -> None:
        """Test creating a distribution over multiple variables."""
        from src.models.discrete_distribution import DiscreteDistribution

        # Create a joint distribution over X and Y (2x2 = 4 values)
        dist = DiscreteDistribution(
            variables=["X", "Y"],
            cardinality=[2, 2],
            values=[0.25, 0.25, 0.25, 0.25],
            state_names={"X": ["0", "1"], "Y": ["0", "1"]},
        )

        # Verify creation succeeded
        assert dist.variables == ["X", "Y"]
        assert len(dist.get_values()) == 4
