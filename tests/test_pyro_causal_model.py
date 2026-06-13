# Unit tests for the Pyro causal model module.
"""
Tests for TransportationCausalModel and ImportanceSamplingInference.
"""

import os  # OS operations for temp file handling
import tempfile  # Temporary file creation
from pathlib import Path  # Path manipulation

import pytest  # Testing framework
import torch  # PyTorch for tensor assertions

from src.utils.logger import LoggerFactory  # Logger for test setup


class TestTransportationCausalModel:
    """Test suite for the TransportationCausalModel class."""

    def setup_method(self) -> None:
        """Set up test fixtures with Pyro model config."""
        LoggerFactory.reset()
        self._temp_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: WARNING\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "model:\n"
            "  pyro:\n"
            "    num_samples: 100\n"
            "    random_seed: 42\n"
            "dag:\n"
            "  outcome_labels:\n"
            "    A: ['young', 'adult', 'old']\n"
            "    S: ['M', 'F']\n"
            "    E: ['high', 'uni']\n"
            "    O: ['emp', 'self']\n"
            "    R: ['small', 'big']\n"
            "    T: ['car', 'train', 'other']\n",
            encoding="utf-8",
        )
        from src.utils.config_loader import get_config
        get_config(config_dir=self._temp_dir)

    def teardown_method(self) -> None:
        """Clean up temporary files."""
        LoggerFactory.reset()
        if self._config_path.exists():
            self._config_path.unlink()
        os.rmdir(self._temp_dir)

    def test_model_initialization(self) -> None:
        """Test that the causal model initializes without errors."""
        from src.models.pyro_causal_model import TransportationCausalModel

        # Create the model
        model = TransportationCausalModel()

        # Verify model was created
        assert model is not None

    def test_model_has_correct_labels(self) -> None:
        """Test that the model loads correct outcome labels from config."""
        from src.models.pyro_causal_model import TransportationCausalModel

        # Create the model
        model = TransportationCausalModel()

        # Verify labels are loaded correctly
        assert model._age_labels == ["young", "adult", "old"]
        assert model._sex_labels == ["M", "F"]
        assert model._education_labels == ["high", "uni"]
        assert model._transport_labels == ["car", "train", "other"]

    def test_generative_model_returns_dict(self) -> None:
        """Test that the generative model returns a dictionary of samples."""
        from src.models.pyro_causal_model import TransportationCausalModel

        # Create and run the model
        model = TransportationCausalModel()
        result = model.generative_model()

        # Verify result is a dictionary with all expected keys
        assert isinstance(result, dict)
        assert "A" in result
        assert "S" in result
        assert "E" in result
        assert "O" in result
        assert "R" in result
        assert "T" in result

    def test_generative_model_returns_tensors(self) -> None:
        """Test that all sampled values are torch tensors."""
        from src.models.pyro_causal_model import TransportationCausalModel

        # Create and run the model
        model = TransportationCausalModel()
        result = model.generative_model()

        # Verify all values are tensors
        for key, value in result.items():
            assert isinstance(value, torch.Tensor)

    def test_generative_model_values_in_valid_range(self) -> None:
        """Test that sampled category indices are within valid ranges."""
        from src.models.pyro_causal_model import TransportationCausalModel

        # Create and run the model multiple times
        model = TransportationCausalModel()

        # Run multiple times to check consistency
        for _ in range(10):
            result = model.generative_model()

            # Age should be 0, 1, or 2
            assert 0 <= result["A"].item() <= 2

            # Sex should be 0 or 1
            assert 0 <= result["S"].item() <= 1

            # Education should be 0 or 1
            assert 0 <= result["E"].item() <= 1

            # Occupation should be 0 or 1
            assert 0 <= result["O"].item() <= 1

            # Residence should be 0 or 1
            assert 0 <= result["R"].item() <= 1

            # Transportation should be 0, 1, or 2
            assert 0 <= result["T"].item() <= 2

    def test_probability_tables_valid(self) -> None:
        """Test that all probability tables sum to 1 along the correct axis."""
        from src.models.pyro_causal_model import TransportationCausalModel

        # Create model
        model = TransportationCausalModel()

        # Verify age probabilities sum to 1
        assert abs(model._age_probs.sum().item() - 1.0) < 1e-6

        # Verify sex probabilities sum to 1
        assert abs(model._sex_probs.sum().item() - 1.0) < 1e-6

        # Verify occupation probabilities sum to 1 for each education level
        for edu_level in range(2):
            row_sum = model._occupation_probs[edu_level].sum().item()
            assert abs(row_sum - 1.0) < 1e-6

        # Verify residence probabilities sum to 1 for each education level
        for edu_level in range(2):
            row_sum = model._residence_probs[edu_level].sum().item()
            assert abs(row_sum - 1.0) < 1e-6


class TestImportanceSamplingInference:
    """Test suite for the ImportanceSamplingInference class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        LoggerFactory.reset()
        self._temp_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: WARNING\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "model:\n"
            "  pyro:\n"
            "    num_samples: 200\n"
            "    random_seed: 42\n"
            "dag:\n"
            "  outcome_labels:\n"
            "    A: ['young', 'adult', 'old']\n"
            "    S: ['M', 'F']\n"
            "    E: ['high', 'uni']\n"
            "    O: ['emp', 'self']\n"
            "    R: ['small', 'big']\n"
            "    T: ['car', 'train', 'other']\n",
            encoding="utf-8",
        )
        from src.utils.config_loader import get_config
        get_config(config_dir=self._temp_dir)

    def teardown_method(self) -> None:
        """Clean up temporary files."""
        LoggerFactory.reset()
        if self._config_path.exists():
            self._config_path.unlink()
        os.rmdir(self._temp_dir)

    def test_inference_initialization(self) -> None:
        """Test that the inference engine initializes correctly."""
        from src.models.pyro_causal_model import (
            ImportanceSamplingInference,
            TransportationCausalModel,
        )

        # Create model and inference engine
        model = TransportationCausalModel()
        inference = ImportanceSamplingInference(model)

        # Verify initialization
        assert inference is not None
        assert inference._num_samples == 200

    def test_compute_posterior_returns_dict(self) -> None:
        """Test that compute_posterior returns a probability dictionary."""
        from src.models.pyro_causal_model import (
            ImportanceSamplingInference,
            TransportationCausalModel,
        )

        # Create model and inference
        model = TransportationCausalModel()
        inference = ImportanceSamplingInference(model)

        # Compute posterior P(Education | Transportation=train)
        posterior = inference.compute_posterior(
            target_variable="education",
            evidence={"transportation": 1.0},
            num_samples=100,  # Small for fast test
        )

        # Verify result is a dictionary
        assert isinstance(posterior, dict)

        # Verify probabilities are non-negative
        for prob in posterior.values():
            assert prob >= 0.0

        # Verify probabilities approximately sum to 1
        total = sum(posterior.values())
        assert abs(total - 1.0) < 0.05  # Allow some sampling error

    def test_compute_posterior_with_custom_samples(self) -> None:
        """Test that custom num_samples parameter is respected."""
        from src.models.pyro_causal_model import (
            ImportanceSamplingInference,
            TransportationCausalModel,
        )

        # Create model and inference
        model = TransportationCausalModel()
        inference = ImportanceSamplingInference(model)

        # Compute with very few samples (just to verify it runs)
        posterior = inference.compute_posterior(
            target_variable="education",
            evidence={"transportation": 0.0},
            num_samples=50,
        )

        # Verify we get a result
        assert len(posterior) > 0

    def test_get_label_map_known_variable(self) -> None:
        """Test that _get_label_map returns correct mappings for known variables."""
        from src.models.pyro_causal_model import (
            ImportanceSamplingInference,
            TransportationCausalModel,
        )

        # Create model and inference
        model = TransportationCausalModel()
        inference = ImportanceSamplingInference(model)

        # Get label map for education
        label_map = inference._get_label_map("education")

        # Verify mappings
        assert label_map[0] == "high"
        assert label_map[1] == "uni"

    def test_get_label_map_unknown_variable(self) -> None:
        """Test that _get_label_map returns empty dict for unknown variables."""
        from src.models.pyro_causal_model import (
            ImportanceSamplingInference,
            TransportationCausalModel,
        )

        # Create model and inference
        model = TransportationCausalModel()
        inference = ImportanceSamplingInference(model)

        # Get label map for unknown variable
        label_map = inference._get_label_map("nonexistent")

        # Verify empty dict
        assert label_map == {}
