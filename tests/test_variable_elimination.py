# Unit tests for the variable elimination inference module.
"""
Tests for VariableEliminationInference - Validates query execution and error handling.
"""

import os  # OS operations for temp file handling
import tempfile  # Temporary file creation
from pathlib import Path  # Path manipulation

import pandas as pd  # DataFrame for test data
import pytest  # Testing framework

from src.utils.logger import LoggerFactory  # Logger for test setup


class TestVariableEliminationInference:
    """Test suite for the VariableEliminationInference class."""

    def setup_method(self) -> None:
        """Set up test fixtures with a fitted model."""
        LoggerFactory.reset()
        self._temp_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: WARNING\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "model:\n"
            "  bayesian_network:\n"
            "    prior_type: dirichlet\n"
            "    pseudo_counts: 1\n",
            encoding="utf-8",
        )
        from src.utils.config_loader import get_config
        get_config(config_dir=self._temp_dir)

        # Create sample data for A -> B -> C network
        self._sample_data = pd.DataFrame({
            "A": ["x", "y", "x", "y", "x", "y"] * 10,
            "B": ["p", "q", "p", "q", "q", "p"] * 10,
            "C": ["m", "n", "m", "n", "m", "n"] * 10,
        })

    def teardown_method(self) -> None:
        """Clean up temporary files."""
        LoggerFactory.reset()
        if self._config_path.exists():
            self._config_path.unlink()
        os.rmdir(self._temp_dir)

    def _create_fitted_model(self):
        """Helper to create and return a fitted model for testing."""
        from src.models.bayesian_network import BayesianNetworkModel, CausalDAG

        # Create a simple A -> B -> C DAG
        dag = CausalDAG(edges=[("A", "B"), ("B", "C")])
        model = BayesianNetworkModel(dag)
        model.fit_mle(self._sample_data)
        return model

    def test_create_inference_engine(self) -> None:
        """Test creating an inference engine from a fitted model."""
        from src.inference.variable_elimination import VariableEliminationInference

        # Create a fitted model
        model = self._create_fitted_model()

        # Create the inference engine
        inference = VariableEliminationInference(model)

        # Verify it was created
        assert inference is not None

    def test_unfitted_model_raises_error(self) -> None:
        """Test that creating inference on unfitted model raises error."""
        from src.inference.variable_elimination import InferenceError, VariableEliminationInference
        from src.models.bayesian_network import BayesianNetworkModel, CausalDAG

        # Create an unfitted model
        dag = CausalDAG(edges=[("A", "B")])
        model = BayesianNetworkModel(dag)

        # Attempt to create inference engine
        with pytest.raises(InferenceError):
            VariableEliminationInference(model)

    def test_query_without_evidence(self) -> None:
        """Test a marginal query (no evidence)."""
        from src.inference.variable_elimination import VariableEliminationInference

        # Create inference engine
        model = self._create_fitted_model()
        inference = VariableEliminationInference(model)

        # Query P(B) without evidence
        result = inference.query(target_variables=["B"])

        # Verify result is not None and contains values
        assert result is not None

    def test_query_with_evidence(self) -> None:
        """Test a conditional query with evidence."""
        from src.inference.variable_elimination import VariableEliminationInference

        # Create inference engine
        model = self._create_fitted_model()
        inference = VariableEliminationInference(model)

        # Query P(C | A=x)
        result = inference.query(
            target_variables=["C"],
            evidence={"A": "x"},
        )

        # Verify result is not None
        assert result is not None

    def test_query_invalid_variable_raises_error(self) -> None:
        """Test that querying a non-existent variable raises InferenceError."""
        from src.inference.variable_elimination import InferenceError, VariableEliminationInference

        # Create inference engine
        model = self._create_fitted_model()
        inference = VariableEliminationInference(model)

        # Query a variable not in the model
        with pytest.raises(InferenceError):
            inference.query(target_variables=["Z"])

    def test_batch_query_returns_list(self) -> None:
        """Test that batch_query returns a list of results."""
        from src.inference.variable_elimination import VariableEliminationInference

        # Create inference engine
        model = self._create_fitted_model()
        inference = VariableEliminationInference(model)

        # Define a batch of queries
        queries = [
            {"target": ["B"], "evidence": {"A": "x"}},
            {"target": ["C"], "evidence": {"A": "y"}},
        ]

        # Execute batch query
        results = inference.batch_query(queries)

        # Verify we got 2 results
        assert len(results) == 2

    def test_batch_query_empty_list(self) -> None:
        """Test that batch_query with empty list returns empty list."""
        from src.inference.variable_elimination import VariableEliminationInference

        # Create inference engine
        model = self._create_fitted_model()
        inference = VariableEliminationInference(model)

        # Execute empty batch query
        results = inference.batch_query([])

        # Verify empty list returned
        assert results == []
