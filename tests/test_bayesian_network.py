# Unit tests for the Bayesian network model module.
"""
Tests for BayesianNetworkModel and CausalDAG - Validates DAG creation and fitting.
"""

import os  # OS operations for temp file handling
import tempfile  # Temporary file creation
from pathlib import Path  # Path manipulation

import pandas as pd  # DataFrame for test data creation
import pytest  # Testing framework

from src.utils.logger import LoggerFactory  # Logger for test setup


class TestCausalDAG:
    """Test suite for the CausalDAG class."""

    def setup_method(self) -> None:
        """Set up test fixtures with minimal config."""
        LoggerFactory.reset()
        self._temp_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: WARNING\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "dag:\n"
            "  transportation_edges:\n"
            "    - ['A', 'E']\n"
            "    - ['S', 'E']\n"
            "    - ['E', 'O']\n"
            "    - ['E', 'R']\n"
            "    - ['O', 'T']\n"
            "    - ['R', 'T']\n",
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

    def test_create_dag_with_valid_edges(self) -> None:
        """Test creating a DAG with a valid edge list."""
        from src.models.bayesian_network import CausalDAG

        # Create a simple DAG with two edges
        dag = CausalDAG(edges=[("A", "B"), ("B", "C")])

        # Verify edges are stored correctly
        assert len(dag.edges) == 2
        assert ("A", "B") in dag.edges

    def test_dag_extracts_all_variables(self) -> None:
        """Test that all unique variables are extracted from edges."""
        from src.models.bayesian_network import CausalDAG

        # Create a DAG with overlapping variables
        dag = CausalDAG(edges=[("A", "B"), ("B", "C"), ("A", "C")])

        # Verify all 3 unique variables are extracted
        assert dag.variables == {"A", "B", "C"}

    def test_empty_edges_raises_error(self) -> None:
        """Test that an empty edge list raises BayesianNetworkError."""
        from src.models.bayesian_network import BayesianNetworkError, CausalDAG

        # Attempt to create DAG with no edges
        with pytest.raises(BayesianNetworkError):
            CausalDAG(edges=[])

    def test_from_config_creates_dag(self) -> None:
        """Test that from_config creates a DAG from the config file."""
        from src.models.bayesian_network import CausalDAG

        # Create DAG from configuration
        dag = CausalDAG.from_config()

        # Verify edges were loaded from config
        assert len(dag.edges) == 6
        assert ("A", "E") in dag.edges
        assert ("R", "T") in dag.edges

    def test_edges_returns_copy(self) -> None:
        """Test that the edges property returns a copy, not the original."""
        from src.models.bayesian_network import CausalDAG

        # Create a DAG
        dag = CausalDAG(edges=[("A", "B")])

        # Get edges and mutate
        edges = dag.edges
        edges.append(("C", "D"))

        # Verify original is unchanged
        assert len(dag.edges) == 1

    def test_variables_returns_copy(self) -> None:
        """Test that the variables property returns a copy."""
        from src.models.bayesian_network import CausalDAG

        # Create a DAG
        dag = CausalDAG(edges=[("A", "B")])

        # Get variables and mutate
        variables = dag.variables
        variables.add("Z")

        # Verify original is unchanged
        assert "Z" not in dag.variables


class TestBayesianNetworkModel:
    """Test suite for the BayesianNetworkModel class."""

    def setup_method(self) -> None:
        """Set up test fixtures with minimal config and sample data."""
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

        # Create sample data for a simple A -> B network
        self._sample_data = pd.DataFrame({
            "A": ["yes", "no", "yes", "no", "yes", "no", "yes", "yes"],
            "B": ["high", "low", "high", "low", "high", "low", "low", "high"],
        })

    def teardown_method(self) -> None:
        """Clean up temporary files."""
        LoggerFactory.reset()
        if self._config_path.exists():
            self._config_path.unlink()
        os.rmdir(self._temp_dir)

    def test_create_model_from_dag(self) -> None:
        """Test creating a BN model from a CausalDAG."""
        from src.models.bayesian_network import BayesianNetworkModel, CausalDAG

        # Create a simple DAG
        dag = CausalDAG(edges=[("A", "B")])

        # Create the model
        model = BayesianNetworkModel(dag)

        # Verify model was created and is not fitted
        assert model is not None
        assert model.is_fitted is False

    def test_fit_mle_marks_model_as_fitted(self) -> None:
        """Test that MLE fitting marks the model as fitted."""
        from src.models.bayesian_network import BayesianNetworkModel, CausalDAG

        # Create model and fit with MLE
        dag = CausalDAG(edges=[("A", "B")])
        model = BayesianNetworkModel(dag)
        model.fit_mle(self._sample_data)

        # Verify model is marked as fitted
        assert model.is_fitted is True

    def test_fit_mle_produces_cpds(self) -> None:
        """Test that MLE fitting produces valid CPDs."""
        from src.models.bayesian_network import BayesianNetworkModel, CausalDAG

        # Create and fit model
        dag = CausalDAG(edges=[("A", "B")])
        model = BayesianNetworkModel(dag)
        model.fit_mle(self._sample_data)

        # Get CPDs
        cpds = model.get_cpds()

        # Verify CPDs were created (one per variable)
        assert len(cpds) == 2

    def test_fit_bayesian_with_defaults(self) -> None:
        """Test Bayesian fitting uses config defaults."""
        from src.models.bayesian_network import BayesianNetworkModel, CausalDAG

        # Create and fit with Bayesian estimation
        dag = CausalDAG(edges=[("A", "B")])
        model = BayesianNetworkModel(dag)
        model.fit_bayesian(self._sample_data)

        # Verify model is fitted
        assert model.is_fitted is True

    def test_fit_bayesian_with_custom_params(self) -> None:
        """Test Bayesian fitting with explicit parameters."""
        from src.models.bayesian_network import BayesianNetworkModel, CausalDAG

        # Create and fit with custom parameters
        dag = CausalDAG(edges=[("A", "B")])
        model = BayesianNetworkModel(dag)
        model.fit_bayesian(
            self._sample_data,
            prior_type="dirichlet",
            pseudo_counts=2,
        )

        # Verify model is fitted
        assert model.is_fitted is True

    def test_get_cpds_before_fitting_raises_error(self) -> None:
        """Test that accessing CPDs before fitting raises an error."""
        from src.models.bayesian_network import BayesianNetworkModel, BayesianNetworkError, CausalDAG

        # Create model without fitting
        dag = CausalDAG(edges=[("A", "B")])
        model = BayesianNetworkModel(dag)

        # Attempt to get CPDs without fitting
        with pytest.raises(BayesianNetworkError):
            model.get_cpds()

    def test_model_property_returns_pgmpy_network(self) -> None:
        """Test that the model property returns a pgmpy BayesianNetwork."""
        from pgmpy.models import BayesianNetwork as PgmpyBN
        from src.models.bayesian_network import BayesianNetworkModel, CausalDAG

        # Create model
        dag = CausalDAG(edges=[("A", "B")])
        model = BayesianNetworkModel(dag)

        # Verify model property type
        assert isinstance(model.model, PgmpyBN)


class TestLatentVariableModel:
    """Test suite for the LatentVariableModel class."""

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
            "  latent:\n"
            "    expectation_maximization:\n"
            "      latent_cardinality: 2\n",
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

    def test_create_latent_model_with_valid_latents(self) -> None:
        """Test creating a latent variable model with valid latent set."""
        from src.models.bayesian_network import CausalDAG, LatentVariableModel

        # Create DAG with A -> B -> C, B is latent
        dag = CausalDAG(edges=[("A", "B"), ("B", "C")])
        model = LatentVariableModel(dag, latent_variables={"B"})

        # Verify model was created
        assert model is not None

    def test_invalid_latent_variable_raises_error(self) -> None:
        """Test that specifying a latent not in the DAG raises an error."""
        from src.models.bayesian_network import BayesianNetworkError, CausalDAG, LatentVariableModel

        # Create DAG with A -> B
        dag = CausalDAG(edges=[("A", "B")])

        # Attempt to create with latent "Z" which is not in the DAG
        with pytest.raises(BayesianNetworkError):
            LatentVariableModel(dag, latent_variables={"Z"})
