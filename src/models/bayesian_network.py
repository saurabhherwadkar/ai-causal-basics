# Bayesian Network model definition, parameter learning, and structure management.
"""
Bayesian Network - Defines and learns parameters for causal DAG structures.

Provides classes for building Bayesian networks from edge lists,
fitting parameters using MLE, Bayesian estimation, and EM for latent variables.
Reads DAG structure from configuration for maintainability.
"""

from typing import Dict, List, Optional, Set, Tuple  # Type hints for signatures

import pandas as pd  # Data manipulation library for fitting from data
from pgmpy.estimators import BayesianEstimator  # Bayesian parameter estimation
from pgmpy.estimators import ExpectationMaximization  # EM for latent variables
from pgmpy.models import BayesianNetwork as PgmpyBayesianNetwork  # pgmpy BN class

from src.utils.config_loader import get_config  # Application configuration access
from src.utils.logger import LoggerFactory  # Centralized logging factory

# Initialize module-level logger for this module
logger = LoggerFactory.get_logger(__name__)


class BayesianNetworkError(Exception):
    """Custom exception for Bayesian network operations."""

    pass  # Inherits all behavior from base Exception


class CausalDAG:
    """
    Represents the structure of a causal Directed Acyclic Graph (DAG).

    Manages the edge list that defines parent-child causal relationships
    between variables. Validates that the structure is a valid DAG.
    """

    def __init__(self, edges: List[Tuple[str, str]]) -> None:
        """
        Initialize the causal DAG with a list of directed edges.

        Args:
            edges: List of (parent, child) tuples defining causal relationships.

        Raises:
            BayesianNetworkError: If the edge list is empty or invalid.
        """
        # Validate that the edge list is not empty
        if not edges:
            logger.error("Cannot create DAG with empty edge list")
            raise BayesianNetworkError("Edge list cannot be empty")

        # Store the validated edge list
        self._edges = edges

        # Extract unique variable names from all edges
        self._variables = self._extract_variables()

        # Log the DAG structure creation
        logger.info(
            "CausalDAG created with %d edges and %d variables",
            len(self._edges),
            len(self._variables),
        )

    def _extract_variables(self) -> Set[str]:
        """
        Extract all unique variable names from the edge list.

        Returns:
            Set of unique variable names appearing in any edge.
        """
        # Collect all variable names from both parent and child positions
        variables = set()  # Use a set to ensure uniqueness

        # Iterate through each edge tuple
        for parent, child in self._edges:
            variables.add(parent)  # Add the parent variable
            variables.add(child)   # Add the child variable

        # Log the extracted variables
        logger.debug("Extracted variables: %s", variables)

        return variables  # Return the set of unique variable names

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Return the list of directed edges in the DAG."""
        return self._edges.copy()  # Return copy to prevent external mutation

    @property
    def variables(self) -> Set[str]:
        """Return the set of all variable names in the DAG."""
        return self._variables.copy()  # Return copy to prevent external mutation

    @classmethod
    def from_config(cls) -> "CausalDAG":
        """
        Create a CausalDAG instance from the application configuration file.

        Reads the DAG edge definitions from config.yaml's dag section.

        Returns:
            A new CausalDAG instance with config-defined edges.
        """
        # Load the DAG edges from configuration
        config = get_config()
        edge_list = config.get("dag.transportation_edges", [])

        # Convert the config format (list of lists) to tuples
        edges = [(edge[0], edge[1]) for edge in edge_list]

        # Log the configuration-based creation
        logger.info("Creating CausalDAG from configuration: %d edges", len(edges))

        return cls(edges=edges)  # Return new instance from config edges


class BayesianNetworkModel:
    """
    Wraps pgmpy's BayesianNetwork for parameter learning and model management.

    Supports three parameter estimation methods:
    1. Maximum Likelihood Estimation (MLE) via fit()
    2. Bayesian Estimation with Dirichlet priors
    3. Expectation Maximization for latent (unobserved) variables
    """

    def __init__(self, dag: CausalDAG) -> None:
        """
        Initialize the Bayesian Network model with a DAG structure.

        Args:
            dag: A CausalDAG instance defining the network structure.
        """
        # Store the DAG structure reference
        self._dag = dag

        # Create the underlying pgmpy BayesianNetwork from the edge list
        self._model = PgmpyBayesianNetwork(dag.edges)

        # Track whether the model has been fitted with data
        self._is_fitted = False

        # Log model initialization
        logger.info("BayesianNetworkModel initialized with %d edges", len(dag.edges))

    def fit_mle(self, data: pd.DataFrame) -> None:
        """
        Fit model parameters using Maximum Likelihood Estimation.

        MLE estimates parameters by counting frequencies in the observed data.
        Simple and fast but can produce zero probabilities for unseen combinations.

        Args:
            data: DataFrame containing observed data with columns matching DAG variables.

        Raises:
            BayesianNetworkError: If fitting fails due to data issues.
        """
        try:
            # Log the fitting operation with data dimensions
            logger.info(
                "Fitting model with MLE: %d rows x %d columns",
                len(data),
                len(data.columns),
            )

            # Call pgmpy's fit method with default (MLE) estimator
            self._model.fit(data)

            # Mark the model as fitted
            self._is_fitted = True

            # Log successful fitting
            logger.info("MLE fitting completed successfully")

        except Exception as fit_error:
            # Log the fitting failure with error details
            logger.error("MLE fitting failed: %s", str(fit_error))
            raise BayesianNetworkError(
                f"Failed to fit model with MLE: {fit_error}"
            ) from fit_error

    def fit_bayesian(
        self,
        data: pd.DataFrame,
        prior_type: Optional[str] = None,
        pseudo_counts: Optional[int] = None,
    ) -> None:
        """
        Fit model parameters using Bayesian estimation with Dirichlet priors.

        Bayesian estimation adds pseudo-counts to prevent zero probabilities,
        providing smoother estimates especially with limited data.

        Args:
            data: DataFrame containing observed data.
            prior_type: Type of prior distribution (default from config: 'dirichlet').
            pseudo_counts: Dirichlet pseudo-count hyperparameter (default from config).

        Raises:
            BayesianNetworkError: If Bayesian fitting fails.
        """
        # Load defaults from configuration if not explicitly provided
        config = get_config()
        if prior_type is None:
            prior_type = config.get("model.bayesian_network.prior_type", "dirichlet")
        if pseudo_counts is None:
            pseudo_counts = config.get("model.bayesian_network.pseudo_counts", 1)

        try:
            # Log the Bayesian fitting operation
            logger.info(
                "Fitting model with Bayesian estimation: prior=%s, pseudo_counts=%d",
                prior_type,
                pseudo_counts,
            )

            # Fit using pgmpy's BayesianEstimator with specified prior
            self._model.fit(
                data,
                estimator=BayesianEstimator,     # Use Bayesian estimator class
                prior_type=prior_type,            # Dirichlet prior type
                pseudo_counts=pseudo_counts,      # Prior strength parameter
            )

            # Mark the model as fitted
            self._is_fitted = True

            # Log successful Bayesian fitting
            logger.info("Bayesian estimation completed successfully")

        except Exception as fit_error:
            # Log the fitting failure
            logger.error("Bayesian fitting failed: %s", str(fit_error))
            raise BayesianNetworkError(
                f"Failed to fit model with Bayesian estimation: {fit_error}"
            ) from fit_error

    def get_cpds(self) -> list:
        """
        Retrieve the learned Conditional Probability Distributions (CPDs).

        Returns:
            List of pgmpy TabularCPD objects representing the causal Markov kernels.

        Raises:
            BayesianNetworkError: If the model has not been fitted yet.
        """
        # Check that the model has been fitted before accessing CPDs
        if not self._is_fitted:
            logger.error("Cannot get CPDs: model has not been fitted")
            raise BayesianNetworkError(
                "Model must be fitted before accessing CPDs"
            )

        # Retrieve CPDs from the underlying pgmpy model
        cpds = self._model.get_cpds()

        # Log the number of CPDs retrieved
        logger.debug("Retrieved %d CPDs from fitted model", len(cpds))

        return cpds  # Return the list of CPD objects

    @property
    def model(self) -> PgmpyBayesianNetwork:
        """Return the underlying pgmpy BayesianNetwork instance."""
        return self._model  # Expose for inference operations

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted with data."""
        return self._is_fitted  # Read-only fitted status


class LatentVariableModel:
    """
    Bayesian Network with latent (unobserved) variables using EM estimation.

    Uses the Structural Expectation Maximization algorithm to learn
    parameters for variables that are not directly observed in the data.
    """

    def __init__(
        self,
        dag: CausalDAG,
        latent_variables: Set[str],
    ) -> None:
        """
        Initialize the latent variable model.

        Args:
            dag: A CausalDAG instance defining the network structure.
            latent_variables: Set of variable names that are unobserved.

        Raises:
            BayesianNetworkError: If latent variables are not in the DAG.
        """
        # Validate that all latent variables exist in the DAG
        invalid_latents = latent_variables - dag.variables
        if invalid_latents:
            logger.error("Latent variables not in DAG: %s", invalid_latents)
            raise BayesianNetworkError(
                f"Latent variables not found in DAG: {invalid_latents}"
            )

        # Store the DAG and latent variable configuration
        self._dag = dag
        self._latent_variables = latent_variables

        # Create the pgmpy model with latent variable specification
        self._model = PgmpyBayesianNetwork(
            dag.edges, latents=latent_variables
        )

        # Log model initialization
        logger.info(
            "LatentVariableModel created with latents: %s", latent_variables
        )

    def fit_em(
        self,
        data: pd.DataFrame,
        latent_cardinality: Optional[Dict[str, int]] = None,
    ) -> list:
        """
        Fit parameters using Expectation Maximization for latent variables.

        EM iteratively estimates the parameters of latent variables by
        alternating between expectation and maximization steps.

        Args:
            data: DataFrame containing observed variables only.
            latent_cardinality: Dict mapping latent variable names to their cardinality.
                               Defaults to config value if not provided.

        Returns:
            List of learned CPD parameters for all variables.

        Raises:
            BayesianNetworkError: If EM fitting fails.
        """
        # Load default latent cardinality from config if not provided
        if latent_cardinality is None:
            config = get_config()
            default_card = config.get(
                "model.latent.expectation_maximization.latent_cardinality", 2
            )
            # Build cardinality dict for all latent variables using default
            latent_cardinality = {
                var: default_card for var in self._latent_variables
            }

        try:
            # Log the EM fitting operation
            logger.info(
                "Running EM with latent cardinality: %s", latent_cardinality
            )

            # Create the EM estimator with model and observed data
            estimator = ExpectationMaximization(self._model, data)

            # Run EM to get learned parameters
            learned_cpds = estimator.get_parameters(
                latent_card=latent_cardinality
            )

            # Log successful EM completion
            logger.info("EM estimation completed: %d CPDs learned", len(learned_cpds))

            return learned_cpds  # Return the learned CPD list

        except Exception as em_error:
            # Log the EM failure with details
            logger.error("EM estimation failed: %s", str(em_error))
            raise BayesianNetworkError(
                f"EM estimation failed: {em_error}"
            ) from em_error
