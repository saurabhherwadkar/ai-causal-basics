# Variable Elimination inference module for exact probabilistic queries.
"""
Variable Elimination - Exact inference algorithm for Bayesian networks.

Performs conditional probability queries by systematically eliminating
(marginalizing over) variables not involved in the query. Efficient
for networks with moderate treewidth.
"""

from typing import Dict, List, Optional  # Type hints for method signatures

from pgmpy.inference import VariableElimination as PgmpyVE  # pgmpy VE implementation

from src.models.bayesian_network import BayesianNetworkModel  # Our BN model wrapper
from src.utils.logger import LoggerFactory  # Centralized logging factory

# Initialize module-level logger for this module
logger = LoggerFactory.get_logger(__name__)


class InferenceError(Exception):
    """Custom exception for inference operation failures."""

    pass  # Inherits all behavior from base Exception


class VariableEliminationInference:
    """
    Performs exact inference queries using the Variable Elimination algorithm.

    Variable Elimination works by marginalizing variables one at a time,
    multiplying factors and summing out irrelevant variables to compute
    the desired conditional probability P(query | evidence).
    """

    def __init__(self, bayesian_network: BayesianNetworkModel) -> None:
        """
        Initialize the inference engine with a fitted Bayesian network.

        Args:
            bayesian_network: A fitted BayesianNetworkModel instance.

        Raises:
            InferenceError: If the model has not been fitted.
        """
        # Validate that the model has been fitted before inference
        if not bayesian_network.is_fitted:
            logger.error("Cannot perform inference on unfitted model")
            raise InferenceError(
                "Model must be fitted before performing inference"
            )

        # Store reference to the Bayesian network model
        self._network = bayesian_network

        # Create the pgmpy VariableElimination inference object
        self._inference_engine = PgmpyVE(bayesian_network.model)

        # Log successful initialization
        logger.info("VariableEliminationInference engine initialized")

    def query(
        self,
        target_variables: List[str],
        evidence: Optional[Dict[str, str]] = None,
    ):
        """
        Compute the conditional probability P(target | evidence).

        Args:
            target_variables: List of variable names to compute the posterior for.
            evidence: Dictionary mapping observed variable names to their values.
                      None or empty dict means no conditioning (returns marginal).

        Returns:
            pgmpy DiscreteFactor representing the query result.

        Raises:
            InferenceError: If the query fails due to invalid variables or errors.
        """
        # Default evidence to empty dict if None
        if evidence is None:
            evidence = {}  # No conditioning evidence

        try:
            # Log the inference query being executed
            logger.info(
                "Executing query: P(%s | %s)",
                target_variables,
                evidence if evidence else "no evidence",
            )

            # Execute the variable elimination query
            query_result = self._inference_engine.query(
                variables=target_variables,  # Variables to compute posterior for
                evidence=evidence,            # Observed evidence for conditioning
            )

            # Log successful query completion
            logger.info("Query completed successfully")

            # Log the result at debug level for detailed inspection
            logger.debug("Query result:\n%s", query_result)

            return query_result  # Return the computed probability distribution

        except Exception as query_error:
            # Log the query failure with full error details
            logger.error(
                "Inference query failed for P(%s | %s): %s",
                target_variables,
                evidence,
                str(query_error),
            )
            # Wrap and re-raise as InferenceError
            raise InferenceError(
                f"Query P({target_variables} | {evidence}) failed: {query_error}"
            ) from query_error

    def batch_query(
        self,
        queries: List[Dict],
    ) -> List:
        """
        Execute multiple inference queries in sequence.

        Args:
            queries: List of query dictionaries, each containing:
                     - 'target': List of target variable names
                     - 'evidence': Dict of evidence (optional)

        Returns:
            List of query results in the same order as input queries.

        Raises:
            InferenceError: If any individual query fails.
        """
        # Initialize the results list
        results = []

        # Log the batch query operation
        logger.info("Executing batch query: %d queries", len(queries))

        # Execute each query in sequence
        for query_index, query_spec in enumerate(queries):
            # Extract target and evidence from the query specification
            target = query_spec.get("target", [])  # Target variables list
            evidence = query_spec.get("evidence", None)  # Evidence dict or None

            # Log individual query within the batch
            logger.debug(
                "Batch query %d/%d: P(%s | %s)",
                query_index + 1,
                len(queries),
                target,
                evidence,
            )

            # Execute the query and append result
            result = self.query(target_variables=target, evidence=evidence)
            results.append(result)  # Add result to the list

        # Log batch completion
        logger.info("Batch query completed: %d results", len(results))

        return results  # Return all query results
