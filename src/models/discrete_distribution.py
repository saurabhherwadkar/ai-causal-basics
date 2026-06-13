# Discrete probability distribution module using pgmpy DiscreteFactor.
"""
Discrete Distribution - Wrapper for creating and manipulating discrete probability distributions.

Provides a clean interface for defining probability distributions over
categorical variables, with support for normalization and validation.
Uses pgmpy's DiscreteFactor as the underlying representation.
"""

from typing import Dict, List, Optional  # Type hints for method signatures

from pgmpy.factors.discrete import DiscreteFactor  # pgmpy's discrete factor representation

from src.utils.config_loader import get_config  # Application configuration access
from src.utils.logger import LoggerFactory  # Centralized logging factory

# Initialize module-level logger for this module
logger = LoggerFactory.get_logger(__name__)


class DiscreteDistributionError(Exception):
    """Custom exception for discrete distribution validation or creation failures."""

    pass  # Inherits all behavior from base Exception


class DiscreteDistribution:
    """
    Represents a discrete probability distribution over categorical variables.

    Wraps pgmpy's DiscreteFactor to provide validation, normalization,
    and a clean interface for probability distribution operations.
    Each instance represents a single probability table for one or more variables.
    """

    def __init__(
        self,
        variables: List[str],
        cardinality: List[int],
        values: List[float],
        state_names: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Initialize a discrete probability distribution.

        Args:
            variables: List of variable names in the distribution.
            cardinality: List of cardinalities (number of possible outcomes per variable).
            values: List of probability or phi values for each combination.
            state_names: Optional mapping of variable names to their state labels.

        Raises:
            DiscreteDistributionError: If input validation fails.
        """
        # Store the variable names for this distribution
        self._variables = variables

        # Store the cardinality (number of outcomes) for each variable
        self._cardinality = cardinality

        # Store the raw probability/phi values
        self._values = values

        # Store optional state names for human-readable output
        self._state_names = state_names

        # Log the distribution creation attempt at debug level
        logger.debug(
            "Creating discrete distribution for variables: %s", variables
        )

        # Validate all inputs before creating the underlying factor
        self._validate_inputs()

        # Create the underlying pgmpy DiscreteFactor instance
        self._factor = self._create_factor()

        # Log successful creation at info level
        logger.info(
            "Successfully created distribution for variables: %s", variables
        )

    def _validate_inputs(self) -> None:
        """
        Validate all constructor inputs for correctness and consistency.

        Raises:
            DiscreteDistributionError: If any validation check fails.
        """
        # Check that variables list is not empty
        if not self._variables:
            logger.error("Variables list cannot be empty")  # Log the error
            raise DiscreteDistributionError("Variables list cannot be empty")

        # Check that cardinality list matches the number of variables
        if len(self._cardinality) != len(self._variables):
            logger.error(
                "Cardinality length (%d) does not match variables length (%d)",
                len(self._cardinality),
                len(self._variables),
            )
            raise DiscreteDistributionError(
                f"Cardinality length ({len(self._cardinality)}) must match "
                f"variables length ({len(self._variables)})"
            )

        # Calculate the expected number of values based on cardinality product
        expected_value_count = 1  # Start with multiplicative identity
        for card in self._cardinality:
            expected_value_count *= card  # Multiply each cardinality

        # Check that the values list length matches the expected count
        if len(self._values) != expected_value_count:
            logger.error(
                "Values length (%d) does not match expected count (%d)",
                len(self._values),
                expected_value_count,
            )
            raise DiscreteDistributionError(
                f"Values length ({len(self._values)}) must equal the product "
                f"of cardinalities ({expected_value_count})"
            )

        # Check that all values are non-negative (valid for probability factors)
        for value in self._values:
            if value < 0:
                logger.error("Negative value found in distribution: %f", value)
                raise DiscreteDistributionError(
                    f"All values must be non-negative, found: {value}"
                )

        # Validate state_names if provided
        if self._state_names is not None:
            self._validate_state_names()  # Delegate to state names validator

    def _validate_state_names(self) -> None:
        """
        Validate that state_names are consistent with variables and cardinality.

        Raises:
            DiscreteDistributionError: If state names are inconsistent.
        """
        # Check each variable has matching state names
        for i, variable in enumerate(self._variables):
            # Skip validation if variable is not in state_names dict
            if variable not in self._state_names:
                logger.warning(
                    "Variable '%s' not found in state_names dictionary", variable
                )
                continue  # Skip this variable

            # Get the state names for this variable
            states = self._state_names[variable]

            # Check that the number of state names matches the cardinality
            if len(states) != self._cardinality[i]:
                logger.error(
                    "State names count (%d) for '%s' does not match cardinality (%d)",
                    len(states),
                    variable,
                    self._cardinality[i],
                )
                raise DiscreteDistributionError(
                    f"State names count ({len(states)}) for variable '{variable}' "
                    f"must match its cardinality ({self._cardinality[i]})"
                )

    def _create_factor(self) -> DiscreteFactor:
        """
        Create the underlying pgmpy DiscreteFactor from validated inputs.

        Returns:
            A pgmpy DiscreteFactor instance representing this distribution.

        Raises:
            DiscreteDistributionError: If factor creation fails.
        """
        try:
            # Build the keyword arguments for DiscreteFactor constructor
            factor_kwargs = {
                "variables": self._variables,       # Variable name list
                "cardinality": self._cardinality,   # Cardinality list
                "values": self._values,             # Probability/phi values
            }

            # Add state_names only if provided (optional parameter)
            if self._state_names is not None:
                factor_kwargs["state_names"] = self._state_names

            # Create and return the DiscreteFactor instance
            factor = DiscreteFactor(**factor_kwargs)

            # Log debug information about the created factor
            logger.debug("Created DiscreteFactor: %s", factor)

            return factor  # Return the created factor

        except Exception as creation_error:
            # Log the error with full details for debugging
            logger.error(
                "Failed to create DiscreteFactor: %s", str(creation_error)
            )
            # Wrap the error in our custom exception type
            raise DiscreteDistributionError(
                f"Failed to create distribution factor: {creation_error}"
            ) from creation_error

    def normalize(self) -> "DiscreteDistribution":
        """
        Normalize the distribution so all values sum to 1.0.

        Converts phi values (unnormalized weights) to proper probabilities
        by dividing each value by the total sum.

        Returns:
            Self reference for method chaining.
        """
        # Log the normalization operation
        logger.debug("Normalizing distribution for variables: %s", self._variables)

        # Call pgmpy's normalize method (modifies factor in place)
        self._factor.normalize(inplace=True)

        # Log successful normalization
        logger.info("Distribution normalized successfully")

        return self  # Return self for method chaining

    def get_values(self) -> List[float]:
        """
        Retrieve the current probability values from the distribution.

        Returns:
            List of probability values (normalized or unnormalized).
        """
        # Extract values from the pgmpy factor and convert to Python list
        return self._factor.values.flatten().tolist()

    @property
    def factor(self) -> DiscreteFactor:
        """Return the underlying pgmpy DiscreteFactor instance."""
        return self._factor  # Expose the internal factor as read-only property

    @property
    def variables(self) -> List[str]:
        """Return the list of variable names in this distribution."""
        return self._variables.copy()  # Return copy to prevent external mutation

    def __str__(self) -> str:
        """Return a human-readable string representation of the distribution."""
        return str(self._factor)  # Delegate to pgmpy's string formatting


class DiscreteDistributionFactory:
    """
    Factory class for creating common discrete distributions from configuration.

    Provides convenient methods for creating distributions using values
    from the application configuration file, reducing boilerplate code.
    """

    def __init__(self) -> None:
        """Initialize the factory with application configuration."""
        # Load the probability section from configuration
        self._config = get_config()

        # Log factory initialization
        logger.debug("DiscreteDistributionFactory initialized")

    def create_from_config(self) -> DiscreteDistribution:
        """
        Create a distribution using default values from the configuration file.

        Returns:
            A DiscreteDistribution instance with config-defined parameters.
        """
        # Read default parameters from the probability config section
        variable_name = self._config.get(
            "probability.default_variable_name", "X"
        )
        cardinality = self._config.get("probability.default_cardinality", 3)
        values = self._config.get(
            "probability.default_values", [0.45, 0.30, 0.25]
        )
        state_names = self._config.get(
            "probability.default_state_names", ["1", "2", "3"]
        )

        # Log the configuration values being used
        logger.info(
            "Creating distribution from config: variable=%s, cardinality=%d",
            variable_name,
            cardinality,
        )

        # Build the state_names dictionary from config values
        state_names_dict = {variable_name: state_names}

        # Create and return the distribution using config values
        return DiscreteDistribution(
            variables=[variable_name],       # Wrap variable name in a list
            cardinality=[cardinality],        # Wrap cardinality in a list
            values=values,                    # Pass values directly
            state_names=state_names_dict,     # Pass constructed state names dict
        )

    def create_custom(
        self,
        variables: List[str],
        cardinality: List[int],
        values: List[float],
        state_names: Optional[Dict[str, List[str]]] = None,
    ) -> DiscreteDistribution:
        """
        Create a distribution with custom parameters.

        Args:
            variables: List of variable names.
            cardinality: List of cardinalities per variable.
            values: List of probability values.
            state_names: Optional state name mapping.

        Returns:
            A new DiscreteDistribution instance.
        """
        # Log the custom distribution creation
        logger.info("Creating custom distribution for variables: %s", variables)

        # Delegate to the DiscreteDistribution constructor
        return DiscreteDistribution(
            variables=variables,
            cardinality=cardinality,
            values=values,
            state_names=state_names,
        )
