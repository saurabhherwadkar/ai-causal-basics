# Entry point for the probability primer demonstration module.
"""
Probability Primer Runner - Demonstrates discrete probability distributions.

Executes the probability primer examples showing how to create,
manipulate, and normalize discrete probability distributions using
pgmpy's DiscreteFactor representation.
"""

import sys  # System exit and path manipulation

from src.models.discrete_distribution import (  # Distribution classes
    DiscreteDistribution,
    DiscreteDistributionFactory,
)
from src.utils.config_loader import get_config  # Configuration access
from src.utils.logger import LoggerFactory  # Logging factory

# Initialize module-level logger
logger = LoggerFactory.get_logger(__name__)


def demonstrate_normalized_distribution() -> None:
    """
    Demonstrate creating a properly normalized probability distribution.

    Creates a distribution with values that sum to 1.0, showing that
    pgmpy treats these as valid probabilities without normalization.
    """
    # Log the start of this demonstration section
    logger.info("=== Normalized Distribution Demo ===")

    # Create a distribution with values summing to 1.0 (already normalized)
    distribution = DiscreteDistribution(
        variables=["X"],                    # Single variable named X
        cardinality=[3],                    # Three possible outcomes
        values=[0.45, 0.30, 0.25],         # Pre-normalized probability values
        state_names={"X": ["1", "2", "3"]},  # Human-readable state labels
    )

    # Print the distribution table to console
    print("--- Normalized Distribution P(X) ---")
    print(distribution)  # Delegates to pgmpy's formatted output
    print()  # Add blank line for readability


def demonstrate_unnormalized_distribution() -> None:
    """
    Demonstrate normalizing a distribution from phi (unnormalized) values.

    Creates a distribution with values that don't sum to 1.0 (phi values),
    then normalizes them to produce valid probabilities.
    """
    # Log the start of this demonstration section
    logger.info("=== Unnormalized Distribution Demo ===")

    # Create a distribution with unnormalized values (phi values)
    distribution = DiscreteDistribution(
        variables=["X"],                    # Single variable named X
        cardinality=[3],                    # Three possible outcomes
        values=[45, 30, 25],               # Unnormalized phi values
        state_names={"X": ["1", "2", "3"]},  # Human-readable state labels
    )

    # Print the unnormalized distribution
    print("--- Unnormalized Distribution (phi values) ---")
    print(distribution)  # Shows raw phi values
    print()

    # Normalize the distribution to get proper probabilities
    distribution.normalize()

    # Print the normalized result
    print("--- After Normalization ---")
    print(distribution)  # Shows normalized probabilities summing to 1.0
    print()


def demonstrate_factory_creation() -> None:
    """
    Demonstrate creating a distribution from configuration using the factory.

    Uses the DiscreteDistributionFactory to create a distribution with
    parameters read from the application configuration file.
    """
    # Log the start of this demonstration section
    logger.info("=== Factory Creation Demo ===")

    # Create a factory instance (reads from config)
    factory = DiscreteDistributionFactory()

    # Create a distribution using configuration defaults
    config_distribution = factory.create_from_config()

    # Print the config-created distribution
    print("--- Distribution from Configuration ---")
    print(config_distribution)  # Shows distribution with config values
    print()


def main() -> None:
    """
    Main entry point for the probability primer demonstration.

    Runs all demonstration functions in sequence, with proper
    error handling and logging.
    """
    try:
        # Log application startup
        logger.info("Starting Probability Primer demonstration")

        # Run each demonstration function in sequence
        demonstrate_normalized_distribution()   # Demo 1: pre-normalized values
        demonstrate_unnormalized_distribution()  # Demo 2: phi values + normalization
        demonstrate_factory_creation()           # Demo 3: config-driven creation

        # Log successful completion
        logger.info("Probability Primer demonstration completed successfully")

    except Exception as error:
        # Log any unhandled exceptions at error level
        logger.error("Probability Primer failed: %s", str(error))
        # Print error message to stderr for user visibility
        print(f"Error: {error}", file=sys.stderr)
        # Exit with non-zero status code to indicate failure
        sys.exit(1)


# Standard Python entry point guard
if __name__ == "__main__":
    main()  # Execute the main function when run directly
