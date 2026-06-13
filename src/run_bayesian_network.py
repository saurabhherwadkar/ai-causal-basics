# Entry point for the Bayesian network demonstration module.
"""
Bayesian Network Runner - Demonstrates causal DAG construction and inference.

Executes the full pipeline: defines a DAG, loads data, fits parameters
using MLE and Bayesian estimation, runs inference queries, demonstrates
latent variable handling via EM, and runs Pyro importance sampling.
"""

import sys  # System exit and path manipulation

from src.inference.variable_elimination import VariableEliminationInference  # Exact inference
from src.models.bayesian_network import (  # Bayesian network classes
    BayesianNetworkModel,
    CausalDAG,
    LatentVariableModel,
)
from src.models.pyro_causal_model import (  # Pyro model classes
    ImportanceSamplingInference,
    TransportationCausalModel,
)
from src.utils.config_loader import get_config  # Configuration access
from src.utils.data_loader import DataLoader  # Secure data loading
from src.utils.logger import LoggerFactory  # Logging factory
from src.visualization.dag_visualizer import DAGRenderer, DoWhyDAGSimulator  # Visualization

# Initialize module-level logger
logger = LoggerFactory.get_logger(__name__)


def demonstrate_mle_fitting() -> None:
    """
    Demonstrate Maximum Likelihood Estimation parameter fitting.

    Loads the transportation survey data, defines the DAG structure,
    and fits parameters using frequency-based MLE estimation.
    """
    # Log the start of this section
    logger.info("=== MLE Parameter Fitting Demo ===")
    print("--- Maximum Likelihood Estimation ---")

    # Create the DAG structure from configuration
    dag = CausalDAG.from_config()

    # Load the transportation survey dataset
    data_loader = DataLoader()
    config = get_config()
    data_url = config.get("data.transportation_survey_url")
    data = data_loader.load_csv(data_url)

    # Create the Bayesian network model
    model = BayesianNetworkModel(dag)

    # Fit parameters using MLE
    model.fit_mle(data)

    # Retrieve and display the learned CPDs (causal Markov kernels)
    cpds = model.get_cpds()

    # Print all learned CPDs
    print(f"\nLearned {len(cpds)} Conditional Probability Distributions:")
    for cpd in cpds:
        print(cpd)  # Print each CPD's probability table
        print()  # Blank line between CPDs

    return model  # Return fitted model for reuse


def demonstrate_bayesian_fitting() -> None:
    """
    Demonstrate Bayesian parameter estimation with Dirichlet priors.

    Adds pseudo-counts to prevent zero probabilities, providing
    smoother parameter estimates especially for rare combinations.
    """
    # Log the start of this section
    logger.info("=== Bayesian Estimation Demo ===")
    print("--- Bayesian Estimation with Dirichlet Prior ---")

    # Create the DAG structure from configuration
    dag = CausalDAG.from_config()

    # Load the transportation survey dataset
    data_loader = DataLoader()
    config = get_config()
    data_url = config.get("data.transportation_survey_url")
    data = data_loader.load_csv(data_url)

    # Create and fit the model with Bayesian estimation
    model = BayesianNetworkModel(dag)
    model.fit_bayesian(data)  # Uses config defaults for prior_type and pseudo_counts

    # Retrieve the last CPD (Transportation given Occupation and Residence)
    cpds = model.get_cpds()
    transport_cpd = cpds[-1]  # Last CPD is P(T | O, R)

    # Print the transportation CPD
    print("\nP(T | O, R) with Bayesian estimation:")
    print(transport_cpd)
    print()


def demonstrate_latent_variable_em() -> None:
    """
    Demonstrate Expectation Maximization for latent variables.

    Removes Education from the observed data and uses structural EM
    to learn its parameters as a latent (unobserved) variable.
    """
    # Log the start of this section
    logger.info("=== Latent Variable EM Demo ===")
    print("--- Expectation Maximization for Latent Variables ---")

    # Load the transportation survey dataset
    data_loader = DataLoader()
    config = get_config()
    data_url = config.get("data.transportation_survey_url")
    data = data_loader.load_csv(data_url)

    # Remove the Education column to simulate it being unobserved
    observed_columns = ["A", "S", "O", "R", "T"]
    observed_data = data[observed_columns]

    # Create the DAG with Education marked as latent
    dag = CausalDAG.from_config()
    latent_model = LatentVariableModel(dag, latent_variables={"E"})

    # Run EM to learn parameters for the latent variable
    learned_cpds = latent_model.fit_em(observed_data)

    # Print the learned CPD for Education (latent variable)
    print("\nLearned P(E | A, S) via EM (Education is latent):")
    print(learned_cpds[1].to_factor())  # Convert to factor for readability
    print()


def demonstrate_variable_elimination() -> None:
    """
    Demonstrate exact inference using Variable Elimination.

    Computes conditional probabilities P(Education | Transportation)
    to answer questions about the most likely education level given
    the observed transportation mode.
    """
    # Log the start of this section
    logger.info("=== Variable Elimination Inference Demo ===")
    print("--- Variable Elimination Inference ---")

    # Create and fit the model (reusing MLE fitting)
    dag = CausalDAG.from_config()
    data_loader = DataLoader()
    config = get_config()
    data_url = config.get("data.transportation_survey_url")
    data = data_loader.load_csv(data_url)

    model = BayesianNetworkModel(dag)
    model.fit_mle(data)

    # Create the inference engine
    inference = VariableEliminationInference(model)

    # Query 1: P(Education | Transportation = train)
    result_train = inference.query(
        target_variables=["E"],              # Query Education
        evidence={"T": "train"},             # Given Transportation is train
    )
    print("\nP(Education | Transportation = train):")
    print(result_train)

    # Query 2: P(Education | Transportation = car)
    result_car = inference.query(
        target_variables=["E"],              # Query Education
        evidence={"T": "car"},               # Given Transportation is car
    )
    print("\nP(Education | Transportation = car):")
    print(result_car)
    print()


def demonstrate_pyro_inference() -> None:
    """
    Demonstrate importance sampling inference using Pyro.

    Uses Pyro's probabilistic programming to estimate the posterior
    distribution P(Education | Transportation = train) via importance
    sampling from the generative model.
    """
    # Log the start of this section
    logger.info("=== Pyro Importance Sampling Demo ===")
    print("--- Pyro Importance Sampling Inference ---")

    # Create the Pyro causal model
    causal_model = TransportationCausalModel()

    # Create the importance sampling inference engine
    inference = ImportanceSamplingInference(causal_model)

    # Compute P(Education | Transportation = train)
    # Transportation = "train" corresponds to index 1
    posterior = inference.compute_posterior(
        target_variable="education",         # Query Education variable
        evidence={"transportation": 1.0},    # Evidence: T=train (index 1)
    )

    # Print the estimated posterior distribution
    print("\nP(Education | Transportation = train) via Importance Sampling:")
    for state, probability in posterior.items():
        print(f"  {state}: {probability:.4f}")  # Print each state and probability
    print()


def demonstrate_dowhy_visualization() -> None:
    """
    Demonstrate causal DAG visualization using DoWhy.

    Generates a simulated dataset with configurable causal structure
    and renders the resulting DAG as a visual graph.
    """
    # Log the start of this section
    logger.info("=== DoWhy DAG Visualization Demo ===")
    print("--- DoWhy Causal DAG Visualization ---")

    # Generate the simulated dataset with configured DAG structure
    simulator = DoWhyDAGSimulator()
    simulated_data = simulator.generate_dataset()

    # Create the DAG renderer
    renderer = DAGRenderer()

    # Render the GML graph from the simulated data
    renderer.render_from_gml(
        gml_graph=simulated_data["gml_graph"],  # GML graph string
        title="Simulated Causal DAG (DoWhy)",    # Figure title
        save_path="output/figures/dowhy_dag.png",  # Save location
    )

    print("DAG visualization rendered successfully")
    print()


def main() -> None:
    """
    Main entry point for the Bayesian network demonstration.

    Runs all demonstration sections in sequence with comprehensive
    error handling and logging at each stage.
    """
    try:
        # Log application startup
        logger.info("Starting Bayesian Network demonstration")

        # Run demonstrations in sequence
        demonstrate_mle_fitting()           # Demo 1: MLE parameter learning
        demonstrate_bayesian_fitting()      # Demo 2: Bayesian estimation
        demonstrate_latent_variable_em()    # Demo 3: EM for latent variables
        demonstrate_variable_elimination()  # Demo 4: Exact inference queries
        demonstrate_pyro_inference()        # Demo 5: Sampling-based inference
        demonstrate_dowhy_visualization()   # Demo 6: DAG visualization

        # Log successful completion
        logger.info("Bayesian Network demonstration completed successfully")

    except Exception as error:
        # Log any unhandled exceptions at error level
        logger.error("Bayesian Network demo failed: %s", str(error))
        # Print error message to stderr for user visibility
        print(f"Error: {error}", file=sys.stderr)
        # Exit with non-zero status code
        sys.exit(1)


# Standard Python entry point guard
if __name__ == "__main__":
    main()  # Execute the main function when run directly
