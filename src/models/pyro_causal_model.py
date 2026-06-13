# Pyro-based probabilistic causal model for sampling-based inference.
"""
Pyro Causal Model - Implements causal DAGs as Pyro probabilistic programs.

Translates the transportation survey causal model into a Pyro generative
model, enabling sampling-based inference methods like importance sampling
for computing posterior distributions.
"""

from typing import Dict, List, Optional  # Type hints for method signatures

import numpy as np  # Numerical computing for array operations
import torch  # PyTorch tensor library (Pyro backend)
import pyro  # Pyro probabilistic programming framework
from pyro.distributions import Categorical  # Categorical distribution class
from pyro.infer import EmpiricalMarginal, Importance  # Inference algorithms

from src.utils.config_loader import get_config  # Application configuration access
from src.utils.logger import LoggerFactory  # Centralized logging factory

# Initialize module-level logger for this module
logger = LoggerFactory.get_logger(__name__)


class PyroModelError(Exception):
    """Custom exception for Pyro model creation or inference failures."""

    pass  # Inherits all behavior from base Exception


class TransportationCausalModel:
    """
    Pyro implementation of the transportation survey causal model.

    Defines the generative process: Age and Sex cause Education,
    Education causes Occupation and Residence, and Occupation and
    Residence jointly cause Transportation mode choice.

    Uses torch tensors for conditional probability tables and
    Pyro's sampling primitives for probabilistic inference.
    """

    def __init__(self) -> None:
        """Initialize the causal model with probability tables from config."""
        # Load configuration for model parameters
        config = get_config()

        # Load outcome category labels from configuration
        outcome_labels = config.get("dag.outcome_labels", {})
        self._age_labels = outcome_labels.get("A", ["young", "adult", "old"])
        self._sex_labels = outcome_labels.get("S", ["M", "F"])
        self._education_labels = outcome_labels.get("E", ["high", "uni"])
        self._occupation_labels = outcome_labels.get("O", ["emp", "self"])
        self._residence_labels = outcome_labels.get("R", ["small", "big"])
        self._transport_labels = outcome_labels.get("T", ["car", "train", "other"])

        # Load Pyro-specific configuration
        self._num_samples = config.get("model.pyro.num_samples", 5000)
        self._random_seed = config.get("model.pyro.random_seed", 42)

        # Initialize the conditional probability tables as torch tensors
        self._initialize_probability_tables()

        # Log model initialization
        logger.info(
            "TransportationCausalModel initialized with %d samples",
            self._num_samples,
        )

    def _initialize_probability_tables(self) -> None:
        """
        Initialize all conditional probability tables as torch tensors.

        These probabilities are based on parameters learned from the
        transportation survey dataset via pgmpy's MLE fit method.
        Values are rounded for clarity while maintaining valid distributions.
        """
        # P(Age) - Marginal probability of age categories
        self._age_probs = torch.tensor([0.3, 0.5, 0.2])

        # P(Sex) - Marginal probability of sex categories
        self._sex_probs = torch.tensor([0.6, 0.4])

        # P(Education | Sex, Age) - Conditional probability table
        # Indexed as [Sex][Age][Education]
        self._education_probs = torch.tensor([
            [[0.75, 0.25], [0.72, 0.28], [0.88, 0.12]],  # Sex=M: Age=[young,adult,old]
            [[0.64, 0.36], [0.70, 0.30], [0.90, 0.10]],  # Sex=F: Age=[young,adult,old]
        ])

        # P(Occupation | Education) - Conditional probability table
        # Indexed as [Education][Occupation]
        self._occupation_probs = torch.tensor([
            [0.96, 0.04],  # Education=high: [emp, self]
            [0.92, 0.08],  # Education=uni: [emp, self]
        ])

        # P(Residence | Education) - Conditional probability table
        # Indexed as [Education][Residence]
        self._residence_probs = torch.tensor([
            [0.25, 0.75],  # Education=high: [small, big]
            [0.20, 0.80],  # Education=uni: [small, big]
        ])

        # P(Transportation | Residence, Occupation) - Conditional probability table
        # Indexed as [Residence][Occupation][Transportation]
        self._transport_probs = torch.tensor([
            [[0.48, 0.42, 0.10], [0.56, 0.36, 0.08]],  # Residence=small: Occ=[emp,self]
            [[0.58, 0.24, 0.18], [0.70, 0.21, 0.09]],  # Residence=big: Occ=[emp,self]
        ])

        # Log successful table initialization
        logger.debug("Probability tables initialized successfully")

    def generative_model(self) -> Dict[str, torch.Tensor]:
        """
        Define the Pyro generative model (forward sampling process).

        This function defines the joint distribution P(A, S, E, O, R, T)
        by specifying each variable's distribution conditioned on its parents.
        Pyro's sample statements register each random choice for inference.

        Returns:
            Dictionary mapping variable names to their sampled values.
        """
        # Sample Age from marginal distribution P(A)
        age = pyro.sample("age", Categorical(probs=self._age_probs))

        # Sample Sex from marginal distribution P(S)
        sex = pyro.sample("gender", Categorical(probs=self._sex_probs))

        # Sample Education conditioned on Sex and Age: P(E | S, A)
        education = pyro.sample(
            "education", Categorical(probs=self._education_probs[sex][age])
        )

        # Sample Occupation conditioned on Education: P(O | E)
        occupation = pyro.sample(
            "occupation", Categorical(probs=self._occupation_probs[education])
        )

        # Sample Residence conditioned on Education: P(R | E)
        residence = pyro.sample(
            "residence", Categorical(probs=self._residence_probs[education])
        )

        # Sample Transportation conditioned on Residence and Occupation: P(T | R, O)
        transportation = pyro.sample(
            "transportation",
            Categorical(probs=self._transport_probs[residence][occupation]),
        )

        # Return all sampled values as a dictionary
        return {
            "A": age,           # Age sample
            "S": sex,           # Sex sample
            "E": education,     # Education sample
            "O": occupation,    # Occupation sample
            "R": residence,     # Residence sample
            "T": transportation,  # Transportation sample
        }

    def render_model_graph(self):
        """
        Render the Pyro model's implied DAG structure as a visual graph.

        Uses Pyro's built-in model renderer to generate a graphviz
        representation of the causal DAG.

        Returns:
            The rendered graph object from pyro.render_model.
        """
        # Log the rendering operation
        logger.info("Rendering Pyro model graph")

        # Use Pyro's render_model utility to generate the DAG visualization
        graph = pyro.render_model(self.generative_model)

        return graph  # Return the rendered graph object


class ImportanceSamplingInference:
    """
    Performs posterior inference using importance sampling in Pyro.

    Importance sampling estimates posterior distributions by generating
    weighted samples from the prior and reweighting based on the
    likelihood of the observed evidence.
    """

    def __init__(self, causal_model: TransportationCausalModel) -> None:
        """
        Initialize the importance sampling inference engine.

        Args:
            causal_model: A TransportationCausalModel instance to perform inference on.
        """
        # Store reference to the causal model
        self._model = causal_model

        # Load configuration for sampling parameters
        config = get_config()
        self._num_samples = config.get("model.pyro.num_samples", 5000)

        # Log initialization
        logger.info(
            "ImportanceSamplingInference initialized: num_samples=%d",
            self._num_samples,
        )

    def compute_posterior(
        self,
        target_variable: str,
        evidence: Dict[str, float],
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute the posterior distribution P(target | evidence) via importance sampling.

        Args:
            target_variable: Name of the variable to compute the posterior for.
            evidence: Dict mapping variable names to observed tensor values.
            num_samples: Number of importance samples (overrides config if provided).

        Returns:
            Dictionary mapping state labels to estimated probabilities.

        Raises:
            PyroModelError: If inference computation fails.
        """
        # Use provided sample count or fall back to configured default
        sample_count = num_samples if num_samples is not None else self._num_samples

        try:
            # Log the inference operation
            logger.info(
                "Computing posterior: P(%s | %s) with %d samples",
                target_variable,
                evidence,
                sample_count,
            )

            # Condition the generative model on the provided evidence
            conditioned_model = pyro.condition(
                self._model.generative_model,  # The generative model function
                data={
                    key: torch.tensor(float(val))  # Convert evidence to tensors
                    for key, val in evidence.items()
                },
            )

            # Run importance sampling to generate weighted posterior samples
            posterior = Importance(
                conditioned_model,        # The evidence-conditioned model
                num_samples=sample_count,  # Number of importance samples
            ).run()

            # Extract the marginal distribution for the target variable
            marginal = EmpiricalMarginal(posterior, target_variable)

            # Generate samples from the empirical marginal distribution
            samples = [marginal().item() for _ in range(sample_count)]

            # Compute probability estimates from sample frequencies
            unique_values, counts = np.unique(samples, return_counts=True)

            # Convert counts to normalized probabilities
            probabilities = counts / sample_count

            # Map numeric indices to human-readable state labels
            label_map = self._get_label_map(target_variable)

            # Build the result dictionary with labels and probabilities
            result = {}
            for value, probability in zip(unique_values, probabilities):
                # Get the label for this numeric index
                label = label_map.get(int(value), f"state_{int(value)}")
                result[label] = float(probability)  # Store as Python float

            # Log the computed posterior
            logger.info("Posterior computed: %s", result)

            return result  # Return the labeled probability dictionary

        except Exception as inference_error:
            # Log the inference failure
            logger.error(
                "Importance sampling failed: %s", str(inference_error)
            )
            raise PyroModelError(
                f"Posterior computation failed: {inference_error}"
            ) from inference_error

    def _get_label_map(self, variable_name: str) -> Dict[int, str]:
        """
        Get the mapping from numeric indices to state labels for a variable.

        Args:
            variable_name: The Pyro sample site name.

        Returns:
            Dictionary mapping integer indices to string labels.
        """
        # Map Pyro sample site names to their label lists
        label_mapping = {
            "age": self._model._age_labels,                # Age state labels
            "gender": self._model._sex_labels,             # Sex state labels
            "education": self._model._education_labels,    # Education state labels
            "occupation": self._model._occupation_labels,  # Occupation state labels
            "residence": self._model._residence_labels,    # Residence state labels
            "transportation": self._model._transport_labels,  # Transport state labels
        }

        # Get the labels for the requested variable
        labels = label_mapping.get(variable_name, [])

        # Build and return the index-to-label dictionary
        return {i: label for i, label in enumerate(labels)}
