# DAG visualization module using DoWhy and NetworkX for causal graph rendering.
"""
DAG Visualizer - Renders causal DAG structures as visual graphs.

Uses DoWhy to generate simulated causal datasets with specified structure
(instruments, confounders, effect modifiers, front-door variables) and
NetworkX/Matplotlib to render the resulting DAG as a publication-quality figure.
"""

from pathlib import Path  # Object-oriented filesystem path handling
from typing import Dict, Optional, Tuple  # Type hints for method signatures

import matplotlib.pyplot as plt  # Plotting library for figure rendering
import networkx as nx  # Graph library for DAG representation and layout
from dowhy import datasets  # DoWhy dataset simulation utilities

from src.utils.config_loader import get_config  # Application configuration access
from src.utils.logger import LoggerFactory  # Centralized logging factory

# Initialize module-level logger for this module
logger = LoggerFactory.get_logger(__name__)


class VisualizationError(Exception):
    """Custom exception for visualization operation failures."""

    pass  # Inherits all behavior from base Exception


class DoWhyDAGSimulator:
    """
    Generates simulated causal datasets with configurable DAG structures.

    Uses DoWhy's dataset generation to create datasets with specified
    numbers of treatments, instruments, confounders, effect modifiers,
    and front-door variables. Returns both data and graph structure.
    """

    def __init__(self) -> None:
        """Initialize the simulator with configuration parameters."""
        # Load DoWhy simulation parameters from configuration
        config = get_config()
        dowhy_config = config.get_section("dowhy")

        # Extract simulation parameters with defaults
        self._beta = dowhy_config.get("beta", 10.0)
        self._num_treatments = dowhy_config.get("num_treatments", 1)
        self._num_instruments = dowhy_config.get("num_instruments", 2)
        self._num_effect_modifiers = dowhy_config.get("num_effect_modifiers", 2)
        self._num_common_causes = dowhy_config.get("num_common_causes", 5)
        self._num_frontdoor_variables = dowhy_config.get("num_frontdoor_variables", 1)
        self._num_samples = dowhy_config.get("num_samples", 100)

        # Log initialization with key parameters
        logger.info(
            "DoWhyDAGSimulator initialized: beta=%.1f, treatments=%d, instruments=%d",
            self._beta,
            self._num_treatments,
            self._num_instruments,
        )

    def generate_dataset(self) -> Dict:
        """
        Generate a simulated linear dataset with the configured DAG structure.

        Returns:
            Dictionary containing 'df' (DataFrame), 'gml_graph' (graph markup),
            and other metadata from DoWhy's dataset generator.

        Raises:
            VisualizationError: If dataset generation fails.
        """
        try:
            # Log the dataset generation operation
            logger.info(
                "Generating simulated dataset: %d samples", self._num_samples
            )

            # Generate the linear dataset using DoWhy's simulation utility
            simulated_data = datasets.linear_dataset(
                beta=self._beta,                              # Causal effect strength
                num_treatments=self._num_treatments,          # Treatment variable count
                num_instruments=self._num_instruments,        # Instrumental variable count
                num_effect_modifiers=self._num_effect_modifiers,  # Effect modifier count
                num_common_causes=self._num_common_causes,    # Confounder count
                num_frontdoor_variables=self._num_frontdoor_variables,  # Front-door count
                num_samples=self._num_samples,                # Data sample count
            )

            # Log successful generation
            logger.info("Simulated dataset generated successfully")

            return simulated_data  # Return the complete simulation result

        except Exception as gen_error:
            # Log the generation failure
            logger.error("Dataset generation failed: %s", str(gen_error))
            raise VisualizationError(
                f"Failed to generate simulated dataset: {gen_error}"
            ) from gen_error


class DAGRenderer:
    """
    Renders causal DAG structures as visual graphs using NetworkX and Matplotlib.

    Parses GML graph representations from DoWhy and renders them with
    configurable layout, node styling, and output format options.
    """

    def __init__(self) -> None:
        """Initialize the renderer with visualization configuration."""
        # Load visualization configuration
        config = get_config()
        viz_config = config.get_section("visualization")

        # Extract visualization parameters
        self._dpi = viz_config.get("figure_dpi", 100)
        self._output_format = viz_config.get("figure_format", "png")
        self._output_dir = Path(viz_config.get("output_directory", "output/figures"))
        self._show_plots = viz_config.get("show_plots", True)

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Log renderer initialization
        logger.info(
            "DAGRenderer initialized: format=%s, dpi=%d",
            self._output_format,
            self._dpi,
        )

    def render_from_gml(
        self,
        gml_graph: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Render a DAG from its GML (Graph Modelling Language) representation.

        Args:
            gml_graph: GML string representing the DAG structure.
            title: Optional title to display on the figure.
            save_path: Optional file path to save the rendered figure.

        Raises:
            VisualizationError: If rendering fails.
        """
        try:
            # Parse the GML string into a NetworkX graph object
            dag = nx.parse_gml(gml_graph)

            # Log the graph structure details
            logger.info(
                "Rendering DAG: %d nodes, %d edges",
                dag.number_of_nodes(),
                dag.number_of_edges(),
            )

            # Generate a layout for the graph nodes
            node_positions = self._compute_layout(dag)

            # Define visual styling options for the graph rendering
            style_options = {
                "font_size": 12,            # Label font size in points
                "node_size": 800,           # Node circle diameter
                "node_color": "white",      # Node fill color
                "edgecolors": "black",      # Node border color
                "linewidths": 1,            # Node border width
                "width": 1,                 # Edge line width
            }

            # Create the matplotlib figure and axes
            plt.figure(figsize=(12, 8), dpi=self._dpi)

            # Draw the NetworkX graph with specified positions and styling
            nx.draw_networkx(dag, node_positions, **style_options)

            # Get the current axes and set margins for readability
            axes = plt.gca()
            axes.margins(x=0.40)  # Add horizontal margins

            # Turn off the axis frame (not needed for graph visualization)
            plt.axis("off")

            # Set the title if provided
            if title:
                plt.title(title, fontsize=14, fontweight="bold")

            # Save the figure to disk if a path is specified
            if save_path:
                self._save_figure(save_path)

            # Display the plot interactively if configured to do so
            if self._show_plots:
                plt.show()  # Open the interactive plot window
            else:
                plt.close()  # Close without displaying to free memory

            # Log successful rendering
            logger.info("DAG rendered successfully")

        except Exception as render_error:
            # Close any open figures to prevent memory leaks
            plt.close("all")
            # Log the rendering failure
            logger.error("DAG rendering failed: %s", str(render_error))
            raise VisualizationError(
                f"Failed to render DAG: {render_error}"
            ) from render_error

    def _compute_layout(self, dag: nx.Graph) -> Dict:
        """
        Compute node positions for the DAG layout.

        Uses a predefined layout for known transportation survey variables,
        or falls back to spring layout for arbitrary graphs.

        Args:
            dag: NetworkX graph to compute layout for.

        Returns:
            Dictionary mapping node names to (x, y) position tuples.
        """
        # Define known positions for the transportation survey DAG variables
        known_positions = {
            "X0": (600, 350),    # Effect modifier X0 position
            "X1": (600, 250),    # Effect modifier X1 position
            "FD0": (300, 300),   # Front-door variable position
            "W0": (0, 400),      # Common cause W0 position
            "W1": (150, 400),    # Common cause W1 position
            "W2": (300, 400),    # Common cause W2 position
            "W3": (450, 400),    # Common cause W3 position
            "W4": (600, 400),    # Common cause W4 position
            "Z0": (10, 250),     # Instrument Z0 position
            "Z1": (10, 350),     # Instrument Z1 position
            "v0": (100, 300),    # Treatment variable position
            "y": (500, 300),     # Outcome variable position
        }

        # Get the nodes present in this specific graph
        graph_nodes = set(dag.nodes())

        # Check if all graph nodes have predefined positions
        if graph_nodes.issubset(set(known_positions.keys())):
            # Filter positions to only include nodes in this graph
            positions = {
                node: pos
                for node, pos in known_positions.items()
                if node in graph_nodes
            }
            logger.debug("Using predefined layout for %d nodes", len(positions))
            return positions  # Return the predefined layout

        # Fall back to spring layout for unknown graph structures
        logger.debug("Using spring layout for %d nodes", len(graph_nodes))
        return nx.spring_layout(dag)  # Compute force-directed layout

    def _save_figure(self, save_path: str) -> None:
        """
        Save the current matplotlib figure to a file.

        Args:
            save_path: Path where the figure should be saved.
        """
        # Resolve the full output path
        output_path = Path(save_path)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the figure with configured DPI and tight bounding box
        plt.savefig(
            str(output_path),
            dpi=self._dpi,              # Figure resolution
            bbox_inches="tight",        # Trim whitespace around the figure
            format=self._output_format,  # Output file format
        )

        # Log the successful save operation
        logger.info("Figure saved to: %s", output_path)
