# Unit tests for the DAG visualization module.
"""
Tests for DoWhyDAGSimulator and DAGRenderer - Validates graph generation and rendering.
"""

import os  # OS operations for temp file handling
import tempfile  # Temporary file creation
from pathlib import Path  # Path manipulation
from unittest.mock import patch  # Mocking for plot isolation

import pytest  # Testing framework

from src.utils.logger import LoggerFactory  # Logger for test setup


class TestDoWhyDAGSimulator:
    """Test suite for the DoWhyDAGSimulator class."""

    def setup_method(self) -> None:
        """Set up test fixtures with DoWhy config."""
        LoggerFactory.reset()
        self._temp_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: WARNING\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "dowhy:\n"
            "  beta: 10.0\n"
            "  num_treatments: 1\n"
            "  num_instruments: 2\n"
            "  num_effect_modifiers: 2\n"
            "  num_common_causes: 3\n"
            "  num_frontdoor_variables: 1\n"
            "  num_samples: 50\n",
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

    def test_simulator_initialization(self) -> None:
        """Test that the simulator initializes with config values."""
        from src.visualization.dag_visualizer import DoWhyDAGSimulator

        # Create simulator
        simulator = DoWhyDAGSimulator()

        # Verify config values are loaded
        assert simulator._beta == 10.0
        assert simulator._num_treatments == 1
        assert simulator._num_instruments == 2
        assert simulator._num_samples == 50

    def test_generate_dataset_returns_dict(self) -> None:
        """Test that generate_dataset returns a dictionary with expected keys."""
        from src.visualization.dag_visualizer import DoWhyDAGSimulator

        # Create simulator and generate data
        simulator = DoWhyDAGSimulator()
        result = simulator.generate_dataset()

        # Verify result is a dict with expected keys
        assert isinstance(result, dict)
        assert "df" in result
        assert "gml_graph" in result

    def test_generate_dataset_has_valid_gml(self) -> None:
        """Test that the generated GML graph string is parseable."""
        import networkx as nx
        from src.visualization.dag_visualizer import DoWhyDAGSimulator

        # Generate dataset
        simulator = DoWhyDAGSimulator()
        result = simulator.generate_dataset()

        # Parse the GML graph
        graph = nx.parse_gml(result["gml_graph"])

        # Verify graph has nodes and edges
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0


class TestDAGRenderer:
    """Test suite for the DAGRenderer class."""

    def setup_method(self) -> None:
        """Set up test fixtures with visualization config."""
        LoggerFactory.reset()
        self._temp_dir = tempfile.mkdtemp()
        self._output_dir = tempfile.mkdtemp()
        self._config_path = Path(self._temp_dir) / "config.yaml"
        self._config_path.write_text(
            "logging:\n"
            "  level: WARNING\n"
            "  file_enabled: false\n"
            "  console_enabled: false\n"
            "visualization:\n"
            "  figure_dpi: 72\n"
            "  figure_format: png\n"
            f"  output_directory: '{self._output_dir}'\n"
            "  show_plots: false\n"
            "  color_scheme: default\n",
            encoding="utf-8",
        )
        from src.utils.config_loader import get_config
        get_config(config_dir=self._temp_dir)

    def teardown_method(self) -> None:
        """Clean up temporary files and directories."""
        LoggerFactory.reset()
        # Clean output directory
        output_path = Path(self._output_dir)
        if output_path.exists():
            for file in output_path.iterdir():
                file.unlink()
            output_path.rmdir()
        # Clean config
        if self._config_path.exists():
            self._config_path.unlink()
        os.rmdir(self._temp_dir)

    def test_renderer_initialization(self) -> None:
        """Test that the renderer initializes with config values."""
        from src.visualization.dag_visualizer import DAGRenderer

        # Create renderer
        renderer = DAGRenderer()

        # Verify config values
        assert renderer._dpi == 72
        assert renderer._output_format == "png"
        assert renderer._show_plots is False

    @patch("matplotlib.pyplot.show")
    def test_render_from_gml_simple_graph(self, mock_show) -> None:
        """Test rendering a simple GML graph."""
        from src.visualization.dag_visualizer import DAGRenderer

        # Create a simple GML string
        gml_string = (
            'graph [\n'
            '  directed 1\n'
            '  node [ id 0 label "A" ]\n'
            '  node [ id 1 label "B" ]\n'
            '  edge [ source 0 target 1 ]\n'
            ']\n'
        )

        # Create renderer (show_plots is False, so no interactive window)
        renderer = DAGRenderer()

        # Render should not raise
        renderer.render_from_gml(gml_string, title="Test DAG")

    @patch("matplotlib.pyplot.show")
    def test_render_from_gml_saves_file(self, mock_show) -> None:
        """Test that rendering saves a figure file when save_path is specified."""
        from src.visualization.dag_visualizer import DAGRenderer

        # Create a simple GML string
        gml_string = (
            'graph [\n'
            '  directed 1\n'
            '  node [ id 0 label "A" ]\n'
            '  node [ id 1 label "B" ]\n'
            '  edge [ source 0 target 1 ]\n'
            ']\n'
        )

        # Define save path
        save_path = os.path.join(self._output_dir, "test_output.png")

        # Render with save path
        renderer = DAGRenderer()
        renderer.render_from_gml(gml_string, save_path=save_path)

        # Verify file was created
        assert Path(save_path).exists()

    def test_compute_layout_spring_fallback(self) -> None:
        """Test that unknown nodes trigger spring layout fallback."""
        import networkx as nx
        from src.visualization.dag_visualizer import DAGRenderer

        # Create a graph with unknown node names
        graph = nx.DiGraph()
        graph.add_edge("unknown_1", "unknown_2")

        # Create renderer and compute layout
        renderer = DAGRenderer()
        positions = renderer._compute_layout(graph)

        # Verify positions were computed for both nodes
        assert "unknown_1" in positions
        assert "unknown_2" in positions

    def test_render_invalid_gml_raises_error(self) -> None:
        """Test that invalid GML string raises VisualizationError."""
        from src.visualization.dag_visualizer import DAGRenderer, VisualizationError

        # Create renderer
        renderer = DAGRenderer()

        # Attempt to render invalid GML
        with pytest.raises(VisualizationError):
            renderer.render_from_gml("this is not valid GML")
