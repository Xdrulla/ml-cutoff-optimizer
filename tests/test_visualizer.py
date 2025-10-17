"""
Unit tests for threshold visualizer.
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from src.ml_cutoff_optimizer.visualizer import ThresholdVisualizer


class TestThresholdVisualizerInit:
    """Test suite for ThresholdVisualizer initialization."""

    def test_initialization_with_valid_data(self):
        """Test that visualizer initializes correctly with valid data."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba, step=0.1)

        assert viz is not None
        assert isinstance(viz.y_true, np.ndarray)
        assert isinstance(viz.y_proba, np.ndarray)
        assert viz.step == 0.1

    def test_bins_creation_with_default_step(self):
        """Test that bins are created correctly with default step (0.1)."""
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.9, 0.1])

        viz = ThresholdVisualizer(y_true, y_proba)  # Default step=0.1

        # With step=0.1, bins should be [0, 0.1, 0.2, ..., 1.0] = 11 edges
        expected_bins = np.arange(0, 1.1, 0.1)
        np.testing.assert_array_almost_equal(viz.bins, expected_bins, decimal=5)

    def test_bins_creation_with_custom_step(self):
        """Test bins creation with custom step size."""
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.9, 0.1])

        viz = ThresholdVisualizer(y_true, y_proba, step=0.05)

        # With step=0.05, bins should be [0, 0.05, 0.10, ..., 1.0] = 21 edges
        expected_bins = np.arange(0, 1.05, 0.05)
        np.testing.assert_array_almost_equal(viz.bins, expected_bins, decimal=5)

    def test_class_separation(self):
        """Test that data is correctly separated by class."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)

        # Positive class probabilities
        np.testing.assert_array_equal(viz.y_proba_positive, np.array([0.6, 0.8]))

        # Negative class probabilities
        np.testing.assert_array_equal(viz.y_proba_negative, np.array([0.2, 0.4]))

    def test_invalid_step_raises_error(self):
        """Test that invalid step size raises error."""
        y_true = np.array([0, 1])
        y_proba = np.array([0.2, 0.8])

        with pytest.raises(ValueError):
            ThresholdVisualizer(y_true, y_proba, step=0.0)  # Invalid step

        with pytest.raises(ValueError):
            ThresholdVisualizer(y_true, y_proba, step=-0.1)  # Invalid step

    def test_invalid_inputs_raise_error(self):
        """Test that invalid inputs raise appropriate errors."""
        # Different lengths
        with pytest.raises(ValueError):
            ThresholdVisualizer([0, 1], [0.2, 0.8, 0.9])

        # Empty arrays
        with pytest.raises(ValueError):
            ThresholdVisualizer([], [])

        # Non-binary labels
        with pytest.raises(ValueError):
            ThresholdVisualizer([0, 1, 2], [0.2, 0.8, 0.9])


class TestPlotDistributions:
    """Test suite for plot_distributions method."""

    def test_plot_creates_figure_and_axes(self):
        """Test that plot creates matplotlib figure and axes objects."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)
        fig, ax = viz.plot_distributions()

        # Should return figure and axes
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Should store them in visualizer
        assert viz.fig is fig
        assert viz.ax is ax

        plt.close(fig)  # Clean up

    def test_plot_with_custom_figsize(self):
        """Test plot with custom figure size."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)
        fig, ax = viz.plot_distributions(figsize=(10, 5))

        # Check figure size
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 5

        plt.close(fig)

    def test_plot_with_custom_title(self):
        """Test plot with custom title."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)
        fig, ax = viz.plot_distributions(title="My Custom Title")

        # Check title
        assert ax.get_title() == "My Custom Title"

        plt.close(fig)

    def test_plot_has_correct_labels(self):
        """Test that plot has correct axis labels."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)
        fig, ax = viz.plot_distributions()

        # Check axis labels
        assert "Probability" in ax.get_xlabel()
        assert "Population" in ax.get_ylabel() or "Percentage" in ax.get_ylabel()

        plt.close(fig)

    def test_plot_with_larger_dataset(self):
        """Test plot with larger dataset to ensure it handles scale."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_proba = np.random.random(1000)

        viz = ThresholdVisualizer(y_true, y_proba, step=0.05)
        fig, ax = viz.plot_distributions()

        # Should create plot without errors
        assert fig is not None
        assert ax is not None

        plt.close(fig)


class TestAddCutoffLines:
    """Test suite for add_cutoff_lines method."""

    def test_add_single_cutoff_line(self):
        """Test adding a single cutoff line."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)
        fig, ax = viz.plot_distributions()

        # Add cutoff line
        viz.add_cutoff_lines({"cutoff": 0.5})

        # Should not raise error
        assert viz.ax is not None

        plt.close(fig)

    def test_add_multiple_cutoff_lines(self):
        """Test adding multiple cutoff lines."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)
        fig, ax = viz.plot_distributions()

        # Add multiple cutoff lines
        viz.add_cutoff_lines({"negative_cutoff": 0.3, "positive_cutoff": 0.7})

        # Should not raise error
        assert viz.ax is not None

        plt.close(fig)

    def test_add_cutoff_without_plot_raises_error(self):
        """Test that adding cutoff before plotting raises error."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)

        # Try to add cutoff without plotting first
        with pytest.raises(ValueError, match="Must call plot_distributions"):
            viz.add_cutoff_lines({"cutoff": 0.5})

    def test_add_cutoff_with_custom_labels(self):
        """Test adding cutoffs with custom labels."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)
        fig, ax = viz.plot_distributions()

        # Add with custom labels
        viz.add_cutoff_lines(
            cutoffs={"my_cutoff": 0.5}, labels={"my_cutoff": "Custom Label"}
        )

        # Should not raise error
        assert viz.ax is not None

        plt.close(fig)


class TestSavePlot:
    """Test suite for save_plot method."""

    def test_save_plot_without_plotting_raises_error(self):
        """Test that saving without plotting raises error."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)

        with pytest.raises(ValueError, match="No plot to save"):
            viz.save_plot("test.png")

    def test_save_plot_after_plotting(self, tmp_path):
        """Test saving plot after creating it."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        viz = ThresholdVisualizer(y_true, y_proba)
        fig, ax = viz.plot_distributions()

        # Save to temporary file
        filepath = tmp_path / "test_plot.png"
        viz.save_plot(str(filepath))

        # File should exist
        assert filepath.exists()

        plt.close(fig)
