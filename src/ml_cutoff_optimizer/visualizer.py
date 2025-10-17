"""
Visualization tools for threshold analysis in binary classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from .utils import validate_binary_inputs, validate_step
from .metrics import MetricsCalculator


class ThresholdVisualizer:
    """
    Create professional visualizations for binary classification threshold analysis.

    This class generates overlapping histograms showing how predicted probabilities
    distribute across the population, with separate visualization for positive class.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1)
    y_proba : array-like
        Predicted probabilities for positive class (between 0 and 1)
    step : float, default=0.1
        Step size for probability bins (e.g., 0.1 = 10% bins, 0.05 = 5% bins)

    Attributes
    ----------
    y_true : np.ndarray
        Validated true labels
    y_proba : np.ndarray
        Validated predicted probabilities
    step : float
        Bin step size
    bins : np.ndarray
        Bin edges for histogram

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)  # doctest: +SKIP
    >>> y_proba = model.predict_proba(X_test)[:, 1]  # doctest: +SKIP
    >>> viz = ThresholdVisualizer(y_test, y_proba, step=0.05)  # doctest: +SKIP
    >>> viz.plot_distributions()  # doctest: +SKIP
    """

    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray, step: float = 0.1):
        """Initialize the visualizer with data and parameters."""
        # Validate inputs
        self.y_true, self.y_proba = validate_binary_inputs(y_true, y_proba)
        self.step = validate_step(step)

        # Create bins based on step size
        # Example: step=0.1 → bins=[0, 0.1, 0.2, ..., 1.0]
        self.bins = np.arange(0, 1 + step, step)

        # Separate probabilities by class
        self.y_proba_positive = self.y_proba[self.y_true == 1]  # Only class 1
        self.y_proba_negative = self.y_proba[self.y_true == 0]  # Only class 0

        # Store figure and axes for later use
        self.fig = None
        self.ax = None

    def plot_distributions(
        self,
        figsize: Tuple[int, int] = (14, 7),
        title: str = "Probability Distribution Analysis",
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot overlapping histograms of probability distributions.

        Creates a professional visualization with:
        - Blue bars: Overall population distribution
        - Red bars: Positive class (y=1) distribution
        - Grid for easy reading
        - Percentage on Y-axis
        - Probability (0-100%) on X-axis

        Parameters
        ----------
        figsize : tuple, default=(14, 7)
            Figure size in inches (width, height)
        title : str, default="Probability Distribution Analysis"
            Plot title
        save_path : str, optional
            If provided, saves the plot to this path

        Returns
        -------
        tuple
            (figure, axes) matplotlib objects

        Examples
        --------
        >>> viz = ThresholdVisualizer([0, 1, 1, 0], [0.2, 0.8, 0.9, 0.1])
        >>> fig, ax = viz.plot_distributions()  # doctest: +SKIP
        """
        # Set style
        sns.set_style("whitegrid")

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Calculate histograms (counts and percentages)
        total_counts, _ = np.histogram(self.y_proba, bins=self.bins)
        positive_counts, _ = np.histogram(self.y_proba_positive, bins=self.bins)

        # Convert to percentages
        total_percentages = (total_counts / len(self.y_proba)) * 100
        positive_percentages = (positive_counts / len(self.y_proba)) * 100

        # Calculate bin centers for plotting
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        bin_width = self.step * 0.8  # 80% of step to avoid overlap

        # Plot overall population (blue)
        self.ax.bar(
            bin_centers,
            total_percentages,
            width=bin_width,
            alpha=0.6,
            color="steelblue",
            edgecolor="navy",
            label="Overall Population",
            linewidth=1.2,
        )

        # Plot positive class (red)
        self.ax.bar(
            bin_centers,
            positive_percentages,
            width=bin_width,
            alpha=0.7,
            color="crimson",
            edgecolor="darkred",
            label="Positive Class (y=1)",
            linewidth=1.2,
        )

        # Formatting
        self.ax.set_xlabel("Predicted Probability (%)", fontsize=12, fontweight="bold")
        self.ax.set_ylabel("Population Percentage (%)", fontsize=12, fontweight="bold")
        self.ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # X-axis: show as percentages (0%, 10%, 20%, ..., 100%)
        x_ticks = np.arange(0, 1.1, 0.1)
        self.ax.set_xticks(x_ticks)
        self.ax.set_xticklabels([f"{int(x*100)}%" for x in x_ticks])

        # Grid
        self.ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)

        # Legend
        self.ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

        # Add statistics box
        self._add_statistics_box()

        # Tight layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            self.fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Plot saved to: {save_path}")

        return self.fig, self.ax

    def add_cutoff_lines(
        self, cutoffs: Dict[str, float], labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add vertical lines indicating suggested cutoff points.

        Parameters
        ----------
        cutoffs : dict
            Dictionary with cutoff values, e.g.:
            {'negative_cutoff': 0.3, 'positive_cutoff': 0.7}
        labels : dict, optional
            Custom labels for each cutoff line

        Examples
        --------
        >>> viz = ThresholdVisualizer([0, 1, 1, 0], [0.2, 0.8, 0.9, 0.1])
        >>> fig, ax = viz.plot_distributions()  # doctest: +SKIP
        >>> viz.add_cutoff_lines({'negative': 0.3, 'positive': 0.7})  # doctest: +SKIP
        """
        if self.ax is None:
            raise ValueError("Must call plot_distributions() first")

        # Default labels
        if labels is None:
            labels = {}

        # Color mapping for different cutoff types
        colors = {
            "negative_cutoff": "green",
            "positive_cutoff": "orange",
            "manual_lower": "purple",
            "manual_upper": "purple",
        }

        # Add vertical lines
        for key, value in cutoffs.items():
            color = colors.get(key, "black")
            label = labels.get(key, f"{key}: {value:.2%}")

            self.ax.axvline(
                x=value,
                color=color,
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
                label=label,
            )

        # Update legend
        self.ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    def _add_statistics_box(self) -> None:
        """Add a text box with summary statistics."""
        total_samples = len(self.y_proba)
        positive_samples = len(self.y_proba_positive)
        negative_samples = len(self.y_proba_negative)

        positive_pct = (positive_samples / total_samples) * 100
        negative_pct = (negative_samples / total_samples) * 100

        # Statistics text
        stats_text = (
            f"Total Samples: {total_samples:,}\n"
            f"Positive (y=1): {positive_samples:,} ({positive_pct:.1f}%)\n"
            f"Negative (y=0): {negative_samples:,} ({negative_pct:.1f}%)"
        )

        # Add text box
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        self.ax.text(
            0.02,
            0.98,
            stats_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            family="monospace",
        )

    def save_plot(self, filepath: str, dpi: int = 300) -> None:
        """
        Save the current plot to a file.

        Parameters
        ----------
        filepath : str
            Path where to save the plot (e.g., 'output/plot.png')
        dpi : int, default=300
            Resolution in dots per inch

        Examples
        --------
        >>> viz = ThresholdVisualizer([0, 1], [0.2, 0.8])
        >>> viz.plot_distributions()  # doctest: +SKIP
        >>> viz.save_plot('my_plot.png')  # doctest: +SKIP
        """
        if self.fig is None:
            raise ValueError("No plot to save. Call plot_distributions() first.")

        self.fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"✅ Plot saved to: {filepath}")
