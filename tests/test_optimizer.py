"""
Unit tests for cutoff optimizer.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from src.ml_cutoff_optimizer.optimizer import CutoffOptimizer


class TestCutoffOptimizerInit:
    """Test suite for CutoffOptimizer initialization."""

    def test_initialization_with_valid_data(self):
        """Test that optimizer initializes correctly."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)

        assert optimizer is not None
        assert isinstance(optimizer.y_true, np.ndarray)
        assert isinstance(optimizer.y_proba, np.ndarray)

    def test_initialization_validates_inputs(self):
        """Test that initialization validates inputs."""
        # Different lengths
        with pytest.raises(ValueError):
            CutoffOptimizer([0, 1], [0.2, 0.8, 0.9])

        # Empty arrays
        with pytest.raises(ValueError):
            CutoffOptimizer([], [])


class TestCalculateMetricsMatrix:
    """Test suite for calculate_metrics_matrix method."""

    def test_returns_dataframe(self):
        """Test that method returns a pandas DataFrame."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        df = optimizer.calculate_metrics_matrix()

        assert isinstance(df, pd.DataFrame)

    def test_default_creates_101_rows(self):
        """Test that default threshold range creates 101 rows."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        df = optimizer.calculate_metrics_matrix()

        # Should have 101 thresholds (0.00 to 1.00)
        assert len(df) == 101

    def test_custom_thresholds(self):
        """Test with custom threshold range."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        custom = np.array([0.3, 0.5, 0.7])
        df = optimizer.calculate_metrics_matrix(thresholds=custom)

        assert len(df) == 3
        np.testing.assert_array_equal(df["threshold"].values, custom)

    def test_stores_metrics_df(self):
        """Test that metrics DataFrame is stored in optimizer."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        df = optimizer.calculate_metrics_matrix()

        # Should store in optimizer.metrics_df
        assert optimizer.metrics_df is not None
        assert optimizer.metrics_df is df


class TestSuggestThreeZones:
    """Test suite for suggest_three_zones method."""

    def test_returns_required_keys(self):
        """Test that result contains all required keys."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones()

        # Check required keys
        required_keys = [
            "negative_cutoff",
            "positive_cutoff",
            "manual_zone",
            "justification",
            "metrics",
            "population",
        ]
        for key in required_keys:
            assert key in result

    def test_cutoffs_are_valid(self):
        """Test that suggested cutoffs are in valid range."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones()

        neg_cutoff = result["negative_cutoff"]
        pos_cutoff = result["positive_cutoff"]

        # Both should be between 0 and 1
        assert 0 <= neg_cutoff <= 1
        assert 0 <= pos_cutoff <= 1

        # Negative cutoff should be <= positive cutoff
        assert neg_cutoff <= pos_cutoff

    def test_manual_zone_tuple(self):
        """Test that manual_zone is a tuple with two values."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones()

        manual_zone = result["manual_zone"]

        # Should be a tuple
        assert isinstance(manual_zone, tuple)
        assert len(manual_zone) == 2

        # First value should be negative_cutoff, second should be positive_cutoff
        assert manual_zone[0] == result["negative_cutoff"]
        assert manual_zone[1] == result["positive_cutoff"]

    def test_population_distribution_sums_to_100(self):
        """Test that population percentages sum to approximately 100%."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.15, 0.85])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones()

        pop = result["population"]
        total_pct = (
            pop["negative_zone_pct"] + pop["manual_zone_pct"] + pop["positive_zone_pct"]
        )

        # Should sum to 100% (with small floating point tolerance)
        assert abs(total_pct - 100.0) < 0.01

    def test_population_counts_sum_to_total(self):
        """Test that population counts sum to total samples."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.15, 0.85])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones()

        pop = result["population"]
        total_count = (
            pop["negative_zone_count"]
            + pop["manual_zone_count"]
            + pop["positive_zone_count"]
        )

        assert total_count == len(y_true)

    def test_metrics_structure(self):
        """Test that metrics dictionary has correct structure."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones()

        metrics = result["metrics"]

        # Should have negative_zone and positive_zone
        assert "negative_zone" in metrics
        assert "positive_zone" in metrics

        # Each should have required metric keys
        for zone in ["negative_zone", "positive_zone"]:
            zone_metrics = metrics[zone]
            assert "precision" in zone_metrics
            assert "recall" in zone_metrics
            assert "specificity" in zone_metrics
            assert "accuracy" in zone_metrics

    def test_justification_is_string(self):
        """Test that justification is a non-empty string."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones()

        justification = result["justification"]

        assert isinstance(justification, str)
        assert len(justification) > 0

    def test_custom_min_metric_value(self):
        """Test with custom minimum metric value."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        optimizer = CutoffOptimizer(y_true, y_proba)

        # Lower min_metric_value should generally give wider zones
        result_strict = optimizer.suggest_three_zones(min_metric_value=0.95)
        result_lenient = optimizer.suggest_three_zones(min_metric_value=0.70)

        # Both should be valid
        assert 0 <= result_strict["negative_cutoff"] <= 1
        assert 0 <= result_lenient["negative_cutoff"] <= 1

    def test_max_manual_zone_width_constraint(self):
        """Test that manual zone width respects maximum."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones(max_manual_zone_width=0.2)

        manual_width = result["positive_cutoff"] - result["negative_cutoff"]

        # Manual zone width should be <= 0.2 (20%)
        assert manual_width <= 0.2 + 0.01  # Small tolerance for floating point

    def test_perfect_separation_case(self):
        """Test with perfectly separated classes."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        optimizer = CutoffOptimizer(y_true, y_proba)
        result = optimizer.suggest_three_zones(min_metric_value=0.99)

        # With perfect separation and high min_metric_value,
        # should find cutoffs that separate classes well
        assert result["negative_cutoff"] < 0.5
        assert result["positive_cutoff"] > 0.5

    def test_difficult_case_all_similar_probabilities(self):
        """Test with difficult case where all probabilities are similar."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.48, 0.49, 0.51, 0.52, 0.50, 0.49])

        optimizer = CutoffOptimizer(y_true, y_proba)

        # Should not raise error even with difficult data
        result = optimizer.suggest_three_zones(min_metric_value=0.5)

        # Should still return valid cutoffs
        assert 0 <= result["negative_cutoff"] <= 1
        assert 0 <= result["positive_cutoff"] <= 1


class TestPlotMetricsEvolution:
    """Test suite for plot_metrics_evolution method."""

    def test_creates_plot(self):
        """Test that plot is created without errors."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        fig, ax = optimizer.plot_metrics_evolution()

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        plt.close(fig)

    def test_plot_with_custom_metrics(self):
        """Test plot with custom metrics list."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)
        fig, ax = optimizer.plot_metrics_evolution(metrics=["precision", "recall"])

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_calculates_metrics_if_not_done(self):
        """Test that metrics are calculated if not already done."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        optimizer = CutoffOptimizer(y_true, y_proba)

        # Should be None initially
        assert optimizer.metrics_df is None

        # Plot should calculate metrics
        fig, ax = optimizer.plot_metrics_evolution()

        # Now should not be None
        assert optimizer.metrics_df is not None

        plt.close(fig)
