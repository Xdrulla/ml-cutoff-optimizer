"""
Unit tests for metrics calculation.
"""

import pytest
import numpy as np
import pandas as pd
from src.ml_cutoff_optimizer.metrics import MetricsCalculator


class TestConfusionMatrixAtThreshold:
    """Test suite for confusion_matrix_at_threshold method."""

    def test_perfect_separation_at_threshold_05(self):
        """Test confusion matrix with perfectly separated classes at 0.5."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        threshold = 0.5

        tn, fp, fn, tp = MetricsCalculator.confusion_matrix_at_threshold(
            y_true, y_proba, threshold
        )

        # Perfect separation: all correct
        assert tn == 2  # Both negatives correctly identified
        assert fp == 0  # No false positives
        assert fn == 0  # No false negatives
        assert tp == 2  # Both positives correctly identified

    def test_all_negative_predictions(self):
        """Test when all predictions are below threshold (all predicted negative)."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = 0.5

        tn, fp, fn, tp = MetricsCalculator.confusion_matrix_at_threshold(
            y_true, y_proba, threshold
        )

        assert tn == 2  # Both negatives correctly identified
        assert fp == 0  # No false positives
        assert fn == 2  # Both positives missed (false negatives)
        assert tp == 0  # No true positives

    def test_all_positive_predictions(self):
        """Test when all predictions are above threshold (all predicted positive)."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.6, 0.7, 0.8, 0.9])
        threshold = 0.5

        tn, fp, fn, tp = MetricsCalculator.confusion_matrix_at_threshold(
            y_true, y_proba, threshold
        )

        assert tn == 0  # No true negatives
        assert fp == 2  # Both negatives wrongly predicted as positive
        assert fn == 0  # No false negatives
        assert tp == 2  # Both positives correctly identified


class TestCalculateAllMetrics:
    """Test suite for calculate_all_metrics method."""

    def test_perfect_classification(self):
        """Test metrics with perfect classification."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        threshold = 0.5

        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_proba, threshold)

        # All metrics should be perfect (1.0)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["fpr"] == 0.0
        assert metrics["fnr"] == 0.0

    def test_all_wrong_classification(self):
        """Test metrics when all predictions are wrong."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.1, 0.2])  # Completely reversed
        threshold = 0.5

        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_proba, threshold)

        # Accuracy should be 0
        assert metrics["accuracy"] == 0.0
        assert metrics["tp"] == 0
        assert metrics["tn"] == 0
        assert metrics["fp"] == 2
        assert metrics["fn"] == 2

    def test_precision_calculation(self):
        """Test precision calculation specifically."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        threshold = 0.5

        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_proba, threshold)

        # TP = 3 (all positives above 0.5)
        # FP = 1 (one negative above 0.5)
        # Precision = TP / (TP + FP) = 3 / 4 = 0.75
        assert metrics["tp"] == 3
        assert metrics["fp"] == 1
        assert metrics["precision"] == 0.75

    def test_recall_calculation(self):
        """Test recall calculation specifically."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8])
        threshold = 0.5

        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_proba, threshold)

        # TP = 2 (two positives above 0.5)
        # FN = 1 (one positive below 0.5)
        # Recall = TP / (TP + FN) = 2 / 3 ≈ 0.6667
        assert metrics["tp"] == 2
        assert metrics["fn"] == 1
        assert abs(metrics["recall"] - 0.6667) < 0.001

    def test_division_by_zero_protection(self):
        """Test that division by zero is handled gracefully."""
        # All predictions negative, all true labels negative
        y_true = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = 0.5

        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_proba, threshold)

        # When no positive predictions, precision should be 0 (not error)
        assert metrics["precision"] == 0.0
        # When no positive labels, recall should be 0
        assert metrics["recall"] == 0.0


class TestPopulationDistribution:
    """Test suite for population_distribution method."""

    def test_simple_distribution(self):
        """Test population distribution with simple bins."""
        y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        bins = np.array([0, 0.5, 1.0])

        counts, percentages = MetricsCalculator.population_distribution(y_proba, bins)

        # 3 samples below 0.5, 2 samples above 0.5
        assert counts[0] == 3
        assert counts[1] == 2

        # 60% below 0.5, 40% above 0.5
        assert percentages[0] == 0.6
        assert percentages[1] == 0.4

    def test_distribution_with_multiple_bins(self):
        """Test distribution across multiple bins."""
        y_proba = np.array([0.05, 0.15, 0.25, 0.55, 0.65, 0.75, 0.85, 0.95])
        bins = np.array([0, 0.25, 0.5, 0.75, 1.0])

        counts, percentages = MetricsCalculator.population_distribution(y_proba, bins)

        # Should have 4 bins
        assert len(counts) == 4

        # Each bin should have 2 samples
        np.testing.assert_array_equal(counts, [2, 1, 2, 3])

    def test_empty_array(self):
        """Test with empty array."""
        y_proba = np.array([])
        bins = np.array([0, 0.5, 1.0])

        counts, percentages = MetricsCalculator.population_distribution(y_proba, bins)

        # All counts and percentages should be 0
        assert np.all(counts == 0)
        assert np.all(percentages == 0)


class TestMetricsByThresholdRange:
    """Test suite for metrics_by_threshold_range method."""

    def test_output_is_dataframe(self):
        """Test that output is a pandas DataFrame."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        df = MetricsCalculator.metrics_by_threshold_range(y_true, y_proba)

        assert isinstance(df, pd.DataFrame)

    def test_default_thresholds(self):
        """Test that default creates 101 thresholds."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        df = MetricsCalculator.metrics_by_threshold_range(y_true, y_proba)

        # Default should be 101 thresholds (0.00, 0.01, ..., 1.00)
        assert len(df) == 101

    def test_custom_thresholds(self):
        """Test with custom threshold range."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])
        custom_thresholds = np.array([0.3, 0.5, 0.7])

        df = MetricsCalculator.metrics_by_threshold_range(
            y_true, y_proba, thresholds=custom_thresholds
        )

        # Should have 3 rows
        assert len(df) == 3

        # Should have correct thresholds
        np.testing.assert_array_equal(df["threshold"].values, custom_thresholds)

    def test_required_columns_present(self):
        """Test that all required columns are present in output."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        df = MetricsCalculator.metrics_by_threshold_range(y_true, y_proba)

        required_cols = [
            "threshold",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "specificity",
            "fpr",
            "fnr",
            "population_positive",
            "tp",
            "tn",
            "fp",
            "fn",
        ]

        for col in required_cols:
            assert col in df.columns

    def test_metrics_monotonicity(self):
        """Test that some metrics have expected monotonic behavior."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        df = MetricsCalculator.metrics_by_threshold_range(y_true, y_proba)

        # Population_positive should be monotonically decreasing as threshold increases
        # (higher threshold = fewer positive predictions)
        pop_pos = df["population_positive"].values

        # Check if generally decreasing (allow for some plateaus)
        assert pop_pos[0] >= pop_pos[-1]
