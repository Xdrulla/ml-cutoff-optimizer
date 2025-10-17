"""
Metrics calculation for binary classification threshold optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix
from .utils import validate_binary_inputs, validate_threshold


class MetricsCalculator:
    """
    Calculate comprehensive metrics for binary classification at any threshold.

    This class provides static methods to compute confusion matrix,
    classification metrics (precision, recall, F1, etc.), and population
    distributions across probability bins.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_proba = np.array([0.2, 0.4, 0.6, 0.9])
    >>> metrics = MetricsCalculator.calculate_all_metrics(y_true, y_proba, threshold=0.5)
    >>> metrics['accuracy']  # doctest: +SKIP
    1.0
    """

    @staticmethod
    def confusion_matrix_at_threshold(
        y_true: np.ndarray, y_proba: np.ndarray, threshold: float
    ) -> Tuple[int, int, int, int]:
        """
        Calculate confusion matrix at a specific threshold.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (0 or 1)
        y_proba : np.ndarray
            Predicted probabilities for positive class
        threshold : float
            Classification threshold (between 0 and 1)

        Returns
        -------
        tuple of int
            (tn, fp, fn, tp) - Confusion matrix values

        Examples
        --------
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_proba = np.array([0.2, 0.4, 0.6, 0.9])
        >>> tn, fp, fn, tp = MetricsCalculator.confusion_matrix_at_threshold(
        ...     y_true, y_proba, threshold=0.5
        ... )
        >>> print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        TP=2, TN=2, FP=0, FN=0
        """
        # Validate inputs
        y_true, y_proba = validate_binary_inputs(y_true, y_proba)
        threshold = validate_threshold(threshold)

        # Convert probabilities to predictions using threshold
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate confusion matrix with explicit labels to handle edge cases
        # sklearn returns: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return tn, fp, fn, tp

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_proba: np.ndarray, threshold: float
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics at a specific threshold.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        y_proba : np.ndarray
            Predicted probabilities for positive class
        threshold : float
            Classification threshold

        Returns
        -------
        dict
            Dictionary containing all metrics:
            - tp, tn, fp, fn: Confusion matrix values
            - accuracy: Overall accuracy
            - precision: Positive predictive value
            - recall: Sensitivity / True positive rate
            - specificity: True negative rate
            - f1: F1-score (harmonic mean of precision and recall)
            - fpr: False positive rate
            - fnr: False negative rate
            - population_positive: Percentage of positive predictions
            - population_negative: Percentage of negative predictions

        Examples
        --------
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_proba = np.array([0.2, 0.4, 0.6, 0.9])
        >>> metrics = MetricsCalculator.calculate_all_metrics(y_true, y_proba, 0.5)
        >>> round(metrics['accuracy'], 2)
        1.0
        >>> round(metrics['precision'], 2)
        1.0
        """
        # Get confusion matrix
        tn, fp, fn, tp = MetricsCalculator.confusion_matrix_at_threshold(
            y_true, y_proba, threshold
        )

        # Total samples
        total = len(y_true)

        # Calculate metrics (with division by zero protection)
        accuracy = (tp + tn) / total if total > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # F1 score (harmonic mean of precision and recall)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Error rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

        # Population distribution (predictions)
        y_pred = (y_proba >= threshold).astype(int)
        population_positive = np.sum(y_pred) / total if total > 0 else 0.0
        population_negative = 1 - population_positive

        return {
            # Confusion matrix
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            # Main metrics
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            # Error rates
            "fpr": float(fpr),  # False Positive Rate
            "fnr": float(fnr),  # False Negative Rate
            # Population
            "population_positive": float(population_positive),
            "population_negative": float(population_negative),
        }

    @staticmethod
    def population_distribution(
        y_proba: np.ndarray, bins: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate population percentage distribution across probability bins.

        Parameters
        ----------
        y_proba : np.ndarray
            Predicted probabilities
        bins : np.ndarray
            Bin edges (e.g., [0, 0.1, 0.2, ..., 1.0])

        Returns
        -------
        tuple of np.ndarray
            (counts, percentages) for each bin

        Examples
        --------
        >>> y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        >>> bins = np.array([0, 0.5, 1.0])
        >>> counts, percentages = MetricsCalculator.population_distribution(y_proba, bins)
        >>> percentages  # doctest: +SKIP
        array([0.6, 0.4])  # 60% below 0.5, 40% above 0.5
        """
        # Count how many samples fall in each bin
        counts, _ = np.histogram(y_proba, bins=bins)

        # Calculate percentages
        total = len(y_proba)
        percentages = counts / total if total > 0 else np.zeros_like(counts)

        return counts, percentages

    @staticmethod
    def metrics_by_threshold_range(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Calculate metrics for a range of thresholds.

        This is useful for analyzing how metrics change as threshold varies,
        which helps in choosing optimal cutoff points.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        y_proba : np.ndarray
            Predicted probabilities
        thresholds : np.ndarray, optional
            Array of thresholds to evaluate. If None, uses 100 evenly spaced
            thresholds between 0 and 1.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: threshold, precision, recall, f1, accuracy,
            specificity, fpr, fnr, population_positive

        Examples
        --------
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_proba = np.array([0.2, 0.4, 0.6, 0.9])
        >>> df = MetricsCalculator.metrics_by_threshold_range(y_true, y_proba)
        >>> df.shape[0] > 0
        True
        """
        # Validate inputs
        y_true, y_proba = validate_binary_inputs(y_true, y_proba)

        # Default thresholds: 0.00, 0.01, 0.02, ..., 1.00
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)

        # Calculate metrics for each threshold
        results = []
        for threshold in thresholds:
            metrics = MetricsCalculator.calculate_all_metrics(
                y_true, y_proba, threshold
            )
            metrics["threshold"] = threshold
            results.append(metrics)

        # Convert to DataFrame and reorder columns
        df = pd.DataFrame(results)
        cols = [
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
        df = df[cols]

        return df
