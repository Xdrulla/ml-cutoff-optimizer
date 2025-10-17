"""
Optimization algorithms for finding optimal classification thresholds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from .utils import validate_binary_inputs
from .metrics import MetricsCalculator


class CutoffOptimizer:
    """
    Find optimal threshold cutoffs for three-zone binary classification.

    This class analyzes the relationship between thresholds and metrics
    to suggest intelligent cutoff points that divide predictions into:
    - Negative Zone: High confidence for class 0
    - Manual Zone: Uncertain predictions requiring human review
    - Positive Zone: High confidence for class 1

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1)
    y_proba : array-like
        Predicted probabilities for positive class

    Attributes
    ----------
    y_true : np.ndarray
        Validated true labels
    y_proba : np.ndarray
        Validated predicted probabilities
    metrics_df : pd.DataFrame
        Metrics calculated for all thresholds (populated after calling methods)

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> model = LogisticRegression()
    >>> model.fit(X, y)  # doctest: +SKIP
    >>> y_proba = model.predict_proba(X)[:, 1]  # doctest: +SKIP
    >>> optimizer = CutoffOptimizer(y, y_proba)  # doctest: +SKIP
    >>> cutoffs = optimizer.suggest_three_zones()  # doctest: +SKIP
    """

    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray):
        """Initialize optimizer with data."""
        # Validate inputs
        self.y_true, self.y_proba = validate_binary_inputs(y_true, y_proba)

        # Will store metrics DataFrame
        self.metrics_df = None

    def calculate_metrics_matrix(
        self, thresholds: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Calculate metrics for all possible thresholds.

        Parameters
        ----------
        thresholds : np.ndarray, optional
            Custom thresholds to evaluate. If None, uses 101 evenly spaced
            thresholds from 0 to 1.

        Returns
        -------
        pd.DataFrame
            DataFrame with metrics for each threshold

        Examples
        --------
        >>> optimizer = CutoffOptimizer([0, 1, 1, 0], [0.2, 0.8, 0.9, 0.1])
        >>> df = optimizer.calculate_metrics_matrix()
        >>> 'threshold' in df.columns
        True
        """
        # Use MetricsCalculator to get metrics for all thresholds
        self.metrics_df = MetricsCalculator.metrics_by_threshold_range(
            self.y_true, self.y_proba, thresholds
        )

        return self.metrics_df

    def suggest_three_zones(
        self,
        negative_zone_metric: str = "specificity",
        positive_zone_metric: str = "recall",
        min_metric_value: float = 0.80,
        max_manual_zone_width: float = 0.40,
    ) -> Dict:
        """
        Suggest optimal cutoffs for three decision zones.

        Algorithm:
        1. Calculate metrics for all thresholds (0.00 to 1.00)
        2. Find negative cutoff: highest threshold where specificity >= target
        3. Find positive cutoff: lowest threshold where recall >= target
        4. Manual zone = everything between the two cutoffs

        Parameters
        ----------
        negative_zone_metric : str, default='specificity'
            Metric to optimize for negative zone ('specificity', 'precision', 'accuracy')
        positive_zone_metric : str, default='recall'
            Metric to optimize for positive zone ('recall', 'precision', 'f1')
        min_metric_value : float, default=0.80
            Minimum acceptable value for the target metrics (0-1)
        max_manual_zone_width : float, default=0.40
            Maximum width of manual zone (0-1). If exceeded, adjusts cutoffs.

        Returns
        -------
        dict
            Dictionary with keys:
            - negative_cutoff: Upper bound for negative zone (0 to this value)
            - positive_cutoff: Lower bound for positive zone (this value to 1)
            - manual_zone: Tuple (lower, upper) for manual review zone
            - justification: Human-readable explanation
            - metrics: Detailed metrics for each zone

        Examples
        --------
        >>> optimizer = CutoffOptimizer([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        >>> result = optimizer.suggest_three_zones()
        >>> 'negative_cutoff' in result
        True
        >>> 'positive_cutoff' in result
        True
        """
        # Calculate metrics if not already done
        if self.metrics_df is None:
            self.calculate_metrics_matrix()

        df = self.metrics_df.copy()

        # --- FIND NEGATIVE CUTOFF ---
        # Goal: Find highest threshold where we still capture most negatives
        # (high specificity = good at identifying negatives)

        # Filter rows where negative metric is >= min_metric_value
        negative_candidates = df[df[negative_zone_metric] >= min_metric_value]

        if len(negative_candidates) > 0:
            # Choose the HIGHEST threshold (most restrictive for negative zone)
            negative_cutoff = negative_candidates["threshold"].max()
        else:
            # Fallback: use threshold where metric is maximized
            negative_cutoff = df.loc[df[negative_zone_metric].idxmax(), "threshold"]

        # --- FIND POSITIVE CUTOFF ---
        # Goal: Find lowest threshold where we still capture most positives
        # (high recall = good at identifying positives)

        # Filter rows where positive metric is >= min_metric_value
        positive_candidates = df[df[positive_zone_metric] >= min_metric_value]

        if len(positive_candidates) > 0:
            # Choose the LOWEST threshold (most restrictive for positive zone)
            positive_cutoff = positive_candidates["threshold"].min()
        else:
            # Fallback: use threshold where metric is maximized
            positive_cutoff = df.loc[df[positive_zone_metric].idxmax(), "threshold"]

        # --- VALIDATE MANUAL ZONE WIDTH ---
        manual_zone_width = positive_cutoff - negative_cutoff

        if manual_zone_width < 0:
            # Zones overlap! Adjust to create small manual zone
            midpoint = (negative_cutoff + positive_cutoff) / 2
            negative_cutoff = midpoint - 0.05
            positive_cutoff = midpoint + 0.05
            manual_zone_width = 0.10

        if manual_zone_width > max_manual_zone_width:
            # Manual zone too wide, narrow it
            center = (negative_cutoff + positive_cutoff) / 2
            half_width = max_manual_zone_width / 2
            negative_cutoff = center - half_width
            positive_cutoff = center + half_width
            manual_zone_width = max_manual_zone_width

        # --- CALCULATE ZONE METRICS ---
        negative_metrics = MetricsCalculator.calculate_all_metrics(
            self.y_true, self.y_proba, negative_cutoff
        )
        positive_metrics = MetricsCalculator.calculate_all_metrics(
            self.y_true, self.y_proba, positive_cutoff
        )

        # Calculate population in each zone
        in_negative_zone = np.sum(self.y_proba < negative_cutoff)
        in_manual_zone = np.sum(
            (self.y_proba >= negative_cutoff) & (self.y_proba < positive_cutoff)
        )
        in_positive_zone = np.sum(self.y_proba >= positive_cutoff)

        total = len(self.y_proba)
        pct_negative = (in_negative_zone / total) * 100
        pct_manual = (in_manual_zone / total) * 100
        pct_positive = (in_positive_zone / total) * 100

        # --- GENERATE JUSTIFICATION ---
        justification = self._generate_justification(
            negative_cutoff,
            positive_cutoff,
            negative_metrics,
            positive_metrics,
            pct_negative,
            pct_manual,
            pct_positive,
            negative_zone_metric,
            positive_zone_metric,
        )

        # --- RETURN RESULTS ---
        return {
            "negative_cutoff": float(negative_cutoff),
            "positive_cutoff": float(positive_cutoff),
            "manual_zone": (float(negative_cutoff), float(positive_cutoff)),
            "justification": justification,
            "metrics": {
                "negative_zone": negative_metrics,
                "positive_zone": positive_metrics,
            },
            "population": {
                "negative_zone_pct": float(pct_negative),
                "manual_zone_pct": float(pct_manual),
                "positive_zone_pct": float(pct_positive),
                "negative_zone_count": int(in_negative_zone),
                "manual_zone_count": int(in_manual_zone),
                "positive_zone_count": int(in_positive_zone),
            },
        }

    def _generate_justification(
        self,
        negative_cutoff: float,
        positive_cutoff: float,
        negative_metrics: Dict,
        positive_metrics: Dict,
        pct_negative: float,
        pct_manual: float,
        pct_positive: float,
        negative_metric_name: str,
        positive_metric_name: str,
    ) -> str:
        """
        Generate human-readable justification for cutoff choices.

        Parameters
        ----------
        (various parameters from suggest_three_zones)

        Returns
        -------
        str
            Multi-line justification text
        """
        justification = f"""
CUTOFF ANALYSIS REPORT
{'=' * 60}

SUGGESTED CUTOFFS:
  • Negative Zone: 0% - {negative_cutoff*100:.1f}%
  • Manual Zone:   {negative_cutoff*100:.1f}% - {positive_cutoff*100:.1f}%
  • Positive Zone: {positive_cutoff*100:.1f}% - 100%

POPULATION DISTRIBUTION:
  • {pct_negative:.1f}% of samples fall in Negative Zone
  • {pct_manual:.1f}% of samples require Manual Review
  • {pct_positive:.1f}% of samples fall in Positive Zone

NEGATIVE ZONE PERFORMANCE (threshold < {negative_cutoff:.2f}):
  • {negative_metric_name.capitalize()}: {negative_metrics[negative_metric_name]:.2%}
  • Specificity: {negative_metrics['specificity']:.2%}
  • False Positive Rate: {negative_metrics['fpr']:.2%}
  • Captures {negative_metrics['tn']} true negatives, {negative_metrics['fp']} false positives

POSITIVE ZONE PERFORMANCE (threshold >= {positive_cutoff:.2f}):
  • {positive_metric_name.capitalize()}: {positive_metrics[positive_metric_name]:.2%}
  • Recall: {positive_metrics['recall']:.2%}
  • Precision: {positive_metrics['precision']:.2%}
  • False Negative Rate: {positive_metrics['fnr']:.2%}
  • Captures {positive_metrics['tp']} true positives, {positive_metrics['fn']} false negatives

RECOMMENDATION:
  The negative cutoff ({negative_cutoff*100:.0f}%) ensures high confidence in 
  rejecting negative cases with {negative_metrics['specificity']:.1%} specificity.
  
  The positive cutoff ({positive_cutoff*100:.0f}%) ensures high confidence in 
  accepting positive cases with {positive_metrics['recall']:.1%} recall.
  
  The manual zone ({pct_manual:.1f}% of cases) represents uncertain predictions
  that would benefit from human review to minimize errors.
{'=' * 60}
        """
        return justification.strip()

    def plot_metrics_evolution(
        self, metrics: list = None, figsize: Tuple[int, int] = (14, 8)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot how metrics evolve as threshold changes.

        Parameters
        ----------
        metrics : list, optional
            List of metrics to plot. Default: ['precision', 'recall', 'f1', 'accuracy']
        figsize : tuple, default=(14, 8)
            Figure size

        Returns
        -------
        tuple
            (figure, axes) matplotlib objects

        Examples
        --------
        >>> optimizer = CutoffOptimizer([0, 1, 1, 0], [0.2, 0.8, 0.9, 0.1])
        >>> fig, ax = optimizer.plot_metrics_evolution()  # doctest: +SKIP
        """
        if self.metrics_df is None:
            self.calculate_metrics_matrix()

        if metrics is None:
            metrics = ["precision", "recall", "f1", "accuracy"]

        fig, ax = plt.subplots(figsize=figsize)

        for metric in metrics:
            ax.plot(
                self.metrics_df["threshold"],
                self.metrics_df[metric],
                label=metric.capitalize(),
                linewidth=2.5,
                alpha=0.8,
            )

        ax.set_xlabel("Threshold", fontsize=12, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
        ax.set_title(
            "Metrics Evolution by Threshold", fontsize=14, fontweight="bold", pad=20
        )
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        return fig, ax
