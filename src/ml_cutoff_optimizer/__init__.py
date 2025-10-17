"""
ML Cutoff Optimizer - Professional toolkit for binary classification threshold optimization.

This package provides tools for:
- Visualizing probability distributions
- Optimizing decision thresholds
- Calculating comprehensive metrics
- Suggesting intelligent three-zone classifications
"""

__version__ = "0.1.0"
__author__ = "Luan Drulla"
__email__ = "serighelli003@gmail.com"

# Import main classes for easy access
from .visualizer import ThresholdVisualizer
from .optimizer import CutoffOptimizer
from .metrics import MetricsCalculator

# Define what gets imported with "from ml_cutoff_optimizer import *"
__all__ = [
    "ThresholdVisualizer",
    "CutoffOptimizer",
    "MetricsCalculator",
]
