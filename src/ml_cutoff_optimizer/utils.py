"""
Utility functions for validation and data processing.
"""

import numpy as np
from typing import Tuple, Union


def validate_binary_inputs(
    y_true: Union[list, np.ndarray], y_proba: Union[list, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert binary classification inputs to numpy arrays.

    Parameters
    ----------
    y_true : array-like
        True binary labels (must contain only 0s and 1s)
    y_proba : array-like
        Predicted probabilities for positive class (must be between 0 and 1)

    Returns
    -------
    tuple of np.ndarray
        Validated (y_true, y_proba) as numpy arrays

    Raises
    ------
    ValueError
        If inputs are invalid (wrong shape, wrong values, etc.)

    Examples
    --------
    >>> y_true = [0, 1, 1, 0]
    >>> y_proba = [0.2, 0.8, 0.9, 0.1]
    >>> y_true_valid, y_proba_valid = validate_binary_inputs(y_true, y_proba)
    >>> type(y_true_valid)
    <class 'numpy.ndarray'>
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Check if arrays are not empty
    if y_true.size == 0 or y_proba.size == 0:
        raise ValueError("Input arrays cannot be empty")

    # Check if arrays have the same length
    if len(y_true) != len(y_proba):
        raise ValueError(
            f"y_true and y_proba must have the same length. "
            f"Got {len(y_true)} and {len(y_proba)}"
        )

    # Flatten arrays (convert [[0, 1]] to [0, 1])
    y_true = y_true.flatten()
    y_proba = y_proba.flatten()

    # Check if y_true contains only 0s and 1s
    unique_values = np.unique(y_true)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"y_true must contain only 0s and 1s. Got unique values: {unique_values}"
        )

    # Validate probabilities
    y_proba = validate_probabilities(y_proba)

    return y_true, y_proba


def validate_probabilities(y_proba: np.ndarray) -> np.ndarray:
    """
    Validate that probabilities are in valid range [0, 1].

    Parameters
    ----------
    y_proba : np.ndarray
        Array of probabilities

    Returns
    -------
    np.ndarray
        Validated probabilities

    Raises
    ------
    ValueError
        If probabilities are outside [0, 1] range

    Examples
    --------
    >>> probs = np.array([0.1, 0.5, 0.9])
    >>> validated = validate_probabilities(probs)
    >>> np.all((validated >= 0) & (validated <= 1))
    True
    """
    if np.any(y_proba < 0) or np.any(y_proba > 1):
        raise ValueError(
            f"Probabilities must be between 0 and 1. "
            f"Got min={y_proba.min():.4f}, max={y_proba.max():.4f}"
        )

    return y_proba


def validate_step(step: float) -> float:
    """
    Validate step size for probability bins.

    Parameters
    ----------
    step : float
        Step size (e.g., 0.1 means 10% bins)

    Returns
    -------
    float
        Validated step size

    Raises
    ------
    ValueError
        If step is not in valid range (0, 1]

    Examples
    --------
    >>> validate_step(0.1)
    0.1
    >>> validate_step(1.5)  # doctest: +SKIP
    ValueError: Step must be between 0 and 1
    """
    if step <= 0 or step > 1:
        raise ValueError(f"Step must be between 0 and 1 (exclusive of 0). Got {step}")

    return step


def validate_threshold(threshold: float) -> float:
    """
    Validate a single threshold value.

    Parameters
    ----------
    threshold : float
        Threshold value to validate

    Returns
    -------
    float
        Validated threshold

    Raises
    ------
    ValueError
        If threshold is not in valid range [0, 1]

    Examples
    --------
    >>> validate_threshold(0.5)
    0.5
    """
    if threshold < 0 or threshold > 1:
        raise ValueError(f"Threshold must be between 0 and 1. Got {threshold}")

    return threshold
