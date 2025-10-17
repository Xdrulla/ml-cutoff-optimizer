"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from src.ml_cutoff_optimizer.utils import (
    validate_binary_inputs,
    validate_probabilities,
    validate_step,
    validate_threshold,
)


class TestValidateBinaryInputs:
    """Test suite for validate_binary_inputs function."""

    def test_valid_inputs_as_lists(self):
        """Test that valid lists are accepted and converted to arrays."""
        y_true = [0, 1, 1, 0]
        y_proba = [0.2, 0.8, 0.9, 0.1]

        y_true_out, y_proba_out = validate_binary_inputs(y_true, y_proba)

        # Should return numpy arrays
        assert isinstance(y_true_out, np.ndarray)
        assert isinstance(y_proba_out, np.ndarray)

        # Should have same length
        assert len(y_true_out) == len(y_proba_out)

    def test_valid_inputs_as_numpy_arrays(self):
        """Test that numpy arrays are accepted."""
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.9, 0.1])

        y_true_out, y_proba_out = validate_binary_inputs(y_true, y_proba)

        assert isinstance(y_true_out, np.ndarray)
        assert isinstance(y_proba_out, np.ndarray)

    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_binary_inputs([], [])

    def test_different_lengths_raise_error(self):
        """Test that arrays with different lengths raise ValueError."""
        y_true = [0, 1, 1]
        y_proba = [0.2, 0.8]

        with pytest.raises(ValueError, match="must have the same length"):
            validate_binary_inputs(y_true, y_proba)

    def test_non_binary_labels_raise_error(self):
        """Test that labels with values other than 0/1 raise ValueError."""
        y_true = [0, 1, 2, 0]  # 2 is invalid
        y_proba = [0.2, 0.8, 0.9, 0.1]

        with pytest.raises(ValueError, match="must contain only 0s and 1s"):
            validate_binary_inputs(y_true, y_proba)

    def test_probabilities_out_of_range_raise_error(self):
        """Test that probabilities outside [0,1] raise ValueError."""
        y_true = [0, 1, 1, 0]
        y_proba = [0.2, 0.8, 1.5, 0.1]  # 1.5 is invalid

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_binary_inputs(y_true, y_proba)


class TestValidateProbabilities:
    """Test suite for validate_probabilities function."""

    def test_valid_probabilities(self):
        """Test that valid probabilities are accepted."""
        probs = np.array([0.0, 0.5, 1.0, 0.3, 0.7])
        result = validate_probabilities(probs)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, probs)

    def test_negative_probabilities_raise_error(self):
        """Test that negative probabilities raise ValueError."""
        probs = np.array([0.5, -0.1, 0.8])

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_probabilities(probs)

    def test_probabilities_above_one_raise_error(self):
        """Test that probabilities > 1 raise ValueError."""
        probs = np.array([0.5, 1.1, 0.8])

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_probabilities(probs)


class TestValidateStep:
    """Test suite for validate_step function."""

    def test_valid_step_sizes(self):
        """Test that valid step sizes are accepted."""
        assert validate_step(0.1) == 0.1
        assert validate_step(0.05) == 0.05
        assert validate_step(0.01) == 0.01
        assert validate_step(1.0) == 1.0

    def test_zero_step_raises_error(self):
        """Test that step=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_step(0.0)

    def test_negative_step_raises_error(self):
        """Test that negative step raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_step(-0.1)

    def test_step_above_one_raises_error(self):
        """Test that step > 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_step(1.5)


class TestValidateThreshold:
    """Test suite for validate_threshold function."""

    def test_valid_thresholds(self):
        """Test that valid thresholds are accepted."""
        assert validate_threshold(0.0) == 0.0
        assert validate_threshold(0.5) == 0.5
        assert validate_threshold(1.0) == 1.0

    def test_negative_threshold_raises_error(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_threshold(-0.1)

    def test_threshold_above_one_raises_error(self):
        """Test that threshold > 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_threshold(1.5)
