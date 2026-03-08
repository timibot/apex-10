"""Tests for isotonic calibration."""
import numpy as np

from apex10.models.calibration import calibrate, calibration_improvement, fit_calibrator


class TestFitCalibrator:
    def test_returns_fitted_calibrator(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        cal = fit_calibrator(y_true, y_prob)
        assert cal is not None

    def test_calibrator_predict_returns_array(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.8, 0.6, 0.7, 0.4, 0.9, 0.2])
        cal = fit_calibrator(y_true, y_prob)
        result = calibrate(cal, y_prob)
        assert len(result) == len(y_prob)


class TestCalibrate:
    def test_output_clipped_to_valid_range(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        cal = fit_calibrator(y_true, y_prob)
        result = calibrate(cal, np.array([0.0, 1.0, 0.5]))
        assert result.min() >= 0.01
        assert result.max() <= 0.99

    def test_output_same_length_as_input(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3, 0.9])
        cal = fit_calibrator(y_true, y_prob)
        result = calibrate(cal, y_prob)
        assert len(result) == len(y_prob)


class TestCalibrationImprovement:
    def test_returns_required_keys(self):
        y_true = np.array([1, 0, 1, 0])
        y_raw = np.array([0.6, 0.6, 0.6, 0.6])
        y_cal = np.array([0.7, 0.3, 0.7, 0.3])
        result = calibration_improvement(y_true, y_raw, y_cal)
        assert all(
            k in result
            for k in ["brier_raw", "brier_calibrated", "improvement", "improved"]
        )

    def test_improved_flag_is_bool(self):
        y_true = np.array([1, 0, 1, 0])
        y_raw = np.array([0.5, 0.5, 0.5, 0.5])
        y_cal = np.array([0.8, 0.2, 0.8, 0.2])
        result = calibration_improvement(y_true, y_raw, y_cal)
        assert isinstance(result["improved"], (bool, np.bool_))
