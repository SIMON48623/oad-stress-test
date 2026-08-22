import numpy as np
import pytest

from oad_stress_test.utils.threshold_calibration import (
    finite_values,
    probabilities_from_scores,
    quantile_thresholds,
    score_signals,
    threshold_grid_from_signals,
)
from scripts.probe_confidence_thresholds import format_threshold_report, load_score_array


def test_quantile_threshold_computation():
    rows = quantile_thresholds(np.array([0.1, 0.2, 0.5, 0.9]), quantiles=[0.5, 1.0])

    assert rows == [
        {"quantile": 0.5, "threshold": 0.35},
        {"quantile": 1.0, "threshold": 0.9},
    ]


def test_nan_inf_are_ignored_by_default_and_can_raise():
    clean = finite_values([0.1, np.nan, np.inf, 0.3])

    np.testing.assert_allclose(clean, np.array([0.1, 0.3]))
    with pytest.raises(ValueError, match="non-finite"):
        finite_values([0.1, np.nan], ignore_nonfinite=False)


def test_score_signals_from_logits():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)

    signals = score_signals(logits, input_kind="logits")

    assert set(signals) == {"max_prob", "entropy", "top2_margin"}
    assert signals["max_prob"].shape == (2,)
    assert signals["entropy"][0] > 0
    assert signals["top2_margin"][0] > 0


def test_probability_detection_and_threshold_grid():
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]], dtype=np.float64)

    inferred = probabilities_from_scores(probs, input_kind="auto")
    grid = threshold_grid_from_signals(score_signals(probs, input_kind="probabilities"), quantiles=[0.5])

    np.testing.assert_allclose(inferred, probs)
    assert grid["max_prob"][0]["threshold"] == pytest.approx(0.6)
    assert "entropy" in grid
    assert "top2_margin" in grid


def test_probe_loads_npz_score_key_and_formats_report(tmp_path):
    path = tmp_path / "scores.npz"
    np.savez_compressed(path, logits=np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32))

    loaded = load_score_array(path, score_key="logits")
    report = format_threshold_report(threshold_grid_from_signals(score_signals(loaded, input_kind="logits"), [0.5]))

    assert loaded.shape == (2, 2)
    assert "max_prob" in report
    assert "q=0.500" in report
