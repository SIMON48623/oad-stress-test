import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_boxcar_runner():
    path = Path(__file__).resolve().parents[1] / "postprocessors" / "causal_boxcar_replication.py"
    spec = importlib.util.spec_from_file_location("boxcar_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load causal boxcar runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


causal_boxcar_probabilities = load_boxcar_runner().causal_boxcar_probabilities


def test_boxcar_w3_uses_only_current_and_past_values() -> None:
    scores = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    changed_future = scores.copy()
    changed_future[3] = [0.0, 1.0]
    first = causal_boxcar_probabilities(scores, 3)
    second = causal_boxcar_probabilities(changed_future, 3)
    np.testing.assert_allclose(first[:3], second[:3], atol=0.0, rtol=0.0)


def test_boxcar_w3_left_pads_with_first_probability() -> None:
    scores = np.asarray([[0.9, 0.1], [0.3, 0.7], [0.0, 1.0]], dtype=np.float64)
    actual = causal_boxcar_probabilities(scores, 3)
    expected = np.asarray(
        [
            [0.9, 0.1],
            [(0.9 + 0.9 + 0.3) / 3.0, (0.1 + 0.1 + 0.7) / 3.0],
            [(0.9 + 0.3 + 0.0) / 3.0, (0.1 + 0.7 + 1.0) / 3.0],
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_w3_matches_primary_ema_kernel_moments() -> None:
    alpha = 0.5
    ema_mean_age = alpha / (1.0 - alpha)
    ema_squared_weight_sum = (1.0 - alpha) / (1.0 + alpha)
    boxcar_mean_age = (3.0 - 1.0) / 2.0
    boxcar_squared_weight_sum = 1.0 / 3.0
    assert ema_mean_age == boxcar_mean_age
    assert ema_squared_weight_sum == boxcar_squared_weight_sum
