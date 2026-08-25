from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "cmert_thumos" / "run_cmert_thumos.py"
SPEC = importlib.util.spec_from_file_location("run_cmert_thumos", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_ema_is_causal_and_matches_frozen_recursion() -> None:
    scores = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    actual = MODULE.smooth(scores, "ema_alpha_0.50")
    expected = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
    np.testing.assert_allclose(actual, expected)
    changed_future = scores.copy()
    changed_future[2] = [1.0, 0.0]
    np.testing.assert_allclose(
        MODULE.smooth(scores, "ema_alpha_0.50")[:2],
        MODULE.smooth(changed_future, "ema_alpha_0.50")[:2],
    )


def test_boxcar_uses_only_current_and_past_values() -> None:
    scores = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    actual = MODULE.smooth(scores, "boxcar_w3")
    expected = np.asarray(
        [[1.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]]
    )
    np.testing.assert_allclose(actual, expected)


def test_ambiguous_class_splits_valid_runs() -> None:
    valid = np.asarray([True, True, False, True, False, True, True])
    assert MODULE.valid_runs(valid) == [(0, 2), (3, 4), (5, 7)]


def test_accuracy_boundary_cannot_be_reported_as_full_inversion() -> None:
    lookup = {
        "accuracy": {"ci_low": -0.0052, "ci_high": -0.0019},
        "global_ece": {"ci_low": -0.0116, "ci_high": -0.0079},
        "mean_tfi": {"ci_low": -0.4213, "ci_high": -0.3203},
        "mean_transition_delay": {"ci_low": 0.7399, "ci_high": 0.9173},
        "missed_transition_rate": {"ci_low": 0.0267, "ci_high": 0.0404},
    }
    checks = MODULE.gate_checks(lookup)
    assert checks["global_ece_improves"]
    assert checks["tfi_improves"]
    assert checks["delay_worsens"]
    assert checks["miss_rate_worsens"]
    assert not checks["accuracy_noninferior"]
    assert not checks["full_ranking_inversion"]


def test_protocol_forbids_posthoc_parameter_search() -> None:
    assert MODULE.EMA_ALPHA == 0.50
    assert MODULE.BOXCAR_WINDOW == 3
    assert MODULE.BOOTSTRAP_REPETITIONS == 2000
    assert MODULE.BOOTSTRAP_SEED == 48629
    assert MODULE.ACCURACY_NONINFERIORITY_MARGIN == -0.002
