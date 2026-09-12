from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from oad_stress_test.metrics.ek100 import top1_in_active_set


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]


def _load_testra_runner():
    path = REPOSITORY_ROOT / "reproducibility" / "paper_v2" / "testra_ek100" / "run_testra_ek100.py"
    spec = importlib.util.spec_from_file_location("paper_v2_testra_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_active_set_top1_accepts_membership_without_requiring_exact_set_match():
    result = top1_in_active_set(
        np.asarray([2, 1]),
        [{1, 2}, {2}],
    )

    np.testing.assert_array_equal(result, np.asarray([True, False]))


def test_clean_transition_filter_requires_single_active_labels_on_both_sides():
    runner = _load_testra_runner()
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    predictions = labels.copy()
    active_counts = np.asarray([1, 2, 1, 1, 1, 1])

    result = runner.transition_stats(labels, predictions, active_counts, horizon=16)

    assert result == {"transition_count": 1, "delay_sum": 0.0, "missed_sum": 0.0}


def test_transition_window_stops_at_next_change_and_a_miss_contributes_h16():
    runner = _load_testra_runner()
    labels = np.asarray([0, 0, 1, 1, 0, 0])
    predictions = np.asarray([0, 0, 0, 0, 0, 0])
    active_counts = np.ones_like(labels)

    result = runner.transition_stats(labels, predictions, active_counts, horizon=16)

    assert result == {"transition_count": 2, "delay_sum": 16.0, "missed_sum": 1.0}


def test_metric_protocol_freezes_horizon_and_miss_value_at_16():
    protocol = json.loads((PACKAGE_ROOT / "protocols" / "metric_protocol.json").read_text(encoding="utf-8"))

    assert protocol["transition_horizon_feature_timesteps"] == 16
    assert "assign delay 16" in protocol["miss_rule"]


def test_clean_transition_filter_rejects_a_multi_active_post_transition_timestep():
    runner = _load_testra_runner()
    labels = np.asarray([0, 0, 1, 1])
    predictions = labels.copy()
    active_counts = np.asarray([1, 1, 2, 1])

    result = runner.transition_stats(labels, predictions, active_counts, horizon=16)

    assert result == {"transition_count": 0, "delay_sum": 0.0, "missed_sum": 0.0}


def test_transition_window_does_not_accept_an_old_target_prediction_after_the_next_boundary():
    runner = _load_testra_runner()
    labels = np.asarray([0, 0, 1, 1, 0, 0])
    predictions = np.asarray([0, 0, 0, 0, 1, 1])
    active_counts = np.asarray([1, 1, 1, 1, 2, 1])

    result = runner.transition_stats(labels, predictions, active_counts, horizon=16)

    assert result == {"transition_count": 1, "delay_sum": 16.0, "missed_sum": 1.0}


def test_transition_horizon_is_half_open_and_includes_only_offsets_zero_through_fifteen():
    runner = _load_testra_runner()
    labels = np.asarray([0, 0] + [1] * 20)
    active_counts = np.ones_like(labels)
    hit_at_fifteen = np.zeros_like(labels)
    hit_at_fifteen[17] = 1
    hit_at_sixteen = np.zeros_like(labels)
    hit_at_sixteen[18] = 1

    included = runner.transition_stats(labels, hit_at_fifteen, active_counts, horizon=16)
    excluded = runner.transition_stats(labels, hit_at_sixteen, active_counts, horizon=16)

    assert included == {"transition_count": 1, "delay_sum": 15.0, "missed_sum": 0.0}
    assert excluded == {"transition_count": 1, "delay_sum": 16.0, "missed_sum": 1.0}


def test_sequence_without_transitions_has_zero_transition_statistics():
    runner = _load_testra_runner()
    labels = np.asarray([2, 2, 2, 2])
    predictions = np.asarray([2, 1, 2, 1])
    active_counts = np.ones_like(labels)

    result = runner.transition_stats(labels, predictions, active_counts, horizon=16)

    assert result == {"transition_count": 0, "delay_sum": 0.0, "missed_sum": 0.0}
