from __future__ import annotations

import numpy as np

from diagnostic_nonredundancy import (
    bootstrap_indices,
    clustered_mean_interval,
    paired_transition_rows,
    transition_level_rows,
    transition_type,
)


def test_transition_type_mapping() -> None:
    assert transition_type(0, 2) == "background_to_action"
    assert transition_type(2, 0) == "action_to_background"
    assert transition_type(2, 3) == "action_to_action"


def test_paired_rows_preserve_event_identity() -> None:
    labels = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], dtype=float)
    predictions = [{"video_id": "v", "labels": labels, "scores": scores}]
    rows = transition_level_rows(predictions, alphas=(0.0, 0.5), horizons=(16,))
    paired = paired_transition_rows(rows)
    assert len(paired) == 1
    assert paired[0]["event_id"] == "v:2"
    assert paired[0]["raw_delay"] == paired[0]["smoothed_delay"] == 0
    assert paired[0]["equal_detected_delay"] == 1


def test_cluster_interval_resamples_video_not_events() -> None:
    rows = [
        {"video_id": "a", "ptsm_delta": 1.0},
        {"video_id": "a", "ptsm_delta": 1.0},
        {"video_id": "b", "ptsm_delta": 3.0},
    ]
    indices = bootstrap_indices(["a", "b"], repetitions=20, seed=1)
    point, low, high, samples, n_events, n_videos = clustered_mean_interval(
        rows, "ptsm_delta", ["a", "b"], indices
    )
    assert point == 5.0 / 3.0
    assert n_events == 3
    assert n_videos == 2
    assert len(samples) == 20
    assert low <= point <= high
