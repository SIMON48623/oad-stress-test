import numpy as np
import pandas as pd
import pytest

from tools.bootstrap_transition_misalignment import (
    _corrected_odds_ratio_from_counts,
    bootstrap_summary,
    compute_video_counts,
    run_bootstrap,
)


def _row(video_id, distance, raw_error, selective_error, is_abstain, max_prob=0.8):
    return {
        "video_id": video_id,
        "distance_to_transition": distance,
        "raw_error": raw_error,
        "selective_error": selective_error,
        "is_abstain": is_abstain,
        "max_prob": max_prob,
        "model": "toy",
        "policy": "confidence_threshold",
        "uncertainty_mode": "max_prob",
        "threshold": 0.7,
        "budget": 0.25,
        "shift_type": "clean",
    }


def test_per_video_metric_calculation():
    frame = pd.DataFrame([
        _row("v1", 0, 1, 1, True, 0.6),
        _row("v1", 2, 0, 0, False, 0.8),
        _row("v1", 40, 0, 0, False, 0.9),
        _row("v1", 41, 0, 0, False, 0.7),
        _row("v2", 0, 1, 1, False, 0.5),
        _row("v2", 1, 1, np.nan, True, 0.7),
        _row("v2", 45, 0, 0, False, 0.8),
        _row("v2", 46, 1, 1, False, 0.6),
    ])

    counts = compute_video_counts(frame, near_window=8, far_window=32)
    v1 = counts[counts["video_id"].eq("v1")].iloc[0]

    assert v1["near_frames"] == 2
    assert v1["far_frames"] == 2
    assert v1["TEG_raw"] == pytest.approx(0.5)
    assert v1["TAG"] == pytest.approx(0.5)
    assert v1["ATM_raw"] == pytest.approx(0.0)
    assert v1["confidence_gap"] == pytest.approx(-0.1)


def test_bootstrap_resamples_video_clusters_and_is_deterministic():
    frame = pd.DataFrame([
        _row("v1", 0, 1, 1, True),
        _row("v1", 40, 0, 0, False),
        _row("v2", 0, 1, 1, False),
        _row("v2", 40, 1, 1, False),
        _row("v3", 0, 0, 0, False),
        _row("v3", 40, 0, 0, True),
    ])
    counts = compute_video_counts(frame, near_window=8, far_window=32)

    first = bootstrap_summary(counts, bootstrap=25, seed=123)
    second = bootstrap_summary(counts, bootstrap=25, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) >= {"TEG_raw", "TAG", "ATM_raw", "TEG_selective"}
    assert first["n_videos"].unique().tolist() == [3]
    assert first["valid_bootstrap_samples"].min() > 0


def test_numpy_backend_matches_legacy_backend_with_fixed_seed():
    frame = pd.DataFrame([
        _row("v1", 0, 1, 1, True, 0.6),
        _row("v1", 2, 0, 0, False, 0.7),
        _row("v1", 40, 0, 0, False, 0.9),
        _row("v2", 0, 1, np.nan, False, 0.5),
        _row("v2", 40, 1, 1, True, 0.6),
        _row("v3", 0, 0, 0, False, 0.8),
        _row("v3", 45, 0, np.nan, False, 0.9),
    ])
    counts = compute_video_counts(frame, near_window=8, far_window=32)

    legacy = bootstrap_summary(counts, bootstrap=50, seed=99, backend="legacy")
    numpy_backend = bootstrap_summary(counts, bootstrap=50, seed=99, backend="numpy")

    pd.testing.assert_frame_equal(legacy, numpy_backend, check_exact=False, atol=1e-12, rtol=1e-12)


def test_continuity_correction_prevents_infinite_odds_ratio():
    corrected = _corrected_odds_ratio_from_counts(
        event_near=2,
        total_near=2,
        event_far=0,
        total_far=2,
    )

    assert corrected == pytest.approx((2.5 / 0.5) / (0.5 / 2.5))


def test_video_with_no_far_frames_is_handled_without_crash():
    frame = pd.DataFrame([
        _row("near_only", 0, 1, 1, False),
        _row("near_only", 1, 0, 0, True),
    ])

    counts = compute_video_counts(frame, near_window=8, far_window=32)
    row = counts.iloc[0]

    assert row["near_frames"] == 2
    assert row["far_frames"] == 0
    assert np.isnan(row["TEG_raw"])
    assert np.isnan(row["Error_OR_raw"])


def test_numpy_backend_handles_missing_far_and_no_selected_frames():
    frame = pd.DataFrame([
        _row("near_only", 0, 1, np.nan, False),
        _row("near_only", 1, 0, np.nan, True),
        _row("far_only", 40, 0, np.nan, False),
        _row("far_only", 41, 1, np.nan, False),
    ])
    counts = compute_video_counts(frame, near_window=8, far_window=32)

    summary = bootstrap_summary(counts, bootstrap=20, seed=5, backend="numpy")
    selective = summary[summary["metric"].eq("TEG_selective")].iloc[0]
    raw = summary[summary["metric"].eq("TEG_raw")].iloc[0]

    assert np.isnan(selective["point_estimate"])
    assert selective["valid_bootstrap_samples"] == 0
    assert np.isfinite(raw["point_estimate"])


def test_run_bootstrap_writes_expected_outputs(tmp_path):
    frame = pd.DataFrame([
        _row("v1", 0, 1, 1, True),
        _row("v1", 40, 0, 0, False),
        _row("v2", 0, 0, 0, False),
        _row("v2", 40, 0, 0, False),
    ])
    input_path = tmp_path / "frame_with_transition_features.csv"
    frame.to_csv(input_path, index=False)

    outputs = run_bootstrap(input_path, tmp_path / "out", bootstrap=10, seed=7, backend="numpy")

    for path in outputs.values():
        assert path.exists()
    summary = pd.read_csv(outputs["bootstrap_summary"])
    assert "point_estimate" in summary.columns
    assert "ci_2_5" in summary.columns
    assert "sign_rate_positive" in summary.columns
