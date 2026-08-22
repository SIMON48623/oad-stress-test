from __future__ import annotations

import numpy as np

from confirmatory_cluster_bootstrap import (
    ECE_BINS,
    aggregate_records,
    bootstrap_differences,
    ece_from_sufficient_stats,
    fixed_bin_sufficient_stats,
    split_records_by_alpha,
    video_sufficient_stats,
)


def direct_ece(correct: np.ndarray, confidence: np.ndarray, bins: int) -> float:
    counts, correct_sums, confidence_sums = fixed_bin_sufficient_stats(correct, confidence, bins)
    return ece_from_sufficient_stats(counts, correct_sums, confidence_sums)


def test_fixed_bin_ece_reconstruction_matches_direct_concatenation() -> None:
    correct_a = np.asarray([1, 0, 1, 1], dtype=float)
    confidence_a = np.asarray([0.9, 0.6, 0.7, 1.0], dtype=float)
    correct_b = np.asarray([0, 1, 0], dtype=float)
    confidence_b = np.asarray([0.2, 0.8, 0.55], dtype=float)
    a = fixed_bin_sufficient_stats(correct_a, confidence_a, ECE_BINS)
    b = fixed_bin_sufficient_stats(correct_b, confidence_b, ECE_BINS)
    reconstructed = ece_from_sufficient_stats(a[0] + b[0], a[1] + b[1], a[2] + b[2])
    direct = direct_ece(np.concatenate([correct_a, correct_b]), np.concatenate([confidence_a, confidence_b]), ECE_BINS)
    assert reconstructed == direct


def toy_records() -> list[dict[str, object]]:
    labels_a = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    labels_b = np.asarray([1, 1, 0, 0, 0], dtype=np.int64)
    scores_a = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.1, 0.9]])
    scores_b = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1], [0.9, 0.1]])
    records: list[dict[str, object]] = []
    for video_id, labels, scores in (("a", labels_a, scores_a), ("b", labels_b, scores_b)):
        for alpha in (0.0, 0.25, 0.5):
            records.append(video_sufficient_stats(video_id, labels, scores, alpha, ECE_BINS, 16))
    return records


def test_aggregate_has_expected_denominators() -> None:
    by_alpha = split_records_by_alpha(toy_records(), (0.0, 0.25, 0.5))
    aggregate = aggregate_records(by_alpha[0.0], ECE_BINS)
    assert aggregate["n_videos"] == 2
    assert aggregate["n_frames"] == 10
    assert aggregate["n_transitions"] == 2
    assert aggregate["accuracy"] == 1.0


def test_paired_bootstrap_is_deterministic() -> None:
    by_alpha = split_records_by_alpha(toy_records(), (0.0, 0.25, 0.5))
    rows_a, samples_a = bootstrap_differences(by_alpha, ECE_BINS, repetitions=25, seed=9)
    rows_b, samples_b = bootstrap_differences(by_alpha, ECE_BINS, repetitions=25, seed=9)
    assert rows_a == rows_b
    assert samples_a.keys() == samples_b.keys()
    for key in samples_a:
        np.testing.assert_array_equal(samples_a[key], samples_b[key])

