from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from confirmatory_cluster_bootstrap import (
    BOOTSTRAP_SEED,
    ECE_BINS,
    FROZEN_ALPHAS,
    ema_probabilities,
    ece_from_sufficient_stats,
    fixed_bin_sufficient_stats,
    load_prediction_cache,
    sha256_file,
    transition_indices,
    write_csv,
)


PRIMARY_ALPHA = 0.50
SENSITIVITY_ALPHA = 0.25
BASELINE_ALPHA = 0.0
PRIMARY_HORIZON = 16
HORIZONS = (8, 16, 32)
ECE_BIN_COUNTS = (10, 15, 20, 30)
N_BOOTSTRAP = 2000
G2_SEED = 48624


def transition_type(old_label: int, new_label: int, background_label: int = 0) -> str:
    if old_label == background_label and new_label != background_label:
        return "background_to_action"
    if old_label != background_label and new_label == background_label:
        return "action_to_background"
    if old_label != background_label and new_label != background_label:
        return "action_to_action"
    raise ValueError("A ground-truth transition cannot be background_to_background")


def transition_level_rows(
    predictions: list[dict[str, object]],
    alphas: tuple[float, ...] = FROZEN_ALPHAS,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        video_id = str(prediction["video_id"])
        labels = np.asarray(prediction["labels"], dtype=np.int64)
        raw_scores = np.asarray(prediction["scores"], dtype=np.float64)
        boundaries = transition_indices(labels)
        for alpha in alphas:
            scores = ema_probabilities(raw_scores, alpha)
            predicted = np.argmax(scores, axis=1)
            for horizon in horizons:
                for boundary_number, boundary in enumerate(boundaries.tolist()):
                    next_boundary = int(boundaries[boundary_number + 1]) if boundary_number + 1 < len(boundaries) else len(labels)
                    end = min(len(labels), next_boundary, boundary + horizon)
                    if end <= boundary:
                        continue
                    old_label = int(labels[boundary - 1])
                    new_label = int(labels[boundary])
                    window_prediction = predicted[boundary:end]
                    matches = np.flatnonzero(window_prediction == new_label)
                    detected = bool(len(matches))
                    delay = int(matches[0]) if detected else int(horizon)
                    old_probability = scores[boundary:end, old_label]
                    new_probability = scores[boundary:end, new_label]
                    margin = np.maximum(old_probability - new_probability, 0.0)
                    rows.append(
                        {
                            "event_id": f"{video_id}:{boundary}",
                            "video_id": video_id,
                            "transition_frame": int(boundary),
                            "old_label": old_label,
                            "new_label": new_label,
                            "transition_type": transition_type(old_label, new_label),
                            "ema_alpha": float(alpha),
                            "horizon": int(horizon),
                            "window_length": int(end - boundary),
                            "detected": int(detected),
                            "delay": delay,
                            "missed": int(not detected),
                            "event_ptsm": float(margin.mean()),
                            "ptsm_sum": float(margin.sum()),
                            "old_probability_mean": float(old_probability.mean()),
                            "new_probability_mean": float(new_probability.mean()),
                            "stale_old_label_rate": float(np.mean(window_prediction == old_label)),
                        }
                    )
    return rows


def paired_transition_rows(transition_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    raw_lookup: dict[tuple[str, int], dict[str, object]] = {}
    for row in transition_rows:
        if np.isclose(float(row["ema_alpha"]), BASELINE_ALPHA):
            raw_lookup[(str(row["event_id"]), int(row["horizon"]))] = row
    paired: list[dict[str, object]] = []
    for row in transition_rows:
        alpha = float(row["ema_alpha"])
        if np.isclose(alpha, BASELINE_ALPHA):
            continue
        raw = raw_lookup[(str(row["event_id"]), int(row["horizon"]))]
        if int(raw["old_label"]) != int(row["old_label"]) or int(raw["new_label"]) != int(row["new_label"]):
            raise ValueError("Transition identity mismatch")
        paired.append(
            {
                "event_id": row["event_id"],
                "video_id": row["video_id"],
                "transition_frame": row["transition_frame"],
                "transition_type": row["transition_type"],
                "ema_alpha": alpha,
                "horizon": int(row["horizon"]),
                "window_length": int(row["window_length"]),
                "raw_detected": int(raw["detected"]),
                "smoothed_detected": int(row["detected"]),
                "raw_delay": int(raw["delay"]),
                "smoothed_delay": int(row["delay"]),
                "delay_delta": float(row["delay"]) - float(raw["delay"]),
                "delay_equal": int(int(row["delay"]) == int(raw["delay"])),
                "equal_detected_delay": int(
                    int(raw["detected"]) == 1
                    and int(row["detected"]) == 1
                    and int(row["delay"]) == int(raw["delay"])
                ),
                "equal_missed": int(int(raw["missed"]) == 1 and int(row["missed"]) == 1),
                "raw_ptsm": float(raw["event_ptsm"]),
                "smoothed_ptsm": float(row["event_ptsm"]),
                "ptsm_delta": float(row["event_ptsm"]) - float(raw["event_ptsm"]),
                "raw_stale_old_label_rate": float(raw["stale_old_label_rate"]),
                "smoothed_stale_old_label_rate": float(row["stale_old_label_rate"]),
                "stale_old_label_rate_delta": float(row["stale_old_label_rate"]) - float(raw["stale_old_label_rate"]),
                "raw_missed": int(raw["missed"]),
                "smoothed_missed": int(row["missed"]),
                "miss_delta": float(row["missed"]) - float(raw["missed"]),
            }
        )
    return paired


def bootstrap_indices(video_ids: list[str], repetitions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(video_ids), size=(repetitions, len(video_ids)), endpoint=False)


def clustered_mean_interval(
    rows: list[dict[str, object]],
    value_field: str,
    all_video_ids: list[str],
    sampled_indices: np.ndarray,
) -> tuple[float, float, float, np.ndarray, int, int]:
    position = {video_id: idx for idx, video_id in enumerate(all_video_ids)}
    sums = np.zeros(len(all_video_ids), dtype=np.float64)
    counts = np.zeros(len(all_video_ids), dtype=np.float64)
    for row in rows:
        idx = position[str(row["video_id"])]
        sums[idx] += float(row[value_field])
        counts[idx] += 1.0
    n_events = int(counts.sum())
    n_videos = int(np.count_nonzero(counts))
    if n_events == 0:
        return float("nan"), float("nan"), float("nan"), np.asarray([], dtype=np.float64), 0, 0
    boot_sums = sums[sampled_indices].sum(axis=1)
    boot_counts = counts[sampled_indices].sum(axis=1)
    valid = boot_counts > 0
    samples = boot_sums[valid] / boot_counts[valid]
    point = float(sums.sum() / counts.sum())
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), samples, n_events, n_videos


def summarize_subset(
    rows: list[dict[str, object]],
    label: str,
    alpha: float,
    all_video_ids: list[str],
    sampled_indices: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    point, ci_low, ci_high, samples, n_events, n_videos = clustered_mean_interval(
        rows, "ptsm_delta", all_video_ids, sampled_indices
    )
    positive_fraction = float(np.mean([float(row["ptsm_delta"]) > 0.0 for row in rows])) if rows else float("nan")
    return (
        {
            "subset": label,
            "ema_alpha": alpha,
            "n_events": n_events,
            "n_videos": n_videos,
            "mean_ptsm_delta": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "event_positive_fraction": positive_fraction,
            "bootstrap_repetitions": len(samples),
            "resampling_unit": "video",
        },
        samples,
    )


def equal_delay_summaries(
    paired: list[dict[str, object]],
    all_video_ids: list[str],
    sampled_indices: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    summaries: list[dict[str, object]] = []
    strata: list[dict[str, object]] = []
    samples: dict[str, np.ndarray] = {}
    for alpha in (PRIMARY_ALPHA, SENSITIVITY_ALPHA):
        base = [
            row for row in paired
            if np.isclose(float(row["ema_alpha"]), alpha) and int(row["horizon"]) == PRIMARY_HORIZON
        ]
        detected_equal = [row for row in base if int(row["equal_detected_delay"]) == 1]
        summary, values = summarize_subset(
            detected_equal, "equal_detected_delay", alpha, all_video_ids, sampled_indices
        )
        summaries.append(summary)
        samples[f"equal_detected_delay_alpha_{alpha:.2f}"] = values
        equal_missed = [row for row in base if int(row["equal_missed"]) == 1]
        summary, values = summarize_subset(equal_missed, "equal_missed_sensitivity", alpha, all_video_ids, sampled_indices)
        summaries.append(summary)
        samples[f"equal_missed_alpha_{alpha:.2f}"] = values
        for delay in range(PRIMARY_HORIZON):
            selected = [
                row for row in detected_equal
                if int(row["raw_delay"]) == delay and int(row["smoothed_delay"]) == delay
            ]
            if not selected:
                continue
            point, ci_low, ci_high, _, n_events, n_videos = clustered_mean_interval(
                selected, "ptsm_delta", all_video_ids, sampled_indices
            )
            strata.append(
                {
                    "ema_alpha": alpha,
                    "fixed_delay": delay,
                    "n_events": n_events,
                    "n_videos": n_videos,
                    "mean_ptsm_delta": point,
                    "ci_low_descriptive": ci_low,
                    "ci_high_descriptive": ci_high,
                    "event_positive_fraction": float(np.mean([float(row["ptsm_delta"]) > 0.0 for row in selected])),
                    "gate_role": "descriptive_only",
                }
            )
    return summaries, strata, samples


def transition_type_summaries(
    paired: list[dict[str, object]],
    all_video_ids: list[str],
    sampled_indices: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    samples: dict[str, np.ndarray] = {}
    types = ("background_to_action", "action_to_background", "action_to_action")
    for alpha in (PRIMARY_ALPHA, SENSITIVITY_ALPHA):
        for kind in types:
            selected = [
                row for row in paired
                if np.isclose(float(row["ema_alpha"]), alpha)
                and int(row["horizon"]) == PRIMARY_HORIZON
                and str(row["transition_type"]) == kind
            ]
            point, ci_low, ci_high, values, n_events, n_videos = clustered_mean_interval(
                selected, "ptsm_delta", all_video_ids, sampled_indices
            )
            delay_point, delay_low, delay_high, _, _, _ = clustered_mean_interval(
                selected, "delay_delta", all_video_ids, sampled_indices
            )
            rows.append(
                {
                    "ema_alpha": alpha,
                    "transition_type": kind,
                    "n_events": n_events,
                    "n_videos": n_videos,
                    "mean_ptsm_delta": point,
                    "ptsm_ci_low": ci_low,
                    "ptsm_ci_high": ci_high,
                    "mean_delay_delta": delay_point,
                    "delay_ci_low": delay_low,
                    "delay_ci_high": delay_high,
                }
            )
            samples[f"type_{kind}_alpha_{alpha:.2f}"] = values
    return rows, samples


def horizon_summaries(
    paired: list[dict[str, object]],
    all_video_ids: list[str],
    sampled_indices: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    samples: dict[str, np.ndarray] = {}
    for alpha in (PRIMARY_ALPHA, SENSITIVITY_ALPHA):
        for horizon in HORIZONS:
            selected = [
                row for row in paired
                if np.isclose(float(row["ema_alpha"]), alpha) and int(row["horizon"]) == horizon
            ]
            point, ci_low, ci_high, values, n_events, n_videos = clustered_mean_interval(
                selected, "ptsm_delta", all_video_ids, sampled_indices
            )
            delay_point, delay_low, delay_high, _, _, _ = clustered_mean_interval(
                selected, "delay_delta", all_video_ids, sampled_indices
            )
            miss_point, miss_low, miss_high, _, _, _ = clustered_mean_interval(
                selected, "miss_delta", all_video_ids, sampled_indices
            )
            rows.append(
                {
                    "ema_alpha": alpha,
                    "horizon": horizon,
                    "n_events": n_events,
                    "n_videos": n_videos,
                    "mean_event_ptsm_delta": point,
                    "ptsm_ci_low": ci_low,
                    "ptsm_ci_high": ci_high,
                    "mean_delay_delta": delay_point,
                    "delay_ci_low": delay_low,
                    "delay_ci_high": delay_high,
                    "mean_miss_delta": miss_point,
                    "miss_ci_low": miss_low,
                    "miss_ci_high": miss_high,
                }
            )
            samples[f"horizon_{horizon}_alpha_{alpha:.2f}"] = values
    return rows, samples


def per_video_ece_arrays(
    predictions: list[dict[str, object]], alpha: float, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts: list[np.ndarray] = []
    correct_sums: list[np.ndarray] = []
    confidence_sums: list[np.ndarray] = []
    for prediction in predictions:
        labels = np.asarray(prediction["labels"], dtype=np.int64)
        scores = ema_probabilities(np.asarray(prediction["scores"]), alpha)
        predicted = np.argmax(scores, axis=1)
        confidence = np.max(scores, axis=1)
        c, y, p = fixed_bin_sufficient_stats(predicted == labels, confidence, bins)
        counts.append(c)
        correct_sums.append(y)
        confidence_sums.append(p)
    return np.asarray(counts), np.asarray(correct_sums), np.asarray(confidence_sums)


def ece_values_from_matrices(counts: np.ndarray, correct_sums: np.ndarray, confidence_sums: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=1)
    nonempty = counts > 0
    accuracy = np.zeros_like(correct_sums, dtype=np.float64)
    confidence = np.zeros_like(confidence_sums, dtype=np.float64)
    np.divide(correct_sums, counts, out=accuracy, where=nonempty)
    np.divide(confidence_sums, counts, out=confidence, where=nonempty)
    weights = counts / total[:, None]
    return np.sum(weights * np.abs(accuracy - confidence), axis=1)


def ece_bin_summaries(
    predictions: list[dict[str, object]], sampled_indices: np.ndarray
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    samples: dict[str, np.ndarray] = {}
    for bins in ECE_BIN_COUNTS:
        stats = {alpha: per_video_ece_arrays(predictions, alpha, bins) for alpha in FROZEN_ALPHAS}
        point: dict[float, float] = {}
        boot: dict[float, np.ndarray] = {}
        for alpha, (counts, correct_sums, confidence_sums) in stats.items():
            point[alpha] = ece_from_sufficient_stats(counts.sum(axis=0), correct_sums.sum(axis=0), confidence_sums.sum(axis=0))
            boot_counts = counts[sampled_indices].sum(axis=1)
            boot_correct = correct_sums[sampled_indices].sum(axis=1)
            boot_confidence = confidence_sums[sampled_indices].sum(axis=1)
            boot[alpha] = ece_values_from_matrices(boot_counts, boot_correct, boot_confidence)
        for alpha in (PRIMARY_ALPHA, SENSITIVITY_ALPHA):
            difference = boot[alpha] - boot[BASELINE_ALPHA]
            rows.append(
                {
                    "ece_bins": bins,
                    "ema_alpha": alpha,
                    "raw_ece": point[BASELINE_ALPHA],
                    "smoothed_ece": point[alpha],
                    "ece_delta": point[alpha] - point[BASELINE_ALPHA],
                    "ci_low": float(np.quantile(difference, 0.025)),
                    "ci_high": float(np.quantile(difference, 0.975)),
                }
            )
            samples[f"ece_bins_{bins}_alpha_{alpha:.2f}"] = difference
    return rows, samples


def evaluate_gate(
    equal_delay: list[dict[str, object]],
    transition_types: list[dict[str, object]],
    horizons: list[dict[str, object]],
    ece_bins: list[dict[str, object]],
) -> dict[str, object]:
    primary_equal = next(
        row for row in equal_delay
        if str(row["subset"]) == "equal_detected_delay" and np.isclose(float(row["ema_alpha"]), PRIMARY_ALPHA)
    )
    primary_test = (
        int(primary_equal["n_events"]) >= 500
        and int(primary_equal["n_videos"]) >= 30
        and float(primary_equal["ci_low"]) > 0.0
    )
    passing_types = [
        row for row in transition_types
        if np.isclose(float(row["ema_alpha"]), PRIMARY_ALPHA)
        and int(row["n_events"]) >= 100
        and float(row["ptsm_ci_low"]) > 0.0
    ]
    transition_type_test = len(passing_types) >= 2
    primary_horizons = [row for row in horizons if np.isclose(float(row["ema_alpha"]), PRIMARY_ALPHA)]
    horizon_test = len(primary_horizons) == len(HORIZONS) and all(float(row["ptsm_ci_low"]) > 0.0 for row in primary_horizons)
    primary_bins = [row for row in ece_bins if np.isclose(float(row["ema_alpha"]), PRIMARY_ALPHA)]
    ece_test = len(primary_bins) == len(ECE_BIN_COUNTS) and all(float(row["ci_high"]) < 0.0 for row in primary_bins)
    return {
        "primary_equal_delay_test": primary_test,
        "transition_type_test": transition_type_test,
        "passing_transition_types": [str(row["transition_type"]) for row in passing_types],
        "horizon_test": horizon_test,
        "ece_bin_test": ece_test,
        "overall": primary_test and transition_type_test and horizon_test and ece_test,
    }


def fmt(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    gate: dict[str, object],
    equal_delay: list[dict[str, object]],
    transition_types: list[dict[str, object]],
    horizons: list[dict[str, object]],
    ece_bins: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    primary_equal = next(
        row for row in equal_delay
        if str(row["subset"]) == "equal_detected_delay" and np.isclose(float(row["ema_alpha"]), PRIMARY_ALPHA)
    )
    lines = [
        "# G2 diagnostic nonredundancy gate",
        "",
        f"- Decision: **{'PASS' if gate['overall'] else 'FAIL_DOWNGRADE_PTSM'}**",
        f"- Frozen prediction cache: `{metadata['prediction_cache']}`",
        f"- Videos: {metadata['videos']}",
        f"- Transition events at each horizon/alpha: {metadata['transitions_per_condition']}",
        f"- Bootstrap: {N_BOOTSTRAP} paired video-cluster resamples, seed {G2_SEED}",
        "",
        "## Primary equal-delay test",
        "",
        "This test retains only the same ground-truth transitions for which raw and EMA 0.50 have exactly the same detected argmax delay. Any remaining paired PTSM difference cannot be encoded by a delay difference for those events.",
        "",
        f"- Eligible events: {primary_equal['n_events']} from {primary_equal['n_videos']} videos",
        f"- Mean paired PTSM difference: **{fmt(primary_equal['mean_ptsm_delta'])}**",
        f"- 95% CI: **[{fmt(primary_equal['ci_low'])}, {fmt(primary_equal['ci_high'])}]**",
        f"- Fraction of eligible events with positive PTSM difference: {fmt(primary_equal['event_positive_fraction'], 4)}",
        f"- Primary test: **{'PASS' if gate['primary_equal_delay_test'] else 'FAIL'}**",
        "",
        "## Transition-type check",
        "",
        "| Type | Events | PTSM delta | 95% CI | Delay delta | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in transition_types:
        if not np.isclose(float(row["ema_alpha"]), PRIMARY_ALPHA):
            continue
        lines.append(
            f"| {row['transition_type']} | {row['n_events']} | {fmt(row['mean_ptsm_delta'])} | "
            f"[{fmt(row['ptsm_ci_low'])}, {fmt(row['ptsm_ci_high'])}] | {fmt(row['mean_delay_delta'])} | "
            f"[{fmt(row['delay_ci_low'])}, {fmt(row['delay_ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            f"- Passing types: {', '.join(gate['passing_transition_types']) if gate['passing_transition_types'] else 'none'}",
            f"- Transition-type test: **{'PASS' if gate['transition_type_test'] else 'FAIL'}**",
            "",
            "## Horizon sensitivity",
            "",
            "| Horizon | PTSM delta | 95% CI | Delay delta | Miss delta |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in horizons:
        if np.isclose(float(row["ema_alpha"]), PRIMARY_ALPHA):
            lines.append(
                f"| {row['horizon']} | {fmt(row['mean_event_ptsm_delta'])} | "
                f"[{fmt(row['ptsm_ci_low'])}, {fmt(row['ptsm_ci_high'])}] | "
                f"{fmt(row['mean_delay_delta'])} | {fmt(row['mean_miss_delta'])} |"
            )
    lines.extend(
        [
            "",
            f"- Horizon test: **{'PASS' if gate['horizon_test'] else 'FAIL'}**",
            "",
            "## ECE-bin sensitivity",
            "",
            "| Bins | Raw ECE | EMA 0.50 ECE | Delta | 95% CI |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ece_bins:
        if np.isclose(float(row["ema_alpha"]), PRIMARY_ALPHA):
            lines.append(
                f"| {row['ece_bins']} | {fmt(row['raw_ece'])} | {fmt(row['smoothed_ece'])} | "
                f"{fmt(row['ece_delta'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |"
            )
    lines.extend(
        [
            "",
            f"- ECE-bin test: **{'PASS' if gate['ece_bin_test'] else 'FAIL'}**",
            "",
            "## Interpretation boundary",
            "",
            "Passing supports the narrow claim that PTSM retains probability-magnitude information beyond discrete argmax delay in the frozen setting. It does not show that PTSM is theoretically unique, optimal, or a replacement for delay and missed-transition metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen G2 PTSM nonredundancy and bounded sensitivity analysis.")
    parser.add_argument("--prediction-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.prediction_cache.exists() or not args.protocol.exists():
        raise FileNotFoundError("Prediction cache or protocol is missing")
    output_names = (
        "transition_level_statistics.csv",
        "paired_transition_deltas.csv",
        "equal_delay_summary.csv",
        "individual_delay_strata.csv",
        "transition_type_summary.csv",
        "horizon_sensitivity.csv",
        "ece_bin_sensitivity.csv",
        "bootstrap_samples.npz",
        "metadata.json",
        "gate_G2.md",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in output_names):
        raise FileExistsError("G2 outputs already exist; use a new directory to preserve immutability")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    actual_cache_hash = sha256_file(args.prediction_cache)
    if actual_cache_hash != str(protocol["input_cache_sha256"]):
        raise ValueError("Prediction cache hash does not match the frozen protocol")
    predictions = load_prediction_cache(args.prediction_cache)
    all_video_ids = sorted(str(row["video_id"]) for row in predictions)
    sampled_indices = bootstrap_indices(all_video_ids, N_BOOTSTRAP, G2_SEED)
    transitions = transition_level_rows(predictions)
    paired = paired_transition_rows(transitions)
    equal_delay, delay_strata, samples_equal = equal_delay_summaries(paired, all_video_ids, sampled_indices)
    type_rows, samples_type = transition_type_summaries(paired, all_video_ids, sampled_indices)
    horizon_rows, samples_horizon = horizon_summaries(paired, all_video_ids, sampled_indices)
    ece_rows, samples_ece = ece_bin_summaries(predictions, sampled_indices)
    gate = evaluate_gate(equal_delay, type_rows, horizon_rows, ece_rows)
    write_csv(args.output_dir / "transition_level_statistics.csv", transitions)
    write_csv(args.output_dir / "paired_transition_deltas.csv", paired)
    write_csv(args.output_dir / "equal_delay_summary.csv", equal_delay)
    write_csv(args.output_dir / "individual_delay_strata.csv", delay_strata)
    write_csv(args.output_dir / "transition_type_summary.csv", type_rows)
    write_csv(args.output_dir / "horizon_sensitivity.csv", horizon_rows)
    write_csv(args.output_dir / "ece_bin_sensitivity.csv", ece_rows)
    all_samples = {**samples_equal, **samples_type, **samples_horizon, **samples_ece}
    np.savez_compressed(args.output_dir / "bootstrap_samples.npz", **all_samples)
    condition_counts = {}
    for row in transitions:
        key = f"alpha_{float(row['ema_alpha']):.2f}_h{int(row['horizon'])}"
        condition_counts[key] = condition_counts.get(key, 0) + 1
    metadata: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "confirmatory",
        "videos": len(predictions),
        "transitions_per_condition": int(len(transitions) / (len(FROZEN_ALPHAS) * len(HORIZONS))),
        "condition_counts": condition_counts,
        "prediction_cache": str(args.prediction_cache),
        "prediction_cache_sha256": actual_cache_hash,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "script_sha256": sha256_file(Path(__file__)),
        "bootstrap_repetitions": N_BOOTSTRAP,
        "bootstrap_seed": G2_SEED,
        "gate": gate,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.output_dir / "gate_G2.md", gate, equal_delay, type_rows, horizon_rows, ece_rows, metadata)
    print(args.output_dir / "gate_G2.md")


if __name__ == "__main__":
    main()

