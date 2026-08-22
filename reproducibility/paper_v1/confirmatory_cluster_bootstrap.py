from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


FROZEN_ALPHAS = (0.0, 0.25, 0.50)
BASELINE_ALPHA = 0.0
PRIMARY_ALPHA = 0.50
SENSITIVITY_ALPHA = 0.25
HORIZON = 16
ECE_BINS = 15
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 48623
ACCURACY_NONINFERIORITY_MARGIN = -0.002
METRICS = (
    "accuracy",
    "global_ece",
    "mean_tfi",
    "mean_predicted_switches",
    "mean_transition_delay",
    "ptsm",
    "missed_transition_rate",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ema_probabilities(scores: np.ndarray, alpha: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must have shape [frames, classes]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    if len(scores) <= 1 or alpha == 0.0:
        return scores.copy()
    smoothed = np.empty_like(scores)
    smoothed[0] = scores[0]
    for frame_idx in range(1, len(scores)):
        smoothed[frame_idx] = alpha * smoothed[frame_idx - 1] + (1.0 - alpha) * scores[frame_idx]
    smoothed /= np.maximum(smoothed.sum(axis=1, keepdims=True), 1e-12)
    return smoothed


def transition_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return np.flatnonzero(labels[1:] != labels[:-1]) + 1


def temporal_fragmentation_index(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if not len(labels):
        return float("nan")
    ground_truth_segments = 1 + int(np.count_nonzero(labels[1:] != labels[:-1]))
    starts = np.concatenate(([0], np.flatnonzero(predictions[1:] != predictions[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(predictions)]))
    disagreement = 0.0
    for start, end in zip(starts.tolist(), ends.tolist()):
        disagreement += float(np.mean(predictions[start:end] != labels[start:end]))
    return disagreement / float(ground_truth_segments)


def fixed_bin_sufficient_stats(correct: np.ndarray, confidence: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if len(correct) != len(confidence):
        raise ValueError("correct and confidence must have equal length")
    indices = np.floor(confidence * bins).astype(np.int64)
    indices = np.clip(indices, 0, bins - 1)
    counts = np.bincount(indices, minlength=bins).astype(np.int64)
    correct_sums = np.bincount(indices, weights=correct, minlength=bins).astype(np.float64)
    confidence_sums = np.bincount(indices, weights=confidence, minlength=bins).astype(np.float64)
    return counts, correct_sums, confidence_sums


def ece_from_sufficient_stats(counts: np.ndarray, correct_sums: np.ndarray, confidence_sums: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    correct_sums = np.asarray(correct_sums, dtype=np.float64)
    confidence_sums = np.asarray(confidence_sums, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    nonempty = counts > 0
    accuracy = np.zeros_like(counts)
    confidence = np.zeros_like(counts)
    accuracy[nonempty] = correct_sums[nonempty] / counts[nonempty]
    confidence[nonempty] = confidence_sums[nonempty] / counts[nonempty]
    return float(np.sum((counts[nonempty] / total) * np.abs(accuracy[nonempty] - confidence[nonempty])))


def transition_sufficient_stats(labels: np.ndarray, scores: np.ndarray, horizon: int) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.argmax(scores, axis=1)
    boundaries = transition_indices(labels)
    delay_sum = 0.0
    missed_sum = 0.0
    ptsm_sum = 0.0
    ptsm_count = 0
    old_label_frame_sum = 0.0
    eligible_transitions = 0
    for boundary_number, boundary in enumerate(boundaries.tolist()):
        next_boundary = int(boundaries[boundary_number + 1]) if boundary_number + 1 < len(boundaries) else len(labels)
        end = min(len(labels), next_boundary, boundary + int(horizon))
        if end <= boundary:
            continue
        old_label = int(labels[boundary - 1])
        new_label = int(labels[boundary])
        window_predictions = predictions[boundary:end]
        new_label_matches = np.flatnonzero(window_predictions == new_label)
        if len(new_label_matches):
            delay_sum += float(new_label_matches[0])
        else:
            delay_sum += float(horizon)
            missed_sum += 1.0
        old_prob = scores[boundary:end, old_label]
        new_prob = scores[boundary:end, new_label]
        ptsm_sum += float(np.maximum(old_prob - new_prob, 0.0).sum())
        ptsm_count += int(end - boundary)
        old_label_frame_sum += float(np.count_nonzero(window_predictions == old_label))
        eligible_transitions += 1
    return {
        "transition_count": eligible_transitions,
        "delay_sum": delay_sum,
        "missed_sum": missed_sum,
        "ptsm_sum": ptsm_sum,
        "ptsm_count": ptsm_count,
        "stale_old_label_frame_sum": old_label_frame_sum,
    }


def video_sufficient_stats(video_id: str, labels: np.ndarray, raw_scores: np.ndarray, alpha: float, bins: int, horizon: int) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = ema_probabilities(raw_scores, alpha)
    predictions = np.argmax(scores, axis=1)
    confidence = np.max(scores, axis=1)
    correct = predictions == labels
    bin_counts, bin_correct, bin_confidence = fixed_bin_sufficient_stats(correct, confidence, bins)
    transition = transition_sufficient_stats(labels, scores, horizon)
    row: dict[str, object] = {
        "video_id": video_id,
        "ema_alpha": float(alpha),
        "n_frames": int(len(labels)),
        "correct_sum": int(correct.sum()),
        "tfi": temporal_fragmentation_index(labels, predictions),
        "predicted_switches": int(np.count_nonzero(predictions[1:] != predictions[:-1])),
        **transition,
    }
    for bin_idx in range(bins):
        prefix = f"ece_bin_{bin_idx:02d}"
        row[f"{prefix}_count"] = int(bin_counts[bin_idx])
        row[f"{prefix}_correct_sum"] = float(bin_correct[bin_idx])
        row[f"{prefix}_confidence_sum"] = float(bin_confidence[bin_idx])
    return row


def aggregate_records(records: list[dict[str, object]], bins: int, sampled_indices: np.ndarray | None = None) -> dict[str, float]:
    if sampled_indices is None:
        sampled_indices = np.arange(len(records), dtype=np.int64)
    sampled_indices = np.asarray(sampled_indices, dtype=np.int64)
    selected = [records[int(idx)] for idx in sampled_indices]
    frames = float(sum(int(row["n_frames"]) for row in selected))
    correct = float(sum(int(row["correct_sum"]) for row in selected))
    transition_count = float(sum(int(row["transition_count"]) for row in selected))
    ptsm_count = float(sum(int(row["ptsm_count"]) for row in selected))
    counts = np.asarray(
        [sum(int(row[f"ece_bin_{bin_idx:02d}_count"]) for row in selected) for bin_idx in range(bins)],
        dtype=np.float64,
    )
    correct_sums = np.asarray(
        [sum(float(row[f"ece_bin_{bin_idx:02d}_correct_sum"]) for row in selected) for bin_idx in range(bins)],
        dtype=np.float64,
    )
    confidence_sums = np.asarray(
        [sum(float(row[f"ece_bin_{bin_idx:02d}_confidence_sum"]) for row in selected) for bin_idx in range(bins)],
        dtype=np.float64,
    )
    return {
        "accuracy": correct / frames,
        "global_ece": ece_from_sufficient_stats(counts, correct_sums, confidence_sums),
        "mean_tfi": float(np.mean([float(row["tfi"]) for row in selected])),
        "mean_predicted_switches": float(np.mean([float(row["predicted_switches"]) for row in selected])),
        "mean_transition_delay": float(sum(float(row["delay_sum"]) for row in selected) / transition_count),
        "ptsm": float(sum(float(row["ptsm_sum"]) for row in selected) / ptsm_count),
        "missed_transition_rate": float(sum(float(row["missed_sum"]) for row in selected) / transition_count),
        "n_frames": frames,
        "n_videos": float(len(selected)),
        "n_transitions": transition_count,
        "n_ptsm_frames": ptsm_count,
    }


def split_records_by_alpha(records: list[dict[str, object]], alphas: Iterable[float]) -> dict[float, list[dict[str, object]]]:
    by_alpha: dict[float, list[dict[str, object]]] = {}
    expected_video_ids: list[str] | None = None
    for alpha in alphas:
        group = sorted(
            [row for row in records if np.isclose(float(row["ema_alpha"]), alpha)],
            key=lambda row: str(row["video_id"]),
        )
        ids = [str(row["video_id"]) for row in group]
        if expected_video_ids is None:
            expected_video_ids = ids
        elif ids != expected_video_ids:
            raise ValueError("Video IDs or ordering differ across alpha conditions")
        by_alpha[float(alpha)] = group
    return by_alpha


def bootstrap_differences(
    by_alpha: dict[float, list[dict[str, object]]],
    bins: int,
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    baseline_records = by_alpha[BASELINE_ALPHA]
    n_videos = len(baseline_records)
    rng = np.random.default_rng(seed)
    samples: dict[str, np.ndarray] = {}
    comparison_rows: list[dict[str, object]] = []
    point = {alpha: aggregate_records(rows, bins) for alpha, rows in by_alpha.items()}
    comparison_alphas = (SENSITIVITY_ALPHA, PRIMARY_ALPHA)
    for alpha in comparison_alphas:
        for metric in METRICS:
            key = f"alpha_{alpha:.2f}__{metric}"
            samples[key] = np.empty(repetitions, dtype=np.float64)
    for bootstrap_idx in range(repetitions):
        sampled_indices = rng.integers(0, n_videos, size=n_videos, endpoint=False)
        sampled_metrics = {
            alpha: aggregate_records(records, bins, sampled_indices)
            for alpha, records in by_alpha.items()
        }
        for alpha in comparison_alphas:
            for metric in METRICS:
                key = f"alpha_{alpha:.2f}__{metric}"
                samples[key][bootstrap_idx] = sampled_metrics[alpha][metric] - sampled_metrics[BASELINE_ALPHA][metric]
    expected_sign = {
        "accuracy": "noninferior",
        "global_ece": "negative",
        "mean_tfi": "negative",
        "mean_predicted_switches": "negative",
        "mean_transition_delay": "positive",
        "ptsm": "positive",
        "missed_transition_rate": "positive",
    }
    for alpha in comparison_alphas:
        for metric in METRICS:
            key = f"alpha_{alpha:.2f}__{metric}"
            values = samples[key]
            sign = expected_sign[metric]
            if sign == "negative":
                direction_probability = float(np.mean(values < 0.0))
            elif sign == "positive":
                direction_probability = float(np.mean(values > 0.0))
            else:
                direction_probability = float(np.mean(values >= ACCURACY_NONINFERIORITY_MARGIN))
            comparison_rows.append(
                {
                    "comparison": f"ema_{alpha:.2f}_minus_raw",
                    "ema_alpha": alpha,
                    "metric": metric,
                    "raw_point": point[BASELINE_ALPHA][metric],
                    "smoothed_point": point[alpha][metric],
                    "delta_point": point[alpha][metric] - point[BASELINE_ALPHA][metric],
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "expected_direction": sign,
                    "direction_probability": direction_probability,
                    "bootstrap_repetitions": repetitions,
                    "resampling_unit": "video",
                    "seed": seed,
                }
            )
    return comparison_rows, samples


def leave_one_video_out_differences(by_alpha: dict[float, list[dict[str, object]]], bins: int) -> list[dict[str, object]]:
    n_videos = len(by_alpha[BASELINE_ALPHA])
    rows: list[dict[str, object]] = []
    full_indices = np.arange(n_videos, dtype=np.int64)
    for omitted_idx in range(n_videos):
        retained = full_indices[full_indices != omitted_idx]
        metrics = {alpha: aggregate_records(records, bins, retained) for alpha, records in by_alpha.items()}
        omitted_id = str(by_alpha[BASELINE_ALPHA][omitted_idx]["video_id"])
        omitted_frames = int(by_alpha[BASELINE_ALPHA][omitted_idx]["n_frames"])
        for alpha in (SENSITIVITY_ALPHA, PRIMARY_ALPHA):
            for metric in METRICS:
                rows.append(
                    {
                        "omitted_video_id": omitted_id,
                        "omitted_n_frames": omitted_frames,
                        "ema_alpha": alpha,
                        "metric": metric,
                        "delta_after_omission": metrics[alpha][metric] - metrics[BASELINE_ALPHA][metric],
                    }
                )
    return rows


def evaluate_gate(comparison_rows: list[dict[str, object]], alpha: float) -> dict[str, bool]:
    lookup = {
        str(row["metric"]): row
        for row in comparison_rows
        if np.isclose(float(row["ema_alpha"]), alpha)
    }
    checks = {
        "global_ece_improves": float(lookup["global_ece"]["ci_high"]) < 0.0,
        "ptsm_worsens": float(lookup["ptsm"]["ci_low"]) > 0.0,
        "accuracy_noninferior": float(lookup["accuracy"]["ci_low"]) >= ACCURACY_NONINFERIORITY_MARGIN,
        "delay_worsens": float(lookup["mean_transition_delay"]["ci_low"]) > 0.0,
        "miss_rate_worsens": float(lookup["missed_transition_rate"]["ci_low"]) > 0.0,
    }
    checks["responsiveness_worsens"] = checks["delay_worsens"] or checks["miss_rate_worsens"]
    checks["overall"] = (
        checks["global_ece_improves"]
        and checks["ptsm_worsens"]
        and checks["accuracy_noninferior"]
        and checks["responsiveness_worsens"]
    )
    return checks


def collect_predictions(model: object, dataset: object, expected_videos: int) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for index, video in enumerate(dataset):
        scores = np.asarray(model.predict_scores_sequence(video.features), dtype=np.float32)
        labels = np.asarray(video.labels, dtype=np.int64)
        if len(scores) != len(labels):
            raise ValueError(f"{video.video_id}: score/label length mismatch")
        predictions.append({"video_id": str(video.video_id), "labels": labels, "scores": scores})
        print(f"predicted {index + 1}/{expected_videos}: {video.video_id} ({len(labels)} frames)", flush=True)
    if len(predictions) != expected_videos:
        raise ValueError(f"Expected {expected_videos} videos, found {len(predictions)}")
    if len({str(row['video_id']) for row in predictions}) != len(predictions):
        raise ValueError("Duplicate video IDs in predictions")
    return predictions


def save_prediction_cache(path: Path, predictions: list[dict[str, object]]) -> None:
    offsets = [0]
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    video_ids: list[str] = []
    for row in predictions:
        row_labels = np.asarray(row["labels"], dtype=np.int64)
        row_scores = np.asarray(row["scores"], dtype=np.float32)
        labels.append(row_labels)
        scores.append(row_scores)
        video_ids.append(str(row["video_id"]))
        offsets.append(offsets[-1] + len(row_labels))
    np.savez_compressed(
        path,
        video_ids=np.asarray(video_ids),
        offsets=np.asarray(offsets, dtype=np.int64),
        labels=np.concatenate(labels),
        scores=np.concatenate(scores),
    )


def load_prediction_cache(path: Path) -> list[dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        video_ids = archive["video_ids"].astype(str)
        offsets = archive["offsets"].astype(np.int64)
        labels = archive["labels"].astype(np.int64)
        scores = archive["scores"].astype(np.float32)
    predictions: list[dict[str, object]] = []
    for index, video_id in enumerate(video_ids.tolist()):
        start, end = int(offsets[index]), int(offsets[index + 1])
        predictions.append({"video_id": video_id, "labels": labels[start:end], "scores": scores[start:end]})
    return predictions


def load_model(checkpoint: Path) -> object:
    with checkpoint.open("rb") as handle:
        model = pickle.load(handle)
    model.device = "cpu"
    model.device_used_ = "cpu"
    model.model = model.model.cpu()
    model.reset_sequence()
    return model


def write_schema(path: Path, bins: int) -> None:
    schema = {
        "row_unit": "video x ema_alpha",
        "aggregation_contract": "rows are sampled as paired video clusters; ECE is reconstructed from bin sufficient statistics",
        "base_fields": {
            "video_id": "string cluster identifier",
            "ema_alpha": "frozen EMA coefficient",
            "n_frames": "frame denominator for micro accuracy",
            "correct_sum": "correct-frame numerator",
            "tfi": "per-video temporal fragmentation index",
            "predicted_switches": "per-video count",
            "transition_count": "eligible ground-truth transition count",
            "delay_sum": "sum of transition delays; misses contribute the frozen horizon",
            "missed_sum": "number of missed transitions",
            "ptsm_sum": "sum of positive old-minus-new probability margins",
            "ptsm_count": "eligible post-transition frame count",
            "stale_old_label_frame_sum": "descriptive old-argmax frame count"
        },
        "ece_bin_fields": {
            "bins": bins,
            "pattern": "ece_bin_XX_{count,correct_sum,confidence_sum}",
            "edges": np.linspace(0.0, 1.0, bins + 1).tolist(),
            "right_edge_rule": "confidence 1.0 belongs to the final bin"
        }
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def fmt(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def write_gate_report(
    path: Path,
    point_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    loo_rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    primary_checks = evaluate_gate(comparison_rows, PRIMARY_ALPHA)
    sensitivity_checks = evaluate_gate(comparison_rows, SENSITIVITY_ALPHA)
    if primary_checks["overall"]:
        decision = "PASS_PRIMARY"
        carrier = PRIMARY_ALPHA
    elif sensitivity_checks["overall"]:
        decision = "PASS_PREDECLARED_SENSITIVITY"
        carrier = SENSITIVITY_ALPHA
    else:
        decision = "FAIL_STOP"
        carrier = None
    lookup = {(float(row["ema_alpha"]), str(row["metric"])): row for row in comparison_rows}
    loo_lookup: dict[tuple[float, str], list[float]] = {}
    for row in loo_rows:
        key = (float(row["ema_alpha"]), str(row["metric"]))
        loo_lookup.setdefault(key, []).append(float(row["delta_after_omission"]))
    lines = [
        "# G1 confirmatory ranking-inversion gate",
        "",
        f"- Decision: **{decision}**",
        f"- Gate-carrying alpha: **{carrier if carrier is not None else 'none'}**",
        f"- Videos: {metadata['videos']}",
        f"- Frames: {metadata['frames']}",
        f"- Ground-truth transitions: {metadata['transitions']}",
        f"- Bootstrap: {N_BOOTSTRAP} paired video-cluster resamples, seed {BOOTSTRAP_SEED}",
        "",
        "## Frozen point estimates",
        "",
        "| EMA alpha | Accuracy | Global ECE | TFI | Switches/video | Delay | PTSM | Miss rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(point_rows, key=lambda item: float(item["ema_alpha"])):
        lines.append(
            f"| {fmt(row['ema_alpha'], 2)} | {fmt(row['accuracy'])} | {fmt(row['global_ece'])} | "
            f"{fmt(row['mean_tfi'])} | {fmt(row['mean_predicted_switches'], 3)} | "
            f"{fmt(row['mean_transition_delay'])} | {fmt(row['ptsm'])} | {fmt(row['missed_transition_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences (smoothed minus raw)",
            "",
            "| Alpha | Metric | Delta | 95% CI | Expected direction probability |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for alpha in (PRIMARY_ALPHA, SENSITIVITY_ALPHA):
        for metric in ("accuracy", "global_ece", "ptsm", "mean_transition_delay", "missed_transition_rate", "mean_tfi"):
            row = lookup[(alpha, metric)]
            lines.append(
                f"| {alpha:.2f} | {metric} | {fmt(row['delta_point'])} | "
                f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['direction_probability'], 4)} |"
            )
    lines.extend(["", "## Gate checks", ""])
    for alpha, checks in ((PRIMARY_ALPHA, primary_checks), (SENSITIVITY_ALPHA, sensitivity_checks)):
        lines.append(f"### Alpha {alpha:.2f}")
        lines.append("")
        for name, passed in checks.items():
            lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
        lines.append("")
    lines.extend(["## Leave-one-video-out influence", ""])
    for alpha in (PRIMARY_ALPHA, SENSITIVITY_ALPHA):
        ece_values = np.asarray(loo_lookup[(alpha, "global_ece")])
        ptsm_values = np.asarray(loo_lookup[(alpha, "ptsm")])
        lines.append(
            f"- alpha {alpha:.2f}: ECE delta range [{ece_values.min():.6f}, {ece_values.max():.6f}]; "
            f"PTSM delta range [{ptsm_values.min():.6f}, {ptsm_values.max():.6f}]."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This gate tests one frozen predictor and one dataset. Passing confirms the paired empirical ranking inversion under the registered protocol; it does not establish universal generality or metric superiority.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen G1 paired video-cluster bootstrap for transition-conditioned reliability.")
    parser.add_argument("--project-src", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-videos", type=int, default=211)
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_cache = args.output_dir / "frozen_predictions.npz"
    result_paths = [
        args.output_dir / "per_video_statistics.csv",
        args.output_dir / "point_estimates.csv",
        args.output_dir / "bootstrap_deltas.csv",
        args.output_dir / "bootstrap_samples.npz",
        args.output_dir / "leave_one_video_out_deltas.csv",
        args.output_dir / "metadata.json",
        args.output_dir / "gate_G1.md",
    ]
    if any(path.exists() for path in result_paths):
        raise FileExistsError("Confirmatory outputs already exist; use a new output directory to preserve immutability")
    for required in (args.project_src, args.checkpoint, args.feature_dir, args.split_file, args.annotation_dir, args.protocol):
        if not required.exists():
            raise FileNotFoundError(required)
    with args.protocol.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if int(protocol["expected_videos"]) != args.expected_videos:
        raise ValueError("CLI expected-videos conflicts with frozen protocol")
    if args.reuse_predictions:
        if not prediction_cache.exists():
            raise FileNotFoundError(prediction_cache)
        predictions = load_prediction_cache(prediction_cache)
    else:
        if prediction_cache.exists():
            raise FileExistsError("Prediction cache already exists; pass --reuse-predictions or select a new output directory")
        sys.path.insert(0, str(args.project_src))
        from oad_stress_test.datasets.feature_dataset import FeatureDataset

        dataset = FeatureDataset(
            args.feature_dir,
            args.split_file,
            annotation_dir=args.annotation_dir,
            background_label=0,
            end_idx_inclusive=True,
        )
        model = load_model(args.checkpoint)
        predictions = collect_predictions(model, dataset, args.expected_videos)
        save_prediction_cache(prediction_cache, predictions)
    if len(predictions) != args.expected_videos:
        raise ValueError(f"Frozen cache has {len(predictions)} videos; expected {args.expected_videos}")
    records: list[dict[str, object]] = []
    for prediction in predictions:
        for alpha in FROZEN_ALPHAS:
            records.append(
                video_sufficient_stats(
                    str(prediction["video_id"]),
                    np.asarray(prediction["labels"]),
                    np.asarray(prediction["scores"]),
                    alpha,
                    ECE_BINS,
                    HORIZON,
                )
            )
    by_alpha = split_records_by_alpha(records, FROZEN_ALPHAS)
    point_by_alpha = {alpha: aggregate_records(rows, ECE_BINS) for alpha, rows in by_alpha.items()}
    point_rows = [{"ema_alpha": alpha, **metrics} for alpha, metrics in sorted(point_by_alpha.items())]
    comparison_rows, bootstrap_samples = bootstrap_differences(by_alpha, ECE_BINS, N_BOOTSTRAP, BOOTSTRAP_SEED)
    loo_rows = leave_one_video_out_differences(by_alpha, ECE_BINS)
    write_csv(args.output_dir / "per_video_statistics.csv", records)
    write_csv(args.output_dir / "point_estimates.csv", point_rows)
    write_csv(args.output_dir / "bootstrap_deltas.csv", comparison_rows)
    write_csv(args.output_dir / "leave_one_video_out_deltas.csv", loo_rows)
    np.savez_compressed(args.output_dir / "bootstrap_samples.npz", **bootstrap_samples)
    write_schema(args.output_dir / "statistics_schema.json", ECE_BINS)
    baseline_point = point_by_alpha[BASELINE_ALPHA]
    metadata: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "confirmatory",
        "videos": int(baseline_point["n_videos"]),
        "frames": int(baseline_point["n_frames"]),
        "transitions": int(baseline_point["n_transitions"]),
        "ptsm_frames": int(baseline_point["n_ptsm_frames"]),
        "alphas": list(FROZEN_ALPHAS),
        "horizon": HORIZON,
        "ece_bins": ECE_BINS,
        "bootstrap_repetitions": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_unit": "video",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "split_file": str(args.split_file),
        "split_sha256": sha256_file(args.split_file),
        "annotation_file": str(args.annotation_dir / "thumos14.csv"),
        "annotation_sha256": sha256_file(args.annotation_dir / "thumos14.csv"),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "script_sha256": sha256_file(Path(__file__)),
        "prediction_cache": str(prediction_cache),
        "prediction_cache_sha256": sha256_file(prediction_cache),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_gate_report(args.output_dir / "gate_G1.md", point_rows, comparison_rows, loo_rows, metadata)
    print(args.output_dir / "gate_G1.md")


if __name__ == "__main__":
    main()

