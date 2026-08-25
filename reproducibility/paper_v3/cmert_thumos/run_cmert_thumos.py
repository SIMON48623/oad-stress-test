from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CONDITIONS = ("native", "ema_alpha_0.50", "boxcar_w3")
METRICS = (
    "accuracy",
    "global_ece",
    "mean_tfi",
    "mean_predicted_switches",
    "mean_transition_delay",
    "missed_transition_rate",
)
EMA_ALPHA = 0.50
BOXCAR_WINDOW = 3
ECE_BINS = 15
HORIZON = 16
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 48629
ACCURACY_NONINFERIORITY_MARGIN = -0.002
IGNORE_INDEX = 21


@dataclass(frozen=True)
class Sequence:
    sequence_id: str
    scores: np.ndarray
    targets: np.ndarray


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_sequences(input_dir: Path) -> list[Sequence]:
    sequences: list[Sequence] = []
    for path in sorted(input_dir.glob("video_test_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            scores = np.asarray(archive["probabilities"], dtype=np.float64)
            targets = np.asarray(archive["targets"], dtype=np.float64)
        if scores.shape != targets.shape or scores.ndim != 2 or scores.shape[1] != 22:
            raise ValueError(
                f"{path.name}: unexpected score/target shapes {scores.shape}, {targets.shape}"
            )
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(targets)):
            raise ValueError(f"{path.name}: non-finite score or target")
        if np.min(scores) < -1e-7 or not np.allclose(scores.sum(axis=1), 1.0, atol=2e-5):
            raise ValueError(f"{path.name}: exported values are not probability vectors")
        sequences.append(Sequence(path.stem, scores, targets))
    if len(sequences) != 213:
        raise ValueError(f"Expected 213 official THUMOS14 test videos, found {len(sequences)}")
    return sequences


def smooth(scores: np.ndarray, condition: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if condition == "native" or len(values) <= 1:
        return values.copy()
    output = np.empty_like(values)
    if condition == "ema_alpha_0.50":
        output[0] = values[0]
        for index in range(1, len(values)):
            output[index] = EMA_ALPHA * output[index - 1] + (1.0 - EMA_ALPHA) * values[index]
    elif condition == "boxcar_w3":
        padded = np.pad(values, ((BOXCAR_WINDOW - 1, 0), (0, 0)), mode="edge")
        cumulative = np.vstack(
            [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0)]
        )
        output = (cumulative[BOXCAR_WINDOW:] - cumulative[:-BOXCAR_WINDOW]) / BOXCAR_WINDOW
    else:
        raise ValueError(f"Unknown condition: {condition}")
    output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-12)
    return output


def ece_stats(
    correct: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.clip(np.floor(confidence * ECE_BINS).astype(np.int64), 0, ECE_BINS - 1)
    counts = np.bincount(indices, minlength=ECE_BINS).astype(np.int64)
    correct_sum = np.bincount(indices, weights=correct.astype(np.float64), minlength=ECE_BINS)
    confidence_sum = np.bincount(indices, weights=confidence, minlength=ECE_BINS)
    return counts, correct_sum, confidence_sum


def ece_from_stats(
    counts: np.ndarray, correct_sum: np.ndarray, confidence_sum: np.ndarray
) -> float:
    total = float(counts.sum())
    nonempty = counts > 0
    bin_accuracy = correct_sum[nonempty] / counts[nonempty]
    bin_confidence = confidence_sum[nonempty] / counts[nonempty]
    return float(np.sum((counts[nonempty] / total) * np.abs(bin_accuracy - bin_confidence)))


def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(valid.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def fragmentation_parts(
    labels: np.ndarray, predictions: np.ndarray, valid: np.ndarray
) -> tuple[float, int, int]:
    disagreement_sum = 0.0
    ground_truth_segments = 0
    predicted_switches = 0
    for start, end in valid_runs(valid):
        run_labels = labels[start:end]
        run_predictions = predictions[start:end]
        ground_truth_segments += 1 + int(np.count_nonzero(run_labels[1:] != run_labels[:-1]))
        predicted_switches += int(np.count_nonzero(run_predictions[1:] != run_predictions[:-1]))
        pred_starts = np.concatenate(
            ([0], np.flatnonzero(run_predictions[1:] != run_predictions[:-1]) + 1)
        )
        pred_ends = np.concatenate((pred_starts[1:], [len(run_predictions)]))
        for pred_start, pred_end in zip(pred_starts.tolist(), pred_ends.tolist()):
            disagreement_sum += float(
                np.mean(run_predictions[pred_start:pred_end] != run_labels[pred_start:pred_end])
            )
    return disagreement_sum, ground_truth_segments, predicted_switches


def transition_stats(
    labels: np.ndarray,
    predictions: np.ndarray,
    active_counts: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    transition_count = 0
    delay_sum = 0.0
    missed_sum = 0.0
    for number, boundary in enumerate(boundaries.tolist()):
        if not (
            valid[boundary - 1]
            and valid[boundary]
            and active_counts[boundary - 1] == 1
            and active_counts[boundary] == 1
        ):
            continue
        next_boundary = int(boundaries[number + 1]) if number + 1 < len(boundaries) else len(labels)
        end = min(len(labels), next_boundary, boundary + HORIZON)
        invalid = np.flatnonzero(~valid[boundary:end])
        if len(invalid):
            end = boundary + int(invalid[0])
        if end <= boundary:
            continue
        matches = np.flatnonzero(predictions[boundary:end] == int(labels[boundary]))
        if len(matches):
            delay_sum += float(matches[0])
        else:
            delay_sum += float(HORIZON)
            missed_sum += 1.0
        transition_count += 1
    return {
        "transition_count": transition_count,
        "delay_sum": delay_sum,
        "missed_sum": missed_sum,
    }


def sequence_stats(sequence: Sequence, condition: str) -> dict[str, object]:
    scores = smooth(sequence.scores, condition)
    targets = sequence.targets
    labels = np.argmax(targets, axis=1).astype(np.int64)
    active_counts = np.count_nonzero(targets > 0.5, axis=1).astype(np.int64)
    valid = targets[:, IGNORE_INDEX] < 0.5
    predictions = np.argmax(scores, axis=1).astype(np.int64)
    confidence = np.max(scores, axis=1)
    correct = predictions[valid] == labels[valid]
    counts, correct_sum, confidence_sum = ece_stats(correct, confidence[valid])
    disagreement_sum, gt_segments, predicted_switches = fragmentation_parts(
        labels, predictions, valid
    )
    transition = transition_stats(labels, predictions, active_counts, valid)
    row: dict[str, object] = {
        "sequence_id": sequence.sequence_id,
        "condition": condition,
        "n_frames": int(len(labels)),
        "n_valid_frames": int(np.count_nonzero(valid)),
        "n_ignored_ambiguous_frames": int(np.count_nonzero(~valid)),
        "correct_sum": int(np.count_nonzero(correct)),
        "fragmentation_disagreement_sum": disagreement_sum,
        "ground_truth_segments": gt_segments,
        "tfi": float(disagreement_sum / gt_segments),
        "predicted_switches": predicted_switches,
        **transition,
    }
    for index in range(ECE_BINS):
        row[f"ece_bin_{index:02d}_count"] = int(counts[index])
        row[f"ece_bin_{index:02d}_correct_sum"] = float(correct_sum[index])
        row[f"ece_bin_{index:02d}_confidence_sum"] = float(confidence_sum[index])
    return row


def aggregate(rows: list[dict[str, object]], indices: np.ndarray | None = None) -> dict[str, float]:
    if indices is None:
        indices = np.arange(len(rows), dtype=np.int64)
    selected = [rows[int(index)] for index in np.asarray(indices, dtype=np.int64)]
    frames = float(sum(int(row["n_valid_frames"]) for row in selected))
    transitions = float(sum(int(row["transition_count"]) for row in selected))
    counts = np.asarray(
        [sum(int(row[f"ece_bin_{i:02d}_count"]) for row in selected) for i in range(ECE_BINS)],
        dtype=np.float64,
    )
    correct_sum = np.asarray(
        [sum(float(row[f"ece_bin_{i:02d}_correct_sum"]) for row in selected) for i in range(ECE_BINS)],
        dtype=np.float64,
    )
    confidence_sum = np.asarray(
        [
            sum(float(row[f"ece_bin_{i:02d}_confidence_sum"]) for row in selected)
            for i in range(ECE_BINS)
        ],
        dtype=np.float64,
    )
    return {
        "accuracy": float(sum(int(row["correct_sum"]) for row in selected) / frames),
        "global_ece": ece_from_stats(counts, correct_sum, confidence_sum),
        "mean_tfi": float(np.mean([float(row["tfi"]) for row in selected])),
        "mean_predicted_switches": float(
            np.mean([float(row["predicted_switches"]) for row in selected])
        ),
        "mean_transition_delay": float(
            sum(float(row["delay_sum"]) for row in selected) / transitions
        ),
        "missed_transition_rate": float(
            sum(float(row["missed_sum"]) for row in selected) / transitions
        ),
        "n_valid_frames": frames,
        "n_sequences": float(len(selected)),
        "n_transitions": transitions,
    }


def gate_checks(lookup: dict[str, dict[str, object]]) -> dict[str, bool]:
    checks = {
        "accuracy_noninferior": float(lookup["accuracy"]["ci_low"])
        >= ACCURACY_NONINFERIORITY_MARGIN,
        "global_ece_improves": float(lookup["global_ece"]["ci_high"]) < 0.0,
        "tfi_improves": float(lookup["mean_tfi"]["ci_high"]) < 0.0,
        "delay_worsens": float(lookup["mean_transition_delay"]["ci_low"]) > 0.0,
        "miss_rate_worsens": float(lookup["missed_transition_rate"]["ci_low"]) > 0.0,
    }
    checks["full_ranking_inversion"] = all(checks.values())
    return checks


def audit(sequences: list[Sequence], input_dir: Path, output_dir: Path) -> dict[str, object]:
    records = [sequence_stats(sequence, condition) for sequence in sequences for condition in CONDITIONS]
    groups = {
        condition: sorted(
            [row for row in records if row["condition"] == condition],
            key=lambda row: str(row["sequence_id"]),
        )
        for condition in CONDITIONS
    }
    ids = [[str(row["sequence_id"]) for row in groups[condition]] for condition in CONDITIONS]
    if any(group_ids != ids[0] for group_ids in ids[1:]):
        raise ValueError("Sequence IDs differ across conditions")

    points = {condition: aggregate(groups[condition]) for condition in CONDITIONS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {
        f"{condition}__{metric}": np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
        for condition in CONDITIONS[1:]
        for metric in METRICS
    }
    for bootstrap_index in range(BOOTSTRAP_REPETITIONS):
        selected = rng.integers(0, len(ids[0]), size=len(ids[0]), dtype=np.int64)
        native = aggregate(groups["native"], selected)
        for condition in CONDITIONS[1:]:
            current = aggregate(groups[condition], selected)
            for metric in METRICS:
                samples[f"{condition}__{metric}"][bootstrap_index] = (
                    current[metric] - native[metric]
                )

    comparison_rows: list[dict[str, object]] = []
    gates: dict[str, dict[str, bool]] = {}
    for condition in CONDITIONS[1:]:
        lookup: dict[str, dict[str, object]] = {}
        for metric in METRICS:
            values = samples[f"{condition}__{metric}"]
            row = {
                "comparison": f"{condition}_minus_native",
                "metric": metric,
                "native_point": points["native"][metric],
                "smoothed_point": points[condition][metric],
                "delta_point": points[condition][metric] - points["native"][metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "resampling_unit": "official THUMOS14 test video",
            }
            comparison_rows.append(row)
            lookup[metric] = row
        gates[condition] = gate_checks(lookup)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_video_statistics.csv", records)
    write_csv(
        output_dir / "point_estimates.csv",
        [{"condition": key, **value} for key, value in points.items()],
    )
    write_csv(output_dir / "bootstrap_deltas.csv", comparison_rows)
    np.savez_compressed(output_dir / "bootstrap_samples.npz", **samples)
    summary = {
        "frozen_protocol": {
            "conditions": list(CONDITIONS),
            "ema_alpha": EMA_ALPHA,
            "boxcar_window": BOXCAR_WINDOW,
            "ece_bins": ECE_BINS,
            "transition_horizon_feature_timesteps": HORIZON,
            "transition_horizon_seconds": HORIZON / 4.0,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "accuracy_noninferiority_margin": ACCURACY_NONINFERIORITY_MARGIN,
            "ignore_index": IGNORE_INDEX,
            "parameter_search_after_observing_results": False,
        },
        "input_inventory": {
            "sessions": len(sequences),
            "frames": int(sum(len(sequence.scores) for sequence in sequences)),
            "file_sha256": {
                f"{sequence.sequence_id}.npz": sha256_file(
                    input_dir / f"{sequence.sequence_id}.npz"
                )
                for sequence in sequences
            },
        },
        "points": points,
        "gates": gates,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen CMeRT-THUMOS14 current-output probabilities"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    summary = audit(load_sequences(input_dir), input_dir, args.output_dir.resolve())
    print(json.dumps({"points": summary["points"], "gates": summary["gates"]}, indent=2))


if __name__ == "__main__":
    main()
