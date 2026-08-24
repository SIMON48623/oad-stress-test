from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


WINDOW = 3
ECE_BINS = 15
HORIZON = 16
BOOTSTRAP_REPETITIONS = 2000
ACCURACY_NONINFERIORITY_MARGIN = -0.002
CONDITIONS = ("native", "boxcar_w3")
METRICS = (
    "accuracy",
    "global_ece",
    "mean_tfi",
    "mean_predicted_switches",
    "mean_transition_delay",
    "missed_transition_rate",
)


@dataclass(frozen=True)
class SequencePrediction:
    sequence_id: str
    labels: np.ndarray
    scores: np.ndarray
    active_counts: np.ndarray | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
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


def causal_boxcar_probabilities(scores: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Causal finite-window average with p_0 left padding.

    For W=3, r_t=(p_t+p_{t-1}+p_{t-2})/3. At t=0 and t=1,
    unavailable history is replaced by p_0, matching the EMA initialization.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape [timesteps, classes]")
    if int(window) < 1:
        raise ValueError("window must be positive")
    if len(values) == 0 or int(window) == 1:
        return values.copy()
    padded = np.pad(values, ((int(window) - 1, 0), (0, 0)), mode="edge")
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0, dtype=np.float64)]
    )
    smoothed = (cumulative[int(window) :] - cumulative[: -int(window)]) / float(window)
    smoothed /= np.maximum(smoothed.sum(axis=1, keepdims=True), 1e-12)
    return smoothed


def transition_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    return np.flatnonzero(labels[1:] != labels[:-1]) + 1


def temporal_fragmentation_index(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    if not len(labels):
        return float("nan")
    ground_truth_segments = 1 + int(np.count_nonzero(labels[1:] != labels[:-1]))
    starts = np.concatenate(([0], np.flatnonzero(predictions[1:] != predictions[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(predictions)]))
    disagreement = 0.0
    for start, end in zip(starts.tolist(), ends.tolist()):
        disagreement += float(np.mean(predictions[start:end] != labels[start:end]))
    return disagreement / float(ground_truth_segments)


def fixed_bin_sufficient_stats(
    correct: np.ndarray,
    confidence: np.ndarray,
    bins: int = ECE_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    indices = np.clip(np.floor(confidence * int(bins)).astype(np.int64), 0, int(bins) - 1)
    counts = np.bincount(indices, minlength=int(bins)).astype(np.int64)
    correct_sums = np.bincount(indices, weights=correct, minlength=int(bins)).astype(np.float64)
    confidence_sums = np.bincount(indices, weights=confidence, minlength=int(bins)).astype(np.float64)
    return counts, correct_sums, confidence_sums


def ece_from_sufficient_stats(
    counts: np.ndarray,
    correct_sums: np.ndarray,
    confidence_sums: np.ndarray,
) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    correct_sums = np.asarray(correct_sums, dtype=np.float64)
    confidence_sums = np.asarray(confidence_sums, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    nonempty = counts > 0
    accuracy = correct_sums[nonempty] / counts[nonempty]
    confidence = confidence_sums[nonempty] / counts[nonempty]
    return float(np.sum((counts[nonempty] / total) * np.abs(accuracy - confidence)))


def transition_sufficient_stats(
    labels: np.ndarray,
    predictions: np.ndarray,
    horizon: int,
    active_counts: np.ndarray | None,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    active = None if active_counts is None else np.asarray(active_counts, dtype=np.int64)
    boundaries = transition_indices(labels)
    delay_sum = 0.0
    missed_sum = 0.0
    eligible = 0
    for number, boundary in enumerate(boundaries.tolist()):
        if active is not None and not (active[boundary - 1] == 1 and active[boundary] == 1):
            continue
        next_boundary = int(boundaries[number + 1]) if number + 1 < len(boundaries) else len(labels)
        end = min(len(labels), next_boundary, boundary + int(horizon))
        if end <= boundary:
            continue
        new_label = int(labels[boundary])
        matches = np.flatnonzero(predictions[boundary:end] == new_label)
        if len(matches):
            delay_sum += float(matches[0])
        else:
            delay_sum += float(horizon)
            missed_sum += 1.0
        eligible += 1
    return {
        "transition_count": int(eligible),
        "delay_sum": float(delay_sum),
        "missed_sum": float(missed_sum),
    }


def sequence_sufficient_stats(
    prediction: SequencePrediction,
    condition: str,
    bins: int = ECE_BINS,
    horizon: int = HORIZON,
) -> dict[str, object]:
    if condition == "native":
        scores = np.asarray(prediction.scores, dtype=np.float64)
    elif condition == "boxcar_w3":
        scores = causal_boxcar_probabilities(prediction.scores, WINDOW)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    labels = np.asarray(prediction.labels, dtype=np.int64)
    if len(labels) != len(scores):
        raise ValueError(f"{prediction.sequence_id}: score/label length mismatch")
    predictions = np.argmax(scores, axis=1).astype(np.int64)
    confidence = np.max(scores, axis=1)
    correct = predictions == labels
    bin_counts, bin_correct, bin_confidence = fixed_bin_sufficient_stats(correct, confidence, bins)
    transition = transition_sufficient_stats(
        labels,
        predictions,
        horizon,
        prediction.active_counts,
    )
    row: dict[str, object] = {
        "sequence_id": prediction.sequence_id,
        "condition": condition,
        "n_frames": int(len(labels)),
        "correct_sum": int(correct.sum()),
        "tfi": temporal_fragmentation_index(labels, predictions),
        "predicted_switches": int(np.count_nonzero(predictions[1:] != predictions[:-1])),
        **transition,
    }
    for bin_idx in range(int(bins)):
        prefix = f"ece_bin_{bin_idx:02d}"
        row[f"{prefix}_count"] = int(bin_counts[bin_idx])
        row[f"{prefix}_correct_sum"] = float(bin_correct[bin_idx])
        row[f"{prefix}_confidence_sum"] = float(bin_confidence[bin_idx])
    return row


def aggregate_records(
    records: list[dict[str, object]],
    bins: int = ECE_BINS,
    sampled_indices: np.ndarray | None = None,
) -> dict[str, float]:
    if sampled_indices is None:
        sampled_indices = np.arange(len(records), dtype=np.int64)
    selected = [records[int(index)] for index in np.asarray(sampled_indices, dtype=np.int64)]
    frames = float(sum(int(row["n_frames"]) for row in selected))
    transitions = float(sum(int(row["transition_count"]) for row in selected))
    counts = np.asarray(
        [sum(int(row[f"ece_bin_{idx:02d}_count"]) for row in selected) for idx in range(int(bins))],
        dtype=np.float64,
    )
    correct_sums = np.asarray(
        [sum(float(row[f"ece_bin_{idx:02d}_correct_sum"]) for row in selected) for idx in range(int(bins))],
        dtype=np.float64,
    )
    confidence_sums = np.asarray(
        [sum(float(row[f"ece_bin_{idx:02d}_confidence_sum"]) for row in selected) for idx in range(int(bins))],
        dtype=np.float64,
    )
    return {
        "accuracy": float(sum(int(row["correct_sum"]) for row in selected) / frames),
        "global_ece": ece_from_sufficient_stats(counts, correct_sums, confidence_sums),
        "mean_tfi": float(np.mean([float(row["tfi"]) for row in selected])),
        "mean_predicted_switches": float(np.mean([float(row["predicted_switches"]) for row in selected])),
        "mean_transition_delay": float(sum(float(row["delay_sum"]) for row in selected) / transitions),
        "missed_transition_rate": float(sum(float(row["missed_sum"]) for row in selected) / transitions),
        "n_frames": frames,
        "n_sequences": float(len(selected)),
        "n_transitions": transitions,
    }


def paired_bootstrap(
    by_condition: dict[str, list[dict[str, object]]],
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    native_ids = [str(row["sequence_id"]) for row in by_condition["native"]]
    boxcar_ids = [str(row["sequence_id"]) for row in by_condition["boxcar_w3"]]
    if native_ids != boxcar_ids:
        raise ValueError("Sequence IDs or ordering differ across conditions")
    point = {condition: aggregate_records(rows) for condition, rows in by_condition.items()}
    samples = {metric: np.empty(int(repetitions), dtype=np.float64) for metric in METRICS}
    rng = np.random.default_rng(int(seed))
    n_sequences = len(native_ids)
    for bootstrap_index in range(int(repetitions)):
        indices = rng.integers(0, n_sequences, size=n_sequences, dtype=np.int64)
        native = aggregate_records(by_condition["native"], sampled_indices=indices)
        boxcar = aggregate_records(by_condition["boxcar_w3"], sampled_indices=indices)
        for metric in METRICS:
            samples[metric][bootstrap_index] = boxcar[metric] - native[metric]
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        values = samples[metric]
        rows.append(
            {
                "comparison": "boxcar_w3_minus_native",
                "metric": metric,
                "native_point": point["native"][metric],
                "boxcar_point": point["boxcar_w3"][metric],
                "delta_point": point["boxcar_w3"][metric] - point["native"][metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_repetitions": int(repetitions),
                "bootstrap_seed": int(seed),
            }
        )
    return rows, samples, point


def load_thumos_cache(path: Path) -> list[SequencePrediction]:
    with np.load(path, allow_pickle=False) as archive:
        video_ids = archive["video_ids"].astype(str)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        scores = np.asarray(archive["scores"], dtype=np.float32)
    sequences: list[SequencePrediction] = []
    for index, video_id in enumerate(video_ids.tolist()):
        start, end = int(offsets[index]), int(offsets[index + 1])
        sequences.append(SequencePrediction(video_id, labels[start:end], scores[start:end]))
    return sequences


def save_ek100_cache(path: Path, predictions: list[SequencePrediction]) -> None:
    payload: dict[str, np.ndarray] = {}
    for prediction in predictions:
        payload[f"{prediction.sequence_id}__scores"] = np.asarray(prediction.scores, dtype=np.float32)
        payload[f"{prediction.sequence_id}__primary_labels"] = np.asarray(prediction.labels, dtype=np.int16)
        payload[f"{prediction.sequence_id}__active_counts"] = np.asarray(prediction.active_counts, dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_ek100_cache(path: Path) -> list[SequencePrediction]:
    with np.load(path, allow_pickle=False) as archive:
        ids = sorted(key[: -len("__scores")] for key in archive.files if key.endswith("__scores"))
        return [
            SequencePrediction(
                sequence_id,
                np.asarray(archive[f"{sequence_id}__primary_labels"], dtype=np.int64),
                np.asarray(archive[f"{sequence_id}__scores"], dtype=np.float32),
                np.asarray(archive[f"{sequence_id}__active_counts"], dtype=np.int64),
            )
            for sequence_id in ids
        ]


def common_session_ids(rgb_dir: Path, flow_dir: Path, target_dir: Path) -> list[str]:
    def stems(path: Path) -> set[str]:
        return {item.stem for item in path.glob("*.npy") if item.is_file()}

    ids = sorted(stems(rgb_dir) & stems(flow_dir) & stems(target_dir))
    if not ids:
        raise ValueError("No aligned EK100 session IDs were found")
    return ids


def rebuild_ek100_predictions(
    model_path: Path,
    data_root: Path,
    project_root: Path,
    protocol_path: Path,
    cache_path: Path,
) -> list[SequencePrediction]:
    import torch

    project_src = project_root / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from oad_stress_test.models.causal_gru import _CausalGRUNet

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rgb_dir = data_root / protocol["inputs"]["rgb_subdir"]
    flow_dir = data_root / protocol["inputs"]["flow_subdir"]
    target_dir = data_root / protocol["inputs"]["verb_target_subdir"]
    ids = common_session_ids(rgb_dir, flow_dir, target_dir)
    train_count = int(protocol["dataset"]["train_sessions"])
    eval_count = int(protocol["dataset"]["evaluation_sessions"])
    eval_ids = ids[train_count : train_count + eval_count]
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    model = _CausalGRUNet(
        input_dim=int(len(feature_mean)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=0.0,
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.gru.flatten_parameters()
    predictions: list[SequencePrediction] = []
    chunk_size = 2048
    for sequence_index, sequence_id in enumerate(eval_ids, start=1):
        rgb = np.load(rgb_dir / f"{sequence_id}.npy", mmap_mode="r")
        flow = np.load(flow_dir / f"{sequence_id}.npy", mmap_mode="r")
        target = np.load(target_dir / f"{sequence_id}.npy", mmap_mode="r")
        length = min(len(rgb), len(flow), len(target))
        active = np.asarray(target[:length] > 0, dtype=bool)
        labels = np.zeros(length, dtype=np.int64)
        has_active = active.any(axis=1)
        labels[has_active] = np.argmax(active[has_active], axis=1).astype(np.int64)
        active_counts = active.sum(axis=1).astype(np.int64)
        score_chunks: list[np.ndarray] = []
        hidden = None
        with torch.no_grad():
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                features = np.concatenate(
                    [np.asarray(rgb[start:end], dtype=np.float32), np.asarray(flow[start:end], dtype=np.float32)],
                    axis=1,
                )
                features = ((features - feature_mean) / feature_std).astype(np.float32)
                logits, hidden = model(torch.from_numpy(features).unsqueeze(0), hidden)
                hidden = hidden.detach()
                scores = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
                score_chunks.append(scores)
        predictions.append(
            SequencePrediction(
                sequence_id,
                labels,
                np.concatenate(score_chunks, axis=0),
                active_counts,
            )
        )
        if sequence_index % 10 == 0 or sequence_index == len(eval_ids):
            print(f"Rebuilt EK100 probabilities {sequence_index}/{len(eval_ids)}", flush=True)
    expected_timesteps = int(protocol["dataset"]["expected_evaluation_timesteps"])
    observed_timesteps = int(sum(len(item.labels) for item in predictions))
    if observed_timesteps != expected_timesteps:
        raise ValueError(f"EK100 timestep inventory mismatch: {observed_timesteps} vs {expected_timesteps}")
    save_ek100_cache(cache_path, predictions)
    return predictions


def leave_one_out_ranges(by_condition: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, float]]:
    n_sequences = len(by_condition["native"])
    values = {metric: [] for metric in METRICS}
    for omitted in range(n_sequences):
        retained = np.asarray([idx for idx in range(n_sequences) if idx != omitted], dtype=np.int64)
        native = aggregate_records(by_condition["native"], sampled_indices=retained)
        boxcar = aggregate_records(by_condition["boxcar_w3"], sampled_indices=retained)
        for metric in METRICS:
            values[metric].append(boxcar[metric] - native[metric])
    return {
        metric: {"min": float(np.min(metric_values)), "max": float(np.max(metric_values))}
        for metric, metric_values in values.items()
    }


def gate_checks(comparison_rows: list[dict[str, object]]) -> dict[str, bool]:
    lookup = {str(row["metric"]): row for row in comparison_rows}
    checks = {
        "accuracy_noninferior": float(lookup["accuracy"]["ci_low"]) >= ACCURACY_NONINFERIORITY_MARGIN,
        "global_ece_improves": float(lookup["global_ece"]["ci_high"]) < 0.0,
        "tfi_improves": float(lookup["mean_tfi"]["ci_high"]) < 0.0,
        "delay_worsens": float(lookup["mean_transition_delay"]["ci_low"]) > 0.0,
        "miss_rate_worsens": float(lookup["missed_transition_rate"]["ci_low"]) > 0.0,
    }
    checks["aggregate_prefers_boxcar"] = (
        checks["accuracy_noninferior"] and checks["global_ece_improves"] and checks["tfi_improves"]
    )
    checks["transition_prefers_native"] = checks["delay_worsens"] and checks["miss_rate_worsens"]
    checks["full_ranking_inversion"] = checks["aggregate_prefers_boxcar"] and checks["transition_prefers_native"]
    return checks


def evaluate_instance(
    name: str,
    predictions: list[SequencePrediction],
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    records = [
        sequence_sufficient_stats(prediction, condition)
        for prediction in predictions
        for condition in CONDITIONS
    ]
    by_condition = {
        condition: sorted(
            [row for row in records if row["condition"] == condition],
            key=lambda row: str(row["sequence_id"]),
        )
        for condition in CONDITIONS
    }
    comparison_rows, samples, point = paired_bootstrap(
        by_condition,
        BOOTSTRAP_REPETITIONS,
        seed,
    )
    loo_ranges = leave_one_out_ranges(by_condition)
    checks = gate_checks(comparison_rows)
    instance_dir = output_dir / name
    instance_dir.mkdir(parents=True, exist_ok=True)
    write_csv(instance_dir / "per_sequence_statistics.csv", records)
    write_csv(instance_dir / "bootstrap_deltas.csv", comparison_rows)
    write_csv(
        instance_dir / "point_estimates.csv",
        [{"condition": condition, **values} for condition, values in point.items()],
    )
    np.savez_compressed(instance_dir / "bootstrap_samples.npz", **samples)
    (instance_dir / "leave_one_out_ranges.json").write_text(
        json.dumps(loo_ranges, indent=2) + "\n",
        encoding="utf-8",
    )
    (instance_dir / "gate_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    lookup = {str(row["metric"]): row for row in comparison_rows}
    return {
        "instance": name,
        "sequences": len(predictions),
        "frames": int(point["native"]["n_frames"]),
        "transitions": int(point["native"]["n_transitions"]),
        "checks": checks,
        "metrics": {
            metric: {
                "native": float(lookup[metric]["native_point"]),
                "boxcar": float(lookup[metric]["boxcar_point"]),
                "delta": float(lookup[metric]["delta_point"]),
                "ci_low": float(lookup[metric]["ci_low"]),
                "ci_high": float(lookup[metric]["ci_high"]),
                "loo_min": float(loo_ranges[metric]["min"]),
                "loo_max": float(loo_ranges[metric]["max"]),
            }
            for metric in METRICS
        },
    }


def report_markdown(results: list[dict[str, object]]) -> str:
    lines = [
        "# Predeclared causal boxcar W=3 replication",
        "",
        "The only added postprocessor is the three-timestep causal boxcar average. Its mean kernel age and squared-weight sum both match the primary EMA with alpha=0.50. No other window was evaluated.",
        "",
        "| Instance | Delta accuracy | Delta ECE | Delta TFI | Delta delay | Delta miss rate | Full inversion |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        metrics = dict(result["metrics"])
        checks = dict(result["checks"])
        lines.append(
            f"| {result['instance']} | {metrics['accuracy']['delta']:+.6f} | "
            f"{metrics['global_ece']['delta']:+.6f} | {metrics['mean_tfi']['delta']:+.6f} | "
            f"{metrics['mean_transition_delay']['delta']:+.6f} | "
            f"{metrics['missed_transition_rate']['delta']:+.6f} | "
            f"{'PASS' if checks['full_ranking_inversion'] else 'FAIL'} |"
        )
    lines.extend(["", "## Paired 95% cluster-bootstrap intervals", ""])
    for result in results:
        lines.extend([f"### {result['instance']}", ""])
        for metric in METRICS:
            values = dict(result["metrics"])[metric]
            lines.append(
                f"- {metric}: {values['delta']:+.6f} "
                f"[{values['ci_low']:+.6f}, {values['ci_high']:+.6f}]"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision rule",
            "",
            "A full replication requires accuracy noninferiority, significant ECE and TFI improvement, and significant worsening of both transition delay and missed-transition rate. A failed direction is reported as a boundary result and does not authorize another window.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-cache", required=True, type=Path)
    parser.add_argument("--g4-cache", required=True, type=Path)
    parser.add_argument("--ek100-model", required=True, type=Path)
    parser.add_argument("--ek100-cache", required=True, type=Path)
    parser.add_argument("--ek100-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--ek100-protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "postprocessor": "causal_boxcar_probability_average",
        "window": WINDOW,
        "left_initialization": "repeat p_0 for unavailable history",
        "matched_primary_ema_alpha": 0.50,
        "matching": {
            "mean_kernel_age_timesteps": 1.0,
            "sum_squared_weights": 1.0 / 3.0,
        },
        "ece_bins": ECE_BINS,
        "transition_horizon_feature_timesteps": HORIZON,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "accuracy_noninferiority_margin": ACCURACY_NONINFERIORITY_MARGIN,
        "no_other_window_evaluated": True,
        "inputs": {
            "g1_cache": str(args.g1_cache.resolve()),
            "g1_cache_sha256": sha256_file(args.g1_cache),
            "g4_cache": str(args.g4_cache.resolve()),
            "g4_cache_sha256": sha256_file(args.g4_cache),
            "ek100_model": str(args.ek100_model.resolve()),
            "ek100_model_sha256": sha256_file(args.ek100_model),
            "ek100_protocol": str(args.ek100_protocol.resolve()),
            "ek100_protocol_sha256": sha256_file(args.ek100_protocol),
        },
    }
    (output_dir / "boxcar_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    g1_predictions = load_thumos_cache(args.g1_cache)
    print("Evaluating THUMOS14 causal GRU", flush=True)
    results = [evaluate_instance("thumos14_causal_gru", g1_predictions, 48623, output_dir)]

    g4_predictions = load_thumos_cache(args.g4_cache)
    print("Evaluating THUMOS14 linear probe", flush=True)
    results.append(evaluate_instance("thumos14_linear_probe", g4_predictions, 48625, output_dir))

    if args.ek100_cache.exists():
        print("Loading rebuilt EK100 prediction cache", flush=True)
        ek100_predictions = load_ek100_cache(args.ek100_cache)
    else:
        print("Rebuilding EK100 prediction cache from the frozen formal model", flush=True)
        ek100_predictions = rebuild_ek100_predictions(
            args.ek100_model,
            args.ek100_root,
            args.project_root,
            args.ek100_protocol,
            args.ek100_cache,
        )
    print("Evaluating EK100 causal GRU", flush=True)
    results.append(evaluate_instance("ek100_causal_gru", ek100_predictions, 48627, output_dir))

    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (output_dir / "BOXCAR_W3_REPORT.md").write_text(report_markdown(results), encoding="utf-8")
    print(output_dir / "BOXCAR_W3_REPORT.md")


if __name__ == "__main__":
    main()
