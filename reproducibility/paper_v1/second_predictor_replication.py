from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from confirmatory_cluster_bootstrap import (
    ACCURACY_NONINFERIORITY_MARGIN,
    BASELINE_ALPHA,
    ECE_BINS,
    FROZEN_ALPHAS,
    HORIZON,
    METRICS,
    PRIMARY_ALPHA,
    SENSITIVITY_ALPHA,
    aggregate_records,
    bootstrap_differences,
    leave_one_video_out_differences,
    save_prediction_cache,
    sha256_file,
    split_records_by_alpha,
    video_sufficient_stats,
    write_csv,
    write_schema,
)


N_BOOTSTRAP = 2000
G4_SEED = 48625


def vectorized_linear_scores(model: object, features: np.ndarray) -> np.ndarray:
    if getattr(model, "scaler", None) is None or getattr(model, "model", None) is None:
        raise RuntimeError("Linear probe is not fitted")
    x = np.asarray(features, dtype=np.float32)
    x_scaled = model.scaler.transform(x)
    margins = np.asarray(model.model.decision_function(x_scaled), dtype=np.float64)
    classes = np.asarray(model.model.classes_, dtype=np.int64)
    if margins.ndim == 1:
        if len(classes) != 2:
            raise RuntimeError("One-dimensional decision margins require exactly two classes")
        margins = np.stack([-margins, margins], axis=1)
    if margins.shape[1] != len(classes):
        raise RuntimeError("Decision-margin width does not match classifier classes")
    margins = np.nan_to_num(margins, nan=0.0, posinf=50.0, neginf=-50.0)
    score_mode = str(model.score_mode)
    if score_mode == "softmax":
        centered = margins - np.max(margins, axis=1, keepdims=True)
        partial = np.exp(np.clip(centered, -50.0, 50.0))
        partial /= np.maximum(partial.sum(axis=1, keepdims=True), 1e-12)
        fill_value = 0.0
    elif score_mode == "sigmoid":
        partial = 1.0 / (1.0 + np.exp(-np.clip(margins, -50.0, 50.0)))
        fill_value = 0.0
    elif score_mode == "raw_margin":
        partial = margins
        fill_value = -1e9
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")
    scores = np.full((len(x), int(model.num_classes)), fill_value, dtype=np.float64)
    valid = (classes >= 0) & (classes < int(model.num_classes))
    scores[:, classes[valid]] = partial[:, valid]
    if score_mode == "softmax":
        scores /= np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
    return scores.astype(np.float32)


def inference_equivalence(model: object, features: np.ndarray, frames: int = 20) -> dict[str, object]:
    take = min(frames, len(features))
    batch = vectorized_linear_scores(model, features[:take])
    reference = np.stack([model.predict_scores(feature) for feature in features[:take]]).astype(np.float32)
    difference = np.abs(batch - reference)
    result = {
        "frames_checked": take,
        "classes": int(batch.shape[1]),
        "max_absolute_difference": float(difference.max(initial=0.0)),
        "mean_absolute_difference": float(difference.mean()),
        "tolerance": 1e-7,
    }
    result["pass"] = bool(result["max_absolute_difference"] <= result["tolerance"])
    return result


def collect_predictions(model: object, dataset: object, expected_videos: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    predictions: list[dict[str, object]] = []
    equivalence: dict[str, object] | None = None
    for index, video in enumerate(dataset):
        if equivalence is None:
            batch_candidate = inference_equivalence(model, video.features, frames=20)
            reference = np.stack([model.predict_scores(feature) for feature in video.features[:20]]).astype(np.float32)
            selected = np.stack([model.predict_scores(feature) for feature in video.features[:20]]).astype(np.float32)
            selected_difference = np.abs(selected - reference)
            equivalence = {
                "selected_method": "original_per_frame_predict_scores",
                "selected_frames_checked": int(len(selected)),
                "selected_max_absolute_difference": float(selected_difference.max(initial=0.0)),
                "selected_pass": bool(selected_difference.max(initial=0.0) == 0.0),
                "rejected_batch_candidate": batch_candidate,
            }
            if not equivalence["selected_pass"]:
                raise RuntimeError(f"Original-interface equivalence failed: {equivalence}")
        scores = np.stack([model.predict_scores(feature) for feature in video.features]).astype(np.float32)
        labels = np.asarray(video.labels, dtype=np.int64)
        if len(scores) != len(labels):
            raise ValueError(f"{video.video_id}: score/label length mismatch")
        predictions.append({"video_id": str(video.video_id), "labels": labels, "scores": scores})
        print(f"predicted {index + 1}/{expected_videos}: {video.video_id} ({len(labels)} frames)", flush=True)
    if len(predictions) != expected_videos:
        raise ValueError(f"Expected {expected_videos} videos, found {len(predictions)}")
    if equivalence is None:
        raise RuntimeError("Dataset is empty")
    return predictions, equivalence


def gate_checks(comparison_rows: list[dict[str, object]], alpha: float) -> dict[str, bool]:
    lookup = {
        str(row["metric"]): row
        for row in comparison_rows
        if np.isclose(float(row["ema_alpha"]), alpha)
    }
    checks = {
        "global_ece_improves": float(lookup["global_ece"]["ci_high"]) < 0.0,
        "tfi_improves": float(lookup["mean_tfi"]["ci_high"]) < 0.0,
        "accuracy_noninferior": float(lookup["accuracy"]["ci_low"]) >= ACCURACY_NONINFERIORITY_MARGIN,
        "delay_worsens": float(lookup["mean_transition_delay"]["ci_low"]) > 0.0,
        "miss_rate_worsens": float(lookup["missed_transition_rate"]["ci_low"]) > 0.0,
    }
    checks["responsiveness_worsens"] = checks["delay_worsens"] or checks["miss_rate_worsens"]
    checks["overall"] = (
        checks["global_ece_improves"]
        and checks["tfi_improves"]
        and checks["accuracy_noninferior"]
        and checks["responsiveness_worsens"]
    )
    return checks


def fmt(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    point_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    loo_rows: list[dict[str, object]],
    equivalence: dict[str, object],
    metadata: dict[str, object],
) -> None:
    primary = gate_checks(comparison_rows, PRIMARY_ALPHA)
    sensitivity = gate_checks(comparison_rows, SENSITIVITY_ALPHA)
    if primary["overall"]:
        decision = "PASS_PRIMARY"
        carrier: float | None = PRIMARY_ALPHA
    elif sensitivity["overall"]:
        decision = "PASS_PREDECLARED_SENSITIVITY"
        carrier = SENSITIVITY_ALPHA
    else:
        decision = "FAIL_NO_THIRD_SIMILAR_PREDICTOR"
        carrier = None
    lookup = {(float(row["ema_alpha"]), str(row["metric"])): row for row in comparison_rows}
    loo_lookup: dict[tuple[float, str], list[float]] = {}
    for row in loo_rows:
        loo_lookup.setdefault((float(row["ema_alpha"]), str(row["metric"])), []).append(float(row["delta_after_omission"]))
    lines = [
        "# G4 second-predictor replication gate",
        "",
        f"- Decision: **{decision}**",
        f"- Gate-carrying alpha: **{carrier if carrier is not None else 'none'}**",
        f"- Predictor: existing THUMOS14 multinomial linear probe",
        f"- Videos / frames / transitions: {metadata['videos']} / {metadata['frames']} / {metadata['transitions']}",
        f"- Selected original-interface equivalence max absolute difference: {equivalence['selected_max_absolute_difference']:.3e} (**{'PASS' if equivalence['selected_pass'] else 'FAIL'}**)",
        f"- Rejected batch candidate max absolute difference: {equivalence['rejected_batch_candidate']['max_absolute_difference']:.3e}",
        f"- Bootstrap: {N_BOOTSTRAP} paired video-cluster resamples, seed {G4_SEED}",
        "",
        "## Point estimates",
        "",
        "| Alpha | Accuracy | Global ECE | TFI | Delay | Miss rate | PTSM (descriptive) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(point_rows, key=lambda item: float(item["ema_alpha"])):
        lines.append(
            f"| {fmt(row['ema_alpha'], 2)} | {fmt(row['accuracy'])} | {fmt(row['global_ece'])} | "
            f"{fmt(row['mean_tfi'])} | {fmt(row['mean_transition_delay'])} | "
            f"{fmt(row['missed_transition_rate'])} | {fmt(row['ptsm'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences (smoothed minus raw)",
            "",
            "| Alpha | Metric | Delta | 95% CI |",
            "|---:|---|---:|---:|",
        ]
    )
    for alpha in (PRIMARY_ALPHA, SENSITIVITY_ALPHA):
        for metric in ("accuracy", "global_ece", "mean_tfi", "mean_transition_delay", "missed_transition_rate", "ptsm"):
            row = lookup[(alpha, metric)]
            lines.append(
                f"| {alpha:.2f} | {metric} | {fmt(row['delta_point'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |"
            )
    lines.extend(["", "## Gate checks", ""])
    for alpha, checks in ((PRIMARY_ALPHA, primary), (SENSITIVITY_ALPHA, sensitivity)):
        lines.append(f"### Alpha {alpha:.2f}")
        lines.append("")
        for name, passed in checks.items():
            lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
        lines.append("")
    lines.extend(["## Leave-one-video-out direction ranges", ""])
    for alpha in (PRIMARY_ALPHA, SENSITIVITY_ALPHA):
        ece_values = np.asarray(loo_lookup[(alpha, "global_ece")])
        delay_values = np.asarray(loo_lookup[(alpha, "mean_transition_delay")])
        lines.append(
            f"- alpha {alpha:.2f}: ECE delta [{ece_values.min():.6f}, {ece_values.max():.6f}]; "
            f"delay delta [{delay_values.min():.6f}, {delay_values.max():.6f}]."
        )
    lines.extend(
        [
            "",
            "PTSM is reported descriptively and does not carry this gate. Passing supports replication across two fitted predictors on THUMOS14, not cross-dataset generality.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen G4 linear-probe replication.")
    parser.add_argument("--project-src", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.project_src, args.checkpoint, args.feature_dir, args.split_file, args.annotation_dir, args.protocol):
        if not required.exists():
            raise FileNotFoundError(required)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha256_file(args.checkpoint) != str(protocol["predictor"]["checkpoint_sha256"]):
        raise ValueError("Linear-probe checkpoint hash does not match frozen protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_names = (
        "inference_equivalence.json",
        "frozen_predictions.npz",
        "per_video_statistics.csv",
        "point_estimates.csv",
        "bootstrap_deltas.csv",
        "bootstrap_samples.npz",
        "leave_one_video_out_deltas.csv",
        "statistics_schema.json",
        "metadata.json",
        "gate_G4.md",
    )
    if any((args.output_dir / name).exists() for name in output_names):
        raise FileExistsError("G4 outputs already exist; use a new directory to preserve immutability")
    sys.path.insert(0, str(args.project_src))
    from oad_stress_test.datasets.feature_dataset import FeatureDataset

    with args.checkpoint.open("rb") as handle:
        model = pickle.load(handle)
    dataset = FeatureDataset(
        args.feature_dir,
        args.split_file,
        annotation_dir=args.annotation_dir,
        background_label=0,
        end_idx_inclusive=True,
    )
    predictions, equivalence = collect_predictions(model, dataset, int(protocol["expected_videos"]))
    (args.output_dir / "inference_equivalence.json").write_text(json.dumps(equivalence, indent=2), encoding="utf-8")
    prediction_cache = args.output_dir / "frozen_predictions.npz"
    save_prediction_cache(prediction_cache, predictions)
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
    point = {alpha: aggregate_records(rows, ECE_BINS) for alpha, rows in by_alpha.items()}
    point_rows = [{"ema_alpha": alpha, **metrics} for alpha, metrics in sorted(point.items())]
    comparison_rows, bootstrap_samples = bootstrap_differences(by_alpha, ECE_BINS, N_BOOTSTRAP, G4_SEED)
    loo_rows = leave_one_video_out_differences(by_alpha, ECE_BINS)
    write_csv(args.output_dir / "per_video_statistics.csv", records)
    write_csv(args.output_dir / "point_estimates.csv", point_rows)
    write_csv(args.output_dir / "bootstrap_deltas.csv", comparison_rows)
    write_csv(args.output_dir / "leave_one_video_out_deltas.csv", loo_rows)
    np.savez_compressed(args.output_dir / "bootstrap_samples.npz", **bootstrap_samples)
    write_schema(args.output_dir / "statistics_schema.json", ECE_BINS)
    baseline = point[BASELINE_ALPHA]
    metadata: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "confirmatory_replication",
        "predictor": "linear_probe_multinomial_softmax",
        "videos": int(baseline["n_videos"]),
        "frames": int(baseline["n_frames"]),
        "transitions": int(baseline["n_transitions"]),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "script_sha256": sha256_file(Path(__file__)),
        "prediction_cache": str(prediction_cache),
        "prediction_cache_sha256": sha256_file(prediction_cache),
        "bootstrap_repetitions": N_BOOTSTRAP,
        "bootstrap_seed": G4_SEED,
        "inference_equivalence": equivalence,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.output_dir / "gate_G4.md", point_rows, comparison_rows, loo_rows, equivalence, metadata)
    print(args.output_dir / "gate_G4.md")


if __name__ == "__main__":
    main()
