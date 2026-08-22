from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from oad_stress_test.config import load_config
from oad_stress_test.datasets.feature_dataset import FeatureDataset
from oad_stress_test.datasets.schema import VideoSequence


TRANSITION_TYPES = ["background_to_action", "action_to_background", "action_to_action"]


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if len(array) == 0:
        return None
    return float(np.percentile(array, float(percentile)))


def transition_indices(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(values) <= 1:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero(values[1:] != values[:-1]).astype(np.int64) + 1


def transition_type_counts(labels: np.ndarray, background_label: int = 0) -> dict[str, int]:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = {name: 0 for name in TRANSITION_TYPES}
    for idx in transition_indices(values):
        prev_label = int(values[idx - 1])
        next_label = int(values[idx])
        if prev_label == int(background_label) and next_label != int(background_label):
            counts["background_to_action"] += 1
        elif prev_label != int(background_label) and next_label == int(background_label):
            counts["action_to_background"] += 1
        elif prev_label != int(background_label) and next_label != int(background_label):
            counts["action_to_action"] += 1
    return counts


def distance_to_nearest_transition(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    transitions = transition_indices(values)
    if len(transitions) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    timeline = np.arange(len(values), dtype=np.int64)
    distances = np.abs(timeline[:, None] - transitions[None, :]).min(axis=1)
    return distances.astype(np.float32)


def _feature_step_seconds(video: VideoSequence, feature_fps: float | None) -> float | None:
    if video.timestamps is not None:
        timestamps = np.asarray(video.timestamps, dtype=np.float64).reshape(-1)
        diffs = np.diff(timestamps)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs) > 0:
            return float(np.median(diffs))
    if feature_fps is not None and float(feature_fps) > 0:
        return 1.0 / float(feature_fps)
    return None


def candidate_windows(distances: np.ndarray, seconds_per_step: float | None = None) -> dict[str, object]:
    valid = np.asarray(distances, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return {
            "near_step_percentiles": {},
            "far_step_percentiles": {},
            "near_second_percentiles": {},
            "far_second_percentiles": {},
            "note": "No transitions were found; near/far windows cannot be proposed.",
        }
    near_percentiles = {str(p): _percentile(valid, p) for p in [5, 10, 25]}
    far_percentiles = {str(p): _percentile(valid, p) for p in [50, 75, 90]}
    out: dict[str, object] = {
        "near_step_percentiles": near_percentiles,
        "far_step_percentiles": far_percentiles,
        "near_second_percentiles": {},
        "far_second_percentiles": {},
        "note": (
            "Candidates are data-scale audit values, not final protocol choices. "
            "Select near/far windows after checking TVSeries stride and label density."
        ),
    }
    if seconds_per_step is not None:
        out["near_second_percentiles"] = {
            key: None if value is None else float(value) * float(seconds_per_step)
            for key, value in near_percentiles.items()
        }
        out["far_second_percentiles"] = {
            key: None if value is None else float(value) * float(seconds_per_step)
            for key, value in far_percentiles.items()
        }
    return out


def audit_dataset(dataset: FeatureDataset, max_videos: int | None = None) -> dict[str, object]:
    if max_videos is not None and int(max_videos) <= 0:
        raise ValueError("max_videos must be positive")
    video_ids = list(dataset.video_ids if max_videos is None else dataset.video_ids[: int(max_videos)])
    background_label = int(dataset.background_label)
    lengths: list[int] = []
    feature_dims: list[int] = []
    class_counts: dict[int, int] = {}
    action_frames = 0
    background_frames = 0
    transition_counts: list[int] = []
    transition_types = {name: 0 for name in TRANSITION_TYPES}
    all_distances: list[float] = []
    seconds_per_step_values: list[float] = []
    per_video_rows = []

    for video_id in video_ids:
        video = dataset.load_video(video_id)
        labels = np.asarray(video.labels, dtype=np.int64)
        features = np.asarray(video.features, dtype=np.float32)
        lengths.append(int(len(labels)))
        feature_dims.append(int(features.shape[1]))
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique.tolist(), counts.tolist()):
            class_counts[int(label)] = class_counts.get(int(label), 0) + int(count)
        bg_count = int(np.sum(labels == background_label))
        background_frames += bg_count
        action_frames += int(len(labels) - bg_count)
        transitions = transition_indices(labels)
        type_counts = transition_type_counts(labels, background_label=background_label)
        for key, value in type_counts.items():
            transition_types[key] += int(value)
        distances = distance_to_nearest_transition(labels)
        all_distances.extend([float(value) for value in distances[np.isfinite(distances)]])
        step_seconds = _feature_step_seconds(video, dataset.feature_fps)
        if step_seconds is not None:
            seconds_per_step_values.append(float(step_seconds))
        transition_counts.append(int(len(transitions)))
        per_video_rows.append({
            "video_id": video_id,
            "length": int(len(labels)),
            "feature_dim": int(features.shape[1]),
            "transition_count": int(len(transitions)),
            **type_counts,
        })

    total_frames = int(sum(lengths))
    seconds_per_step = None if not seconds_per_step_values else float(np.median(seconds_per_step_values))
    class_ids = sorted(class_counts)
    label_distribution = {str(label): int(class_counts[label]) for label in class_ids}
    length_array = np.asarray(lengths, dtype=np.float64)
    transition_array = np.asarray(transition_counts, dtype=np.float64)
    distance_array = np.asarray(all_distances, dtype=np.float64)
    return {
        "num_videos": int(len(video_ids)),
        "video_ids_audited": video_ids,
        "num_classes_in_labels": int(len(class_ids)),
        "class_ids": class_ids,
        "feature_dim_values": sorted(set(feature_dims)),
        "feature_step_seconds_estimate": seconds_per_step,
        "length_steps": {
            "min": None if len(length_array) == 0 else int(np.min(length_array)),
            "median": None if len(length_array) == 0 else float(np.median(length_array)),
            "mean": None if len(length_array) == 0 else float(np.mean(length_array)),
            "max": None if len(length_array) == 0 else int(np.max(length_array)),
        },
        "total_frames": total_frames,
        "background_frames": int(background_frames),
        "action_frames": int(action_frames),
        "background_ratio": None if total_frames == 0 else float(background_frames / total_frames),
        "action_ratio": None if total_frames == 0 else float(action_frames / total_frames),
        "label_distribution": label_distribution,
        "transition_counts_per_video": {
            "min": None if len(transition_array) == 0 else int(np.min(transition_array)),
            "median": None if len(transition_array) == 0 else float(np.median(transition_array)),
            "mean": None if len(transition_array) == 0 else float(np.mean(transition_array)),
            "max": None if len(transition_array) == 0 else int(np.max(transition_array)),
        },
        "total_transitions": int(sum(transition_counts)),
        "transition_types": transition_types,
        "distance_to_transition_steps": {
            "p05": _percentile(distance_array, 5),
            "p10": _percentile(distance_array, 10),
            "p25": _percentile(distance_array, 25),
            "p50": _percentile(distance_array, 50),
            "p75": _percentile(distance_array, 75),
            "p90": _percentile(distance_array, 90),
            "p95": _percentile(distance_array, 95),
        },
        "candidate_windows": candidate_windows(distance_array, seconds_per_step),
        "per_video": per_video_rows,
    }


def required_tvseries_schema_note() -> str:
    return "\n".join([
        "# TVSeries Data Audit Requirements",
        "",
        "No TVSeries data/configuration was detected automatically. Do not run a formal v1.8 protocol until these files are present and audited.",
        "",
        "Expected minimal feature-level schema:",
        "",
        "```text",
        "data/tvseries/",
        "  features/",
        "    <video_or_episode_id>.npy or <video_or_episode_id>.npz",
        "  annotations/",
        "    tvseries.csv",
        "  splits/",
        "    train.txt",
        "    test.txt",
        "```",
        "",
        "Feature files must provide one feature array per video/episode with shape `[T, D]`. `.npz` files may also include `labels` and optional `timestamps`.",
        "",
        "If labels are not embedded in `.npz` files, annotations must contain:",
        "",
        "- `video_id`",
        "- `label`",
        "- either `start_idx,end_idx` in feature steps, or `start_time,end_time` plus a defensible `feature_fps` / timestamp mapping.",
        "",
        "The audit must establish the TVSeries feature stride and transition-distance distribution before choosing near/far windows.",
    ])


def format_audit_markdown(audit: dict[str, object]) -> str:
    seconds = audit.get("feature_step_seconds_estimate")
    seconds_text = "unknown" if seconds is None else f"{float(seconds):.4f}"
    windows = audit["candidate_windows"]
    return "\n".join([
        "# TVSeries Transition Timing Audit",
        "",
        f"- videos / episodes audited: {audit['num_videos']}",
        f"- classes observed in labels: {audit['num_classes_in_labels']}",
        f"- class ids: {audit['class_ids']}",
        f"- feature dimensions: {audit['feature_dim_values']}",
        f"- estimated seconds per feature step: {seconds_text}",
        f"- length in feature steps: {audit['length_steps']}",
        f"- total frames / feature steps: {audit['total_frames']}",
        f"- background frames: {audit['background_frames']}",
        f"- action frames: {audit['action_frames']}",
        f"- background ratio: {audit['background_ratio']}",
        f"- action ratio: {audit['action_ratio']}",
        f"- total transitions: {audit['total_transitions']}",
        f"- transitions per video: {audit['transition_counts_per_video']}",
        f"- transition types: {audit['transition_types']}",
        f"- distance-to-transition percentiles in steps: {audit['distance_to_transition_steps']}",
        "",
        "## Candidate Near/Far Windows",
        "",
        f"- near step percentiles: {windows['near_step_percentiles']}",
        f"- far step percentiles: {windows['far_step_percentiles']}",
        f"- near second percentiles: {windows['near_second_percentiles']}",
        f"- far second percentiles: {windows['far_second_percentiles']}",
        f"- note: {windows['note']}",
        "",
        "These are audit candidates only. v1.8 should choose TVSeries near/far windows after checking feature stride, label density, and transition distribution.",
    ])


def _load_dataset_from_config(config_path: str | Path, split_name: str) -> FeatureDataset:
    cfg = load_config(config_path)
    d = cfg.dataset
    split_dir = Path(d.get("split_dir", ""))
    split_key = "train_split_file" if split_name == "train" else "split_file"
    fallback = d.get("train_split", "train.txt") if split_name == "train" else d.get("test_split", "test.txt")
    split_file = Path(d.get(split_key) or (split_dir / fallback))
    return FeatureDataset(
        d["feature_dir"],
        split_file,
        annotation_dir=d.get("annotation_dir"),
        annotation_file=d.get("annotation_file"),
        background_label=int(d.get("background_label", 0)),
        feature_fps=d.get("feature_fps"),
        label_map=d.get("label_map"),
        end_idx_inclusive=bool(d.get("end_idx_inclusive", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TVSeries feature-level transition timing before choosing near/far bins.")
    parser.add_argument("--config", default="configs/tvseries.yaml")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output", default=None, help="optional markdown output path")
    parser.add_argument("--json-output", default=None, help="optional JSON audit output path")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        note = required_tvseries_schema_note()
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(note, encoding="utf-8")
        print(note)
        raise SystemExit(2)

    dataset = _load_dataset_from_config(config_path, args.split)
    audit = audit_dataset(dataset, max_videos=args.max_videos)
    markdown = format_audit_markdown(audit)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
