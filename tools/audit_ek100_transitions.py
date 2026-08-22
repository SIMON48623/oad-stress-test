from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


ARRAY_SUFFIXES = {".npy", ".npz"}
DEFAULT_CANDIDATE_BINS = [(8, 32), (4, 16)]


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if len(array) == 0:
        return None
    return float(np.percentile(array, float(percentile)))


def _shape_key(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _bump(mapping: dict[str, int], key: str, amount: int = 1) -> None:
    mapping[str(key)] = int(mapping.get(str(key), 0) + int(amount))


def _read_split_file(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    split_path = Path(path)
    ids: list[str] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        ids.append(Path(value.split()[0]).stem)
    return ids


def _array_files(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Array directory not found: {root}")
    files: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ARRAY_SUFFIXES):
        key = path.stem
        if key in files:
            duplicates.append(key)
        files[key] = path
    if duplicates:
        raise ValueError(f"Duplicate array stems in {root}: {sorted(set(duplicates))[:10]}")
    return files


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, mmap_mode="r"))
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        for key in ["target", "targets", "labels", "features", "arr_0"]:
            if key in data.files:
                return np.asarray(data[key])
        if not data.files:
            raise ValueError(f"{path} contains no arrays")
        return np.asarray(data[data.files[0]])
    raise ValueError(f"Unsupported array suffix: {path.suffix}")


def active_mask_from_target(target: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    array = np.asarray(target)
    if array.ndim != 2:
        raise ValueError(f"target_perframe array must have shape [L, C], got {array.shape}")
    return np.asarray(array, dtype=np.float32) > float(threshold)


def transition_indices_from_active(active: np.ndarray) -> np.ndarray:
    mask = np.asarray(active, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"active mask must have shape [L, C], got {mask.shape}")
    if mask.shape[0] <= 1:
        return np.asarray([], dtype=np.int64)
    changed = np.any(mask[1:] != mask[:-1], axis=1)
    return np.flatnonzero(changed).astype(np.int64) + 1


def distance_to_nearest_transition(active: np.ndarray) -> np.ndarray:
    transitions = transition_indices_from_active(active)
    length = int(np.asarray(active).shape[0])
    if len(transitions) == 0:
        return np.full(length, np.nan, dtype=np.float32)
    timeline = np.arange(length, dtype=np.int64)
    distances = np.abs(timeline[:, None] - transitions[None, :]).min(axis=1)
    return distances.astype(np.float32)


def transition_event_counts(active: np.ndarray) -> dict[str, int]:
    mask = np.asarray(active, dtype=bool)
    counts = {
        "transition_events": 0,
        "action_entry_events": 0,
        "action_exit_events": 0,
        "action_to_action_events": 0,
        "label_entry_count": 0,
        "label_exit_count": 0,
    }
    for idx in transition_indices_from_active(mask):
        prev = mask[idx - 1]
        curr = mask[idx]
        prev_count = int(prev.sum())
        curr_count = int(curr.sum())
        added = np.logical_and(curr, np.logical_not(prev))
        removed = np.logical_and(prev, np.logical_not(curr))
        counts["transition_events"] += 1
        counts["label_entry_count"] += int(added.sum())
        counts["label_exit_count"] += int(removed.sum())
        if prev_count == 0 and curr_count > 0:
            counts["action_entry_events"] += 1
        elif prev_count > 0 and curr_count == 0:
            counts["action_exit_events"] += 1
        elif prev_count > 0 and curr_count > 0:
            counts["action_to_action_events"] += 1
    return counts


def active_label_count_summary(active_counts: np.ndarray) -> dict[str, object]:
    values = np.asarray(active_counts, dtype=np.float64)
    total = int(len(values))
    zero_count = int(np.sum(values == 0))
    multi_count = int(np.sum(values > 1))
    return {
        "min": None if total == 0 else int(np.min(values)),
        "p50": _percentile(values, 50),
        "mean": None if total == 0 else float(np.mean(values)),
        "p90": _percentile(values, 90),
        "max": None if total == 0 else int(np.max(values)),
        "zero_active_frames": zero_count,
        "multi_label_frames": multi_count,
        "zero_active_ratio": None if total == 0 else float(zero_count / total),
        "multi_label_ratio": None if total == 0 else float(multi_count / total),
    }


def candidate_bin_stats(
    distances: np.ndarray,
    *,
    feature_stride_seconds: float | None = None,
    min_far_frames: int = 1000,
    min_far_ratio: float = 0.01,
) -> dict[str, dict[str, object]]:
    valid = np.asarray(distances, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    total = int(len(valid))
    out: dict[str, dict[str, object]] = {}
    for near_window, far_window in DEFAULT_CANDIDATE_BINS:
        near_count = int(np.sum(valid <= float(near_window))) if total else 0
        far_count = int(np.sum(valid > float(far_window))) if total else 0
        key = f"near_le_{near_window}_far_gt_{far_window}"
        entry: dict[str, object] = {
            "near_window_steps": int(near_window),
            "far_window_steps": int(far_window),
            "valid_distance_frames": total,
            "near_frames": near_count,
            "far_frames": far_count,
            "near_ratio": None if total == 0 else float(near_count / total),
            "far_ratio": None if total == 0 else float(far_count / total),
            "far_sufficient": bool(far_count >= int(min_far_frames) and (total > 0 and far_count / total >= float(min_far_ratio))),
            "min_far_frames": int(min_far_frames),
            "min_far_ratio": float(min_far_ratio),
        }
        if feature_stride_seconds is not None:
            entry["near_window_seconds"] = float(near_window) * float(feature_stride_seconds)
            entry["far_window_seconds"] = float(far_window) * float(feature_stride_seconds)
        out[key] = entry
    return out


def recommendation_from_bins(bins: dict[str, dict[str, object]]) -> str:
    thumos = bins.get("near_le_8_far_gt_32", {})
    compact = bins.get("near_le_4_far_gt_16", {})
    if bool(thumos.get("far_sufficient", False)):
        return "keep_thumos_style_near_le_8_far_gt_32_for_comparability"
    if bool(compact.get("far_sufficient", False)):
        return "use_ek100_specific_near_le_4_far_gt_16_with_caveat"
    return "prefer_continuous_proximity_diagnostic_over_binned_atm_claim"


def _inspect_optional_features(video_ids: list[str], feature_dir: str | Path | None) -> dict[str, object] | None:
    if feature_dir is None:
        return None
    files = _array_files(feature_dir)
    shape_distribution: dict[str, int] = {}
    missing: list[str] = []
    for video_id in video_ids:
        path = files.get(video_id)
        if path is None:
            missing.append(video_id)
            continue
        shape = tuple(int(value) for value in load_array(path).shape)
        _bump(shape_distribution, _shape_key(shape))
    return {
        "directory": str(feature_dir),
        "matched_files": int(len(video_ids) - len(missing)),
        "missing_files": int(len(missing)),
        "missing_preview": missing[:10],
        "shape_distribution": shape_distribution,
    }


def audit_ek100_paths(
    *,
    target_perframe_dir: str | Path,
    split_file: str | Path | None = None,
    rgb_feature_dir: str | Path | None = None,
    flow_feature_dir: str | Path | None = None,
    feature_stride_seconds: float | None = None,
    target_threshold: float = 0.5,
    max_videos: int | None = None,
    min_far_frames: int = 1000,
    min_far_ratio: float = 0.01,
) -> dict[str, object]:
    if max_videos is not None and int(max_videos) <= 0:
        raise ValueError("max_videos must be positive")
    target_files = _array_files(target_perframe_dir)
    split_ids = _read_split_file(split_file)
    requested_ids = sorted(target_files) if split_ids is None else split_ids
    if max_videos is not None:
        requested_ids = requested_ids[: int(max_videos)]
    missing_target_ids = [video_id for video_id in requested_ids if video_id not in target_files]
    video_ids = [video_id for video_id in requested_ids if video_id in target_files]
    if not video_ids:
        raise ValueError("No target_perframe files matched the requested split/list")

    target_shapes: dict[str, int] = {}
    lengths: list[int] = []
    class_counts = np.zeros(0, dtype=np.int64)
    active_counts_all: list[int] = []
    transition_counts: list[int] = []
    distance_values: list[float] = []
    event_totals = {
        "transition_events": 0,
        "action_entry_events": 0,
        "action_exit_events": 0,
        "action_to_action_events": 0,
        "label_entry_count": 0,
        "label_exit_count": 0,
    }
    per_video = []

    for video_id in video_ids:
        target = load_array(target_files[video_id])
        if target.ndim != 2:
            raise ValueError(f"{target_files[video_id]} target must have shape [L, C], got {target.shape}")
        _bump(target_shapes, _shape_key(tuple(int(value) for value in target.shape)))
        active = active_mask_from_target(target, threshold=target_threshold)
        active_counts = active.sum(axis=1).astype(np.int64)
        if class_counts.shape[0] < active.shape[1]:
            padded = np.zeros(active.shape[1], dtype=np.int64)
            padded[: class_counts.shape[0]] = class_counts
            class_counts = padded
        class_counts[: active.shape[1]] += active.sum(axis=0).astype(np.int64)
        distances = distance_to_nearest_transition(active)
        events = transition_event_counts(active)
        for key, value in events.items():
            event_totals[key] += int(value)
        lengths.append(int(active.shape[0]))
        active_counts_all.extend([int(value) for value in active_counts.tolist()])
        transition_counts.append(int(events["transition_events"]))
        distance_values.extend([float(value) for value in distances[np.isfinite(distances)]])
        per_video.append({
            "video_id": video_id,
            "length": int(active.shape[0]),
            "num_classes": int(active.shape[1]),
            "zero_active_frames": int(np.sum(active_counts == 0)),
            "multi_label_frames": int(np.sum(active_counts > 1)),
            **events,
        })

    distances_array = np.asarray(distance_values, dtype=np.float64)
    active_counts_array = np.asarray(active_counts_all, dtype=np.int64)
    length_array = np.asarray(lengths, dtype=np.float64)
    transition_array = np.asarray(transition_counts, dtype=np.float64)
    bins = candidate_bin_stats(
        distances_array,
        feature_stride_seconds=feature_stride_seconds,
        min_far_frames=min_far_frames,
        min_far_ratio=min_far_ratio,
    )
    observed_class_ids = np.flatnonzero(class_counts > 0).astype(int).tolist()
    total_frames = int(sum(lengths))
    active_summary = active_label_count_summary(active_counts_array)
    return {
        "dataset": "ek100",
        "target_perframe_dir": str(target_perframe_dir),
        "split_file": None if split_file is None else str(split_file),
        "num_videos": int(len(video_ids)),
        "video_ids_audited": video_ids,
        "missing_target_ids": missing_target_ids,
        "target_shape_distribution": target_shapes,
        "target_num_classes_values": sorted({int(row["num_classes"]) for row in per_video}),
        "observed_action_class_count": int(len(observed_class_ids)),
        "observed_action_class_ids_preview": observed_class_ids[:20],
        "background_no_action_convention": (
            "all_zero_target_row_inferred_as_no_action"
            if int(active_summary["zero_active_frames"]) > 0
            else "no_zero_target_rows_observed_no_background_convention_inferred"
        ),
        "feature_stride_seconds": None if feature_stride_seconds is None else float(feature_stride_seconds),
        "length_steps": {
            "min": int(np.min(length_array)),
            "median": float(np.median(length_array)),
            "mean": float(np.mean(length_array)),
            "max": int(np.max(length_array)),
        },
        "total_frames": total_frames,
        "active_label_count_distribution": active_summary,
        "no_action_frames": int(active_summary["zero_active_frames"]),
        "no_action_ratio": None if total_frames == 0 else float(int(active_summary["zero_active_frames"]) / total_frames),
        "multi_label_frames": int(active_summary["multi_label_frames"]),
        "multi_label_ratio": None if total_frames == 0 else float(int(active_summary["multi_label_frames"]) / total_frames),
        "transition_counts_per_video": {
            "min": int(np.min(transition_array)),
            "median": float(np.median(transition_array)),
            "mean": float(np.mean(transition_array)),
            "max": int(np.max(transition_array)),
        },
        "transition_event_totals": event_totals,
        "distance_to_transition_steps": {
            "p05": _percentile(distances_array, 5),
            "p10": _percentile(distances_array, 10),
            "p25": _percentile(distances_array, 25),
            "p50": _percentile(distances_array, 50),
            "p75": _percentile(distances_array, 75),
            "p90": _percentile(distances_array, 90),
            "p95": _percentile(distances_array, 95),
        },
        "candidate_bins": bins,
        "far_frame_recommendation": recommendation_from_bins(bins),
        "rgb_feature_audit": _inspect_optional_features(video_ids, rgb_feature_dir),
        "flow_feature_audit": _inspect_optional_features(video_ids, flow_feature_dir),
        "per_video": per_video,
    }


def required_ek100_schema_note() -> str:
    return "\n".join([
        "# EK100 Transition Audit Requirements",
        "",
        "No EK100 target_perframe directory was provided. Do not run a formal EK100 protocol until TeSTra-style labels and features are available locally and audited.",
        "",
        "Expected TeSTra-style layout:",
        "",
        "```text",
        "data/ek100/",
        "  target_perframe/",
        "    <video_or_session_id>.npy or .npz  # shape L x 3807",
        "  rgb_kinetics_bninception/            # optional audit of RGB feature shape",
        "  flow_kinetics_bninception/           # optional audit of Flow feature shape",
        "  noun_perframe/                       # optional",
        "  verb_perframe/                       # optional",
        "  splits/",
        "    train.txt",
        "    test.txt",
        "```",
        "",
        "The audit treats all-zero target rows as no-action/background when such rows exist. It does not download or create EK100 data.",
    ])


def format_audit_markdown(audit: dict[str, object]) -> str:
    stride = audit.get("feature_stride_seconds")
    stride_text = "unknown" if stride is None else f"{float(stride):.4f}s"
    lines = [
        "# EK100 Transition Density Audit",
        "",
        f"- videos / sessions audited: {audit['num_videos']}",
        f"- target shape distribution: {audit['target_shape_distribution']}",
        f"- target class dimension values: {audit['target_num_classes_values']}",
        f"- observed action classes: {audit['observed_action_class_count']}",
        f"- observed class id preview: {audit['observed_action_class_ids_preview']}",
        f"- background / no-action convention: {audit['background_no_action_convention']}",
        f"- feature stride seconds: {stride_text}",
        f"- length in feature steps: {audit['length_steps']}",
        f"- total frames / feature steps: {audit['total_frames']}",
        f"- active label count distribution: {audit['active_label_count_distribution']}",
        f"- no-action ratio: {audit['no_action_ratio']}",
        f"- multi-label ratio: {audit['multi_label_ratio']}",
        f"- transition counts per video: {audit['transition_counts_per_video']}",
        f"- transition event totals: {audit['transition_event_totals']}",
        f"- distance-to-transition percentiles in steps: {audit['distance_to_transition_steps']}",
        f"- far-frame recommendation: {audit['far_frame_recommendation']}",
        "",
        "## Candidate Near/Far Bins",
        "",
    ]
    for name, stats in audit["candidate_bins"].items():
        lines.extend([
            f"### {name}",
            "",
            f"- near frames: {stats['near_frames']} ({stats['near_ratio']})",
            f"- far frames: {stats['far_frames']} ({stats['far_ratio']})",
            f"- far sufficient: {stats['far_sufficient']}",
            f"- minimum far frames criterion: {stats['min_far_frames']}",
            f"- minimum far ratio criterion: {stats['min_far_ratio']}",
        ])
        if "near_window_seconds" in stats:
            lines.extend([
                f"- near window seconds: {stats['near_window_seconds']}",
                f"- far window seconds: {stats['far_window_seconds']}",
            ])
        lines.append("")
    for key in ["rgb_feature_audit", "flow_feature_audit"]:
        value = audit.get(key)
        if value is not None:
            lines.extend([
                f"## {key}",
                "",
                f"- directory: {value['directory']}",
                f"- matched files: {value['matched_files']}",
                f"- missing files: {value['missing_files']}",
                f"- shape distribution: {value['shape_distribution']}",
                "",
            ])
    lines.append("This audit is pre-training only. It decides whether binned near/far ATM is defensible on EK100 or whether a continuous proximity diagnostic is needed.")
    return "\n".join(lines)


def _config_value(config: dict[str, object], *keys: str):
    dataset = config.get("dataset", {}) if isinstance(config.get("dataset", {}), dict) else {}
    for key in keys:
        if key in config:
            return config[key]
        if key in dataset:
            return dataset[key]
    return None


def _load_config(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TeSTra-style EK100 target_perframe transition density before training.")
    parser.add_argument("--config", default=None, help="optional YAML config with EK100 paths")
    parser.add_argument("--target-perframe-dir", default=None)
    parser.add_argument("--rgb-feature-dir", default=None)
    parser.add_argument("--flow-feature-dir", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--feature-stride-seconds", type=float, default=None)
    parser.add_argument("--feature-fps", type=float, default=None)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--min-far-frames", type=int, default=1000)
    parser.add_argument("--min-far-ratio", type=float, default=0.01)
    parser.add_argument("--output", default=None, help="optional markdown output path")
    parser.add_argument("--json-output", default=None, help="optional JSON audit output path")
    args = parser.parse_args()

    config = _load_config(args.config)
    stride = args.feature_stride_seconds
    if stride is None and args.feature_fps is not None and args.feature_fps > 0:
        stride = 1.0 / float(args.feature_fps)
    if stride is None:
        config_stride = _config_value(config, "feature_stride_seconds", "feature_step_seconds")
        config_fps = _config_value(config, "feature_fps")
        if config_stride is not None:
            stride = float(config_stride)
        elif config_fps is not None and float(config_fps) > 0:
            stride = 1.0 / float(config_fps)

    target_dir = args.target_perframe_dir or _config_value(config, "target_perframe_dir", "target_dir")
    if target_dir is None:
        note = required_ek100_schema_note()
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(note, encoding="utf-8")
        print(note)
        raise SystemExit(2)

    audit = audit_ek100_paths(
        target_perframe_dir=target_dir,
        split_file=args.split_file or _config_value(config, "split_file"),
        rgb_feature_dir=args.rgb_feature_dir or _config_value(config, "rgb_feature_dir"),
        flow_feature_dir=args.flow_feature_dir or _config_value(config, "flow_feature_dir"),
        feature_stride_seconds=stride,
        target_threshold=float(args.target_threshold),
        max_videos=args.max_videos,
        min_far_frames=int(args.min_far_frames),
        min_far_ratio=float(args.min_far_ratio),
    )
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
