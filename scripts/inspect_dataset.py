from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from oad_stress_test.config import load_config
from oad_stress_test.datasets.feature_dataset import FeatureDataset


def _dedupe_datasets(datasets: Iterable[tuple[str, FeatureDataset]]) -> list[tuple[str, FeatureDataset, str]]:
    seen = set()
    out = []
    for split_name, dataset in datasets:
        for video_id in dataset.video_ids:
            if video_id in seen:
                continue
            seen.add(video_id)
            out.append((split_name, dataset, video_id))
    return out


def _dataset_kwargs(cfg):
    d = cfg.dataset
    return {
        "annotation_dir": d.get("annotation_dir"),
        "annotation_file": d.get("annotation_file"),
        "background_label": int(d.get("background_label", 0)),
        "feature_fps": d.get("feature_fps"),
        "label_map": d.get("label_map"),
        "end_idx_inclusive": bool(d.get("end_idx_inclusive", True)),
    }


def _configured_datasets(cfg) -> list[tuple[str, FeatureDataset]]:
    d = cfg.dataset
    feature_dir = Path(d["feature_dir"])
    split_dir = Path(d.get("split_dir", ""))
    split_specs = [
        ("train", Path(d["train_split_file"]) if d.get("train_split_file") else split_dir / d.get("train_split", "train.txt")),
        ("test", Path(d["split_file"]) if d.get("split_file") else split_dir / d.get("test_split", "test.txt")),
    ]
    datasets = [
        (split_name, FeatureDataset(feature_dir, split_file, **_dataset_kwargs(cfg)))
        for split_name, split_file in split_specs
        if split_file.exists()
    ]
    if not datasets:
        expected = ", ".join(str(split_file) for _, split_file in split_specs)
        raise FileNotFoundError(f"No split files found. Expected one of: {expected}")
    return datasets


def inspect_config(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    datasets = _configured_datasets(cfg)
    entries = _dedupe_datasets(datasets)
    background_label = int(cfg.dataset.get("background_label", 0))

    lengths = []
    label_values = set()
    background_steps = 0
    action_steps = 0
    load_errors = []

    for split_name, dataset, video_id in entries:
        try:
            lengths.append(dataset.feature_length(video_id))
            video = dataset.load_video(video_id)
        except Exception as exc:  # noqa: BLE001 - this script reports dataset health.
            load_errors.append((split_name, video_id, str(exc)))
            continue
        labels = video.labels.astype(np.int64)
        label_values.update(int(v) for v in np.unique(labels))
        background_steps += int(np.sum(labels == background_label))
        action_steps += int(np.sum(labels != background_label))

    annotation_issues = []
    for _, dataset in datasets:
        annotation_issues.extend(dataset.annotation_bounds_issues())
    total_steps = background_steps + action_steps
    mean_length = float(np.mean(lengths)) if lengths else float("nan")
    action_ratio = action_steps / total_steps if total_steps else float("nan")
    background_ratio = background_steps / total_steps if total_steps else float("nan")

    print(f"Config: {Path(config_path)}")
    print(f"Videos: {len(entries)}")
    print(f"Mean video length: {mean_length:.2f} timesteps")
    print(f"Num classes: {len(label_values)}")
    print(f"Action/background steps: {action_steps}/{background_steps}")
    print(f"Action/background ratio: {action_ratio:.4f}/{background_ratio:.4f}")
    print(f"Annotation out-of-bounds issues: {len(annotation_issues)}")

    for issue in annotation_issues[:10]:
        print(
            "  "
            f"{issue['video_id']} row={issue['row_index']} "
            f"start={issue['start']} end={issue['end']} "
            f"length={issue['length']} issue={issue['issue']}"
        )
    if len(annotation_issues) > 10:
        print(f"  ... {len(annotation_issues) - 10} more")

    print(f"Load errors: {len(load_errors)}")
    for split_name, video_id, error in load_errors[:10]:
        print(f"  {split_name}/{video_id}: {error}")
    if len(load_errors) > 10:
        print(f"  ... {len(load_errors) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a feature-level OAD dataset.")
    parser.add_argument("--config", default="configs/thumos14.yaml")
    args = parser.parse_args()
    inspect_config(args.config)


if __name__ == "__main__":
    main()
