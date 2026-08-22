from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _make_piecewise_labels(rng: np.random.Generator, length: int, num_classes: int) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int64)
    t = 0
    current = 0
    while t < length:
        seg_len = int(rng.integers(12, 35))
        if rng.random() < 0.65:
            current = int(rng.integers(1, num_classes))
        else:
            current = 0
        labels[t:min(length, t + seg_len)] = current
        t += seg_len
    return labels


def generate_dummy_dataset(
    root: str | Path,
    num_train: int = 8,
    num_test: int = 4,
    length: int = 140,
    dim: int = 16,
    num_classes: int = 4,
    seed: int = 7,
) -> Tuple[Path, Path]:
    """Creates a tiny synthetic OAD-like feature dataset.

    Class means are separated enough that a prototype classifier produces
    meaningful, non-random curves, but still noisy around transitions.
    """
    root = Path(root)
    feature_dir = root / "features"
    split_dir = root / "splits"
    feature_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    class_means = rng.normal(0, 2.0, size=(num_classes, dim)).astype(np.float32)
    class_means[0] = 0.0

    def write_video(video_id: str) -> None:
        labels = _make_piecewise_labels(rng, length=length, num_classes=num_classes)
        features = class_means[labels] + rng.normal(0, 0.9, size=(length, dim)).astype(np.float32)
        # Add transition noise to make delay metrics meaningful.
        transition_idx = np.where(labels[1:] != labels[:-1])[0] + 1
        for t0 in transition_idx:
            lo, hi = max(0, t0 - 2), min(length, t0 + 3)
            features[lo:hi] += rng.normal(0, 1.2, size=(hi - lo, dim)).astype(np.float32)
        np.savez_compressed(feature_dir / f"{video_id}.npz", features=features, labels=labels)

    train_ids = [f"train_{i:03d}" for i in range(num_train)]
    test_ids = [f"test_{i:03d}" for i in range(num_test)]
    for vid in train_ids + test_ids:
        write_video(vid)

    (split_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    return feature_dir, split_dir
