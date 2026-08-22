from __future__ import annotations

import json
import re
from typing import Dict

import numpy as np
import pandas as pd


SCORE_COLUMN_RE = re.compile(r"^score_class_(\d+)$")


def score_class_columns(logs: pd.DataFrame) -> dict[int, str]:
    columns: dict[int, str] = {}
    for column in logs.columns:
        match = SCORE_COLUMN_RE.match(str(column))
        if match:
            columns[int(match.group(1))] = str(column)
    return dict(sorted(columns.items()))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked_true = y_true[order]
    true_positives = np.cumsum(ranked_true)
    ranks = np.arange(1, len(ranked_true) + 1, dtype=np.float64)
    precision_at_hits = true_positives[ranked_true] / ranks[ranked_true]
    return float(np.sum(precision_at_hits) / positives)


def per_class_ap(logs: pd.DataFrame, background_label: int = 0) -> Dict[int, float]:
    if len(logs) == 0:
        return {}
    columns = score_class_columns(logs)
    labels = logs["gt_label"].to_numpy()
    out: Dict[int, float] = {}
    for class_id, column in columns.items():
        if class_id == int(background_label):
            continue
        scores = logs[column].fillna(0.0).to_numpy(dtype=np.float64)
        out[class_id] = average_precision(labels == class_id, scores)
    return out


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def frame_map(logs: pd.DataFrame, background_label: int = 0) -> float:
    values = np.asarray(list(per_class_ap(logs, background_label=background_label).values()), dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _train_positive_counts_array(train_positive_counts: object, num_classes: int) -> np.ndarray | None:
    if train_positive_counts is None:
        return None
    if isinstance(train_positive_counts, dict):
        counts = np.zeros(num_classes, dtype=np.int64)
        for key, value in train_positive_counts.items():
            class_id = int(key)
            if 0 <= class_id < num_classes:
                counts[class_id] = int(value)
        return counts
    arr = np.asarray(train_positive_counts, dtype=np.int64).reshape(-1)
    if len(arr) < num_classes:
        padded = np.zeros(num_classes, dtype=np.int64)
        padded[:len(arr)] = arr
        return padded
    return arr[:num_classes]


def summarize_frame_map(
    logs: pd.DataFrame,
    background_label: int = 0,
    train_positive_counts: object = None,
) -> Dict[str, float | str]:
    aps = per_class_ap(logs, background_label=background_label)
    class_ids = sorted(aps)
    num_classes = (max(class_ids) + 1) if class_ids else int(background_label) + 1
    train_counts = _train_positive_counts_array(train_positive_counts, num_classes)
    if train_counts is None:
        seen_action_classes = class_ids
        no_train_positive_classes: list[int] = []
    else:
        seen_action_classes = [
            class_id for class_id in class_ids
            if class_id != int(background_label) and train_counts[class_id] > 0
        ]
        no_train_positive_classes = [
            class_id for class_id in class_ids
            if class_id != int(background_label) and train_counts[class_id] == 0
        ]
    map_20 = _nanmean(list(aps.values()))
    map_seen = _nanmean([aps[class_id] for class_id in seen_action_classes])
    serializable = {
        str(class_id): (None if np.isnan(ap) else float(ap))
        for class_id, ap in aps.items()
    }
    return {
        "frame_mAP": map_20,
        "frame_mAP_20": map_20,
        "frame_mAP_seen_classes": map_seen,
        "num_seen_action_classes": int(len(seen_action_classes)),
        "no_train_positive_classes": json.dumps(no_train_positive_classes),
        "per_class_AP": json.dumps(serializable, sort_keys=True),
    }
