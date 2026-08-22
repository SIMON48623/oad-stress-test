from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np
import pandas as pd


DECISIONS = {"predict", "wait", "abstain"}
REQUIRED_PER_FRAME_FIELDS = {
    "video_id",
    "frame_idx",
    "gt_label",
    "pred_label",
    "decision",
    "max_prob",
    "entropy",
    "top2_margin",
    "budget",
    "model",
    "policy",
    "threshold",
    "shift_type",
    "split",
}


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def safe_int(x: Any, default: int = -1) -> int:
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return safe_float(value)
    if isinstance(value, float):
        return safe_float(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def open_jsonl_writer(path: str | Path) -> TextIO:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def write_jsonl_row(file_handle: TextIO, row: dict[str, Any]) -> None:
    file_handle.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def uncertainty_from_probabilities(scores: Iterable[Any] | None, eps: float = 1e-12) -> dict[str, float]:
    if scores is None:
        return {"max_prob": 0.0, "entropy": 0.0, "top2_margin": 0.0}
    values = np.asarray(list(scores), dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"max_prob": 0.0, "entropy": 0.0, "top2_margin": 0.0}
    clipped = np.clip(values, eps, 1.0)
    max_prob = float(np.max(clipped))
    entropy = float(-np.sum(clipped * np.log(clipped + eps)))
    if len(clipped) < 2:
        top2_margin = max_prob
    else:
        top2 = np.partition(clipped, -2)[-2:]
        top2_margin = float(np.max(top2) - np.min(top2))
    return {
        "max_prob": safe_float(max_prob),
        "entropy": safe_float(entropy),
        "top2_margin": safe_float(top2_margin),
    }


def _score_columns(logs: pd.DataFrame) -> list[str]:
    columns = [column for column in logs.columns if str(column).startswith("score_class_")]
    return sorted(columns, key=lambda name: int(str(name).rsplit("_", 1)[-1]))


def iter_per_frame_rows(
    logs: pd.DataFrame,
    *,
    model: str,
    policy: str,
    threshold: float | None,
    shift_type: str = "clean",
    split: str = "test",
    run_id: str | None = None,
    checkpoint_path: str | None = None,
    frame_global_start: int = 0,
    uncertainty_mode: str | None = None,
) -> Iterable[dict[str, Any]]:
    score_columns = _score_columns(logs)
    num_classes = len(score_columns)
    for local_idx, (_, row) in enumerate(logs.iterrows()):
        scores = [row[column] for column in score_columns if pd.notna(row[column])]
        uncertainty = uncertainty_from_probabilities(scores)
        pred_label = safe_int(row.get("prediction"), default=-1)
        gt_label = safe_int(row.get("gt_label"), default=-1)
        decision = str(row.get("action_type", "wait"))
        if decision not in DECISIONS:
            raise ValueError(f"Unknown decision type: {decision}")
        output = {
            "video_id": str(row.get("video_id", "")),
            "frame_idx": safe_int(row.get("t", row.get("frame_idx")), default=0),
            "gt_label": gt_label,
            "pred_label": pred_label,
            "decision": decision,
            "max_prob": uncertainty["max_prob"],
            "entropy": uncertainty["entropy"],
            "top2_margin": uncertainty["top2_margin"],
            "budget": safe_float(row.get("budget"), default=0.0),
            "model": str(model),
            "policy": str(policy),
            "threshold": None if threshold is None else safe_float(threshold),
            "shift_type": str(shift_type),
            "split": str(split),
            "frame_global_idx": int(frame_global_start + local_idx),
            "is_correct": bool(pred_label >= 0 and pred_label == gt_label),
            "logit_dim": int(num_classes),
            "num_classes": int(num_classes),
            "run_id": None if run_id is None else str(run_id),
            "checkpoint_path": checkpoint_path,
        }
        if uncertainty_mode is not None:
            output["uncertainty_mode"] = str(uncertainty_mode)
        yield output


class PerFrameLogStats:
    def __init__(self) -> None:
        self.total_frames = 0
        self.video_ids: set[str] = set()
        self.decision_counts = {decision: 0 for decision in sorted(DECISIONS)}
        self.sum_max_prob = 0.0
        self.sum_entropy = 0.0
        self.sum_top2_margin = 0.0

    def update(self, row: dict[str, Any]) -> None:
        self.total_frames += 1
        self.video_ids.add(str(row["video_id"]))
        decision = str(row["decision"])
        if decision not in self.decision_counts:
            self.decision_counts[decision] = 0
        self.decision_counts[decision] += 1
        self.sum_max_prob += safe_float(row["max_prob"])
        self.sum_entropy += safe_float(row["entropy"])
        self.sum_top2_margin += safe_float(row["top2_margin"])

    def as_dict(self) -> dict[str, Any]:
        denom = max(1, self.total_frames)
        return {
            "total_frames": int(self.total_frames),
            "num_videos": int(len(self.video_ids)),
            "predict_frames": int(self.decision_counts.get("predict", 0)),
            "wait_frames": int(self.decision_counts.get("wait", 0)),
            "abstain_frames": int(self.decision_counts.get("abstain", 0)),
            "mean_max_prob": self.sum_max_prob / denom,
            "mean_entropy": self.sum_entropy / denom,
            "mean_top2_margin": self.sum_top2_margin / denom,
        }
