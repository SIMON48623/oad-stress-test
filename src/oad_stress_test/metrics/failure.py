from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .frame_map import score_class_columns


def overconfident_misfire_rate(logs: pd.DataFrame, threshold: float = 0.8) -> float:
    pred = logs[logs["action_type"].eq("predict") & logs["prediction"].ge(0)]
    high = pred[pred["confidence"] >= threshold]
    if len(high) == 0:
        return float("nan")
    wrong = high["prediction"].to_numpy() != high["gt_label"].to_numpy()
    return float(np.mean(wrong))


def abstention_failure_proxy(logs: pd.DataFrame, easy_conf_threshold: float = 0.8, hard_conf_threshold: float = 0.5) -> float:
    """Proxy for reversed selectivity.

    Higher values mean more abstentions at high confidence plus predictions at
    low confidence. This is only a first-pass diagnostic; paper version should
    use threshold sweeps.
    """
    if len(logs) == 0:
        return float("nan")
    easy_abstain = logs[logs["confidence"] >= easy_conf_threshold]["action_type"].eq("abstain").mean()
    hard_predict = logs[logs["confidence"] <= hard_conf_threshold]["action_type"].eq("predict").mean()
    vals = [v for v in [easy_abstain, hard_predict] if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_failure_modes(logs: pd.DataFrame) -> Dict[str, float]:
    calibration = summarize_calibration(logs)
    return {
        "overconfident_misfire_rate@0.8": overconfident_misfire_rate(logs, threshold=0.8),
        "accepted_risk_conf_gt_0.90": accepted_risk_at_confidence(logs, threshold=0.90),
        "accepted_risk_conf_gt_0.95": accepted_risk_at_confidence(logs, threshold=0.95),
        "abstention_failure_proxy": abstention_failure_proxy(logs),
        **calibration,
    }


def _score_matrix_and_labels(logs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    columns = score_class_columns(logs)
    if len(logs) == 0 or not columns:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.int64), []
    class_ids = sorted(columns)
    scores = logs[[columns[class_id] for class_id in class_ids]].fillna(0.0).to_numpy(dtype=np.float64)
    row_sums = scores.sum(axis=1, keepdims=True)
    if np.nanmin(scores) < 0.0 or np.any(row_sums <= 0.0):
        centered = scores - np.nanmax(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(centered, -50.0, 50.0))
        scores = exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
    else:
        scores = scores / np.maximum(row_sums, 1e-12)
    labels = logs["gt_label"].to_numpy(dtype=np.int64)
    return scores, labels, class_ids


def expected_calibration_error(logs: pd.DataFrame, n_bins: int = 10) -> float:
    scores, labels, class_ids = _score_matrix_and_labels(logs)
    if len(scores) == 0:
        return float("nan")
    id_to_col = {class_id: idx for idx, class_id in enumerate(class_ids)}
    pred_cols = np.argmax(scores, axis=1)
    pred_labels = np.asarray([class_ids[idx] for idx in pred_cols], dtype=np.int64)
    confidences = np.max(scores, axis=1)
    correct = pred_labels == labels
    ece = 0.0
    for bin_idx in range(int(n_bins)):
        lo = bin_idx / float(n_bins)
        hi = (bin_idx + 1) / float(n_bins)
        if bin_idx == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def negative_log_likelihood(logs: pd.DataFrame) -> float:
    scores, labels, class_ids = _score_matrix_and_labels(logs)
    if len(scores) == 0:
        return float("nan")
    id_to_col = {class_id: idx for idx, class_id in enumerate(class_ids)}
    probs = []
    for row_idx, label in enumerate(labels):
        col = id_to_col.get(int(label))
        probs.append(1e-12 if col is None else max(float(scores[row_idx, col]), 1e-12))
    return float(np.mean(-np.log(np.asarray(probs, dtype=np.float64))))


def brier_score(logs: pd.DataFrame) -> float:
    scores, labels, class_ids = _score_matrix_and_labels(logs)
    if len(scores) == 0:
        return float("nan")
    targets = np.zeros_like(scores)
    id_to_col = {class_id: idx for idx, class_id in enumerate(class_ids)}
    for row_idx, label in enumerate(labels):
        col = id_to_col.get(int(label))
        if col is not None:
            targets[row_idx, col] = 1.0
    return float(np.mean(np.sum((scores - targets) ** 2, axis=1)))


def accepted_risk_at_confidence(logs: pd.DataFrame, threshold: float) -> float:
    pred = logs[logs["action_type"].eq("predict") & logs["prediction"].ge(0)]
    high = pred[pred["confidence"] > float(threshold)]
    if len(high) == 0:
        return float("nan")
    wrong = high["prediction"].to_numpy() != high["gt_label"].to_numpy()
    return float(np.mean(wrong))


def summarize_calibration(logs: pd.DataFrame) -> Dict[str, float]:
    return {
        "ece_10": expected_calibration_error(logs, n_bins=10),
        "nll": negative_log_likelihood(logs),
        "brier": brier_score(logs),
    }
