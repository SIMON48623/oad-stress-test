from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def frame_accuracy(logs: pd.DataFrame) -> float:
    pred_mask = logs["action_type"].eq("predict") & logs["prediction"].ge(0)
    if pred_mask.sum() == 0:
        return float("nan")
    return float((logs.loc[pred_mask, "prediction"].to_numpy() == logs.loc[pred_mask, "gt_label"].to_numpy()).mean())


def action_accuracy(logs: pd.DataFrame, background_label: int = 0) -> float:
    pred_mask = (
        logs["action_type"].eq("predict")
        & logs["prediction"].ge(0)
        & logs["gt_label"].ne(background_label)
    )
    if pred_mask.sum() == 0:
        return float("nan")
    return float((logs.loc[pred_mask, "prediction"].to_numpy() == logs.loc[pred_mask, "gt_label"].to_numpy()).mean())


def accepted_action_mask(logs: pd.DataFrame, background_label: int = 0) -> pd.Series:
    if len(logs) == 0:
        return pd.Series(dtype=bool)
    return (
        logs["action_type"].eq("predict")
        & logs["prediction"].ge(0)
        & logs["gt_label"].ne(background_label)
    )


def accepted_background_mask(logs: pd.DataFrame, background_label: int = 0) -> pd.Series:
    if len(logs) == 0:
        return pd.Series(dtype=bool)
    return (
        logs["action_type"].eq("predict")
        & logs["prediction"].ge(0)
        & logs["gt_label"].eq(background_label)
    )


def action_only_selective_risk(logs: pd.DataFrame, background_label: int = 0) -> float:
    mask = accepted_action_mask(logs, background_label=background_label)
    if mask.sum() == 0:
        return float("nan")
    wrong = logs.loc[mask, "prediction"].to_numpy() != logs.loc[mask, "gt_label"].to_numpy()
    return float(np.mean(wrong))


def high_conf_action_error_rate(logs: pd.DataFrame, threshold: float, background_label: int = 0) -> float:
    if len(logs) == 0:
        return float("nan")
    mask = accepted_action_mask(logs, background_label=background_label) & logs["confidence"].gt(float(threshold))
    if mask.sum() == 0:
        return float("nan")
    wrong = logs.loc[mask, "prediction"].to_numpy() != logs.loc[mask, "gt_label"].to_numpy()
    return float(np.mean(wrong))


def high_conf_action_count(logs: pd.DataFrame, threshold: float, background_label: int = 0) -> int:
    if len(logs) == 0:
        return 0
    mask = accepted_action_mask(logs, background_label=background_label) & logs["confidence"].gt(float(threshold))
    return int(mask.sum())


def action_precision(logs: pd.DataFrame, background_label: int = 0) -> float:
    pred_mask = logs["action_type"].eq("predict") & logs["prediction"].ge(0)
    action_pred = pred_mask & logs["prediction"].ne(background_label)
    if action_pred.sum() == 0:
        return float("nan")
    return float(logs.loc[action_pred, "gt_label"].ne(background_label).mean())


def action_recall(logs: pd.DataFrame, background_label: int = 0) -> float:
    gt_action = logs["gt_label"].ne(background_label)
    if gt_action.sum() == 0:
        return float("nan")
    action_pred_on_action = (
        logs["action_type"].eq("predict")
        & logs["prediction"].ge(0)
        & logs["prediction"].ne(background_label)
        & gt_action
    )
    return float(action_pred_on_action.sum() / gt_action.sum())


def processed_ratio(logs: pd.DataFrame) -> float:
    if len(logs) == 0:
        return float("nan")
    return float(logs["observed"].mean())


def coverage(logs: pd.DataFrame) -> float:
    if len(logs) == 0:
        return float("nan")
    return float(logs["action_type"].eq("predict").mean())


def abstain_rate(logs: pd.DataFrame) -> float:
    if len(logs) == 0:
        return float("nan")
    return float(logs["action_type"].eq("abstain").mean())


def _mean_confidence_for(logs: pd.DataFrame, mask: pd.Series | None = None) -> float:
    if len(logs) == 0:
        return float("nan")
    subset = logs if mask is None else logs.loc[mask]
    if len(subset) == 0:
        return float("nan")
    return float(np.nanmean(subset["confidence"].to_numpy()))


def num_accepted_decisions(logs: pd.DataFrame) -> int:
    return int(logs["action_type"].eq("predict").sum()) if len(logs) else 0


def num_abstained_decisions(logs: pd.DataFrame) -> int:
    return int(logs["action_type"].eq("abstain").sum()) if len(logs) else 0


def num_candidate_decisions(logs: pd.DataFrame) -> int:
    return num_accepted_decisions(logs) + num_abstained_decisions(logs)


def summarize_basic(logs: pd.DataFrame, background_label: int = 0) -> Dict[str, float]:
    accepted_mask = logs["action_type"].eq("predict") if len(logs) else pd.Series(dtype=bool)
    abstained_mask = logs["action_type"].eq("abstain") if len(logs) else pd.Series(dtype=bool)
    confidence_all = _mean_confidence_for(logs)
    accepted_actions = accepted_action_mask(logs, background_label=background_label)
    accepted_background = accepted_background_mask(logs, background_label=background_label)
    accepted_action_count = int(accepted_actions.sum())
    return {
        "frame_accuracy_on_predicted": frame_accuracy(logs),
        "action_accuracy": action_accuracy(logs, background_label=background_label),
        "action_only_action_accuracy": action_accuracy(logs, background_label=background_label),
        "action_only_selective_risk": action_only_selective_risk(logs, background_label=background_label),
        "action_precision": action_precision(logs, background_label=background_label),
        "action_recall": action_recall(logs, background_label=background_label),
        "processed_ratio": processed_ratio(logs),
        "coverage": coverage(logs),
        "abstain_rate": abstain_rate(logs),
        "mean_confidence": confidence_all,
        "mean_confidence_all": confidence_all,
        "mean_confidence_accepted": _mean_confidence_for(logs, accepted_mask),
        "mean_confidence_abstained": _mean_confidence_for(logs, abstained_mask),
        "num_candidate_decisions": num_candidate_decisions(logs),
        "num_accepted_decisions": num_accepted_decisions(logs),
        "num_abstained_decisions": num_abstained_decisions(logs),
        "accepted_action_count": accepted_action_count,
        "action_accepted_count": accepted_action_count,
        "background_accepted_count": int(accepted_background.sum()),
        "high_conf_action_count_090": high_conf_action_count(logs, threshold=0.90, background_label=background_label),
        "high_conf_action_error_rate_090": high_conf_action_error_rate(logs, threshold=0.90, background_label=background_label),
        "high_conf_action_count_095": high_conf_action_count(logs, threshold=0.95, background_label=background_label),
        "high_conf_action_error_rate_095": high_conf_action_error_rate(logs, threshold=0.95, background_label=background_label),
    }
