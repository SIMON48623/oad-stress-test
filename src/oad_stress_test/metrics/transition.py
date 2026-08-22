from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _stable_correct_after(group: pd.DataFrame, start_t: int, new_label: int, stable_steps: int) -> int | None:
    max_t = int(group["t"].max())
    for t in range(start_t, max_t + 1):
        window = group[(group["t"] >= t) & (group["t"] < t + stable_steps)]
        if len(window) < stable_steps:
            return None
        ok = (window["action_type"].eq("predict") & window["prediction"].eq(new_label)).all()
        if bool(ok):
            return int(t - start_t)
    return None


def _transition_points(labels: np.ndarray) -> np.ndarray:
    return np.where(labels[1:] != labels[:-1])[0] + 1


def transition_count_total(logs: pd.DataFrame) -> int:
    total = 0
    for _, group in logs.groupby("video_id"):
        labels = group.sort_values("t")["gt_label"].to_numpy()
        total += int(len(_transition_points(labels)))
    return total


def _stable_delays_from_predictions(labels: np.ndarray, predictions: np.ndarray, stable_steps: int) -> List[int]:
    delays: List[int] = []
    transition_points = _transition_points(labels)
    if len(transition_points) == 0:
        return delays

    valid_start_cache: Dict[int, np.ndarray] = {}
    for new_label in np.unique(labels[transition_points]):
        correct = predictions == int(new_label)
        if len(correct) < stable_steps:
            valid_start_cache[int(new_label)] = np.asarray([], dtype=np.int64)
            continue
        counts = np.convolve(correct.astype(np.int8), np.ones(stable_steps, dtype=np.int8), mode="valid")
        valid_start_cache[int(new_label)] = np.flatnonzero(counts == stable_steps)

    for t0 in transition_points:
        new_label = int(labels[t0])
        valid_starts = valid_start_cache[new_label]
        match_idx = int(np.searchsorted(valid_starts, int(t0)))
        if match_idx < len(valid_starts):
            delays.append(int(valid_starts[match_idx] - t0))
    return delays


def transition_delays(logs: pd.DataFrame, stable_steps: int = 3) -> List[int]:
    delays: List[int] = []
    for _, group in logs.groupby("video_id"):
        group = group.sort_values("t").reset_index(drop=True)
        labels = group["gt_label"].to_numpy()
        actions = group["action_type"].to_numpy()
        predictions = group["prediction"].to_numpy()
        stable_predictions = np.where(actions == "predict", predictions, -1)
        delays.extend(_stable_delays_from_predictions(labels, stable_predictions, stable_steps=stable_steps))
    return delays


def observed_transition_delays(logs: pd.DataFrame) -> List[int]:
    delays: List[int] = []
    for _, group in logs.groupby("video_id"):
        group = group.sort_values("t").reset_index(drop=True)
        labels = group["gt_label"].to_numpy()
        transition_points = _transition_points(labels)
        if len(transition_points) == 0:
            continue
        observed_predict = (
            group["observed"].to_numpy(dtype=bool)
            & group["action_type"].eq("predict").to_numpy()
            & group["prediction"].ge(0).to_numpy()
        )
        predictions = group["prediction"].to_numpy()
        for t0 in transition_points:
            new_label = int(labels[t0])
            matches = np.flatnonzero(observed_predict[int(t0):] & (predictions[int(t0):] == new_label))
            if len(matches):
                delays.append(int(matches[0]))
    return delays


def carried_state_transition_delays(logs: pd.DataFrame, stable_steps: int = 3) -> List[int]:
    delays: List[int] = []
    for _, group in logs.groupby("video_id"):
        group = group.sort_values("t").reset_index(drop=True)
        labels = group["gt_label"].to_numpy()
        predictions = group["prediction"].to_numpy()
        actions = group["action_type"].to_numpy()
        carried = np.full(len(group), -1, dtype=np.int64)
        last_prediction = -1
        for i, (action, prediction) in enumerate(zip(actions, predictions)):
            if action == "predict" and int(prediction) >= 0:
                last_prediction = int(prediction)
            carried[i] = last_prediction
        delays.extend(_stable_delays_from_predictions(labels, carried, stable_steps=stable_steps))
    return delays


def _summarize_delays(delays: List[int], delay_prefix: str, count_key: str) -> Dict[str, float]:
    if not delays:
        return {
            f"{delay_prefix}_median": float("nan"),
            f"{delay_prefix}_p90": float("nan"),
            count_key: 0,
        }
    arr = np.asarray(delays, dtype=np.float32)
    return {
        f"{delay_prefix}_median": float(np.median(arr)),
        f"{delay_prefix}_p90": float(np.percentile(arr, 90)),
        count_key: int(len(arr)),
    }


def summarize_transition_delay(logs: pd.DataFrame, stable_steps: int = 3) -> Dict[str, float]:
    delays = transition_delays(logs, stable_steps=stable_steps)
    observed_delays = observed_transition_delays(logs)
    carried_delays = carried_state_transition_delays(logs, stable_steps=stable_steps)
    total_transitions = transition_count_total(logs)
    if not delays:
        out = {
            "transition_delay_median": float("nan"),
            "transition_delay_p90": float("nan"),
            "transition_delay_mean": float("nan"),
            "transition_count_detected": 0,
            "stable_transition_delay_median": float("nan"),
            "stable_transition_delay_p90": float("nan"),
            "stable_transition_count_detected": 0,
        }
    else:
        arr = np.asarray(delays, dtype=np.float32)
        median = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        out = {
            "transition_delay_median": median,
            "transition_delay_p90": p90,
            "transition_delay_mean": float(np.mean(arr)),
            "transition_count_detected": int(len(arr)),
            "stable_transition_delay_median": median,
            "stable_transition_delay_p90": p90,
            "stable_transition_count_detected": int(len(arr)),
        }
    stable_detected = int(len(delays))
    stable_missed = max(0, int(total_transitions) - stable_detected)
    out.update({
        "transition_count_total": int(total_transitions),
        "stable_window_transition_detected_count": stable_detected,
        "stable_window_transition_missed_count": stable_missed,
        "stable_window_transition_detection_rate": (
            float(stable_detected / total_transitions) if total_transitions > 0 else float("nan")
        ),
        "stable_window_transition_delay_median": out["stable_transition_delay_median"],
    })
    out.update(_summarize_delays(observed_delays, "observed_transition_delay", "observed_transition_count_detected"))
    out.update(_summarize_delays(carried_delays, "carried_transition_delay", "carried_transition_count_detected"))
    out["observed_transition_detected_count"] = int(out["observed_transition_count_detected"])
    out["carried_transition_detected_count"] = int(out["carried_transition_count_detected"])
    return out
