from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def selective_risk(logs: pd.DataFrame) -> float:
    pred = logs[logs["action_type"].eq("predict") & logs["prediction"].ge(0)]
    if len(pred) == 0:
        return float("nan")
    err = pred["prediction"].to_numpy() != pred["gt_label"].to_numpy()
    return float(np.mean(err))


def mean_decision_latency(logs: pd.DataFrame) -> float:
    """Proxy latency: mean number of wait steps before each predict/abstain.

    This is a feature-level proxy. It is not real wall-clock latency.
    """
    latencies = []
    for _, group in logs.groupby("video_id"):
        waits = 0
        for _, row in group.sort_values("t").iterrows():
            if row["action_type"] == "wait":
                waits += 1
            else:
                latencies.append(waits)
                waits = 0
    if not latencies:
        return float("nan")
    return float(np.mean(latencies))


def summarize_selective(logs: pd.DataFrame) -> Dict[str, float]:
    return {
        "selective_risk": selective_risk(logs),
        "mean_decision_latency_steps": mean_decision_latency(logs),
    }
