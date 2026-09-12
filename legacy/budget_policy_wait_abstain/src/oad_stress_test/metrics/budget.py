from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def degradation_area(summary: pd.DataFrame, metric: str = "frame_accuracy_on_predicted") -> float:
    clean = summary[["budget", metric]].dropna().sort_values("budget")
    if len(clean) < 2:
        return float("nan")
    values = clean[metric].to_numpy()
    budgets = clean["budget"].to_numpy()
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, budgets))
    return float(np.trapz(values, budgets))


def degradation_slope(summary: pd.DataFrame, metric: str = "frame_accuracy_on_predicted") -> float:
    clean = summary[["budget", metric]].dropna().sort_values("budget")
    if len(clean) < 2:
        return float("nan")
    x = clean["budget"].to_numpy(dtype=float)
    y = clean[metric].to_numpy(dtype=float)
    return float((y[-1] - y[0]) / max(1e-8, x[-1] - x[0]))


def summarize_budget_curve(summary: pd.DataFrame, metric: str = "frame_accuracy_on_predicted") -> Dict[str, float]:
    return {
        f"{metric}_budget_auc": degradation_area(summary, metric),
        f"{metric}_budget_slope": degradation_slope(summary, metric),
    }
