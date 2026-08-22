from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _series_groups(summary: pd.DataFrame):
    group_cols = ["policy"]
    if "threshold" in summary.columns and summary["threshold"].notna().any():
        group_cols.append("threshold")
    for key, group in summary.groupby(group_cols, dropna=False):
        if isinstance(key, tuple):
            policy, threshold = key
        else:
            policy, threshold = key, None
        label = str(policy)
        if threshold is not None and pd.notna(threshold):
            label = f"{label} threshold={float(threshold):.2f}"
        yield label, group


def plot_budget_curve(summary: pd.DataFrame, out: str | Path, metric: str = "frame_accuracy_on_predicted") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, group in _series_groups(summary):
        group = group.sort_values("budget")
        ax.plot(group["budget"], group[metric], marker="o", label=label)
    ax.set_xlabel("Feature-level budget")
    ax.set_ylabel(metric)
    ax.set_title("Budget Degradation Curve")
    ax.legend()
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_transition_delay(
    summary: pd.DataFrame,
    out: str | Path,
    metric: str = "transition_delay_median",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, group in _series_groups(summary):
        group = group.sort_values("budget")
        ax.plot(group["budget"], group[metric], marker="o", label=label)
    ax.set_xlabel("Feature-level budget")
    ax.set_ylabel(metric)
    ax.set_title("Transition Delay")
    ax.legend()
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_selective_frontier(
    summary: pd.DataFrame,
    out: str | Path,
    x_metric: str = "mean_decision_latency_steps",
    y_metric: str = "selective_risk",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, group in _series_groups(summary):
        ax.scatter(group[x_metric], group[y_metric], label=label)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title("Selective Frontier")
    ax.legend()
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_threshold_curve(
    summary: pd.DataFrame,
    out: str | Path,
    metric: str = "abstain_rate",
    policy: str = "confidence_threshold",
) -> None:
    if "threshold" not in summary.columns:
        raise ValueError("summary must contain a threshold column")
    data = summary[summary["policy"].eq(policy) & summary["threshold"].notna()].copy()
    if data.empty:
        raise ValueError(f"No threshold rows found for policy={policy}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for budget, group in data.groupby("budget"):
        group = group.sort_values("threshold")
        ax.plot(group["threshold"], group[metric], marker="o", label=f"budget={float(budget):.2f}")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Threshold")
    ax.legend()
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
