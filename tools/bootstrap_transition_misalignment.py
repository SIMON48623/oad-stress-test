from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tools.analyze_transition_misalignment import add_transition_features, read_input
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution from tools/.
    from analyze_transition_misalignment import add_transition_features, read_input


DEFAULT_GROUP_COLUMNS = ["model", "policy", "uncertainty_mode", "threshold", "budget", "shift_type"]
METRICS = [
    "TEG_raw",
    "TAG",
    "ATM_raw",
    "Error_OR_raw",
    "Abstain_OR",
    "TEG_selective",
    "confidence_gap",
]
POSITIVE_SIGN_METRICS = {"TEG_raw", "ATM_raw", "TEG_selective"}
COUNT_COLUMNS = [
    "near_frames",
    "raw_error_near_sum",
    "raw_error_near_count",
    "selective_error_near_sum",
    "selective_error_near_count",
    "abstain_near_sum",
    "confidence_near_sum",
    "confidence_near_count",
    "far_frames",
    "raw_error_far_sum",
    "raw_error_far_count",
    "selective_error_far_sum",
    "selective_error_far_count",
    "abstain_far_sum",
    "confidence_far_sum",
    "confidence_far_count",
]


class BootstrapError(ValueError):
    pass


def load_transition_frame(path: str | Path) -> pd.DataFrame:
    """Load an analyzer frame CSV or a per-frame JSONL and ensure transition features exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        frame = read_input(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise BootstrapError(f"Unsupported input suffix {path.suffix}; expected .csv or .jsonl")

    if "distance_to_transition" not in frame.columns or "raw_error" not in frame.columns:
        frame = add_transition_features(frame)

    missing = {
        "video_id",
        "distance_to_transition",
        "raw_error",
        "selective_error",
        "is_abstain",
        "max_prob",
    } - set(frame.columns)
    if missing:
        raise BootstrapError(f"Input is missing required transition columns: {sorted(missing)}")

    out = frame.copy()
    out["video_id"] = out["video_id"].astype(str)
    out["distance_to_transition"] = pd.to_numeric(out["distance_to_transition"], errors="coerce")
    for column in ["raw_error", "selective_error", "max_prob"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["is_abstain"] = _to_bool(out["is_abstain"])
    for column in ["threshold", "budget"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def available_group_columns(frame: pd.DataFrame, group_columns: Iterable[str] | None = None) -> list[str]:
    requested = list(group_columns or DEFAULT_GROUP_COLUMNS)
    return [column for column in requested if column in frame.columns]


def compute_video_counts(
    frame: pd.DataFrame,
    near_window: float = 8.0,
    far_window: float = 32.0,
    group_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute per-video near/far counts used for cluster bootstrap."""
    group_cols = available_group_columns(frame, group_columns)
    valid = frame[frame["distance_to_transition"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=[*group_cols, "video_id"])

    rows = []
    for key, group in valid.groupby([*group_cols, "video_id"], dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip([*group_cols, "video_id"], key))
        near = group["distance_to_transition"].le(float(near_window))
        far = group["distance_to_transition"].gt(float(far_window))
        row.update(_region_counts(group, near, "near"))
        row.update(_region_counts(group, far, "far"))
        row.update(metrics_from_counts(pd.DataFrame([row])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values([*group_cols, "video_id"]).reset_index(drop=True)


def bootstrap_summary(
    video_counts: pd.DataFrame,
    bootstrap: int = 1000,
    seed: int = 42,
    tag_epsilon: float = 0.01,
    group_columns: Iterable[str] | None = None,
    backend: str = "legacy",
) -> pd.DataFrame:
    """Cluster-bootstrap videos within each model/policy/threshold/budget group."""
    if backend not in {"legacy", "numpy"}:
        raise BootstrapError(f"Unknown bootstrap backend: {backend}")
    if backend == "numpy":
        return bootstrap_summary_numpy(
            video_counts,
            bootstrap=bootstrap,
            seed=seed,
            tag_epsilon=tag_epsilon,
            group_columns=group_columns,
        )
    return bootstrap_summary_legacy(
        video_counts,
        bootstrap=bootstrap,
        seed=seed,
        tag_epsilon=tag_epsilon,
        group_columns=group_columns,
    )


def bootstrap_summary_legacy(
    video_counts: pd.DataFrame,
    bootstrap: int = 1000,
    seed: int = 42,
    tag_epsilon: float = 0.01,
    group_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pandas-loop bootstrap path kept for backward-compatible auditing."""
    if video_counts.empty:
        return pd.DataFrame()
    group_cols = available_group_columns(video_counts, group_columns)
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, group in video_counts.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        group_values = dict(zip(group_cols, key))
        videos = group["video_id"].astype(str).to_numpy()
        n_videos = len(videos)
        point = metrics_from_counts(group)
        samples = {metric: [] for metric in METRICS}
        indices = _bootstrap_indices(rng, n_videos, int(bootstrap))
        for sampled_indices in indices:
            sampled = group.iloc[sampled_indices].reset_index(drop=True)
            values = metrics_from_counts(sampled)
            for metric in METRICS:
                samples[metric].append(values[metric])
        rows.extend(_summary_rows_from_samples(group_values, point, samples, int(bootstrap), int(seed), int(n_videos), tag_epsilon))
    return pd.DataFrame(rows).sort_values([*group_cols, "metric"]).reset_index(drop=True)


def bootstrap_summary_numpy(
    video_counts: pd.DataFrame,
    bootstrap: int = 1000,
    seed: int = 42,
    tag_epsilon: float = 0.01,
    group_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Vectorized cluster bootstrap over per-video sufficient statistics.

    This backend never revisits frame-level data. It resamples video rows in the
    per-video sufficient-stat table and computes all metrics from summed counts.
    """
    if video_counts.empty:
        return pd.DataFrame()
    missing = set(COUNT_COLUMNS) - set(video_counts.columns)
    if missing:
        raise BootstrapError(f"video_counts is missing count columns required by numpy backend: {sorted(missing)}")
    group_cols = available_group_columns(video_counts, group_columns)
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, group in video_counts.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        group_values = dict(zip(group_cols, key))
        n_videos = int(len(group))
        point = metrics_from_counts(group)
        count_matrix = group[COUNT_COLUMNS].to_numpy(dtype=np.float64, copy=True)
        indices = _bootstrap_indices(rng, n_videos, int(bootstrap))
        samples = _metric_samples_from_count_matrix(count_matrix, indices)
        rows.extend(_summary_rows_from_samples(group_values, point, samples, int(bootstrap), int(seed), n_videos, tag_epsilon))
    return pd.DataFrame(rows).sort_values([*group_cols, "metric"]).reset_index(drop=True)


def _bootstrap_indices(rng: np.random.Generator, n_videos: int, bootstrap: int) -> np.ndarray:
    if n_videos <= 0:
        return np.empty((int(bootstrap), 0), dtype=np.int64)
    return rng.integers(0, int(n_videos), size=(int(bootstrap), int(n_videos)), endpoint=False)


def _summary_rows_from_samples(
    group_values: dict[str, object],
    point: dict[str, float],
    samples: dict[str, list[float] | np.ndarray],
    bootstrap: int,
    seed: int,
    n_videos: int,
    tag_epsilon: float,
) -> list[dict[str, object]]:
    rows = []
    for metric in METRICS:
        values = np.asarray(samples[metric], dtype=float)
        finite = values[np.isfinite(values)]
        row = {
            **group_values,
            "metric": metric,
            "point_estimate": float(point[metric]) if math.isfinite(point[metric]) else float("nan"),
            "bootstrap": int(bootstrap),
            "seed": int(seed),
            "n_videos": int(n_videos),
            "valid_bootstrap_samples": int(len(finite)),
            "mean": _nan_stat(finite, "mean"),
            "std": _nan_stat(finite, "std"),
            "ci_2_5": _nan_percentile(finite, 2.5),
            "ci_97_5": _nan_percentile(finite, 97.5),
            "sign_rate_positive": (
                float(np.mean(finite > 0.0))
                if metric in POSITIVE_SIGN_METRICS and len(finite)
                else float("nan")
            ),
            "sign_rate_near_zero": (
                float(np.mean(np.abs(finite) < float(tag_epsilon)))
                if metric == "TAG" and len(finite)
                else float("nan")
            ),
            "tag_epsilon": float(tag_epsilon) if metric == "TAG" else float("nan"),
        }
        rows.append(row)
    return rows


def _metric_samples_from_count_matrix(count_matrix: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    if len(indices) == 0:
        return {metric: np.asarray([], dtype=float) for metric in METRICS}
    totals = count_matrix[indices].sum(axis=1)
    col = {name: idx for idx, name in enumerate(COUNT_COLUMNS)}

    raw_near = _rate_array(totals[:, col["raw_error_near_sum"]], totals[:, col["raw_error_near_count"]])
    raw_far = _rate_array(totals[:, col["raw_error_far_sum"]], totals[:, col["raw_error_far_count"]])
    teg_raw = raw_near - raw_far
    teg_raw[~np.isfinite(raw_near) | ~np.isfinite(raw_far)] = np.nan

    selective_near = _rate_array(
        totals[:, col["selective_error_near_sum"]],
        totals[:, col["selective_error_near_count"]],
    )
    selective_far = _rate_array(
        totals[:, col["selective_error_far_sum"]],
        totals[:, col["selective_error_far_count"]],
    )
    teg_selective = selective_near - selective_far
    teg_selective[~np.isfinite(selective_near) | ~np.isfinite(selective_far)] = np.nan

    abs_near = _rate_array(totals[:, col["abstain_near_sum"]], totals[:, col["near_frames"]])
    abs_far = _rate_array(totals[:, col["abstain_far_sum"]], totals[:, col["far_frames"]])
    tag = abs_near - abs_far
    tag[~np.isfinite(abs_near) | ~np.isfinite(abs_far)] = np.nan

    conf_near = _rate_array(totals[:, col["confidence_near_sum"]], totals[:, col["confidence_near_count"]])
    conf_far = _rate_array(totals[:, col["confidence_far_sum"]], totals[:, col["confidence_far_count"]])
    confidence_gap = conf_near - conf_far
    confidence_gap[~np.isfinite(conf_near) | ~np.isfinite(conf_far)] = np.nan

    return {
        "TEG_raw": teg_raw,
        "TAG": tag,
        "ATM_raw": teg_raw - tag,
        "Error_OR_raw": _corrected_odds_ratio_array(
            totals[:, col["raw_error_near_sum"]],
            totals[:, col["raw_error_near_count"]],
            totals[:, col["raw_error_far_sum"]],
            totals[:, col["raw_error_far_count"]],
        ),
        "Abstain_OR": _corrected_odds_ratio_array(
            totals[:, col["abstain_near_sum"]],
            totals[:, col["near_frames"]],
            totals[:, col["abstain_far_sum"]],
            totals[:, col["far_frames"]],
        ),
        "TEG_selective": teg_selective,
        "confidence_gap": confidence_gap,
    }


def metrics_from_counts(counts: pd.DataFrame) -> dict[str, float]:
    raw_near = _rate(counts["raw_error_near_sum"].sum(), counts["raw_error_near_count"].sum())
    raw_far = _rate(counts["raw_error_far_sum"].sum(), counts["raw_error_far_count"].sum())
    teg_raw = _difference(raw_near, raw_far)

    selective_near = _rate(counts["selective_error_near_sum"].sum(), counts["selective_error_near_count"].sum())
    selective_far = _rate(counts["selective_error_far_sum"].sum(), counts["selective_error_far_count"].sum())
    teg_selective = _difference(selective_near, selective_far)

    abs_near = _rate(counts["abstain_near_sum"].sum(), counts["near_frames"].sum())
    abs_far = _rate(counts["abstain_far_sum"].sum(), counts["far_frames"].sum())
    tag = _difference(abs_near, abs_far)

    conf_near = _rate(counts["confidence_near_sum"].sum(), counts["confidence_near_count"].sum())
    conf_far = _rate(counts["confidence_far_sum"].sum(), counts["confidence_far_count"].sum())
    confidence_gap = _difference(conf_near, conf_far)

    return {
        "err_near_raw": raw_near,
        "err_far_raw": raw_far,
        "TEG_raw": teg_raw,
        "err_near_selective": selective_near,
        "err_far_selective": selective_far,
        "TEG_selective": teg_selective,
        "abs_near": abs_near,
        "abs_far": abs_far,
        "TAG": tag,
        "conf_near": conf_near,
        "conf_far": conf_far,
        "confidence_gap": confidence_gap,
        "Error_OR_raw": _corrected_odds_ratio_from_counts(
            counts["raw_error_near_sum"].sum(),
            counts["raw_error_near_count"].sum(),
            counts["raw_error_far_sum"].sum(),
            counts["raw_error_far_count"].sum(),
        ),
        "Abstain_OR": _corrected_odds_ratio_from_counts(
            counts["abstain_near_sum"].sum(),
            counts["near_frames"].sum(),
            counts["abstain_far_sum"].sum(),
            counts["far_frames"].sum(),
        ),
        "ATM_raw": _difference(teg_raw, tag),
    }


def _region_counts(group: pd.DataFrame, mask: pd.Series, prefix: str) -> dict[str, float]:
    raw = group.loc[mask, "raw_error"].dropna()
    selective = group.loc[mask, "selective_error"].dropna()
    confidence = group.loc[mask, "max_prob"].dropna()
    return {
        f"{prefix}_frames": int(mask.sum()),
        f"raw_error_{prefix}_sum": float(raw.sum()),
        f"raw_error_{prefix}_count": int(len(raw)),
        f"selective_error_{prefix}_sum": float(selective.sum()),
        f"selective_error_{prefix}_count": int(len(selective)),
        f"abstain_{prefix}_sum": int(group.loc[mask, "is_abstain"].sum()),
        f"confidence_{prefix}_sum": float(confidence.sum()),
        f"confidence_{prefix}_count": int(len(confidence)),
    }


def _corrected_odds_ratio_from_counts(event_near: float, total_near: float, event_far: float, total_far: float) -> float:
    if total_near <= 0 or total_far <= 0:
        return float("nan")
    non_event_near = float(total_near) - float(event_near)
    non_event_far = float(total_far) - float(event_far)
    a = float(event_near) + 0.5
    b = non_event_near + 0.5
    c = float(event_far) + 0.5
    d = non_event_far + 0.5
    return float((a / b) / (c / d))


def _corrected_odds_ratio_array(
    event_near: np.ndarray,
    total_near: np.ndarray,
    event_far: np.ndarray,
    total_far: np.ndarray,
) -> np.ndarray:
    out = np.full_like(event_near, np.nan, dtype=np.float64)
    valid = (total_near > 0) & (total_far > 0)
    if not np.any(valid):
        return out
    non_event_near = total_near[valid] - event_near[valid]
    non_event_far = total_far[valid] - event_far[valid]
    a = event_near[valid] + 0.5
    b = non_event_near + 0.5
    c = event_far[valid] + 0.5
    d = non_event_far + 0.5
    out[valid] = (a / b) / (c / d)
    return out


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes"})


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else float("nan")


def _rate_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _difference(left: float, right: float) -> float:
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return float("nan")
    return float(left) - float(right)


def _nan_stat(values: np.ndarray, stat: str) -> float:
    if len(values) == 0:
        return float("nan")
    if stat == "mean":
        return float(np.mean(values))
    if stat == "std":
        return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    raise ValueError(stat)


def _nan_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else float("nan")


def write_readable_summary(path: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        text = "# Bootstrap Transition Misalignment Summary\n\nNo bootstrap rows were generated.\n"
    else:
        keep = [
            column
            for column in [
                "uncertainty_mode",
                "threshold",
                "budget",
                "metric",
                "point_estimate",
                "mean",
                "ci_2_5",
                "ci_97_5",
                "sign_rate_positive",
                "sign_rate_near_zero",
                "n_videos",
            ]
            if column in summary.columns
        ]
        text = "# Bootstrap Transition Misalignment Summary\n\n"
        text += _to_markdown_table(summary[keep])
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_bootstrap(
    input_path: str | Path,
    output_dir: str | Path,
    near_window: float = 8.0,
    far_window: float = 32.0,
    bootstrap: int = 1000,
    seed: int = 42,
    tag_epsilon: float = 0.01,
    backend: str = "legacy",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_transition_frame(input_path)
    video_counts = compute_video_counts(frame, near_window=near_window, far_window=far_window)
    summary = bootstrap_summary(video_counts, bootstrap=bootstrap, seed=seed, tag_epsilon=tag_epsilon, backend=backend)

    per_video_path = output_dir / "per_video_metrics.csv"
    summary_path = output_dir / "bootstrap_summary.csv"
    readable_path = output_dir / "summary_readable.md"
    video_counts.to_csv(per_video_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_readable_summary(readable_path, summary)
    return {
        "per_video_metrics": per_video_path,
        "bootstrap_summary": summary_path,
        "summary_readable": readable_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Video-cluster bootstrap for transition-abstention misalignment metrics.")
    parser.add_argument("--input", required=True, help="frame_with_transition_features.csv or per-frame JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--near-window", type=float, default=8.0)
    parser.add_argument("--far-window", type=float, default=32.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag-epsilon", type=float, default=0.01)
    parser.add_argument(
        "--backend",
        choices=["legacy", "numpy"],
        default="legacy",
        help="bootstrap backend; numpy uses vectorized per-video sufficient statistics",
    )
    args = parser.parse_args(argv)

    outputs = run_bootstrap(
        input_path=args.input,
        output_dir=args.output_dir,
        near_window=args.near_window,
        far_window=args.far_window,
        bootstrap=args.bootstrap,
        seed=args.seed,
        tag_epsilon=args.tag_epsilon,
        backend=args.backend,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
