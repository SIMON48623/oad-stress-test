from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


AGGREGATE_EXCLUDE_COLUMNS = {
    "seed",
    "log_path",
    "video_summary_path",
    "summary_only",
}


def _budget_key(value: float) -> float:
    return round(float(value), 8)


def _summary_files(results_dir: Path, policies: Sequence[str] | None = None) -> list[Path]:
    return sorted(
        path for path in results_dir.glob("*_summary.csv")
        if not path.name.endswith("_video_summary.csv")
    )


def _sort_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "classifier",
        "linear_mode",
        "score_mode",
        "policy",
        "uncertainty_mode",
        "shift_name",
        "shift_severity",
        "is_clean",
        "threshold",
        "budget",
        "seed",
    ]
    return [column for column in preferred if column in frame.columns]


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        column for column in [
            "classifier",
            "linear_mode",
            "score_mode",
            "policy",
            "uncertainty_mode",
            "budget",
            "threshold",
            "shift_name",
            "shift_severity",
            "is_clean",
            "eval_feature_dir",
        ]
        if column in summary.columns
    ]
    if not group_columns:
        raise ValueError("summary must contain at least one grouping column")
    numeric_columns = []
    for column in summary.columns:
        if column in set(group_columns) | AGGREGATE_EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(summary[column]):
            continue
        if pd.api.types.is_numeric_dtype(summary[column]):
            numeric_columns.append(column)

    rows = []
    for key, group in summary.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        row["num_runs"] = int(len(group))
        row["seed_count"] = int(group["seed"].nunique(dropna=True)) if "seed" in group.columns else 0
        for column in numeric_columns:
            values = group[column].dropna()
            row[f"{column}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(_sort_columns(pd.DataFrame(rows))).reset_index(drop=True)


def merge_summaries(
    results_dir: str | Path,
    policies: Sequence[str] | None = None,
    budgets: Sequence[float] | None = None,
    clean_summary: bool = False,
    aggregate: bool = False,
    aggregate_out: str | Path | None = None,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    files = _summary_files(results_dir, policies=policies)
    if not files:
        raise FileNotFoundError(f"No matching *_summary.csv files found in {results_dir}")

    frames = []
    policy_set = set(policies) if policies else None
    budget_set = {_budget_key(budget) for budget in budgets} if budgets else None

    for path in files:
        frame = pd.read_csv(path)
        if policy_set is not None:
            frame = frame[frame["policy"].isin(policy_set)]
        if budget_set is not None:
            frame = frame[frame["budget"].map(_budget_key).isin(budget_set)]
        if len(frame):
            frames.append(frame)

    if not frames:
        raise ValueError("No summary rows matched the requested policies/budgets")

    summary = pd.concat(frames, ignore_index=True).sort_values(_sort_columns(pd.concat(frames, ignore_index=True)))
    out = results_dir / "summary.csv"
    if clean_summary and out.exists():
        out.unlink()
    summary.to_csv(out, index=False)
    print(f"Merged {len(files)} summaries into {out}")
    print(summary)
    if aggregate:
        aggregated = aggregate_summary(summary)
        aggregate_path = Path(aggregate_out) if aggregate_out is not None else results_dir / "summary_aggregated.csv"
        aggregated.to_csv(aggregate_path, index=False)
        print(f"Aggregated summary into {aggregate_path}")
        print(aggregated)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge per-policy summaries into one summary.csv.")
    parser.add_argument("--results-dir", default="results/dummy")
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--budgets", nargs="+", type=float, default=None)
    parser.add_argument("--clean-summary", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--aggregate-out", default=None)
    args = parser.parse_args(argv)
    merge_summaries(
        results_dir=args.results_dir,
        policies=args.policies,
        budgets=args.budgets,
        clean_summary=args.clean_summary,
        aggregate=args.aggregate,
        aggregate_out=args.aggregate_out,
    )


if __name__ == "__main__":
    main()
