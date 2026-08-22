from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
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
DECISIONS = {"predict", "wait", "abstain"}
GROUP_COLUMNS = ["model", "policy", "threshold", "budget", "shift_type"]


class AnalyzerError(ValueError):
    pass


def parse_bins(raw: str | Iterable[float] | None = None) -> list[float]:
    if raw is None:
        return [0.0, 4.0, 8.0, 16.0, 32.0, 999999.0]
    if isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw]
    if len(values) < 2:
        raise AnalyzerError("bins must contain at least two values")
    if any(values[idx] >= values[idx + 1] for idx in range(len(values) - 1)):
        raise AnalyzerError("bins must be strictly increasing")
    return values


def distance_bin_label(distance: float, bins: list[float]) -> str | float:
    if pd.isna(distance):
        return np.nan
    value = float(distance)
    if value <= bins[1]:
        return f"{_fmt_bin(bins[0])}-{_fmt_bin(bins[1])}"
    for idx in range(2, len(bins)):
        lower = bins[idx - 1]
        upper = bins[idx]
        if value <= upper:
            if idx == len(bins) - 1:
                return f">{_fmt_bin(lower)}"
            return f"{_fmt_bin(lower + 1)}-{_fmt_bin(upper)}"
    return f">{_fmt_bin(bins[-2])}"


def distance_bin_order(bins: list[float]) -> list[str]:
    labels = [f"{_fmt_bin(bins[0])}-{_fmt_bin(bins[1])}"]
    for idx in range(2, len(bins)):
        lower = bins[idx - 1]
        upper = bins[idx]
        if idx == len(bins) - 1:
            labels.append(f">{_fmt_bin(lower)}")
        else:
            labels.append(f"{_fmt_bin(lower + 1)}-{_fmt_bin(upper)}")
    return labels


def _fmt_bin(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def read_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input log not found: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise AnalyzerError(f"Unsupported input suffix: {path.suffix}; expected .jsonl or .csv")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise AnalyzerError(f"Input is missing required columns: {sorted(missing)}")
    invalid_decisions = sorted(set(frame["decision"].dropna().astype(str)) - DECISIONS)
    if invalid_decisions:
        raise AnalyzerError(f"Invalid decisions: {invalid_decisions}")
    out = frame.copy()
    out["video_id"] = out["video_id"].astype(str)
    out["decision"] = out["decision"].astype(str)
    out["frame_idx"] = out["frame_idx"].astype(int)
    out["gt_label"] = out["gt_label"].astype(int)
    out["pred_label"] = out["pred_label"].astype(int)
    for column in ["max_prob", "entropy", "top2_margin", "budget", "threshold"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def add_transition_features(frame: pd.DataFrame, tau: float = 8.0, bins: list[float] | None = None) -> pd.DataFrame:
    bins = parse_bins(bins)
    rows = []
    for _, group in frame.sort_values(["video_id", "frame_idx"]).groupby("video_id", sort=False):
        group = group.copy().reset_index(drop=True)
        transitions = group["gt_label"].ne(group["gt_label"].shift(1))
        transitions.iloc[0] = False
        transition_frames = group.loc[transitions, "frame_idx"].to_numpy(dtype=float)
        group["is_transition_frame"] = transitions.astype(bool)
        if len(transition_frames) == 0:
            group["distance_to_transition"] = np.nan
        else:
            frame_values = group["frame_idx"].to_numpy(dtype=float)
            distances = np.min(np.abs(frame_values[:, None] - transition_frames[None, :]), axis=1)
            group["distance_to_transition"] = distances
        group["proximity"] = np.exp(-group["distance_to_transition"] / max(float(tau), 1e-12))
        group.loc[group["distance_to_transition"].isna(), "proximity"] = np.nan
        group["distance_bin"] = group["distance_to_transition"].map(lambda value: distance_bin_label(value, bins))
        valid_pred = group["pred_label"].ge(0)
        computed_raw_error = np.where(valid_pred, group["pred_label"].ne(group["gt_label"]), np.nan)
        if "raw_error" in group.columns:
            provided_raw = pd.to_numeric(group["raw_error"], errors="coerce")
            group["raw_error"] = np.where(valid_pred, provided_raw, np.nan)
        else:
            group["raw_error"] = computed_raw_error
        is_predict = group["decision"].eq("predict")
        if "selective_error" in group.columns:
            provided_selective = pd.to_numeric(group["selective_error"], errors="coerce")
            group["selective_error"] = np.where(is_predict & valid_pred, provided_selective, np.nan)
        else:
            group["selective_error"] = np.where(is_predict & valid_pred, computed_raw_error, np.nan)
        group["is_abstain"] = group["decision"].eq("abstain")
        group["is_wait"] = group["decision"].eq("wait")
        group["is_predict"] = is_predict
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else frame.copy()


def summarize_transition_curves(frame: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    valid = frame[frame["distance_to_transition"].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame(columns=[
            *GROUP_COLUMNS,
            "distance_bin",
            "n_frames",
            "n_videos",
            "raw_error_rate",
            "selective_error_rate",
            "abstain_rate",
            "wait_rate",
            "predict_rate",
            "mean_max_prob",
            "mean_entropy",
            "mean_top2_margin",
            "mean_proximity",
        ])
    label_order = distance_bin_order(bins)
    valid["distance_bin"] = pd.Categorical(valid["distance_bin"], categories=label_order, ordered=True)
    rows = []
    for key, group in valid.groupby([*GROUP_COLUMNS, "distance_bin"], dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip([*GROUP_COLUMNS, "distance_bin"], key))
        row.update({
            "n_frames": int(len(group)),
            "n_videos": int(group["video_id"].nunique()),
            "raw_error_rate": _mean_or_nan(group["raw_error"]),
            "selective_error_rate": _mean_or_nan(group["selective_error"]),
            "abstain_rate": float(group["is_abstain"].mean()),
            "wait_rate": float(group["is_wait"].mean()),
            "predict_rate": float(group["is_predict"].mean()),
            "mean_max_prob": _mean_or_nan(group["max_prob"]),
            "mean_entropy": _mean_or_nan(group["entropy"]),
            "mean_top2_margin": _mean_or_nan(group["top2_margin"]),
            "mean_proximity": _mean_or_nan(group["proximity"]),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values([*GROUP_COLUMNS, "distance_bin"]).reset_index(drop=True)


def summarize_atm(frame: pd.DataFrame, near_window: float = 8.0, far_window: float = 32.0) -> pd.DataFrame:
    valid = frame[frame["distance_to_transition"].notna()].copy()
    rows = []
    for key, group in valid.groupby(GROUP_COLUMNS, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        near = group["distance_to_transition"].le(float(near_window))
        far = group["distance_to_transition"].gt(float(far_window))
        row = dict(zip(GROUP_COLUMNS, key))
        row["n_frames_valid_distance"] = int(len(group))
        row["n_videos"] = int(group["video_id"].nunique())
        row["near_frames"] = int(near.sum())
        row["far_frames"] = int(far.sum())
        row["err_near_raw"] = _mean_or_nan(group.loc[near, "raw_error"])
        row["err_far_raw"] = _mean_or_nan(group.loc[far, "raw_error"])
        row["TEG_raw"] = _difference(row["err_near_raw"], row["err_far_raw"])
        row["err_near_selective"] = _mean_or_nan(group.loc[near, "selective_error"])
        row["err_far_selective"] = _mean_or_nan(group.loc[far, "selective_error"])
        row["TEG_selective"] = _difference(row["err_near_selective"], row["err_far_selective"])
        row["abs_near"] = _mean_or_nan(group.loc[near, "is_abstain"])
        row["abs_far"] = _mean_or_nan(group.loc[far, "is_abstain"])
        row["TAG"] = _difference(row["abs_near"], row["abs_far"])
        row["conf_near"] = _mean_or_nan(group.loc[near, "max_prob"])
        row["conf_far"] = _mean_or_nan(group.loc[far, "max_prob"])
        row["confidence_gap"] = _difference(row["conf_near"], row["conf_far"])
        row["Error_OR_raw"] = odds_ratio(group["raw_error"], near, far)
        row["Error_OR_selective"] = odds_ratio(group["selective_error"], near, far)
        row["Abstain_OR"] = odds_ratio(group["is_abstain"], near, far)
        row["ATM_raw"] = _difference(row["TEG_raw"], row["TAG"])
        row["ATM_selective"] = _difference(row["TEG_selective"], row["TAG"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_COLUMNS).reset_index(drop=True) if rows else pd.DataFrame()


def odds_ratio(values: pd.Series, near_mask: pd.Series, far_mask: pd.Series) -> float:
    near_values = values[near_mask].dropna().astype(bool)
    far_values = values[far_mask].dropna().astype(bool)
    if len(near_values) == 0 or len(far_values) == 0:
        return float("nan")
    a = float(near_values.sum())
    b = float((~near_values).sum())
    c = float(far_values.sum())
    d = float((~far_values).sum())
    if min(a, b, c, d) == 0.0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    return float((a / b) / (c / d))


def _mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _difference(left: Any, right: Any) -> float:
    if pd.isna(left) or pd.isna(right):
        return float("nan")
    return float(left) - float(right)


def fit_logistic_regressions(frame: pd.DataFrame) -> str:
    lines = ["# Logistic Regression Summary", ""]
    zero_predict_note = None
    if "is_predict" in frame.columns and int(frame["is_predict"].sum()) == 0:
        zero_predict_note = "Selective-error regression skipped: zero predict frames, so selective_error is unavailable."
    try:
        import statsmodels.formula.api as smf
        from statsmodels.tools.sm_exceptions import PerfectSeparationError
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        output = [
            *lines,
            f"statsmodels is not available; regression diagnostics skipped. ({exc})",
            "",
        ]
        if zero_predict_note is not None:
            output.extend([zero_predict_note, ""])
        return "\n".join(output)

    valid = frame[frame["distance_to_transition"].notna() & frame["proximity"].notna()].copy()
    if len(valid) == 0:
        return "\n".join([*lines, "No frames with valid transition distance; regressions skipped.", ""])

    valid["budget_factor"] = valid["budget"].astype(str)
    valid["threshold_factor"] = valid["threshold"].astype(str).fillna("null")
    valid["policy_factor"] = valid["policy"].astype(str)

    def formula_for(outcome: str) -> str:
        terms = ["proximity", "C(video_id)"]
        if valid["budget_factor"].nunique(dropna=False) > 1:
            terms.append("C(budget_factor)")
        if valid["policy_factor"].nunique(dropna=False) > 1:
            terms.append("C(policy_factor)")
        if valid["threshold_factor"].nunique(dropna=False) > 1:
            terms.append("C(threshold_factor)")
        return f"{outcome} ~ " + " + ".join(terms)

    def fit_one(title: str, outcome: str, data: pd.DataFrame) -> None:
        lines.extend([f"## {title}", ""])
        clean = data[[outcome, "proximity", "video_id", "budget_factor", "policy_factor", "threshold_factor"]].dropna()
        if len(clean) < 20:
            lines.extend([f"Skipped: only {len(clean)} usable rows.", ""])
            return
        if clean[outcome].nunique(dropna=True) < 2:
            lines.extend(["Skipped: outcome has fewer than two classes.", ""])
            return
        try:
            result = smf.logit(formula_for(outcome), data=clean).fit(disp=False, maxiter=100)
        except PerfectSeparationError as exc:
            lines.extend([f"Skipped: perfect separation ({exc}).", ""])
            return
        except Exception as exc:  # noqa: BLE001 - report model failures without crashing.
            lines.extend([f"Skipped: regression failed ({exc}).", ""])
            return
        coef = result.params.get("proximity", float("nan"))
        pvalue = result.pvalues.get("proximity", float("nan"))
        lines.extend([
            f"formula: `{formula_for(outcome)}`",
            f"n: {int(result.nobs)}",
            f"proximity_coef: {float(coef):.6f}",
            f"proximity_pvalue: {float(pvalue):.6g}",
            "",
        ])

    valid["raw_error_reg"] = pd.to_numeric(valid["raw_error"], errors="coerce")
    valid["is_abstain_reg"] = valid["is_abstain"].astype(int)
    valid["selective_error_reg"] = pd.to_numeric(valid["selective_error"], errors="coerce")
    fit_one("Model 1: raw_error ~ proximity + controls", "raw_error_reg", valid)
    fit_one("Model 2: is_abstain ~ proximity + controls", "is_abstain_reg", valid)
    predict_rows = valid[valid["is_predict"]].copy()
    if len(predict_rows) == 0:
        lines.extend([
            "## Optional selective_error regression",
            "",
            "Skipped: zero predict frames, so selective_error is unavailable.",
            "",
        ])
    else:
        fit_one("Optional selective_error regression", "selective_error_reg", predict_rows)
    return "\n".join(lines)


def skipped_regression_summary(reason: str = "skipped by user flag") -> str:
    return "\n".join([
        "# Logistic Regression Summary",
        "",
        f"Regression diagnostics skipped: {reason}.",
        "",
    ])


def write_figures(curves: pd.DataFrame, output_dir: Path, bin_order: list[str]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    aggregate = _aggregate_curves_for_plot(curves, bin_order)

    def three_panel(path: Path, metrics: list[tuple[str, str]], empty_message: str | None = None) -> None:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 7), sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        if empty_message is not None:
            fig.text(0.5, 0.5, empty_message, ha="center", va="center", fontsize=12)
            for axis in axes:
                axis.axis("off")
        else:
            x = np.arange(len(aggregate))
            for axis, (column, ylabel) in zip(axes, metrics):
                axis.plot(x, aggregate[column], marker="o")
                axis.set_ylabel(ylabel)
                axis.grid(True, alpha=0.3)
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels(aggregate["distance_bin"].astype(str), rotation=30, ha="right")
            axes[-1].set_xlabel("distance_bin")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    three_panel(
        output_dir / "transition_three_line_raw.png",
        [
            ("raw_error_rate", "raw error"),
            ("abstain_rate", "abstain"),
            ("mean_max_prob", "max prob"),
        ],
    )
    if curves["selective_error_rate"].dropna().empty:
        three_panel(
            output_dir / "transition_three_line_selective.png",
            [
                ("selective_error_rate", "selective error"),
                ("abstain_rate", "abstain"),
                ("mean_max_prob", "max prob"),
            ],
            empty_message="No predict frames; selective error is unavailable.",
        )
    else:
        three_panel(
            output_dir / "transition_three_line_selective.png",
            [
                ("selective_error_rate", "selective error"),
                ("abstain_rate", "abstain"),
                ("mean_max_prob", "max prob"),
            ],
        )
    three_panel(
        output_dir / "confidence_vs_transition.png",
        [
            ("mean_max_prob", "max prob"),
            ("mean_entropy", "entropy"),
            ("mean_top2_margin", "top-2 margin"),
        ],
    )
    return paths


def _aggregate_curves_for_plot(curves: pd.DataFrame, bin_order: list[str]) -> pd.DataFrame:
    if len(curves) == 0:
        return pd.DataFrame({"distance_bin": bin_order})
    frame = curves.copy()
    frame["distance_bin"] = pd.Categorical(frame["distance_bin"], categories=bin_order, ordered=True)
    numeric_cols = [
        "raw_error_rate",
        "selective_error_rate",
        "abstain_rate",
        "wait_rate",
        "predict_rate",
        "mean_max_prob",
        "mean_entropy",
        "mean_top2_margin",
        "mean_proximity",
    ]
    out = frame.groupby("distance_bin", observed=False)[numeric_cols].mean().reset_index()
    return out.sort_values("distance_bin").reset_index(drop=True)


def write_summary_readable(
    path: Path,
    *,
    input_path: Path,
    output_dir: Path,
    frame: pd.DataFrame,
    atm: pd.DataFrame,
    near_window: float,
    far_window: float,
    tau: float,
    bins: list[float],
) -> None:
    total_frames = int(len(frame))
    total_videos = int(frame["video_id"].nunique())
    valid_distance = frame["distance_to_transition"].notna()
    videos_with_transitions = int(frame.loc[valid_distance, "video_id"].nunique())
    frames_with_valid_distance = int(valid_distance.sum())
    excluded_videos = total_videos - videos_with_transitions
    excluded_frames = total_frames - frames_with_valid_distance
    counts = frame["decision"].value_counts().to_dict()
    predict_frames = int(frame["is_predict"].sum())
    main = _qualitative_summary(atm)
    lines = [
        "# v0.9 Transition Misalignment Diagnostic Summary",
        "",
        "- version: v0.9_transition_misalignment_diagnostic",
        f"- input_path: {input_path}",
        f"- output_dir: {output_dir}",
        f"- total_frames: {total_frames}",
        f"- total_videos: {total_videos}",
        f"- videos_with_transitions: {videos_with_transitions}",
        f"- videos_without_transitions_excluded: {excluded_videos}",
        f"- frames_with_valid_transition_distance: {frames_with_valid_distance}",
        f"- frames_excluded_no_transition_video: {excluded_frames}",
        f"- decision_counts: {counts}",
        f"- near_window: {near_window:g}",
        f"- far_window: {far_window:g}",
        f"- tau: {tau:g}",
        f"- bins: {', '.join(_fmt_bin(value) for value in bins)}",
        "",
        "## Qualitative Sanity Check",
        "",
        f"- raw TEG: {main['TEG_raw']}",
        f"- TAG: {main['TAG']}",
        f"- confidence_gap: {main['confidence_gap']}",
        "",
    ]
    if predict_frames == 0:
        lines.extend([
            "## Warning",
            "",
            "The input contains zero predict frames. Selective-error analysis and selective-error regression are unavailable.",
            "",
        ])
    elif predict_frames < 100:
        lines.extend([
            "## Warning",
            "",
            f"The input contains only {predict_frames} predict frames. Selective-error analysis may be unstable.",
            "",
        ])
    lines.extend([
        "This diagnostic can be useful for smoke validation, but smoke runs are not scientific evidence.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _qualitative_summary(atm: pd.DataFrame) -> dict[str, str]:
    def classify(column: str) -> str:
        if column not in atm.columns or atm[column].dropna().empty:
            return "unavailable"
        value = float(atm[column].dropna().mean())
        if value > 1e-6:
            return f"positive (mean={value:.6f})"
        if value < -1e-6:
            return f"negative (mean={value:.6f})"
        return f"near 0 (mean={value:.6f})"

    return {
        "TEG_raw": classify("TEG_raw"),
        "TAG": classify("TAG"),
        "confidence_gap": classify("confidence_gap"),
    }


def analyze_transition_misalignment(
    input_path: str | Path,
    output_dir: str | Path,
    near_window: float = 8.0,
    far_window: float = 32.0,
    tau: float = 8.0,
    bins: str | Iterable[float] | None = None,
    no_figures: bool = False,
    skip_regression: bool = False,
) -> dict[str, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_bins = parse_bins(bins)
    raw = read_input(input_path)
    frame = add_transition_features(raw, tau=tau, bins=parsed_bins)
    curves = summarize_transition_curves(frame, parsed_bins)
    atm = summarize_atm(frame, near_window=near_window, far_window=far_window)

    frame_path = output_dir / "frame_with_transition_features.csv"
    curves_path = output_dir / "transition_curves.csv"
    atm_path = output_dir / "atm_summary.csv"
    regression_path = output_dir / "logistic_regression_summary.md"
    summary_path = output_dir / "summary_readable.md"

    frame.to_csv(frame_path, index=False)
    curves.to_csv(curves_path, index=False)
    atm.to_csv(atm_path, index=False)
    if skip_regression:
        regression_path.write_text(skipped_regression_summary(), encoding="utf-8")
    else:
        regression_path.write_text(fit_logistic_regressions(frame), encoding="utf-8")
    figure_paths = [] if no_figures else write_figures(curves, output_dir, distance_bin_order(parsed_bins))
    write_summary_readable(
        summary_path,
        input_path=input_path,
        output_dir=output_dir,
        frame=frame,
        atm=atm,
        near_window=near_window,
        far_window=far_window,
        tau=tau,
        bins=parsed_bins,
    )
    return {
        "frame_with_transition_features": frame_path,
        "transition_curves": curves_path,
        "atm_summary": atm_path,
        "logistic_regression_summary": regression_path,
        "summary_readable": summary_path,
        **{f"figure_{idx}": path for idx, path in enumerate(figure_paths, start=1)},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze transition-abstention misalignment from v0.8 per-frame logs.")
    parser.add_argument("--input", required=True, help="v0.8 per-frame JSONL or CSV")
    parser.add_argument("--output-dir", required=True, help="directory for v0.9 diagnostic outputs")
    parser.add_argument("--near-window", type=float, default=8.0)
    parser.add_argument("--far-window", type=float, default=32.0)
    parser.add_argument("--tau", type=float, default=8.0)
    parser.add_argument("--bins", default="0,4,8,16,32,999999")
    parser.add_argument("--no-figures", action="store_true", help="skip PNG figure generation")
    parser.add_argument("--skip-regression", action="store_true", help="skip optional statsmodels logistic regressions")
    args = parser.parse_args(argv)
    try:
        outputs = analyze_transition_misalignment(
            input_path=args.input,
            output_dir=args.output_dir,
            near_window=args.near_window,
            far_window=args.far_window,
            tau=args.tau,
            bins=args.bins,
            no_figures=args.no_figures,
            skip_regression=args.skip_regression,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports failures clearly.
        print(f"analysis_failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
