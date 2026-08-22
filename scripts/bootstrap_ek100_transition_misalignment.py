from __future__ import annotations

import argparse
import csv
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SUMMARY_NAME = "analyzer_atm_summary_all.csv"
JSONL_RE = re.compile(r"per_frame_(?P<policy>.+)_q(?P<quantile>.+)\.jsonl$")


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    near_window: float
    far_window: float


ANALYSES = (
    AnalysisSpec("primary_near8_far32", near_window=8.0, far_window=32.0),
    AnalysisSpec("sensitivity_near4_far16", near_window=4.0, far_window=16.0),
)

COUNT_COLUMNS = (
    "near_frames",
    "far_frames",
    "raw_near_sum",
    "raw_near_count",
    "raw_far_sum",
    "raw_far_count",
    "abs_near_sum",
    "abs_near_count",
    "abs_far_sum",
    "abs_far_count",
    "selective_near_sum",
    "selective_near_count",
    "selective_far_sum",
    "selective_far_count",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-cluster bootstrap for v1.10 EK100 transition misalignment.")
    parser.add_argument("--input", required=True, help="v1.10 bootstrap input tar.gz containing per_frame/*.jsonl")
    parser.add_argument("--output-csv", required=True, help="lightweight bootstrap summary CSV")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args(argv)


def open_tar_members(tar_path: Path) -> tuple[list[str], pd.DataFrame | None]:
    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
        jsonl_names = sorted([name for name in names if name.endswith(".jsonl")])
        summary_name = next((name for name in names if name.endswith(SUMMARY_NAME)), None)
        summary = None
        if summary_name is not None:
            handle = tf.extractfile(summary_name)
            if handle is not None:
                summary = pd.read_csv(handle)
    return jsonl_names, summary


def read_jsonl_member(tar_path: Path, member_name: str) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:gz") as tf:
        handle = tf.extractfile(member_name)
        if handle is None:
            raise FileNotFoundError(f"Cannot read tar member: {member_name}")
        frame = pd.read_json(handle, lines=True)
    required = {
        "video_id",
        "frame_idx",
        "gt_label",
        "pred_label",
        "decision",
        "threshold",
        "budget",
        "model",
        "policy",
        "shift_type",
        "raw_error",
        "selective_error",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{member_name} is missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["video_id"] = out["video_id"].astype(str)
    out["frame_idx"] = out["frame_idx"].astype(np.int64)
    out["gt_label"] = out["gt_label"].astype(np.int64)
    out["pred_label"] = out["pred_label"].astype(np.int64)
    out["decision"] = out["decision"].astype(str)
    out["raw_error"] = pd.to_numeric(out["raw_error"], errors="coerce")
    out["selective_error"] = pd.to_numeric(out["selective_error"], errors="coerce")
    return out.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)


def distances_to_transitions(group: pd.DataFrame) -> np.ndarray:
    labels = group["gt_label"].to_numpy(dtype=np.int64)
    frame_idx = group["frame_idx"].to_numpy(dtype=np.float64)
    transitions = np.zeros(len(group), dtype=bool)
    if len(labels) > 1:
        transitions[1:] = labels[1:] != labels[:-1]
    transition_frames = frame_idx[transitions]
    if transition_frames.size == 0:
        return np.full(len(group), np.nan, dtype=np.float64)
    return np.min(np.abs(frame_idx[:, None] - transition_frames[None, :]), axis=1)


def per_video_counts(frame: pd.DataFrame, spec: AnalysisSpec) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for video_id, group in frame.groupby("video_id", sort=False):
        distances = distances_to_transitions(group)
        valid = np.isfinite(distances)
        near = valid & (distances <= spec.near_window)
        far = valid & (distances > spec.far_window)
        raw = group["raw_error"].to_numpy(dtype=np.float64)
        selective = group["selective_error"].to_numpy(dtype=np.float64)
        abstain = group["decision"].eq("abstain").to_numpy(dtype=np.float64)
        rows.append({
            "video_id": str(video_id),
            "near_frames": float(np.sum(near)),
            "far_frames": float(np.sum(far)),
            "raw_near_sum": _nan_sum(raw[near]),
            "raw_near_count": float(np.sum(np.isfinite(raw[near]))),
            "raw_far_sum": _nan_sum(raw[far]),
            "raw_far_count": float(np.sum(np.isfinite(raw[far]))),
            "abs_near_sum": float(np.sum(abstain[near])),
            "abs_near_count": float(np.sum(near)),
            "abs_far_sum": float(np.sum(abstain[far])),
            "abs_far_count": float(np.sum(far)),
            "selective_near_sum": _nan_sum(selective[near]),
            "selective_near_count": float(np.sum(np.isfinite(selective[near]))),
            "selective_far_sum": _nan_sum(selective[far]),
            "selective_far_count": float(np.sum(np.isfinite(selective[far]))),
        })
    return pd.DataFrame(rows)


def _nan_sum(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(valid.sum()) if valid.size else 0.0


def metrics_from_counts(counts: pd.DataFrame | dict[str, float]) -> dict[str, float]:
    if isinstance(counts, pd.DataFrame):
        totals = {column: float(counts[column].sum()) for column in COUNT_COLUMNS}
    else:
        totals = {column: float(counts[column]) for column in COUNT_COLUMNS}
    err_near_raw = _ratio(totals["raw_near_sum"], totals["raw_near_count"])
    err_far_raw = _ratio(totals["raw_far_sum"], totals["raw_far_count"])
    teg_raw = _difference(err_near_raw, err_far_raw)
    abs_near = _ratio(totals["abs_near_sum"], totals["abs_near_count"])
    abs_far = _ratio(totals["abs_far_sum"], totals["abs_far_count"])
    tag = _difference(abs_near, abs_far)
    err_near_selective = _ratio(totals["selective_near_sum"], totals["selective_near_count"])
    err_far_selective = _ratio(totals["selective_far_sum"], totals["selective_far_count"])
    teg_selective = _difference(err_near_selective, err_far_selective)
    return {
        "near_frames": totals["near_frames"],
        "far_frames": totals["far_frames"],
        "err_near_raw": err_near_raw,
        "err_far_raw": err_far_raw,
        "TEG_raw": teg_raw,
        "Error_OR_raw": _odds_ratio_from_counts(totals["raw_near_sum"], totals["raw_near_count"], totals["raw_far_sum"], totals["raw_far_count"]),
        "abs_near": abs_near,
        "abs_far": abs_far,
        "TAG": tag,
        "Abstain_OR": _odds_ratio_from_counts(totals["abs_near_sum"], totals["abs_near_count"], totals["abs_far_sum"], totals["abs_far_count"]),
        "ATM_raw": _difference(teg_raw, tag),
        "err_near_selective": err_near_selective,
        "err_far_selective": err_far_selective,
        "TEG_selective": teg_selective,
        "Error_OR_selective": _odds_ratio_from_counts(totals["selective_near_sum"], totals["selective_near_count"], totals["selective_far_sum"], totals["selective_far_count"]),
        "ATM_selective": _difference(teg_selective, tag),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _difference(left: float, right: float) -> float:
    if math.isnan(left) or math.isnan(right):
        return float("nan")
    return float(left - right)


def _odds_ratio_from_counts(true_near: float, count_near: float, true_far: float, count_far: float) -> float:
    if count_near <= 0 or count_far <= 0:
        return float("nan")
    a = float(true_near)
    b = float(count_near - true_near)
    c = float(true_far)
    d = float(count_far - true_far)
    if min(a, b, c, d) == 0.0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    return float((a / b) / (c / d))


def bootstrap_counts(counts: pd.DataFrame, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = counts[list(COUNT_COLUMNS)].to_numpy(dtype=np.float64)
    n_videos = values.shape[0]
    samples = rng.integers(0, n_videos, size=(int(n_boot), n_videos))
    boot = values[samples].sum(axis=1)
    output: dict[str, list[float]] = {key: [] for key in [
        "TEG_raw",
        "Error_OR_raw",
        "ATM_raw",
        "Abstain_OR",
        "TEG_selective",
        "Error_OR_selective",
        "ATM_selective",
    ]}
    for row in boot:
        metrics = metrics_from_counts(dict(zip(COUNT_COLUMNS, row.tolist())))
        for key in output:
            output[key].append(float(metrics[key]))
    return {key: np.asarray(vals, dtype=np.float64) for key, vals in output.items()}


def ci(values: np.ndarray) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(clean, [2.5, 97.5])
    return float(low), float(high)


def member_stem(member_name: str) -> str:
    return Path(member_name).stem


def run_name_from_stem(stem: str) -> tuple[str, float]:
    match = JSONL_RE.match(stem + ".jsonl")
    if not match:
        return stem, float("nan")
    quantile = match.group("quantile").replace("p", ".")
    return match.group("policy"), float(quantile)


def expected_row(summary: pd.DataFrame | None, analysis: str, per_frame_name: str) -> pd.Series | None:
    if summary is None:
        return None
    if "analysis" in summary.columns and "per_frame_name" in summary.columns:
        rows = summary[(summary["analysis"].eq(analysis)) & (summary["per_frame_name"].eq(per_frame_name))]
    elif "scope" in summary.columns and "run" in summary.columns:
        rows = summary[(summary["scope"].eq(analysis)) & (summary["run"].eq(per_frame_name.replace("per_frame_", "")))]
    else:
        return None
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def validate_point_estimate(point: dict[str, float], expected: pd.Series | None, tolerance: float, label: str) -> None:
    if expected is None:
        return
    checks = ["near_frames", "far_frames", "err_near_raw", "err_far_raw", "TEG_raw", "Error_OR_raw"]
    for key in checks:
        if key not in expected:
            continue
        expected_value = float(expected[key])
        actual_value = float(point[key])
        if key in {"near_frames", "far_frames"}:
            if int(round(actual_value)) != int(round(expected_value)):
                raise ValueError(f"{label}: {key} mismatch: actual {actual_value}, expected {expected_value}")
        elif abs(actual_value - expected_value) > tolerance:
            raise ValueError(f"{label}: {key} mismatch: actual {actual_value}, expected {expected_value}")


def output_row(
    *,
    analysis: str,
    per_frame_name: str,
    policy: str,
    quantile: float,
    threshold: float,
    point: dict[str, float],
    boot: dict[str, np.ndarray],
    n_boot: int,
    n_videos: int,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "analysis": analysis,
        "per_frame_name": per_frame_name,
        "policy": policy,
        "quantile": quantile,
        "threshold": float(threshold),
        "near_frames": int(round(point["near_frames"])),
        "far_frames": int(round(point["far_frames"])),
        "err_near_raw": point["err_near_raw"],
        "err_far_raw": point["err_far_raw"],
        "point_TEG_raw": point["TEG_raw"],
        "point_Error_OR_raw": point["Error_OR_raw"],
        "point_ATM_raw": point["ATM_raw"],
        "point_Abstain_OR": point["Abstain_OR"],
        "point_TEG_selective": point["TEG_selective"],
        "point_Error_OR_selective": point["Error_OR_selective"],
        "point_ATM_selective": point["ATM_selective"],
        "n_boot": int(n_boot),
        "n_videos": int(n_videos),
    }
    for metric in [
        "TEG_raw",
        "Error_OR_raw",
        "ATM_raw",
        "Abstain_OR",
        "TEG_selective",
        "Error_OR_selective",
        "ATM_selective",
    ]:
        low, high = ci(boot[metric])
        row[f"ci_low_{metric}"] = low
        row[f"ci_high_{metric}"] = high
    return row


def write_csv(path: Path, rows: Iterable[dict[str, float | int | str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    tar_path = Path(args.input)
    output_path = Path(args.output_csv)
    jsonl_names, summary = open_tar_members(tar_path)
    if len(jsonl_names) != 15:
        raise ValueError(f"Expected 15 per-frame JSONL files, found {len(jsonl_names)}")
    rows: list[dict[str, float | int | str]] = []
    for member_name in jsonl_names:
        per_frame_name = member_stem(member_name)
        policy, quantile = run_name_from_stem(per_frame_name)
        print(f"reading {per_frame_name}", flush=True)
        frame = read_jsonl_member(tar_path, member_name)
        threshold = float(frame["threshold"].iloc[0])
        for spec in ANALYSES:
            print(f"  analysis {spec.name}", flush=True)
            counts = per_video_counts(frame, spec)
            point = metrics_from_counts(counts)
            expected = expected_row(summary, spec.name, per_frame_name)
            validate_point_estimate(point, expected, float(args.tolerance), f"{spec.name}/{per_frame_name}")
            boot = bootstrap_counts(counts, int(args.n_boot), int(args.seed))
            rows.append(output_row(
                analysis=spec.name,
                per_frame_name=per_frame_name,
                policy=policy,
                quantile=quantile,
                threshold=threshold,
                point=point,
                boot=boot,
                n_boot=int(args.n_boot),
                n_videos=int(counts["video_id"].nunique()),
            ))
        del frame
    write_csv(output_path, rows)
    print(f"wrote {output_path}")


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
