from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ALIASES = {
    "video_id": ["video_id", "video", "vid", "session_video", "clip_id"],
    "frame_idx": ["frame_idx", "frame_index", "t", "idx", "frame"],
    "gt_label": ["gt_label", "y_true", "label", "target", "gt", "true_label", "primary_label"],
    "pred_label": ["pred_label", "y_pred", "pred", "prediction"],
    "max_prob": ["max_prob", "p_hat", "confidence", "conf"],
    "entropy": ["entropy", "predictive_entropy"],
    "top2_margin": ["top2_margin", "margin", "top_2_margin"],
    "raw_error": ["raw_error", "is_error", "error"],
    "decision": ["decision"],
    "budget": ["budget"],
    "model": ["model"],
    "policy": ["policy"],
    "threshold": ["threshold"],
    "shift_type": ["shift_type"],
    "split": ["split"],
    "active_label_set": ["active_label_set", "active_labels"],
}
REQUIRED = (
    "video_id",
    "frame_idx",
    "gt_label",
    "pred_label",
    "max_prob",
    "entropy",
    "top2_margin",
    "raw_error",
    "decision",
    "budget",
    "model",
    "policy",
    "threshold",
    "shift_type",
    "split",
)
METHODS = (
    "global_split_conformal",
    "oracle_mondrian",
    "estimated_regime_mondrian",
    "aci_global",
    "aci_estimated_regime",
)
BOOT_METRICS = (
    "coverage",
    "selective_risk",
    "transition_risk",
    "stable_risk",
    "TEG",
    "Error_OR",
    "miscoverage",
    "delay",
)
GAP_METRICS = ("selective_risk", "transition_risk", "TEG", "Error_OR", "miscoverage")
SCORES = ("one_minus_max_prob", "entropy", "one_minus_top2_margin")
OPTIONAL = ("active_label_set",)


class LadderError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    path: Path
    member: str | None = None

    @property
    def name(self):
        return f"{self.path}::{self.member}" if self.member else str(self.path)


class FenwickQuantile:
    """Exact empirical online quantiles; counts contain only calibration/past frames."""

    def __init__(self, support):
        values = np.asarray(support, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise LadderError("Cannot initialize an empty online quantile")
        self.values = np.unique(values)
        self.tree = np.zeros(len(self.values) + 1, dtype=np.int64)
        self.count = 0

    def add(self, value):
        if not np.isfinite(value):
            return
        index = int(np.searchsorted(self.values, value)) + 1
        while index < len(self.tree):
            self.tree[index] += 1
            index += index & -index
        self.count += 1

    def add_many(self, values):
        for value in values:
            self.add(value)

    def quantile(self, probability):
        if not self.count:
            raise LadderError("Online quantile history is empty")
        rank = min(max(int(math.ceil(probability * self.count)), 1), self.count)
        index = 0
        bit = 1 << (len(self.tree).bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate < len(self.tree) and self.tree[candidate] < rank:
                index = candidate
                rank -= int(self.tree[candidate])
            bit >>= 1
        return float(self.values[min(index, len(self.values) - 1)])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-hoc scalar-score reliability ladder; old policy decisions are ignored."
    )
    parser.add_argument("--input", nargs="+", help=".tar.gz, directory, JSONL, or glob")
    parser.add_argument(
        "--calibration-input",
        nargs="+",
        help="Official calibration JSONL input; requires --evaluation-input",
    )
    parser.add_argument(
        "--evaluation-input",
        nargs="+",
        help="Official evaluation JSONL input; requires --calibration-input",
    )
    parser.add_argument("--member", help="Exact JSONL member path inside a tar archive")
    parser.add_argument("--member-contains", help="Select one tar JSONL member by substring")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-csv")
    parser.add_argument("--summary-md")
    parser.add_argument("--curve-csv")
    parser.add_argument("--near", type=float, default=8.0)
    parser.add_argument("--far", type=float, default=32.0)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--score", choices=SCORES, default="one_minus_max_prob")
    parser.add_argument("--calib-frac", type=float, default=0.50)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--expected-videos", type=int)
    parser.add_argument("--expected-calibration-videos", type=int)
    parser.add_argument("--delay-window", type=int, default=64)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--ema-halflife", type=float, default=8.0)
    parser.add_argument("--flip-rate-thresh", type=float, default=0.25)
    parser.add_argument("--entropy-z-thresh", type=float, default=1.0)
    parser.add_argument("--maxprob-z-thresh", type=float, default=1.0)
    parser.add_argument("--aci-delta", type=float, default=0.01)
    parser.add_argument(
        "--aci-delta-grid",
        nargs="+",
        type=float,
        default=[0.001, 0.005, 0.01, 0.05],
    )
    parser.add_argument(
        "--curve-coverages",
        nargs="+",
        type=float,
        default=[0.70, 0.80, 0.90],
    )
    parser.add_argument("--dry-run-schema", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.coverage < 1:
        parser.error("--coverage must be between 0 and 1")
    if not 0 < args.calib_frac < 1:
        parser.error("--calib-frac must be between 0 and 1")
    if args.near >= args.far:
        parser.error("--near must be smaller than --far")
    if args.member and args.member_contains:
        parser.error("--member and --member-contains are mutually exclusive")
    separate_inputs = args.calibration_input is not None or args.evaluation_input is not None
    if separate_inputs and not (args.calibration_input and args.evaluation_input):
        parser.error("--calibration-input and --evaluation-input must be provided together")
    if separate_inputs and args.input:
        parser.error("--input cannot be combined with separate calibration/evaluation inputs")
    if not separate_inputs and not args.input:
        parser.error("provide --input or separate --calibration-input/--evaluation-input")
    if separate_inputs and (args.member or args.member_contains):
        parser.error("archive member selectors are only supported with --input")
    if args.expected_videos is None:
        defaults = {"thumos14": 211, "ek100": 133}
        args.expected_videos = defaults.get(args.dataset.lower())
    return args


def _paths(specs):
    paths = []
    for spec in specs:
        path = Path(spec)
        if path.exists():
            paths.extend(sorted(path.rglob("*.jsonl"))) if path.is_dir() else paths.append(path)
        else:
            paths.extend(Path(item) for item in glob.glob(spec, recursive=True))
    result = sorted({path.resolve() for path in paths})
    if not result:
        raise FileNotFoundError(f"No inputs matched: {specs}")
    return result


def _sources(paths, member=None, member_contains=None):
    result = []
    for path in paths:
        lower = path.name.lower()
        if lower.endswith((".tar.gz", ".tgz", ".tar")):
            mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(path, mode) as archive:
                members = sorted(
                    member.name
                    for member in archive.getmembers()
                    if member.isfile() and member.name.lower().endswith((".jsonl", ".ndjson"))
                )
            if member:
                members = [name for name in members if name == member]
            elif member_contains:
                members = [name for name in members if member_contains in name]
            result.extend(Source(path, member) for member in members)
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            result.append(Source(path))
        else:
            raise LadderError(f"Unsupported input: {path}")
    if not result:
        selector = member or member_contains
        suffix = f" matching selector {selector!r}" if selector else ""
        raise LadderError(f"No JSONL streams found{suffix}")
    if (member or member_contains) and len(result) != 1:
        raise LadderError(
            f"Member selector must resolve to exactly one JSONL stream; found {len(result)}: "
            f"{[source.name for source in result]}"
        )
    return result


def _open(source):
    if source.member is None:
        return source.path.open("rb"), None
    mode = "r:gz" if source.path.name.lower().endswith((".tar.gz", ".tgz")) else "r:"
    archive = tarfile.open(source.path, mode)
    handle = archive.extractfile(source.member)
    if handle is None:
        archive.close()
        raise LadderError(f"Cannot read {source.name}")
    return handle, archive


def _preview(source, limit=5):
    handle, archive = _open(source)
    rows = []
    try:
        for raw in handle:
            if len(rows) >= limit:
                break
            if raw.strip():
                rows.append(json.loads(raw))
    finally:
        handle.close()
        if archive:
            archive.close()
    if not rows:
        raise LadderError(f"No rows in {source.name}")
    return rows


def detect_schema(keys):
    available = set(keys)
    return {
        name: next(alias for alias in aliases if alias in available)
        for name, aliases in ALIASES.items()
        if any(alias in available for alias in aliases)
    }


def _read(source, mapping):
    handle, archive = _open(source)
    chunks = []
    columns = list(dict.fromkeys(mapping.values()))
    try:
        for index, chunk in enumerate(pd.read_json(handle, lines=True, chunksize=100_000), start=1):
            missing = set(columns) - set(chunk)
            if missing:
                raise LadderError(f"Chunk {index} is missing {sorted(missing)}")
            chunks.append(chunk[columns].copy())
            print(f"  loaded chunk {index}: {len(chunk):,} rows", flush=True)
    finally:
        handle.close()
        if archive:
            archive.close()
    return pd.concat(chunks, ignore_index=True)


def read_jsonl_or_tar(
    specs,
    dry_run_schema=False,
    *,
    member=None,
    member_contains=None,
    expected_videos=None,
):
    sources = _sources(_paths(specs), member=member, member_contains=member_contains)
    source = sources[0]
    preview = _preview(source)
    keys = sorted({key for row in preview for key in row})
    mapping = detect_schema(keys)
    missing = [field for field in REQUIRED if field not in mapping]
    print(f"source: {source.name}")
    print(f"available_keys: {keys}")
    print(f"detected_schema: {mapping}")
    print(f"missing_required: {missing}")
    if missing:
        raise LadderError(f"Missing required fields {missing}; available keys: {keys}")
    if dry_run_schema:
        return None
    if len(sources) > 1:
        raise LadderError(
            "Multiple JSONL streams matched. Select one canonical stream with "
            "--member/--member-contains or pass one JSONL file."
        )
    raw = _read(source, mapping)
    frame = pd.DataFrame(
        {
            name: raw[column]
            for name, column in mapping.items()
            if name in REQUIRED or name in OPTIONAL
        }
    )
    frame["video_id"] = frame["video_id"].astype(str)
    frame["frame_idx"] = pd.to_numeric(frame["frame_idx"], errors="raise").astype(np.int64)
    frame["gt_label"] = frame["gt_label"].astype(str)
    frame["pred_label"] = frame["pred_label"].astype(str)
    if "active_label_set" in frame:
        frame["active_label_set"] = frame["active_label_set"].map(
            canonical_active_label_set
        )
    for field in ("max_prob", "entropy", "top2_margin", "raw_error"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if frame[["max_prob", "entropy", "top2_margin", "raw_error"]].isna().any(axis=None):
        raise LadderError("Required scalar fields contain missing or nonnumeric values")
    if not frame["raw_error"].isin([0, 1]).all():
        raise LadderError("raw_error must be binary")
    duplicate = frame.duplicated(["video_id", "frame_idx"], keep=False)
    if duplicate.any():
        checked = [
            "gt_label",
            "pred_label",
            "max_prob",
            "entropy",
            "top2_margin",
            "raw_error",
        ]
        if "active_label_set" in frame:
            checked.append("active_label_set")
        conflicts = (
            frame.loc[duplicate]
            .groupby(["video_id", "frame_idx"], dropna=False)[checked]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            raise LadderError(
                f"Found {int(conflicts.sum())} duplicate video/frame keys with conflicting "
                "model outputs; refusing to silently merge streams"
            )
        print(f"deduplicating {int(duplicate.sum() - conflicts.sum()):,} repeated rows")
        frame = frame.drop_duplicates(["video_id", "frame_idx"], keep="first")
    frame = frame.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    video_count = int(frame.video_id.nunique())
    if expected_videos is not None and video_count != int(expected_videos):
        raise LadderError(
            f"Expected {int(expected_videos)} videos for formal {str(frame.iloc[0]['model'])} "
            f"input, found {video_count}"
        )
    frame.attrs["source"] = source.name
    frame.attrs["available_keys"] = keys
    frame.attrs["schema_mapping"] = mapping
    frame.attrs["video_count"] = video_count
    return frame


def canonical_active_label_set(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"[]", "()", "{}"}:
            return ()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in text.strip("[](){}").split(",") if item.strip()]
        value = parsed
    if isinstance(value, (set, frozenset, list, tuple, np.ndarray)):
        return tuple(sorted({int(item) for item in value}))
    return (int(value),)


def _transition_mask(labels):
    labels = np.asarray(labels)
    mask = np.zeros(len(labels), dtype=bool)
    if len(labels) > 1:
        mask[1:] = labels[1:] != labels[:-1]
    return mask


def _distance(frame_idx, labels):
    transitions = frame_idx[_transition_mask(labels)]
    if not len(transitions):
        return np.full(len(frame_idx), np.inf)
    insertion = np.searchsorted(transitions, frame_idx)
    left = transitions[np.clip(insertion - 1, 0, len(transitions) - 1)]
    right = transitions[np.clip(insertion, 0, len(transitions) - 1)]
    return np.minimum(np.abs(frame_idx - left), np.abs(frame_idx - right)).astype(float)


def add_oracle_regime(frame, near, far, dataset=None):
    outputs = []
    for _, group in frame.groupby("video_id", sort=False):
        group = group.sort_values("frame_idx").copy()
        if str(dataset).lower() == "tvseries" and "active_label_set" in group:
            oracle_states = group["active_label_set"].to_numpy(object)
        else:
            oracle_states = group["gt_label"].to_numpy()
        distance = _distance(group["frame_idx"].to_numpy(float), oracle_states)
        transition_event = _transition_mask(oracle_states)
        regime = np.full(len(group), -1, dtype=np.int8)
        regime[distance <= near] = 1
        regime[distance > far] = 0
        group["distance_to_transition"] = distance
        group["gt_transition_event"] = transition_event
        group["oracle_regime"] = regime
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)


def causal_transition_regime(
    group,
    *,
    window=8,
    ema_halflife=8.0,
    flip_rate_thresh=0.25,
    entropy_z_thresh=1.0,
    maxprob_z_thresh=1.0,
):
    g = group.sort_values("frame_idx")
    pred = g["pred_label"].to_numpy()
    maxp = g["max_prob"].to_numpy(float)
    ent = g["entropy"].to_numpy(float)
    n = len(g)
    regime = np.zeros(n, dtype=np.int64)
    if n == 0:
        return regime
    alpha = 1.0 - 0.5 ** (1.0 / max(ema_halflife, 1e-6))
    ema_m, ema_e, var_m, var_e = float(maxp[0]), float(ent[0]), 0.0, 0.0
    flips = np.zeros(n)
    if n > 1:
        flips[1:] = pred[1:] != pred[:-1]
    for t in range(n):
        lo = max(0, t - window + 1)
        flip_rate = float(flips[lo : t + 1].mean())
        zm = (ema_m - maxp[t]) / (np.sqrt(var_m) + 1e-6)
        ze = (ent[t] - ema_e) / (np.sqrt(var_e) + 1e-6)
        regime[t] = int(
            flip_rate >= flip_rate_thresh
            or ze >= entropy_z_thresh
            or zm >= maxprob_z_thresh
        )
        d_m = maxp[t] - ema_m
        ema_m += alpha * d_m
        var_m = (1 - alpha) * (var_m + alpha * d_m * d_m)
        d_e = ent[t] - ema_e
        ema_e += alpha * d_e
        var_e = (1 - alpha) * (var_e + alpha * d_e * d_e)
    return regime


def add_estimated_regime(frame, args):
    outputs = []
    for _, group in frame.groupby("video_id", sort=False):
        group = group.sort_values("frame_idx").copy()
        group["estimated_regime"] = causal_transition_regime(
            group,
            window=args.window,
            ema_halflife=args.ema_halflife,
            flip_rate_thresh=args.flip_rate_thresh,
            entropy_z_thresh=args.entropy_z_thresh,
            maxprob_z_thresh=args.maxprob_z_thresh,
        )
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)


def split_calib_eval_by_video(frame, fraction, seed):
    videos = np.asarray(sorted(frame.video_id.unique()), object)
    if len(videos) < 2:
        raise LadderError("At least two videos are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(videos)
    count = min(max(1, round(fraction * len(videos))), len(videos) - 1)
    calib_ids = set(videos[:count])
    calib = frame[frame.video_id.isin(calib_ids)].copy()
    evaluation = frame[~frame.video_id.isin(calib_ids)].copy()
    print(
        f"cluster split: calibration={calib.video_id.nunique()}, "
        f"evaluation={evaluation.video_id.nunique()} videos"
    )
    return calib, evaluation


def official_calibration_evaluation(calibration, evaluation, dataset=None):
    if len(calibration) == 0 or len(evaluation) == 0:
        raise LadderError("Official calibration and evaluation streams must be non-empty")
    calib_ids = set(calibration.video_id.astype(str).unique())
    eval_ids = set(evaluation.video_id.astype(str).unique())
    overlap = sorted(calib_ids & eval_ids)
    if overlap:
        raise LadderError(
            f"Calibration/evaluation video ids must be disjoint; overlap: {overlap[:5]}"
        )
    if str(dataset).lower() == "tvseries":
        calibration_splits = {
            value.lower() for value in calibration["split"].astype(str).unique()
        }
        evaluation_splits = {
            value.lower() for value in evaluation["split"].astype(str).unique()
        }
        if not calibration_splits.issubset({"val", "validation"}):
            raise LadderError(
                "TVSeries calibration input must contain only val/validation rows; "
                f"found {sorted(calibration_splits)}"
            )
        if evaluation_splits != {"test"}:
            raise LadderError(
                "TVSeries evaluation input must contain only test rows; "
                f"found {sorted(evaluation_splits)}"
            )
    calibration = calibration.copy()
    evaluation = evaluation.copy()
    print(
        f"official split: calibration={len(calib_ids)}, "
        f"evaluation={len(eval_ids)} videos"
    )
    return calibration, evaluation


def _scores(frame, name):
    if name == "one_minus_max_prob":
        return 1 - frame["max_prob"].to_numpy(float)
    if name == "entropy":
        return frame["entropy"].to_numpy(float)
    return 1 - frame["top2_margin"].to_numpy(float)


def _quantile(values, coverage):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise LadderError("Cannot fit threshold on an empty group")
    return float(np.quantile(values, coverage, method="higher"))


def fit_global_threshold(calibration, coverage):
    return _quantile(calibration.score, coverage)


def fit_oracle_mondrian_thresholds(calibration, coverage, global_tau):
    thresholds = {-1: global_tau}
    for regime in (0, 1):
        scores = calibration.loc[calibration.oracle_regime == regime, "score"]
        thresholds[regime] = _quantile(scores, coverage) if len(scores) else global_tau
    return thresholds


def fit_estimated_mondrian_thresholds(calibration, coverage, global_tau):
    thresholds = {}
    for regime in (0, 1):
        scores = calibration.loc[calibration.estimated_regime == regime, "score"]
        thresholds[regime] = _quantile(scores, coverage) if len(scores) else global_tau
    return thresholds


def apply_threshold_policy(evaluation, thresholds, regime_column=None):
    output = evaluation[
        [
            "video_id",
            "frame_idx",
            "raw_error",
            "oracle_regime",
            "gt_transition_event",
        ]
    ].copy()
    tau = (
        np.full(len(evaluation), float(thresholds))
        if regime_column is None
        else evaluation[regime_column].map(thresholds).to_numpy(float)
    )
    output["predicted"] = evaluation.score.to_numpy(float) <= tau
    return output


def _aci_row(row, predicted):
    return {
        "video_id": row.video_id,
        "frame_idx": row.frame_idx,
        "raw_error": row.raw_error,
        "oracle_regime": row.oracle_regime,
        "gt_transition_event": row.gt_transition_event,
        "predicted": predicted,
    }


def run_aci_global(calibration, evaluation, coverage, delta):
    support = np.concatenate([calibration.score.to_numpy(), evaluation.score.to_numpy()])
    history = FenwickQuantile(support)
    history.add_many(calibration.score)
    alpha_target, alpha = 1 - coverage, 1 - coverage
    rows = []
    for row in evaluation.sort_values(["video_id", "frame_idx"]).itertuples(index=False):
        predicted = bool(row.score <= history.quantile(1 - alpha))
        rows.append(_aci_row(row, predicted))
        abstained = int(not predicted)
        alpha = float(
            np.clip(
                alpha + delta * (alpha_target - abstained),
                0.001,
                0.999,
            )
        )
        history.add(row.score)
    return pd.DataFrame(rows)


def run_aci_estimated_regime(calibration, evaluation, coverage, delta):
    support = np.concatenate([calibration.score.to_numpy(), evaluation.score.to_numpy()])
    histories = {regime: FenwickQuantile(support) for regime in (0, 1)}
    for regime in (0, 1):
        values = calibration.loc[calibration.estimated_regime == regime, "score"]
        histories[regime].add_many(values if len(values) else calibration.score)
    alpha_target = 1 - coverage
    alpha = {0: alpha_target, 1: alpha_target}
    rows = []
    for row in evaluation.sort_values(["video_id", "frame_idx"]).itertuples(index=False):
        regime = int(row.estimated_regime)
        predicted = bool(row.score <= histories[regime].quantile(1 - alpha[regime]))
        rows.append(_aci_row(row, predicted))
        abstained = int(not predicted)
        alpha[regime] = float(
            np.clip(
                alpha[regime] + delta * (alpha_target - abstained),
                0.001,
                0.999,
            )
        )
        histories[regime].add(row.score)
    return pd.DataFrame(rows)


def select_aci_delta(calibration, coverage, delta_grid, seed, *, estimated=False):
    """Choose a conservative ACI step using calibration videos only."""
    videos = np.asarray(sorted(calibration.video_id.unique()), object)
    if len(videos) < 4:
        return float(min(delta_grid))
    rng = np.random.default_rng(seed + (1 if estimated else 0))
    rng.shuffle(videos)
    split = max(1, len(videos) // 2)
    history_ids = set(videos[:split])
    history = calibration[calibration.video_id.isin(history_ids)]
    validation = calibration[~calibration.video_id.isin(history_ids)]
    candidates = []
    for delta in sorted(set(float(value) for value in delta_grid)):
        outcomes = (
            run_aci_estimated_regime(history, validation, coverage, delta)
            if estimated
            else run_aci_global(history, validation, coverage, delta)
        )
        achieved = float(outcomes.predicted.mean())
        candidates.append((abs(achieved - coverage), delta, achieved))
    _, selected, achieved = min(candidates)
    label = "estimated-regime" if estimated else "global"
    print(
        f"ACI {label} delta selected on calibration only: {selected:g} "
        f"(validation coverage={achieved:.4f}, target={coverage:.4f})",
        flush=True,
    )
    return selected


def _mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def _transition_delays(outcomes, delay_window):
    delays = []
    for _, group in outcomes.groupby("video_id", sort=False):
        group = group.sort_values("frame_idx")
        frame_idx = group.frame_idx.to_numpy(float)
        predicted = group.predicted.to_numpy(bool)
        transitions = np.flatnonzero(group.gt_transition_event.to_numpy(bool))
        for transition_index in transitions:
            transition_frame = frame_idx[transition_index]
            later = np.flatnonzero(
                predicted
                & (frame_idx >= transition_frame)
                & (frame_idx <= transition_frame + delay_window)
            )
            if len(later):
                delays.append(float(frame_idx[later[0]] - transition_frame))
    return np.asarray(delays, dtype=float)


def compute_metrics(outcomes, coverage_target=0.80, delay_window=64):
    predicted = outcomes.predicted.astype(bool)
    transition = outcomes.oracle_regime.eq(1)
    stable = outcomes.oracle_regime.eq(0)
    risk = _mean(outcomes.loc[predicted, "raw_error"])
    transition_risk = _mean(outcomes.loc[predicted & transition, "raw_error"])
    stable_risk = _mean(outcomes.loc[predicted & stable, "raw_error"])
    t_error = outcomes.loc[predicted & transition, "raw_error"].sum()
    t_total = int((predicted & transition).sum())
    s_error = outcomes.loc[predicted & stable, "raw_error"].sum()
    s_total = int((predicted & stable).sum())
    error_or = (
        ((t_error + 0.5) / (t_total - t_error + 0.5))
        / ((s_error + 0.5) / (s_total - s_error + 0.5))
        if t_total and s_total
        else np.nan
    )
    delays = _transition_delays(outcomes, delay_window)
    return {
        "coverage": float(predicted.mean()),
        "selective_risk": risk,
        "transition_risk": transition_risk,
        "stable_risk": stable_risk,
        "TEG": transition_risk - stable_risk
        if np.isfinite(transition_risk) and np.isfinite(stable_risk)
        else np.nan,
        "Error_OR": error_or,
        "miscoverage": risk - (1.0 - coverage_target) if np.isfinite(risk) else np.nan,
        "delay": float(delays.mean()) if len(delays) else np.nan,
        "n_detected_transitions": int(len(delays)),
        "n_frames": len(outcomes),
        "n_predicted": int(predicted.sum()),
        "n_transition_frames": int(transition.sum()),
        "n_stable_frames": int(stable.sum()),
    }


def _video_counts(outcomes, delay_window):
    rows = []
    for video_id, group in outcomes.groupby("video_id", sort=False):
        predicted = group.predicted.astype(bool)
        transition = group.oracle_regime.eq(1)
        stable = group.oracle_regime.eq(0)
        delays = _transition_delays(group, delay_window)
        rows.append(
            [
                video_id,
                len(group),
                predicted.sum(),
                group.loc[predicted, "raw_error"].sum(),
                (predicted & transition).sum(),
                group.loc[predicted & transition, "raw_error"].sum(),
                (predicted & stable).sum(),
                group.loc[predicted & stable, "raw_error"].sum(),
                delays.sum(),
                len(delays),
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "video_id",
            "n",
            "pred",
            "error",
            "t_pred",
            "t_error",
            "s_pred",
            "s_error",
            "delay_sum",
            "delay_count",
        ],
    )


def _count_metrics(counts, coverage_target):
    n, pred, error, t_pred, t_error, s_pred, s_error, delay_sum, delay_count = counts
    risk = error / pred if pred else np.nan
    trisk = t_error / t_pred if t_pred else np.nan
    srisk = s_error / s_pred if s_pred else np.nan
    odds = (
        ((t_error + 0.5) / (t_pred - t_error + 0.5))
        / ((s_error + 0.5) / (s_pred - s_error + 0.5))
        if t_pred and s_pred
        else np.nan
    )
    return {
        "coverage": pred / n,
        "selective_risk": risk,
        "transition_risk": trisk,
        "stable_risk": srisk,
        "TEG": trisk - srisk if np.isfinite(trisk) and np.isfinite(srisk) else np.nan,
        "Error_OR": odds,
        "miscoverage": risk - (1.0 - coverage_target) if np.isfinite(risk) else np.nan,
        "delay": delay_sum / delay_count if delay_count else np.nan,
    }


def cluster_bootstrap(
    outcomes,
    n_boot,
    seed,
    *,
    coverage_target=0.80,
    delay_window=64,
):
    values = _video_counts(outcomes, delay_window).drop(columns="video_id").to_numpy(float)
    rng = np.random.default_rng(seed)
    distributions = {metric: np.full(n_boot, np.nan) for metric in BOOT_METRICS}
    for start in range(0, n_boot, 100):
        size = min(100, n_boot - start)
        sampled = rng.integers(0, len(values), size=(size, len(values)))
        for offset, total in enumerate(values[sampled].sum(axis=1)):
            metrics = _count_metrics(total, coverage_target)
            for metric in BOOT_METRICS:
                distributions[metric][start + offset] = metrics[metric]
    result = {}
    for metric, values in distributions.items():
        values = values[np.isfinite(values)]
        result[metric] = tuple(np.percentile(values, [2.5, 97.5])) if len(values) else (np.nan, np.nan)
    return result


def _precision(truth, prediction):
    return float((truth & prediction).sum() / prediction.sum()) if prediction.sum() else np.nan


def _recall(truth, prediction):
    return float((truth & prediction).sum() / truth.sum()) if truth.sum() else np.nan


def _f1(precision, recall):
    return (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else np.nan
    )


def regime_diagnostics(evaluation, args):
    frame = evaluation[evaluation.oracle_regime.isin([0, 1])]
    truth = frame.oracle_regime.eq(1).to_numpy()
    prediction = frame.estimated_regime.eq(1).to_numpy()
    tp, tr = _precision(truth, prediction), _recall(truth, prediction)
    sp, sr = _precision(~truth, ~prediction), _recall(~truth, ~prediction)
    return pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "model": args.model,
                "window": args.window,
                "near": args.near,
                "far": args.far,
                "transition_precision": tp,
                "transition_recall": tr,
                "transition_f1": _f1(tp, tr),
                "stable_precision": sp,
                "stable_recall": sr,
                "stable_f1": _f1(sp, sr),
                "transition_prevalence_oracle": truth.mean(),
                "transition_prevalence_estimated": prediction.mean(),
                "false_transition_rate": (prediction & ~truth).sum() / (~truth).sum(),
            }
        ]
    )


def _outcomes(
    calib,
    evaluation,
    coverage,
    global_delta,
    estimated_delta,
    aci=True,
):
    global_tau = fit_global_threshold(calib, coverage)
    output = {
        "global_split_conformal": apply_threshold_policy(evaluation, global_tau),
        "oracle_mondrian": apply_threshold_policy(
            evaluation,
            fit_oracle_mondrian_thresholds(calib, coverage, global_tau),
            "oracle_regime",
        ),
        "estimated_regime_mondrian": apply_threshold_policy(
            evaluation,
            fit_estimated_mondrian_thresholds(calib, coverage, global_tau),
            "estimated_regime",
        ),
    }
    if aci:
        output["aci_global"] = run_aci_global(
            calib, evaluation, coverage, global_delta
        )
        output["aci_estimated_regime"] = run_aci_estimated_regime(
            calib, evaluation, coverage, estimated_delta
        )
    return output


def _metadata(method):
    return {
        "global_split_conformal": ("none", "yes", "no"),
        "oracle_mondrian": ("gt transition", "no", "no"),
        "estimated_regime_mondrian": ("causal estimate", "yes", "no"),
        "aci_global": ("none", "yes", "yes"),
        "aci_estimated_regime": ("causal estimate", "yes", "yes"),
    }[method]


METHOD_LABELS = {
    "global_split_conformal": "Global split-conformal",
    "oracle_mondrian": "Oracle Mondrian",
    "estimated_regime_mondrian": "Estimated-regime Mondrian",
    "aci_global": "ACI (global)",
    "aci_estimated_regime": "ACI + estimated regime",
}


def _group_counts(frame):
    return {
        "oracle_stable": int(frame.oracle_regime.eq(0).sum()),
        "oracle_transition_like": int(frame.oracle_regime.eq(1).sum()),
        "oracle_middle": int(frame.oracle_regime.eq(-1).sum()),
        "estimated_stable": int(frame.estimated_regime.eq(0).sum()),
        "estimated_transition_like": int(frame.estimated_regime.eq(1).sum()),
    }


def _format_value(value):
    return "NA" if not np.isfinite(value) else f"{value:.6f}"


def _format_ci(point, interval):
    low, high = interval
    if not np.isfinite(point):
        return "NA"
    return f"{point:.6f} [{low:.6f}, {high:.6f}]"


def _summary_markdown(
    args,
    calib,
    evaluation,
    summary_rows,
    intervals_by_method,
    global_delta,
    estimated_delta,
):
    calibration_counts = _group_counts(calib)
    evaluation_counts = _group_counts(evaluation)
    lines = [
        f"# v1.11 Reliability Ladder: {args.dataset.upper()}",
        "",
        "This is a log-only post-hoc reliability diagnostic. No training, inference, "
        "new data, or new backbone was run.",
        "",
        f"- Input: `{args.selected_source}`",
        f"- Schema audit: PASS ({len(REQUIRED)} required fields)",
        f"- Dataset/model: `{args.dataset}` / `{args.model}`",
        f"- Videos: {args.expected_videos}",
        f"- Calibration/evaluation videos: {calib.video_id.nunique()} / {evaluation.video_id.nunique()}",
        f"- Score and target coverage: `{args.score}`, c={args.coverage:.2f}",
        f"- Near/far: <= {args.near:g} / > {args.far:g}",
        f"- Bootstrap: {args.n_boot} video-cluster resamples, seed={args.seed}",
        f"- ACI deltas selected on calibration only: global={global_delta:g}, estimated={estimated_delta:g}",
        "",
        "The original `decision` and `selective_error` fields are ignored. Every "
        "predict/abstain decision is recomputed from scalar scores, and all errors use "
        "`raw_error`. Oracle middle-band frames use the global threshold and contribute "
        "to overall metrics, but not stable/transition group calibration or risks.",
        "",
        "## Regime Counts",
        "",
        "| Split | Oracle stable | Oracle transition-like | Oracle middle | Estimated stable | Estimated transition-like |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Calibration | {calibration_counts['oracle_stable']:,} | "
            f"{calibration_counts['oracle_transition_like']:,} | "
            f"{calibration_counts['oracle_middle']:,} | "
            f"{calibration_counts['estimated_stable']:,} | "
            f"{calibration_counts['estimated_transition_like']:,} |"
        ),
        (
            f"| Evaluation | {evaluation_counts['oracle_stable']:,} | "
            f"{evaluation_counts['oracle_transition_like']:,} | "
            f"{evaluation_counts['oracle_middle']:,} | "
            f"{evaluation_counts['estimated_stable']:,} | "
            f"{evaluation_counts['estimated_transition_like']:,} |"
        ),
        "",
        "## Primary Target-Coverage Table",
        "",
        "| Method | Group source | Causal | Coverage | Sel.Risk [CI] | Transition Risk [CI] | Stable Risk [CI] | Error_OR [CI] | Miscov [CI] | Delay [CI] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        method = row["method"]
        intervals = intervals_by_method[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['group_source']} | {row['causal']} | "
            f"{row['coverage']:.6f} | "
            f"{_format_ci(row['selective_risk'], intervals['selective_risk'])} | "
            f"{_format_ci(row['transition_risk'], intervals['transition_risk'])} | "
            f"{_format_ci(row['stable_risk'], intervals['stable_risk'])} | "
            f"{_format_ci(row['Error_OR'], intervals['Error_OR'])} | "
            f"{_format_ci(row['miscoverage'], intervals['miscoverage'])} | "
            f"{_format_ci(row['delay'], intervals['delay'])} |"
        )
    lines.extend(
        [
            "",
            "## Method Boundary",
            "",
            "Oracle Mondrian is a non-causal upper bound. Estimated-regime Mondrian is "
            "the causal counterpart. ACI rows are heuristic abstention-feedback online "
            "baselines; no finite-sample coverage guarantee is claimed under evolving "
            "or noisy regimes. Delay is the mean number of frame steps from each ground-"
            "truth transition to the first accepted prediction within the configured "
            f"{args.delay_window}-step window.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(args, calib, evaluation):
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_delta = select_aci_delta(
        calib,
        args.coverage,
        args.aci_delta_grid,
        args.seed,
        estimated=False,
    )
    estimated_delta = select_aci_delta(
        calib,
        args.coverage,
        args.aci_delta_grid,
        args.seed,
        estimated=True,
    )
    outcomes = _outcomes(
        calib,
        evaluation,
        args.coverage,
        global_delta,
        estimated_delta,
    )
    summaries, boot_rows, by_method = [], [], {}
    intervals_by_method = {}
    for method in METHODS:
        print(f"evaluating and bootstrapping: {method}", flush=True)
        metrics = compute_metrics(
            outcomes[method],
            coverage_target=args.coverage,
            delay_window=args.delay_window,
        )
        by_method[method] = metrics
        source, causal, feedback = _metadata(method)
        summaries.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "score": args.score,
                "coverage_target": args.coverage,
                "method": method,
                "group_source": source,
                "causal": causal,
                "uses_label_feedback": feedback,
                **metrics,
            }
        )
        intervals = cluster_bootstrap(
            outcomes[method],
            args.n_boot,
            args.seed,
            coverage_target=args.coverage,
            delay_window=args.delay_window,
        )
        intervals_by_method[method] = intervals
        for metric in BOOT_METRICS:
            low, high = intervals[metric]
            boot_rows.append(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "score": args.score,
                    "coverage_target": args.coverage,
                    "method": method,
                    "metric": metric,
                    "point_estimate": metrics[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "n_boot": args.n_boot,
                    "cluster_unit": "video_id",
                }
            )
    global_m = by_method["global_split_conformal"]
    oracle_m = by_method["oracle_mondrian"]
    estimated_m = by_method["estimated_regime_mondrian"]
    gaps = []
    for metric in GAP_METRICS:
        recoverable = global_m[metric] - oracle_m[metric]
        recovered = global_m[metric] - estimated_m[metric]
        gaps.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "score": args.score,
                "coverage_target": args.coverage,
                "metric": metric,
                "global_value": global_m[metric],
                "oracle_mondrian_value": oracle_m[metric],
                "estimated_mondrian_value": estimated_m[metric],
                "oracle_recoverable_gap": recoverable,
                "online_recovered_gap": recovered,
                "online_availability_ratio": recovered / max(abs(recoverable), 1e-12),
                "residual_oracle_estimated_gap": estimated_m[metric] - oracle_m[metric],
            }
        )
    curves = []
    for target in args.curve_coverages:
        print(f"risk-coverage target: {target:.2f}", flush=True)
        for method, result in _outcomes(
            calib,
            evaluation,
            float(target),
            global_delta,
            estimated_delta,
            False,
        ).items():
            source, causal, feedback = _metadata(method)
            curves.append(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "score": args.score,
                    "coverage_target": target,
                    "method": method,
                    "group_source": source,
                    "causal": causal,
                    "uses_label_feedback": feedback,
                    **compute_metrics(
                        result,
                        coverage_target=float(target),
                        delay_window=args.delay_window,
                    ),
                }
            )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "baseline_ladder_summary.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(output_dir / "baseline_ladder_bootstrap.csv", index=False)
    pd.DataFrame(gaps).to_csv(output_dir / "oracle_vs_estimated_gap.csv", index=False)
    regime_diagnostics(evaluation, args).to_csv(
        output_dir / "regime_estimator_diagnostics.csv", index=False
    )
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(output_dir / "risk_coverage_curve.csv", index=False)
    summary_md = _summary_markdown(
        args,
        calib,
        evaluation,
        summaries,
        intervals_by_method,
        global_delta,
        estimated_delta,
    )
    (output_dir / "README.md").write_text(summary_md, encoding="utf-8")

    formal_rows = []
    for row in summaries:
        intervals = intervals_by_method[row["method"]]
        formal = {
            "dataset": args.dataset,
            "model": args.model,
            "score": args.score,
            "coverage_target": args.coverage,
            "Method": METHOD_LABELS[row["method"]],
            "Group source": row["group_source"],
            "Causal": row["causal"],
            "uses_label_feedback": row["uses_label_feedback"],
            "Coverage": row["coverage"],
            "n_frames": row["n_frames"],
            "n_predicted": row["n_predicted"],
            "n_transition_frames": row["n_transition_frames"],
            "n_stable_frames": row["n_stable_frames"],
            "n_detected_transitions": row["n_detected_transitions"],
        }
        for metric, label in [
            ("selective_risk", "Sel.Risk"),
            ("transition_risk", "Transition Risk"),
            ("stable_risk", "Stable Risk"),
            ("TEG", "TEG"),
            ("Error_OR", "Error_OR"),
            ("miscoverage", "Miscov"),
            ("delay", "Delay"),
        ]:
            formal[label] = row[metric]
            formal[f"{label} CI low"] = intervals[metric][0]
            formal[f"{label} CI high"] = intervals[metric][1]
        formal_rows.append(formal)
    formal_frame = pd.DataFrame(formal_rows)

    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "primary_summary.csv"
    summary_md_path = Path(args.summary_md) if args.summary_md else output_dir / "summary_readable.md"
    curve_csv = Path(args.curve_csv) if args.curve_csv else output_dir / "risk_coverage_curve.csv"
    for path in (summary_csv, summary_md_path, curve_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    formal_frame.to_csv(summary_csv, index=False)
    summary_md_path.write_text(summary_md, encoding="utf-8")
    curve_frame.to_csv(curve_csv, index=False)
    print(f"wrote outputs to: {output_dir}")
    return {
        "summary": formal_frame,
        "curve": curve_frame,
        "calibration_counts": _group_counts(calib),
        "evaluation_counts": _group_counts(evaluation),
        "global_delta": global_delta,
        "estimated_delta": estimated_delta,
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md_path),
        "curve_csv": str(curve_csv),
    }


def run(args):
    is_tvseries = str(args.dataset).lower() == "tvseries"
    if is_tvseries and not (args.calibration_input and args.evaluation_input):
        if not args.dry_run_schema:
            raise LadderError(
                "TVSeries formal analysis requires separate official "
                "--calibration-input (val) and --evaluation-input (test)"
            )
    if args.calibration_input and args.evaluation_input:
        calibration = read_jsonl_or_tar(
            args.calibration_input,
            args.dry_run_schema,
            expected_videos=args.expected_calibration_videos,
        )
        evaluation = read_jsonl_or_tar(
            args.evaluation_input,
            args.dry_run_schema,
            expected_videos=args.expected_videos,
        )
        if args.dry_run_schema:
            return
        calibration, evaluation = official_calibration_evaluation(
            calibration, evaluation, dataset=args.dataset
        )
        args.selected_source = (
            f"calibration={calibration.attrs['source']}; "
            f"evaluation={evaluation.attrs['source']}"
        )
        frame = pd.concat([calibration, evaluation], ignore_index=True)
    else:
        frame = read_jsonl_or_tar(
            args.input,
            args.dry_run_schema,
            member=args.member,
            member_contains=args.member_contains,
            expected_videos=args.expected_videos,
        )
        if args.dry_run_schema:
            return
        args.selected_source = frame.attrs["source"]
        calibration = evaluation = None
    if is_tvseries and "active_label_set" not in frame:
        raise LadderError(
            "TVSeries formal analysis requires active_label_set; "
            "a lossy single-label projection is not accepted"
        )
    logged_models = sorted(frame.model.astype(str).unique())
    if args.model not in logged_models:
        raise LadderError(
            f"Requested model {args.model!r}, but canonical stream contains {logged_models}"
        )
    print(f"canonical stream: {len(frame):,} frames, {frame.video_id.nunique()} videos")
    frame = add_oracle_regime(frame, args.near, args.far, dataset=args.dataset)
    frame = add_estimated_regime(frame, args)
    frame["score"] = _scores(frame, args.score)
    if calibration is None:
        calib, evaluation = split_calib_eval_by_video(
            frame, args.calib_frac, args.seed
        )
    else:
        calib_ids = set(calibration.video_id.astype(str).unique())
        eval_ids = set(evaluation.video_id.astype(str).unique())
        calib = frame[frame.video_id.isin(calib_ids)].copy()
        evaluation = frame[frame.video_id.isin(eval_ids)].copy()
    return write_outputs(args, calib, evaluation)


def main(argv=None):
    try:
        run(parse_args(argv))
    except (LadderError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
