from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from oad_stress_test.datasets.schema import VideoSequence
from oad_stress_test.metrics.ek100 import top1_not_in_active_set_error
from oad_stress_test.models.causal_gru import CausalGRUClassifier
from oad_stress_test.utils.threshold_calibration import DEFAULT_QUANTILES, score_signals, threshold_grid_from_signals


RGB_SUBDIR = "rgb_kinetics_bninception"
FLOW_SUBDIR = "flow_kinetics_bninception"
TARGET_SUBDIR = "target_perframe"


@dataclass
class Ek100SmokeVideo:
    video_id: str
    features: np.ndarray
    target: np.ndarray
    primary_labels: np.ndarray
    state_ids: np.ndarray


def _npy_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing directory: {path}")
    return {file.stem for file in path.glob("*.npy")}


def resolve_ek100_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = Path(args.ek100_root) if args.ek100_root else None
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else (root / RGB_SUBDIR if root else None)
    flow_dir = Path(args.flow_dir) if args.flow_dir else (root / FLOW_SUBDIR if root else None)
    target_dir = Path(args.target_dir) if args.target_dir else (root / TARGET_SUBDIR if root else None)
    if rgb_dir is None or flow_dir is None or target_dir is None:
        raise ValueError("Provide --ek100-root or all of --rgb-dir, --flow-dir, and --target-dir")
    return rgb_dir, flow_dir, target_dir


def common_session_ids(rgb_dir: Path, flow_dir: Path, target_dir: Path) -> list[str]:
    ids = sorted(_npy_ids(rgb_dir) & _npy_ids(flow_dir) & _npy_ids(target_dir))
    if not ids:
        raise ValueError("No overlapping EK100 session IDs across RGB, Flow, and target directories")
    return ids


def primary_labels_from_target(target: np.ndarray) -> np.ndarray:
    active = np.asarray(target) > 0
    if active.ndim != 2:
        raise ValueError(f"target must have shape [T, C], got {active.shape}")
    labels = np.zeros(active.shape[0], dtype=np.int64)
    has_active = active.any(axis=1)
    labels[has_active] = np.argmax(active[has_active], axis=1).astype(np.int64)
    return labels


def active_set_state_ids(target: np.ndarray) -> np.ndarray:
    active = np.asarray(target) > 0
    state_to_id: dict[tuple[int, ...], int] = {}
    states: list[int] = []
    for row in active:
        key = tuple(int(idx) for idx in np.flatnonzero(row))
        if key not in state_to_id:
            state_to_id[key] = len(state_to_id)
        states.append(state_to_id[key])
    return np.asarray(states, dtype=np.int64)


def load_ek100_smoke_video(
    video_id: str,
    *,
    rgb_dir: Path,
    flow_dir: Path,
    target_dir: Path,
    max_frames: int | None,
) -> Ek100SmokeVideo:
    rgb = np.load(rgb_dir / f"{video_id}.npy", mmap_mode="r")
    flow = np.load(flow_dir / f"{video_id}.npy", mmap_mode="r")
    target = np.load(target_dir / f"{video_id}.npy", mmap_mode="r")
    if rgb.ndim != 2 or flow.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"{video_id}: expected RGB/Flow/target arrays with shape [T, D]/[T, C], "
            f"got {rgb.shape}, {flow.shape}, {target.shape}"
        )
    length = min(int(rgb.shape[0]), int(flow.shape[0]), int(target.shape[0]))
    if max_frames is not None:
        length = min(length, int(max_frames))
    if length <= 0:
        raise ValueError(f"{video_id}: no frames available after alignment")
    features = np.concatenate(
        [
            np.asarray(rgb[:length], dtype=np.float32),
            np.asarray(flow[:length], dtype=np.float32),
        ],
        axis=1,
    )
    target_array = np.asarray(target[:length] > 0, dtype=bool)
    return Ek100SmokeVideo(
        video_id=video_id,
        features=features,
        target=target_array,
        primary_labels=primary_labels_from_target(target_array),
        state_ids=active_set_state_ids(target_array),
    )


def to_video_sequence(video: Ek100SmokeVideo) -> VideoSequence:
    return VideoSequence(video_id=video.video_id, features=video.features, labels=video.primary_labels)


def confident_decisions(signal: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if mode == "entropy":
        return values <= float(threshold)
    if mode in {"max_prob", "top2_margin"}:
        return values >= float(threshold)
    raise ValueError(f"Unknown uncertainty mode: {mode}")


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_per_frame_log(
    path: Path,
    *,
    eval_videos: list[Ek100SmokeVideo],
    predictions: dict[str, dict[str, np.ndarray]],
    mode: str,
    quantile: float,
    threshold: float,
    budget: float,
) -> dict[str, float | int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    predict = 0
    abstain = 0
    raw_errors: list[float] = []
    selective_errors: list[float] = []
    with path.open("w", encoding="utf-8") as handle:
        for video in eval_videos:
            pred = predictions[video.video_id]
            scores = pred["scores"]
            labels = pred["pred_label"].astype(np.int64)
            signals = pred["signals"]
            raw_error = top1_not_in_active_set_error(labels, video.target).astype(bool)
            is_predict = confident_decisions(signals[mode], threshold, mode)
            for idx in range(len(labels)):
                decision = "predict" if bool(is_predict[idx]) else "abstain"
                error_value = int(bool(raw_error[idx]))
                if decision == "predict":
                    selective_errors.append(float(error_value))
                    predict += 1
                else:
                    abstain += 1
                raw_errors.append(float(error_value))
                total += 1
                row = {
                    "video_id": video.video_id,
                    "frame_idx": int(idx),
                    "gt_label": int(video.state_ids[idx]),
                    "primary_label": int(video.primary_labels[idx]),
                    "pred_label": int(labels[idx]),
                    "decision": decision,
                    "max_prob": float(signals["max_prob"][idx]),
                    "entropy": float(signals["entropy"][idx]),
                    "top2_margin": float(signals["top2_margin"][idx]),
                    "budget": float(budget),
                    "model": "causal_gru",
                    "policy": "confidence_threshold",
                    "threshold": float(threshold),
                    "threshold_quantile": float(quantile),
                    "uncertainty_mode": mode,
                    "shift_type": "clean",
                    "split": "ek100_smoke",
                    "raw_error": error_value,
                    "selective_error": error_value if decision == "predict" else None,
                    "is_correct": not bool(raw_error[idx]),
                    "active_label_count": int(video.target[idx].sum()),
                }
                handle.write(json.dumps({key: _json_safe(value) for key, value in row.items()}) + "\n")
    return {
        "uncertainty_mode": mode,
        "threshold_quantile": float(quantile),
        "threshold": float(threshold),
        "per_frame_jsonl": str(path),
        "total_frames": int(total),
        "num_videos": int(len(eval_videos)),
        "predict_frames": int(predict),
        "abstain_frames": int(abstain),
        "predict_rate": float(predict / total) if total else float("nan"),
        "abstain_rate": float(abstain / total) if total else float("nan"),
        "raw_error_rate": float(np.mean(raw_errors)) if raw_errors else float("nan"),
        "selective_error_rate": float(np.mean(selective_errors)) if selective_errors else float("nan"),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_smoke(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir, flow_dir, target_dir = resolve_ek100_dirs(args)
    ids = common_session_ids(rgb_dir, flow_dir, target_dir)
    required = int(args.max_train_sessions) + int(args.max_eval_sessions)
    if len(ids) < required:
        raise ValueError(f"Need at least {required} overlapping sessions for smoke, found {len(ids)}")
    train_ids = ids[: int(args.max_train_sessions)]
    eval_ids = ids[int(args.max_train_sessions) : required]

    train_videos = [
        load_ek100_smoke_video(
            video_id,
            rgb_dir=rgb_dir,
            flow_dir=flow_dir,
            target_dir=target_dir,
            max_frames=args.max_frames_per_session,
        )
        for video_id in train_ids
    ]
    eval_videos = [
        load_ek100_smoke_video(
            video_id,
            rgb_dir=rgb_dir,
            flow_dir=flow_dir,
            target_dir=target_dir,
            max_frames=args.max_frames_per_session,
        )
        for video_id in eval_ids
    ]
    num_classes = int(train_videos[0].target.shape[1])
    model = CausalGRUClassifier(
        num_classes=num_classes,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=0.0,
        max_epochs=int(args.epochs),
        lr=float(args.lr),
        chunk_length=int(args.chunk_length),
        device=str(args.device),
        seed=int(args.seed),
        use_class_weights=True,
    )
    start = perf_counter()
    model.fit([to_video_sequence(video) for video in train_videos])
    fit_seconds = perf_counter() - start

    predictions: dict[str, dict[str, np.ndarray]] = {}
    all_scores = []
    for video in eval_videos:
        scores = model.predict_scores_sequence(video.features)
        signals = score_signals(scores, input_kind="probabilities")
        predictions[video.video_id] = {
            "scores": scores,
            "signals": signals,
            "pred_label": np.argmax(scores, axis=1).astype(np.int64),
        }
        all_scores.append(scores)
    all_signals = score_signals(np.concatenate(all_scores, axis=0), input_kind="probabilities")
    quantiles = [float(value) for value in args.quantiles]
    threshold_grid = threshold_grid_from_signals(all_signals, quantiles=quantiles)

    threshold_rows = []
    summary_rows = []
    jsonl_dir = output_dir / "per_frame"
    for mode in ["max_prob", "entropy", "top2_margin"]:
        for item in threshold_grid[mode]:
            quantile = float(item["quantile"])
            threshold = float(item["threshold"])
            threshold_rows.append({"uncertainty_mode": mode, "quantile": quantile, "threshold": threshold})
            safe_q = str(quantile).replace(".", "p")
            path = jsonl_dir / f"per_frame_{mode}_q{safe_q}.jsonl"
            summary_rows.append(
                write_per_frame_log(
                    path,
                    eval_videos=eval_videos,
                    predictions=predictions,
                    mode=mode,
                    quantile=quantile,
                    threshold=threshold,
                    budget=float(args.budget),
                )
            )

    write_csv(output_dir / "thresholds.csv", threshold_rows)
    write_csv(output_dir / "smoke_summary.csv", summary_rows)
    (output_dir / "run_config.md").write_text(
        "\n".join(
            [
                "# v1.9 EK100 Smoke Run Config",
                "",
                "This is a CPU/local smoke check only. It is not a formal EK100 replication result.",
                "",
                f"- rgb_dir: {rgb_dir}",
                f"- flow_dir: {flow_dir}",
                f"- target_dir: {target_dir}",
                f"- train_sessions: {', '.join(train_ids)}",
                f"- eval_sessions: {', '.join(eval_ids)}",
                f"- max_frames_per_session: {args.max_frames_per_session}",
                f"- num_classes: {num_classes}",
                f"- hidden_dim: {args.hidden_dim}",
                f"- num_layers: {args.num_layers}",
                f"- epochs: {args.epochs}",
                f"- device_requested: {args.device}",
                f"- device_used: {model.device_used_}",
                f"- budget: {args.budget}",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "training_log.md").write_text(
        "\n".join(
            [
                "# v1.9 EK100 Smoke Training Log",
                "",
                f"- fit_seconds: {fit_seconds:.6f}",
                f"- train_loss_by_epoch: {list(model.train_loss_by_epoch_)}",
                f"- sample_count_nonzero_classes: {int(np.count_nonzero(model.sample_counts_)) if model.sample_counts_ is not None else 0}",
                f"- train_frames: {sum(len(video.primary_labels) for video in train_videos)}",
                f"- eval_frames: {sum(len(video.primary_labels) for video in eval_videos)}",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "thresholds": output_dir / "thresholds.csv",
        "summary": output_dir / "smoke_summary.csv",
        "run_config": output_dir / "run_config.md",
        "training_log": output_dir / "training_log.md",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny EK100 causal-GRU smoke check for v1.9 plumbing.")
    parser.add_argument("--ek100-root", help="Root containing rgb_kinetics_bninception, flow_kinetics_bninception, target_perframe")
    parser.add_argument("--rgb-dir")
    parser.add_argument("--flow-dir")
    parser.add_argument("--target-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-sessions", type=int, default=10)
    parser.add_argument("--max-eval-sessions", type=int, default=5)
    parser.add_argument("--max-frames-per-session", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--quantiles", nargs="+", type=float, default=list(DEFAULT_QUANTILES))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        outputs = run_smoke(args)
    except Exception as exc:  # noqa: BLE001 - CLI reports smoke failures clearly.
        print(f"ek100_smoke_failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
