from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from oad_stress_test.datasets.schema import VideoSequence
from oad_stress_test.metrics.ek100 import top1_not_in_active_set_error
from oad_stress_test.models.causal_tcn import CausalTCNClassifier, torch
from oad_stress_test.utils.threshold_calibration import score_signals
try:
    from scripts.run_v19_ek100_smoke import (
        active_set_state_ids,
        common_session_ids,
        load_ek100_smoke_video,
        resolve_ek100_dirs,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    from run_v19_ek100_smoke import (
        active_set_state_ids,
        common_session_ids,
        load_ek100_smoke_video,
        resolve_ek100_dirs,
    )


V110_RAW_ERROR = 0.7735905493523266
MIN_MEANINGFUL_ERROR_DROP = 0.02
CANONICAL_JSONL = "per_frame_causal_tcn_ek100_clean.jsonl"


def split_session_ids(
    session_ids: list[str],
    *,
    max_train_sessions: int,
    max_eval_sessions: int,
) -> tuple[list[str], list[str]]:
    required = int(max_train_sessions) + int(max_eval_sessions)
    if len(session_ids) < required:
        raise ValueError(
            f"Need at least {required} aligned EK100 sessions, found {len(session_ids)}"
        )
    return (
        session_ids[: int(max_train_sessions)],
        session_ids[int(max_train_sessions) : required],
    )


def load_train_sequence(
    video_id: str,
    *,
    rgb_dir: Path,
    flow_dir: Path,
    target_dir: Path,
    max_frames: int | None,
) -> VideoSequence:
    rgb = np.load(rgb_dir / f"{video_id}.npy", mmap_mode="r")
    flow = np.load(flow_dir / f"{video_id}.npy", mmap_mode="r")
    target = np.load(target_dir / f"{video_id}.npy", mmap_mode="r")
    if rgb.ndim != 2 or flow.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"{video_id}: expected [T, D] RGB/Flow and [T, C] target arrays"
        )
    length = min(int(rgb.shape[0]), int(flow.shape[0]), int(target.shape[0]))
    if max_frames is not None:
        length = min(length, int(max_frames))
    if length <= 0:
        raise ValueError(f"{video_id}: no aligned frames")
    features = np.concatenate(
        [
            np.asarray(rgb[:length], dtype=np.float32),
            np.asarray(flow[:length], dtype=np.float32),
        ],
        axis=1,
    )
    labels = np.argmax(np.asarray(target[:length]), axis=1).astype(np.int64)
    return VideoSequence(video_id=video_id, features=features, labels=labels)


def iter_train_sequences(
    video_ids: Iterable[str],
    *,
    rgb_dir: Path,
    flow_dir: Path,
    target_dir: Path,
    max_frames: int | None,
) -> Iterable[VideoSequence]:
    for video_id in video_ids:
        yield load_train_sequence(
            video_id,
            rgb_dir=rgb_dir,
            flow_dir=flow_dir,
            target_dir=target_dir,
            max_frames=max_frames,
        )


def topk_in_active_set(prediction_scores: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(prediction_scores)
    active = np.asarray(target, dtype=bool)
    if scores.shape != active.shape:
        raise ValueError(f"score/target shape mismatch: {scores.shape} vs {active.shape}")
    take = min(int(k), int(scores.shape[1]))
    topk = np.argpartition(scores, -take, axis=1)[:, -take:]
    return np.take_along_axis(active, topk, axis=1).any(axis=1)


def write_eval_log(
    path: Path,
    *,
    model: CausalTCNClassifier,
    eval_ids: list[str],
    rgb_dir: Path,
    flow_dir: Path,
    target_dir: Path,
    max_frames: int | None,
    budget: float,
) -> dict[str, float | int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    total_errors = 0
    total_top5_correct = 0
    start = perf_counter()
    with path.open("w", encoding="utf-8") as handle:
        for video_number, video_id in enumerate(eval_ids, start=1):
            video = load_ek100_smoke_video(
                video_id,
                rgb_dir=rgb_dir,
                flow_dir=flow_dir,
                target_dir=target_dir,
                max_frames=max_frames,
            )
            scores = model.predict_scores_sequence(video.features)
            signals = score_signals(scores, input_kind="probabilities")
            labels = np.argmax(scores, axis=1).astype(np.int64)
            raw_error = top1_not_in_active_set_error(labels, video.target).astype(np.int64)
            top5_correct = topk_in_active_set(scores, video.target, 5)
            state_ids = active_set_state_ids(video.target)
            for frame_idx in range(len(labels)):
                row = {
                    "video_id": video.video_id,
                    "frame_idx": int(frame_idx),
                    "gt_label": int(state_ids[frame_idx]),
                    "primary_label": int(video.primary_labels[frame_idx]),
                    "pred_label": int(labels[frame_idx]),
                    "decision": "predict",
                    "max_prob": float(signals["max_prob"][frame_idx]),
                    "entropy": float(signals["entropy"][frame_idx]),
                    "top2_margin": float(signals["top2_margin"][frame_idx]),
                    "raw_error": int(raw_error[frame_idx]),
                    "selective_error": int(raw_error[frame_idx]),
                    "is_correct": bool(not raw_error[frame_idx]),
                    "active_label_count": int(video.target[frame_idx].sum()),
                    "budget": float(budget),
                    "model": "causal_tcn",
                    "policy": "all_frames_reference",
                    "threshold": 0.0,
                    "shift_type": "clean",
                    "split": "test",
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            total_frames += len(labels)
            total_errors += int(raw_error.sum())
            total_top5_correct += int(top5_correct.sum())
            print(
                f"eval {video_number}/{len(eval_ids)}: {video_id} "
                f"frames={len(labels)} cumulative_error={total_errors / total_frames:.6f}",
                flush=True,
            )
    raw_error_rate = total_errors / total_frames if total_frames else float("nan")
    return {
        "num_videos": int(len(eval_ids)),
        "num_frames": int(total_frames),
        "raw_error_rate": float(raw_error_rate),
        "top1_active_set_accuracy": float(1.0 - raw_error_rate),
        "top5_active_set_accuracy": (
            float(total_top5_correct / total_frames) if total_frames else float("nan")
        ),
        "evaluation_seconds": float(perf_counter() - start),
    }


def write_quality_outputs(
    output_dir: Path,
    *,
    quality: dict[str, object],
    diagnostics: dict[str, object],
    command: str,
    quality_doc: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictor_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality))
        writer.writeheader()
        writer.writerow(quality)
    (output_dir / "training_log.md").write_text(
        "\n".join(
            [
                "# v1.12 EK100 Causal TCN Training Log",
                "",
                f"- fit_seconds: {quality['fit_seconds']}",
                f"- train_loss_by_epoch: {diagnostics['train_loss_by_epoch']}",
                f"- sample_count_nonzero_classes: {quality['sample_count_nonzero_classes']}",
                f"- receptive_field: {diagnostics['receptive_field']}",
                f"- device_used: {diagnostics['device_used']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    improvement = float(quality["raw_error_improvement_vs_v110"])
    gate = bool(quality["materially_improved"])
    interpretation = (
        "The stronger predictor materially reduces raw error, so the EK100 "
        "reliability ladder can be used for the planned confound check."
        if gate
        else "The attempted stronger-predictor sanity did not materially improve "
        "predictor quality. The model-capability confound remains unresolved, and "
        "any ladder result must not be overinterpreted."
    )
    markdown = "\n".join(
        [
            "# v1.12 EK100 Stronger-Predictor Quality Gate",
            "",
            "This is a diagnostic reference predictor, not a new method or SOTA claim.",
            "",
            f"- model: causal_tcn",
            f"- exact command: `{command}`",
            "- input features: TeSTra RGB + Flow BN-Inception features",
            f"- videos: {quality['num_videos']}",
            f"- frames: {quality['num_frames']}",
            f"- raw_error / top-1 error: {float(quality['raw_error_rate']):.6f}",
            f"- top-1 active-set accuracy: {float(quality['top1_active_set_accuracy']):.6f}",
            f"- top-5 active-set accuracy: {float(quality['top5_active_set_accuracy']):.6f}",
            f"- v1.10/v1.11 canonical raw error: {V110_RAW_ERROR:.6f}",
            f"- absolute raw-error reduction: {improvement:.6f}",
            f"- pre-specified material-improvement gate: {MIN_MEANINGFUL_ERROR_DROP:.2f}",
            f"- quality gate passed: {str(gate).lower()}",
            "",
            interpretation,
            "",
        ]
    )
    (output_dir / "predictor_quality.md").write_text(markdown, encoding="utf-8")
    if quality_doc is not None:
        quality_doc.parent.mkdir(parents=True, exist_ok=True)
        quality_doc.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace, command: str) -> dict[str, Path]:
    if args.formal:
        if str(args.device) != "cuda":
            raise ValueError("Formal v1.12 requires --device cuda")
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("Formal v1.12 requires CUDA; refusing CPU fallback")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / CANONICAL_JSONL
    if jsonl_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing log: {jsonl_path}")

    rgb_dir, flow_dir, target_dir = resolve_ek100_dirs(args)
    session_ids = common_session_ids(rgb_dir, flow_dir, target_dir)
    train_ids, eval_ids = split_session_ids(
        session_ids,
        max_train_sessions=args.max_train_sessions,
        max_eval_sessions=args.max_eval_sessions,
    )
    first_target = np.load(target_dir / f"{train_ids[0]}.npy", mmap_mode="r")
    num_classes = int(first_target.shape[1])
    model = CausalTCNClassifier(
        num_classes=num_classes,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        kernel_size=int(args.kernel_size),
        dilations=tuple(int(value) for value in args.dilations),
        dropout=float(args.dropout),
        max_epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        chunk_length=int(args.chunk_length),
        device=str(args.device),
        seed=int(args.seed),
        background_label=-1,
        use_class_weights=True,
    )
    fit_start = perf_counter()
    model.fit(
        iter_train_sequences(
            train_ids,
            rgb_dir=rgb_dir,
            flow_dir=flow_dir,
            target_dir=target_dir,
            max_frames=args.max_frames_per_session,
        )
    )
    fit_seconds = perf_counter() - fit_start
    eval_stats = write_eval_log(
        jsonl_path,
        model=model,
        eval_ids=eval_ids,
        rgb_dir=rgb_dir,
        flow_dir=flow_dir,
        target_dir=target_dir,
        max_frames=args.max_frames_per_session,
        budget=float(args.budget),
    )
    diagnostics = model.training_diagnostics()
    raw_error_rate = float(eval_stats["raw_error_rate"])
    improvement = V110_RAW_ERROR - raw_error_rate
    quality: dict[str, object] = {
        "model": "causal_tcn",
        "device": diagnostics["device_used"],
        "num_train_videos": len(train_ids),
        "num_videos": eval_stats["num_videos"],
        "num_frames": eval_stats["num_frames"],
        "raw_error_rate": raw_error_rate,
        "top1_active_set_accuracy": eval_stats["top1_active_set_accuracy"],
        "top5_active_set_accuracy": eval_stats["top5_active_set_accuracy"],
        "v110_raw_error_rate": V110_RAW_ERROR,
        "raw_error_improvement_vs_v110": improvement,
        "materially_improved": improvement >= MIN_MEANINGFUL_ERROR_DROP,
        "fit_seconds": float(fit_seconds),
        "evaluation_seconds": eval_stats["evaluation_seconds"],
        "sample_count_nonzero_classes": int(
            np.count_nonzero(model.sample_counts_)
            if model.sample_counts_ is not None
            else 0
        ),
    }
    write_quality_outputs(
        output_dir,
        quality=quality,
        diagnostics=diagnostics,
        command=command,
        quality_doc=Path(args.quality_doc) if args.quality_doc else None,
    )
    (output_dir / "run_config.md").write_text(
        "\n".join(
            [
                "# v1.12 EK100 Stronger-Predictor Run Config",
                "",
                f"- mode: {'formal' if args.formal else 'smoke'}",
                f"- rgb_dir: {rgb_dir}",
                f"- flow_dir: {flow_dir}",
                f"- target_dir: {target_dir}",
                f"- train/eval videos: {len(train_ids)} / {len(eval_ids)}",
                f"- num_classes: {num_classes}",
                f"- hidden_dim: {args.hidden_dim}",
                f"- num_layers: {args.num_layers}",
                f"- kernel_size: {args.kernel_size}",
                f"- dilations: {args.dilations}",
                f"- dropout: {args.dropout}",
                f"- epochs: {args.epochs}",
                f"- chunk_length: {args.chunk_length}",
                f"- device: {diagnostics['device_used']}",
                f"- budget: {args.budget}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "canonical_jsonl": jsonl_path,
        "quality_csv": output_dir / "predictor_quality.csv",
        "quality_md": output_dir / "predictor_quality.md",
        "training_log": output_dir / "training_log.md",
        "run_config": output_dir / "run_config.md",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated v1.12 EK100 causal-TCN stronger-predictor sanity."
    )
    parser.add_argument("--ek100-root")
    parser.add_argument("--rgb-dir")
    parser.add_argument("--flow-dir")
    parser.add_argument("--target-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality-doc")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-sessions", type=int, default=500)
    parser.add_argument("--max-eval-sessions", type=int, default=133)
    parser.add_argument("--max-frames-per-session", type=int, default=1_000_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilations", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--chunk-length", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command = shlex.join([sys.executable, __file__, *(argv or sys.argv[1:])])
    try:
        outputs = run(args, command)
    except Exception as exc:  # noqa: BLE001 - CLI should expose a clear failure.
        print(f"v112_ek100_stronger_predictor_failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
