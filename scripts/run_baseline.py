from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from oad_stress_test.config import load_config
from oad_stress_test.evaluators.streaming import CausalStreamingEvaluator
from oad_stress_test.metrics.summary import summarize_log
from oad_stress_test.utils.jsonl_logger import (
    PerFrameLogStats,
    iter_per_frame_rows,
    open_jsonl_writer,
    write_jsonl_row,
)
from oad_stress_test.utils.factory import make_classifier, make_datasets, make_policy
from oad_stress_test.utils.io import ensure_dir, write_jsonl
from oad_stress_test.utils.smoke_selection import select_class_overlap_smoke


THRESHOLD_SWEEP_POLICIES = {"confidence_threshold", "uncertainty_threshold", "uncertainty_wait_abstain"}
UNCERTAINTY_MODE_POLICIES = {
    "confidence",
    "confidence_threshold",
    "uncertainty_threshold",
    "uncertainty_wait_abstain",
}


def _limit_dataset(dataset, max_videos: int | None, flag_name: str = "--max-videos"):
    if max_videos is None:
        return dataset
    if max_videos <= 0:
        raise ValueError(f"{flag_name} must be positive")
    dataset.video_ids = dataset.video_ids[:max_videos]
    return dataset


def _threshold_values(policy: str, thresholds: list[float] | None, threshold: float) -> list[float | None]:
    if policy in THRESHOLD_SWEEP_POLICIES:
        return [float(value) for value in (thresholds or [threshold])]
    if policy == "confidence":
        return [float(threshold)]
    return [None]


def _seed_values(policy: str, seed: int, seeds: list[int] | None) -> list[int | None]:
    if policy == "random":
        return [int(value) for value in (seeds or [seed])]
    return [None]


def _shift_fragment(shift_name: str = "clean", shift_severity: float = 0.0) -> str | None:
    if str(shift_name) == "clean" and float(shift_severity) == 0.0:
        return None
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(shift_name))
    return f"{safe_name}_s{float(shift_severity):.3f}"


def _run_name(
    policy: str,
    threshold: float | None,
    seed: int | None = None,
    classifier: str = "prototype",
    linear_mode: str = "multinomial",
    score_mode: str = "raw_margin",
    uncertainty_mode: str = "max_prob",
    shift_name: str = "clean",
    shift_severity: float = 0.0,
) -> str:
    parts = []
    shift = _shift_fragment(shift_name, shift_severity)
    if shift is not None:
        parts.append(shift)
    if classifier == "linear_probe":
        parts.append(classifier)
        parts.append(linear_mode)
        parts.append(score_mode)
    elif classifier != "prototype":
        parts.append(classifier)
    parts.append(policy)
    if policy in UNCERTAINTY_MODE_POLICIES and uncertainty_mode != "max_prob":
        parts.append(uncertainty_mode)
    if policy in THRESHOLD_SWEEP_POLICIES:
        if threshold is None:
            raise ValueError(f"{policy} requires a threshold")
        parts.append(f"thr{threshold:.2f}")
    if policy == "random" and seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(parts)


def _summary_name(
    policy: str,
    classifier: str,
    linear_mode: str = "multinomial",
    score_mode: str = "raw_margin",
    uncertainty_mode: str = "max_prob",
    shift_name: str = "clean",
    shift_severity: float = 0.0,
) -> str:
    prefix = _shift_fragment(shift_name, shift_severity)
    prefix = "" if prefix is None else f"{prefix}_"
    mode_suffix = ""
    if policy in UNCERTAINTY_MODE_POLICIES and uncertainty_mode != "max_prob":
        mode_suffix = f"_{uncertainty_mode}"
    if classifier == "prototype":
        return f"{prefix}{policy}{mode_suffix}_summary.csv"
    if classifier == "linear_probe":
        return f"{prefix}{classifier}_{linear_mode}_{score_mode}_{policy}{mode_suffix}_summary.csv"
    return f"{prefix}{classifier}_{policy}{mode_suffix}_summary.csv"


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _path_metadata(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


def _config_bool(value) -> bool:
    if isinstance(value, str):
        return bool(_bool_arg(value))
    return bool(value)


def _write_per_frame_summary(
    path: Path,
    *,
    command: str,
    dataset: str,
    model: str,
    split: str,
    shift_type: str,
    budgets: list[float],
    policy: str,
    uncertainty_mode: str,
    thresholds: list[float | None],
    output_jsonl: Path,
    stats: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold_text = ", ".join("null" if value is None else f"{float(value):.4g}" for value in thresholds)
    budget_text = ", ".join(f"{float(value):.4g}" for value in budgets)
    lines = [
        "# v0.8 Transition Logging Summary",
        "",
        "- version: v0.8_transition_logging",
        f"- command: `{command}`",
        f"- dataset: {dataset}",
        f"- model: {model}",
        f"- split: {split}",
        f"- shift_type: {shift_type}",
        f"- budgets: {budget_text}",
        f"- policies: {policy} (uncertainty_mode={uncertainty_mode})",
        f"- thresholds: {threshold_text}",
        f"- total_frames: {int(stats['total_frames'])}",
        f"- num_videos: {int(stats['num_videos'])}",
        f"- predict_frames: {int(stats['predict_frames'])}",
        f"- wait_frames: {int(stats['wait_frames'])}",
        f"- abstain_frames: {int(stats['abstain_frames'])}",
        f"- mean_max_prob: {float(stats['mean_max_prob']):.6f}",
        f"- mean_entropy: {float(stats['mean_entropy']):.6f}",
        f"- mean_top2_margin: {float(stats['mean_top2_margin']):.6f}",
        f"- output_jsonl_path: {output_jsonl}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dummy.yaml")
    parser.add_argument(
        "--policy",
        choices=["uniform", "random", "confidence", "confidence_threshold", "uncertainty_threshold", "uncertainty_wait_abstain"],
        required=True,
    )
    parser.add_argument("--classifier", choices=["prototype", "linear_probe", "causal_gru", "causal_tcn"], default="prototype")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--class-balanced-sampling", nargs="?", const=True, default=False, type=_bool_arg)
    parser.add_argument("--linear-class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--linear-mode", choices=["multinomial", "ovr"], default="multinomial")
    parser.add_argument("--score-mode", choices=["raw_margin", "sigmoid", "softmax"], default="raw_margin")
    parser.add_argument("--linear-max-iter", type=int, default=1000)
    parser.add_argument("--linear-tol", type=float, default=1e-3)
    parser.add_argument("--linear-alpha", type=float, default=1e-4)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--gru-num-layers", type=int, default=1)
    parser.add_argument("--gru-dropout", type=float, default=0.0)
    parser.add_argument("--gru-max-epochs", type=int, default=1)
    parser.add_argument("--gru-lr", type=float, default=1e-3)
    parser.add_argument("--gru-weight-decay", type=float, default=0.0)
    parser.add_argument("--gru-chunk-length", type=int, default=512)
    parser.add_argument("--gru-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tcn-hidden-dim", type=int, default=128)
    parser.add_argument("--tcn-num-layers", type=int, default=4)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--tcn-dilations", nargs="+", type=int, default=None)
    parser.add_argument("--tcn-dropout", type=float, default=0.0)
    parser.add_argument("--tcn-max-epochs", type=int, default=20)
    parser.add_argument("--tcn-lr", type=float, default=1e-3)
    parser.add_argument("--tcn-weight-decay", type=float, default=0.0)
    parser.add_argument("--tcn-chunk-length", type=int, default=512)
    parser.add_argument("--tcn-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--background-ratio", type=float, default=1.0)
    parser.add_argument("--min-samples-per-class", type=int, default=200)
    parser.add_argument("--budgets", nargs="+", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--uncertainty-mode", choices=["max_prob", "entropy", "top2_margin"], default="max_prob")
    parser.add_argument("--max-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--allow-test-fit", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--dump-per-frame", action="store_true")
    parser.add_argument("--per-frame-output", default=None)
    parser.add_argument("--split-name", default="test")
    parser.add_argument(
        "--eval-feature-dir",
        "--test-feature-dir",
        dest="eval_feature_dir",
        default=None,
        help="optional shifted/evaluation feature directory; train fitting still uses dataset.feature_dir from config",
    )
    parser.add_argument("--shift-name", default="clean")
    parser.add_argument("--shift-severity", type=float, default=0.0)
    parser.add_argument("--is-clean", nargs="?", const=True, default=None, type=_bool_arg)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--class-overlap-smoke", action="store_true")
    parser.add_argument("--smoke-target-classes", type=int, default=8)
    parser.add_argument("--smoke-min-action-frames-per-class", type=int, default=500)
    parser.add_argument("--smoke-min-overlap-classes", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dump = cfg.evaluation.get("dump_per_frame", cfg.output.get("dump_per_frame", False))
    dump_per_frame = bool(args.dump_per_frame or _config_bool(config_dump))
    train, test = make_datasets(cfg, allow_test_fit=args.allow_test_fit, eval_feature_dir=args.eval_feature_dir)
    results_dir = ensure_dir(args.results_dir or cfg.output.get("results_dir", "results"))
    config_per_frame_output = cfg.evaluation.get("per_frame_output", cfg.output.get("per_frame_output"))
    per_frame_output = Path(args.per_frame_output or config_per_frame_output or (results_dir / "per_frame.jsonl"))
    is_clean = bool(args.shift_name == "clean") if args.is_clean is None else bool(args.is_clean)
    shift_metadata = {
        "shift_name": str(args.shift_name),
        "shift_severity": float(args.shift_severity),
        "is_clean": bool(is_clean),
        "eval_feature_dir": _path_metadata(args.eval_feature_dir or cfg.dataset["feature_dir"]),
    }
    selection_audit = None
    if args.class_overlap_smoke:
        selection_audit = select_class_overlap_smoke(
            train,
            test,
            max_train_videos=args.max_train_videos or 80,
            max_test_videos=args.max_videos or 15,
            background_label=int(cfg.dataset.get("background_label", 0)),
            seed=int(args.seed),
            target_class_count=int(args.smoke_target_classes),
            min_action_frames_per_class=int(args.smoke_min_action_frames_per_class),
            min_overlap_classes=int(args.smoke_min_overlap_classes),
            result_dir=results_dir,
        )
        print("CLASS_OVERLAP_SMOKE_AUDIT=" + json.dumps({
            "selected_train_video_count": len(selection_audit["selected_train_video_ids"]),
            "selected_test_video_count": len(selection_audit["selected_test_video_ids"]),
            "train_classes": selection_audit["selected_train_classes"],
            "test_classes": selection_audit["selected_test_classes"],
            "intersection": selection_audit["selected_intersection"],
            "warning": selection_audit["warning"],
        }, sort_keys=True), flush=True)
    else:
        train = _limit_dataset(train, args.max_train_videos, "--max-train-videos")
        test = _limit_dataset(test, args.max_videos, "--max-videos")
    fit_start = perf_counter()
    clf = make_classifier(
        cfg,
        train,
        classifier=args.classifier,
        max_train_samples=args.max_train_samples,
        class_balanced_sampling=args.class_balanced_sampling,
        linear_class_weight=args.linear_class_weight,
        linear_mode=args.linear_mode,
        score_mode=args.score_mode,
        background_ratio=args.background_ratio,
        min_samples_per_class=args.min_samples_per_class,
        seed=args.seed,
        linear_max_iter=args.linear_max_iter,
        linear_tol=args.linear_tol,
        linear_alpha=args.linear_alpha,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        gru_dropout=args.gru_dropout,
        gru_max_epochs=args.gru_max_epochs,
        gru_lr=args.gru_lr,
        gru_weight_decay=args.gru_weight_decay,
        gru_chunk_length=args.gru_chunk_length,
        gru_device=args.gru_device,
        tcn_hidden_dim=args.tcn_hidden_dim,
        tcn_num_layers=args.tcn_num_layers,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dilations=args.tcn_dilations,
        tcn_dropout=args.tcn_dropout,
        tcn_max_epochs=args.tcn_max_epochs,
        tcn_lr=args.tcn_lr,
        tcn_weight_decay=args.tcn_weight_decay,
        tcn_chunk_length=args.tcn_chunk_length,
        tcn_device=args.tcn_device,
        use_cache=not args.no_cache,
    )
    fit_seconds = perf_counter() - fit_start
    print(
        "CLASSIFIER_FIT_INFO="
        + json.dumps(
            {
                "cache_enabled": not args.no_cache,
                "cache_hit": bool(getattr(clf, "cache_hit_", False)),
                "cache_path": getattr(clf, "cache_path_", None),
                "fit_seconds": fit_seconds,
                "classifier": args.classifier,
                "linear_mode": args.linear_mode if args.classifier == "linear_probe" else None,
                "score_mode": args.score_mode if args.classifier == "linear_probe" else None,
                "max_train_videos": None if args.max_train_videos is None else int(args.max_train_videos),
                "linear_max_iter": int(args.linear_max_iter),
                "linear_tol": float(args.linear_tol),
                "linear_alpha": float(args.linear_alpha),
                "gru_hidden_dim": int(args.gru_hidden_dim) if args.classifier == "causal_gru" else None,
                "gru_num_layers": int(args.gru_num_layers) if args.classifier == "causal_gru" else None,
                "gru_dropout": float(args.gru_dropout) if args.classifier == "causal_gru" else None,
                "gru_max_epochs": int(args.gru_max_epochs) if args.classifier == "causal_gru" else None,
                "gru_lr": float(args.gru_lr) if args.classifier == "causal_gru" else None,
                "gru_weight_decay": float(args.gru_weight_decay) if args.classifier == "causal_gru" else None,
                "gru_chunk_length": int(args.gru_chunk_length) if args.classifier == "causal_gru" else None,
                "gru_device": args.gru_device if args.classifier == "causal_gru" else None,
                "tcn_hidden_dim": int(args.tcn_hidden_dim) if args.classifier == "causal_tcn" else None,
                "tcn_num_layers": int(args.tcn_num_layers) if args.classifier == "causal_tcn" else None,
                "tcn_kernel_size": int(args.tcn_kernel_size) if args.classifier == "causal_tcn" else None,
                "tcn_dilations": args.tcn_dilations if args.classifier == "causal_tcn" else None,
                "tcn_dropout": float(args.tcn_dropout) if args.classifier == "causal_tcn" else None,
                "tcn_max_epochs": int(args.tcn_max_epochs) if args.classifier == "causal_tcn" else None,
                "tcn_lr": float(args.tcn_lr) if args.classifier == "causal_tcn" else None,
                "tcn_weight_decay": float(args.tcn_weight_decay) if args.classifier == "causal_tcn" else None,
                "tcn_chunk_length": int(args.tcn_chunk_length) if args.classifier == "causal_tcn" else None,
                "tcn_device": args.tcn_device if args.classifier == "causal_tcn" else None,
                "device_used": getattr(clf, "device_used_", None),
                "convergence_warning": bool(getattr(clf, "convergence_warning_", False)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    training_diagnostics = clf.training_diagnostics() if hasattr(clf, "training_diagnostics") else None
    if training_diagnostics is not None:
        print("CLASSIFIER_TRAINING_DIAGNOSTICS=" + json.dumps(training_diagnostics, sort_keys=True), flush=True)
        training_log_path = Path(results_dir) / "training_log.md"
        lines = [
            "# Classifier Training Log",
            "",
            f"classifier: {args.classifier}",
            f"cache_enabled: {not args.no_cache}",
            f"cache_hit: {bool(getattr(clf, 'cache_hit_', False))}",
            f"cache_path: {getattr(clf, 'cache_path_', None)}",
            f"fit_seconds: {fit_seconds:.4f}",
            f"device_used: {getattr(clf, 'device_used_', None)}",
            "",
            "## Diagnostics",
            "```json",
            json.dumps(training_diagnostics, indent=2, sort_keys=True),
            "```",
            "",
        ]
        training_log_path.write_text("\n".join(lines), encoding="utf-8")
    train_positive_counts = getattr(clf, "sampling_seen_counts_", None)
    classifier_run_metadata = {
        "classifier_cache_enabled": not args.no_cache,
        "classifier_cache_hit": bool(getattr(clf, "cache_hit_", False)),
        "classifier_cache_path": getattr(clf, "cache_path_", None),
        "classifier_fit_seconds": float(fit_seconds),
        "classifier_convergence_warning": bool(getattr(clf, "convergence_warning_", False)),
        "linear_max_iter": int(args.linear_max_iter),
        "linear_tol": float(args.linear_tol),
        "linear_alpha": float(args.linear_alpha),
        "max_train_videos": None if args.max_train_videos is None else int(args.max_train_videos),
        "gru_hidden_dim": int(args.gru_hidden_dim) if args.classifier == "causal_gru" else None,
        "gru_num_layers": int(args.gru_num_layers) if args.classifier == "causal_gru" else None,
        "gru_dropout": float(args.gru_dropout) if args.classifier == "causal_gru" else None,
        "gru_max_epochs": int(args.gru_max_epochs) if args.classifier == "causal_gru" else None,
        "gru_lr": float(args.gru_lr) if args.classifier == "causal_gru" else None,
        "gru_weight_decay": float(args.gru_weight_decay) if args.classifier == "causal_gru" else None,
        "gru_chunk_length": int(args.gru_chunk_length) if args.classifier == "causal_gru" else None,
        "tcn_hidden_dim": int(args.tcn_hidden_dim) if args.classifier == "causal_tcn" else None,
        "tcn_num_layers": int(args.tcn_num_layers) if args.classifier == "causal_tcn" else None,
        "tcn_kernel_size": int(args.tcn_kernel_size) if args.classifier == "causal_tcn" else None,
        "tcn_dilations": None if args.classifier != "causal_tcn" or args.tcn_dilations is None else json.dumps(args.tcn_dilations),
        "tcn_dropout": float(args.tcn_dropout) if args.classifier == "causal_tcn" else None,
        "tcn_max_epochs": int(args.tcn_max_epochs) if args.classifier == "causal_tcn" else None,
        "tcn_lr": float(args.tcn_lr) if args.classifier == "causal_tcn" else None,
        "tcn_weight_decay": float(args.tcn_weight_decay) if args.classifier == "causal_tcn" else None,
        "tcn_chunk_length": int(args.tcn_chunk_length) if args.classifier == "causal_tcn" else None,
        "tcn_device": args.tcn_device if args.classifier == "causal_tcn" else None,
        "classifier_device_used": getattr(clf, "device_used_", None),
        "class_overlap_smoke": bool(args.class_overlap_smoke),
        "selected_train_video_count": None if selection_audit is None else int(len(selection_audit["selected_train_video_ids"])),
        "selected_test_video_count": None if selection_audit is None else int(len(selection_audit["selected_test_video_ids"])),
        "selected_intersection": None if selection_audit is None else json.dumps(selection_audit["selected_intersection"]),
    }
    classifier_run_metadata.update(shift_metadata)
    evaluator = CausalStreamingEvaluator()
    budgets = args.budgets or cfg.evaluation.get("budgets", [0.10, 0.25, 0.50, 1.00])
    stable_steps = int(cfg.evaluation.get("stable_steps", 3))
    background_label = int(cfg.dataset.get("background_label", 0))
    thresholds = _threshold_values(args.policy, args.thresholds, args.threshold)
    seeds = _seed_values(args.policy, args.seed, args.seeds)
    summary_path = Path(results_dir) / _summary_name(
        args.policy,
        args.classifier,
        args.linear_mode,
        args.score_mode,
        args.uncertainty_mode,
        args.shift_name,
        args.shift_severity,
    )
    run_names = [
        _run_name(
            args.policy,
            threshold,
            seed,
            classifier=args.classifier,
            linear_mode=args.linear_mode,
            score_mode=args.score_mode,
            uncertainty_mode=args.uncertainty_mode,
            shift_name=args.shift_name,
            shift_severity=args.shift_severity,
        )
        for threshold in thresholds
        for seed in seeds
    ]
    if args.summary_only:
        output_paths = [summary_path] + [
            Path(results_dir) / f"{run_name}_budget{budget:.2f}_video_summary.csv"
            for run_name in run_names
            for budget in budgets
        ]
    else:
        output_paths = [summary_path] + [
            Path(results_dir) / f"{run_name}_budget{budget:.2f}.jsonl"
            for run_name in run_names
            for budget in budgets
        ]
    if dump_per_frame:
        output_paths.append(per_frame_output)
        output_paths.append(Path(results_dir) / "summary_readable.md")
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        existing_list = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(f"Output files already exist. Use --overwrite to replace them:\n{existing_list}")

    summaries = []
    per_frame_stats = PerFrameLogStats()
    per_frame_global_idx = 0
    per_frame_handle = open_jsonl_writer(per_frame_output) if dump_per_frame else None
    runs = [(threshold, seed, budget) for threshold in thresholds for seed in seeds for budget in budgets]
    try:
        for threshold, seed, budget in tqdm(runs, desc=f"policy={args.policy}"):
            run_name = _run_name(
                args.policy,
                threshold,
                seed,
                classifier=args.classifier,
                linear_mode=args.linear_mode,
                score_mode=args.score_mode,
                uncertainty_mode=args.uncertainty_mode,
                shift_name=args.shift_name,
                shift_severity=args.shift_severity,
            )
            policy = make_policy(
                args.policy,
                clf,
                threshold=args.threshold if threshold is None else threshold,
                uncertainty_mode=args.uncertainty_mode,
                max_wait=args.max_wait,
                seed=args.seed if seed is None else seed,
            )
            frames = []
            video_summaries = []
            budget_start = perf_counter()
            for video_idx, video in enumerate(test, start=1):
                logs_for_video = evaluator.evaluate_video(video, policy=policy, budget=budget)
                frames.append(logs_for_video)
                video_row = summarize_log(
                    logs_for_video,
                    stable_steps=stable_steps,
                    background_label=background_label,
                    train_positive_counts=train_positive_counts,
                )
                video_row.update({
                    "video_id": video.video_id,
                    "policy": args.policy,
                    "classifier": args.classifier,
                    "linear_mode": args.linear_mode if args.classifier == "linear_probe" else None,
                    "score_mode": args.score_mode if args.classifier == "linear_probe" else None,
                    "budget": float(budget),
                    "timesteps": int(video.length),
                    "seed": None if seed is None else int(seed),
                    "uncertainty_mode": args.uncertainty_mode if args.policy in UNCERTAINTY_MODE_POLICIES else None,
                })
                video_row.update(classifier_run_metadata)
                if threshold is not None:
                    video_row["threshold"] = float(threshold)
                if args.policy in THRESHOLD_SWEEP_POLICIES:
                    video_row["max_wait"] = int(args.max_wait)
                video_summaries.append(video_row)
                if args.progress_every and video_idx % args.progress_every == 0:
                    elapsed = perf_counter() - budget_start
                    print(f"{run_name} budget={budget:.2f}: processed {video_idx}/{len(test)} videos in {elapsed:.1f}s", flush=True)
            logs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if dump_per_frame and per_frame_handle is not None:
                for per_frame_row in iter_per_frame_rows(
                    logs,
                    model=args.classifier,
                    policy=args.policy,
                    threshold=threshold,
                    shift_type=args.shift_name,
                    split=args.split_name,
                    run_id=f"{run_name}_budget{float(budget):.2f}",
                    checkpoint_path=None,
                    frame_global_start=per_frame_global_idx,
                    uncertainty_mode=args.uncertainty_mode if args.policy in UNCERTAINTY_MODE_POLICIES else None,
                ):
                    write_jsonl_row(per_frame_handle, per_frame_row)
                    per_frame_stats.update(per_frame_row)
                    per_frame_global_idx += 1
            log_path = Path(results_dir) / f"{run_name}_budget{budget:.2f}.jsonl"
            video_summary_path = Path(results_dir) / f"{run_name}_budget{budget:.2f}_video_summary.csv"
            if args.summary_only:
                pd.DataFrame(video_summaries).to_csv(video_summary_path, index=False)
            else:
                write_jsonl(logs, log_path)
            row = summarize_log(
                logs,
                stable_steps=stable_steps,
                background_label=background_label,
                train_positive_counts=train_positive_counts,
            )
            row.update({
                "policy": args.policy,
                "classifier": args.classifier,
                "linear_mode": args.linear_mode if args.classifier == "linear_probe" else None,
                "score_mode": args.score_mode if args.classifier == "linear_probe" else None,
                "budget": float(budget),
                "seed": None if seed is None else int(seed),
                "log_path": "" if args.summary_only else str(log_path),
                "video_summary_path": str(video_summary_path) if args.summary_only else "",
                "num_videos": int(len(test)),
                "summary_only": bool(args.summary_only),
                "uncertainty_mode": args.uncertainty_mode if args.policy in UNCERTAINTY_MODE_POLICIES else None,
            })
            row.update(classifier_run_metadata)
            if threshold is not None:
                row["threshold"] = float(threshold)
            if args.policy in THRESHOLD_SWEEP_POLICIES:
                row["max_wait"] = int(args.max_wait)
            summaries.append(row)
    finally:
        if per_frame_handle is not None:
            per_frame_handle.close()

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    if dump_per_frame:
        _write_per_frame_summary(
            Path(results_dir) / "summary_readable.md",
            command=" ".join(sys.argv),
            dataset=str(cfg.dataset.get("name", cfg.path.stem)),
            model=args.classifier,
            split=args.split_name,
            shift_type=args.shift_name,
            budgets=[float(value) for value in budgets],
            policy=args.policy,
            uncertainty_mode=args.uncertainty_mode,
            thresholds=thresholds,
            output_jsonl=per_frame_output,
            stats=per_frame_stats.as_dict(),
        )
    print(f"Wrote logs and summary to {results_dir}")


if __name__ == "__main__":
    main()
