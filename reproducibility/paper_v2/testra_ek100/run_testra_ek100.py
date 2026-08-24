#!/usr/bin/env python3
"""Run the frozen TeSTra-EK100 checkpoint audit.

This script has two deliberately separate responsibilities:

1. reproduce the official EK100 verb-anticipation score as an integrity gate;
2. export the current-timestep verb probabilities and evaluate the frozen
   native/EMA/boxcar comparison used by the OAD stress-test paper.

The checkpoint, split, output index, smoothing parameters, metrics, and decision
rules are fixed in ``protocol/testra_ek100_protocol.json``.  A failed result is
reported; it does not authorize selecting another checkpoint or postprocessor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SequencePrediction:
    sequence_id: str
    labels: np.ndarray
    scores: np.ndarray
    active_counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("full", "export", "audit"), default="full")
    parser.add_argument("--cache", type=Path)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(exponent.sum(axis=-1, keepdims=True), 1e-12)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    bundle = args.bundle_root.resolve()
    inference_checkpoint = bundle / "checkpoints" / "testra_ek100_laplace_inference_only.pth"
    default_checkpoint = (
        inference_checkpoint
        if inference_checkpoint.is_file()
        else bundle / "checkpoints" / "testra_ek100_laplace.pth"
    )
    checkpoint = (args.checkpoint or default_checkpoint).resolve()
    protocol_path = (args.protocol or bundle / "protocols" / "testra_ek100_protocol.json").resolve()
    output_dir = args.output_dir.resolve()
    cache = (args.cache or output_dir / "testra_ek100_current_verb_probabilities.npz").resolve()
    data_root = args.data_root.resolve()
    return bundle, checkpoint, protocol_path, output_dir, cache


def verify_inventory(data_root: Path, sessions: list[str], protocol: dict[str, object]) -> dict[str, object]:
    input_spec = dict(protocol["inputs"])
    directories = {
        "rgb": data_root / str(input_spec["rgb_subdir"]),
        "flow": data_root / str(input_spec["flow_subdir"]),
        "verb": data_root / str(input_spec["verb_target_subdir"]),
    }
    missing: dict[str, list[str]] = {}
    for name, directory in directories.items():
        absent = [session for session in sessions if not (directory / f"{session}.npy").is_file()]
        if absent:
            missing[name] = absent
    if missing:
        preview = {name: values[:10] for name, values in missing.items()}
        raise FileNotFoundError(f"EK100 validation inventory is incomplete: {preview}")
    return {
        "data_root": str(data_root),
        "session_count": len(sessions),
        "directories": {name: str(path) for name, path in directories.items()},
        "missing": 0,
    }


def infer_cfg(bundle: Path, data_root: Path, checkpoint: Path, device_index: str = "0"):
    testra_root = bundle / "vendor" / "TeSTra"
    source_root = testra_root / "src"
    sys.path.insert(0, str(source_root))
    os.chdir(testra_root)

    from rekognition_online_action_detection.config.defaults import get_cfg

    cfg = get_cfg()
    config_path = testra_root / "configs" / "EK100" / "TESTRA" / (
        "testra_long_64_work_5_anti_2_kinetics_2x_mixup_v+n_eql_decay_0.9.yaml"
    )
    cfg.merge_from_file(str(config_path))
    data_info_path = testra_root / "data" / "data_info.json"
    data_info = json.loads(data_info_path.read_text(encoding="utf-8"))["EK100"]
    cfg.GPU = device_index
    cfg.DATA.DATA_ROOT = str(data_root)
    cfg.DATA.CLASS_NAMES = data_info["class_names"]
    cfg.DATA.NUM_CLASSES = int(data_info["num_classes"])
    cfg.DATA.IGNORE_INDEX = int(data_info["ignore_index"])
    cfg.DATA.METRICS = data_info["metrics"]
    cfg.DATA.FPS = int(data_info["fps"])
    cfg.DATA.TRAIN_SESSION_SET = data_info["train_session_set"]
    cfg.DATA.TEST_SESSION_SET = data_info["test_session_set"]
    cfg.DATA.EK_EXT_PATH = str(testra_root / "external" / "rulstm" / "RULSTM" / "data" / "ek100")
    cfg.MODEL.CHECKPOINT = str(checkpoint)
    if cfg.INPUT.MODALITY == "twostream":
        cfg.INPUT.MODALITY = "visual+motion"
    if cfg.INPUT.MODALITY == "threestream":
        cfg.INPUT.MODALITY = "visual+motion+object"

    lstr = cfg.MODEL.LSTR
    lstr.AGES_MEMORY_LENGTH = int(lstr.AGES_MEMORY_SECONDS * cfg.DATA.FPS)
    lstr.LONG_MEMORY_LENGTH = int(lstr.LONG_MEMORY_SECONDS * cfg.DATA.FPS)
    lstr.WORK_MEMORY_LENGTH = int(lstr.WORK_MEMORY_SECONDS * cfg.DATA.FPS)
    lstr.TOTAL_MEMORY_LENGTH = int(
        lstr.AGES_MEMORY_LENGTH + lstr.LONG_MEMORY_LENGTH + lstr.WORK_MEMORY_LENGTH
    )
    lstr.ANTICIPATION_LENGTH = int(lstr.ANTICIPATION_SECONDS * cfg.DATA.FPS)
    lstr.AGES_MEMORY_NUM_SAMPLES = int(lstr.AGES_MEMORY_LENGTH // lstr.AGES_MEMORY_SAMPLE_RATE)
    lstr.LONG_MEMORY_NUM_SAMPLES = int(lstr.LONG_MEMORY_LENGTH // lstr.LONG_MEMORY_SAMPLE_RATE)
    lstr.WORK_MEMORY_NUM_SAMPLES = int(lstr.WORK_MEMORY_LENGTH // lstr.WORK_MEMORY_SAMPLE_RATE)
    lstr.TOTAL_MEMORY_NUM_SAMPLES = int(
        lstr.AGES_MEMORY_NUM_SAMPLES + lstr.LONG_MEMORY_NUM_SAMPLES + lstr.WORK_MEMORY_NUM_SAMPLES
    )
    lstr.ANTICIPATION_NUM_SAMPLES = int(lstr.ANTICIPATION_LENGTH // lstr.ANTICIPATION_SAMPLE_RATE)
    return cfg, data_info, config_path, data_info_path


def build_model(cfg, checkpoint: Path, requested_device: str):
    import torch
    from rekognition_online_action_detection.models import build_model as official_build_model

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(requested_device)
    torch.manual_seed(int(cfg.SEED))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.SEED))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = official_build_model(cfg, device)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" not in payload:
        raise KeyError("Official checkpoint does not contain model_state_dict")
    incompatibility = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompatibility}")
    del payload
    model.eval()
    return model, device


def uniform_sampler(start: int, end: int, num_samples: int, sample_rate: int) -> np.ndarray:
    if start < 0:
        start = (end + 1) % sample_rate
    indices = np.arange(start, end + 1)[::sample_rate]
    padding = int(num_samples) - int(indices.shape[0])
    if padding > 0:
        indices = np.concatenate((np.zeros(padding), indices))
    return np.sort(indices).astype(np.int32)


def sample_indices(cfg, work_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lstr = cfg.MODEL.LSTR
    work_end = work_start + int(lstr.WORK_MEMORY_LENGTH)
    work_indices = np.arange(work_start, work_end).clip(0)
    work_indices = work_indices[:: int(lstr.WORK_MEMORY_SAMPLE_RATE)].astype(np.int32)
    long_start = work_start - int(lstr.LONG_MEMORY_LENGTH)
    long_end = work_start - 1
    long_indices = uniform_sampler(
        long_start,
        long_end,
        int(lstr.LONG_MEMORY_NUM_SAMPLES),
        int(lstr.LONG_MEMORY_SAMPLE_RATE),
    ).clip(0)
    padding_mask = np.zeros(long_indices.shape[0], dtype=np.float32)
    last_zero = bisect_right(long_indices.tolist(), 0) - 1
    if last_zero > 0:
        padding_mask[:last_zero] = float("-inf")
    return long_indices, work_indices, padding_mask


def stack_batch(
    rgb: np.ndarray,
    flow: np.ndarray,
    cfg,
    starts: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    visual: list[np.ndarray] = []
    motion: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    for start in starts:
        long_indices, work_indices, mask = sample_indices(cfg, start)
        all_indices = np.concatenate((long_indices, work_indices))
        visual.append(np.asarray(rgb[all_indices], dtype=np.float32))
        motion.append(np.asarray(flow[all_indices], dtype=np.float32))
        masks.append(mask)
        queries.append(work_indices)
    return np.stack(visual), np.stack(motion), np.stack(masks), queries


def load_validation_segments(path: Path) -> dict[str, list[dict[str, int | str]]]:
    by_video: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        for row in reader:
            if len(row) != 7:
                raise ValueError(f"Unexpected EK100 validation row: {row}")
            segment = {
                "id": row[0].strip(),
                "video": row[1].strip(),
                "start_f": int(row[2]),
                "end_f": int(row[3]),
                "verb": int(row[4]),
                "noun": int(row[5]),
                "action": int(row[6]),
            }
            by_video[str(segment["video"])].append(segment)
    return by_video


def mean_top5_recall_multiple_timesteps(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    results: list[float] = []
    for time_index in range(scores.shape[1]):
        time_scores = scores[:, time_index, :]
        recalls: list[float] = []
        for label in np.unique(labels):
            selected = labels == label
            rankings = time_scores[selected].argsort(axis=1)[:, ::-1]
            recalls.append(float(np.mean((rankings[:, :5] == int(label)).any(axis=1))))
        results.append(float(np.mean(recalls)))
    return np.asarray(results, dtype=np.float64)


def save_cache(path: Path, predictions: list[SequencePrediction]) -> None:
    ids = [item.sequence_id for item in predictions]
    offsets = [0]
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    active_counts: list[np.ndarray] = []
    for item in predictions:
        labels.append(np.asarray(item.labels, dtype=np.int16))
        scores.append(np.asarray(item.scores, dtype=np.float32))
        active_counts.append(np.asarray(item.active_counts, dtype=np.int8))
        offsets.append(offsets[-1] + len(item.labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        video_ids=np.asarray(ids),
        offsets=np.asarray(offsets, dtype=np.int64),
        labels=np.concatenate(labels),
        scores=np.concatenate(scores),
        active_counts=np.concatenate(active_counts),
    )


def load_cache(path: Path) -> list[SequencePrediction]:
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["video_ids"].astype(str)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        scores = np.asarray(archive["scores"], dtype=np.float32)
        active_counts = np.asarray(archive["active_counts"], dtype=np.int64)
    return [
        SequencePrediction(
            sequence_id,
            labels[int(offsets[index]) : int(offsets[index + 1])],
            scores[int(offsets[index]) : int(offsets[index + 1])],
            active_counts[int(offsets[index]) : int(offsets[index + 1])],
        )
        for index, sequence_id in enumerate(ids.tolist())
    ]


def export_predictions(
    model,
    device,
    cfg,
    sessions: list[str],
    segments_by_video: dict[str, list[dict[str, int | str]]],
    data_root: Path,
    protocol: dict[str, object],
    batch_size: int,
    cache_path: Path,
    output_dir: Path,
) -> tuple[list[SequencePrediction], dict[str, object]]:
    import torch

    inputs = dict(protocol["inputs"])
    rgb_dir = data_root / str(inputs["rgb_subdir"])
    flow_dir = data_root / str(inputs["flow_subdir"])
    verb_dir = data_root / str(inputs["verb_target_subdir"])
    work_length = int(cfg.MODEL.LSTR.WORK_MEMORY_LENGTH)
    work_rate = int(cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE)
    work_samples = int(cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES)
    anticipation_samples = int(cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES)
    anticipation_length = int(cfg.MODEL.LSTR.ANTICIPATION_LENGTH)
    num_verbs = int(protocol["model"]["verb_classes_including_background"])
    fps = int(cfg.DATA.FPS)

    predictions: list[SequencePrediction] = []
    segment_scores: list[np.ndarray] = []
    segment_labels: list[int] = []
    discarded_segments = 0
    total_valid = 0
    started = time.time()

    with torch.inference_mode():
        for session_index, session in enumerate(sessions, start=1):
            rgb = np.load(rgb_dir / f"{session}.npy", mmap_mode="r")
            flow = np.load(flow_dir / f"{session}.npy", mmap_mode="r")
            target = np.load(verb_dir / f"{session}.npy", mmap_mode="r")
            length = min(len(rgb), len(flow), len(target))
            if length < work_length:
                raise ValueError(f"{session}: only {length} timesteps, fewer than work memory {work_length}")
            if rgb.shape[1] != 1024 or flow.shape[1] != 1024 or target.shape[1] != num_verbs:
                raise ValueError(
                    f"{session}: unexpected shapes rgb={rgb.shape}, flow={flow.shape}, verb={target.shape}"
                )

            current_logits = np.full((length, num_verbs), np.nan, dtype=np.float32)
            official_future_logits = np.zeros(
                (length, num_verbs, anticipation_samples), dtype=np.float32
            )
            all_starts = list(range(0, length - work_length + 1))
            for chunk_start in range(0, len(all_starts), int(batch_size)):
                starts = all_starts[chunk_start : chunk_start + int(batch_size)]
                visual_np, motion_np, mask_np, queries = stack_batch(rgb, flow, cfg, starts)
                visual = torch.from_numpy(visual_np).to(device, non_blocking=True)
                motion = torch.from_numpy(motion_np).to(device, non_blocking=True)
                padding_mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
                _, verb_logits_t, _ = model(visual, motion, motion, padding_mask)
                verb_logits = verb_logits_t.detach().float().cpu().numpy()
                if verb_logits.shape[1] != work_samples + anticipation_samples:
                    raise ValueError(f"Unexpected verb output shape: {verb_logits.shape}")

                for batch_index, (start, query_indices) in enumerate(zip(starts, queries)):
                    logits = verb_logits[batch_index]
                    if int(query_indices[0]) < work_rate:
                        current_logits[query_indices] = logits[: len(query_indices)]
                        for anticipation_index in range(anticipation_samples):
                            future_indices = np.arange(
                                int(query_indices[-1]) + work_rate,
                                int(query_indices[-1]) + anticipation_index + 1 + work_rate,
                            )
                            full_indices = np.concatenate((query_indices, future_indices))
                            full_indices = full_indices[full_indices < length]
                            official_future_logits[full_indices, :, anticipation_index] = logits[
                                : len(full_indices)
                            ]
                    else:
                        current_logits[int(query_indices[-1])] = logits[work_samples - 1]
                        for anticipation_index in range(anticipation_samples):
                            future_index = int(query_indices[-1]) + anticipation_index + 1
                            if future_index < length:
                                official_future_logits[future_index, :, anticipation_index] = logits[
                                    work_samples + anticipation_index
                                ]

            valid = np.isfinite(current_logits).all(axis=1)
            expected_valid = length - 1
            if int(valid.sum()) != expected_valid:
                absent = np.flatnonzero(~valid).tolist()
                raise ValueError(
                    f"{session}: expected {expected_valid} current outputs, got {int(valid.sum())}; "
                    f"unfilled indices begin {absent[:20]}"
                )
            probabilities = softmax(current_logits[valid]).astype(np.float32)
            target_values = np.asarray(target[:length], dtype=np.float32)
            active = target_values > 0
            labels = np.argmax(target_values, axis=1).astype(np.int64)
            active_counts = active.sum(axis=1).astype(np.int64)
            predictions.append(
                SequencePrediction(session, labels[valid], probabilities, active_counts[valid])
            )
            total_valid += int(valid.sum())

            for segment in segments_by_video.get(session, []):
                start_step = int(np.floor(int(segment["start_f"]) / 30.0 * fps))
                if start_step >= anticipation_length and start_step < length:
                    scores = official_future_logits[start_step, 1:, ::-1].T
                else:
                    scores = np.zeros((anticipation_samples, num_verbs - 1), dtype=np.float32)
                    discarded_segments += 1
                segment_scores.append(scores)
                segment_labels.append(int(segment["verb"]))

            elapsed = time.time() - started
            print(
                f"[{session_index:03d}/{len(sessions):03d}] {session}: "
                f"{int(valid.sum())} current outputs; elapsed {elapsed / 60.0:.1f} min",
                flush=True,
            )

    if len(segment_scores) != int(protocol["official_reproduction"]["expected_validation_segments"]):
        raise ValueError(
            f"Expected {protocol['official_reproduction']['expected_validation_segments']} validation segments, "
            f"found {len(segment_scores)}"
        )
    save_cache(cache_path, predictions)
    scores_array = np.stack(segment_scores)
    labels_array = np.asarray(segment_labels, dtype=np.int64)
    recalls = mean_top5_recall_multiple_timesteps(scores_array, labels_array) * 100.0
    horizons = np.linspace(
        float(cfg.MODEL.LSTR.ANTICIPATION_SECONDS),
        float(cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE) / float(cfg.DATA.FPS),
        anticipation_samples,
    )
    official = {
        "metric": "EK100 overall mean top-5 verb recall",
        "horizon_seconds": horizons.tolist(),
        "values_percent": recalls.tolist(),
        "published_gate_horizon_seconds": float(protocol["official_reproduction"]["gate_horizon_seconds"]),
        "published_value_percent": float(protocol["official_reproduction"]["published_verb_percent"]),
        "tolerance_percentage_points": float(protocol["official_reproduction"]["tolerance_percentage_points"]),
        "validation_segments": len(segment_scores),
        "early_segments_scored_as_uniform": discarded_segments,
        "current_output_timesteps": total_valid,
    }
    gate_horizon = official["published_gate_horizon_seconds"]
    gate_index = int(np.argmin(np.abs(horizons - gate_horizon)))
    error = float(recalls[gate_index] - official["published_value_percent"])
    official["reproduction_error_percentage_points"] = error
    official["gate_pass"] = abs(error) <= official["tolerance_percentage_points"]
    (output_dir / "official_reproduction.json").write_text(
        json.dumps(official, indent=2) + "\n", encoding="utf-8"
    )
    return predictions, official


def ema_probabilities(scores: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) <= 1:
        return values.copy()
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = alpha * smoothed[index - 1] + (1.0 - alpha) * values[index]
    smoothed /= np.maximum(smoothed.sum(axis=1, keepdims=True), 1e-12)
    return smoothed


def boxcar_probabilities(scores: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if window <= 1 or len(values) <= 1:
        return values.copy()
    padded = np.pad(values, ((window - 1, 0), (0, 0)), mode="edge")
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0)]
    )
    smoothed = (cumulative[window:] - cumulative[:-window]) / float(window)
    smoothed /= np.maximum(smoothed.sum(axis=1, keepdims=True), 1e-12)
    return smoothed


def condition_scores(scores: np.ndarray, condition: str, protocol: dict[str, object]) -> np.ndarray:
    if condition == "native":
        return np.asarray(scores, dtype=np.float64)
    smoothing = dict(protocol["smoothing"])
    if condition == "ema_alpha_0.50":
        return ema_probabilities(scores, float(smoothing["ema_alpha"]))
    if condition == "boxcar_w3":
        return boxcar_probabilities(scores, int(smoothing["boxcar_window"]))
    raise ValueError(f"Unknown condition {condition}")


def temporal_fragmentation_index(labels: np.ndarray, predictions: np.ndarray) -> float:
    ground_truth_segments = 1 + int(np.count_nonzero(labels[1:] != labels[:-1]))
    starts = np.concatenate(([0], np.flatnonzero(predictions[1:] != predictions[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(predictions)]))
    disagreement = sum(float(np.mean(predictions[start:end] != labels[start:end])) for start, end in zip(starts, ends))
    return float(disagreement / ground_truth_segments)


def ece_stats(correct: np.ndarray, confidence: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.clip(np.floor(confidence * bins).astype(np.int64), 0, bins - 1)
    counts = np.bincount(indices, minlength=bins).astype(np.int64)
    correct_sum = np.bincount(indices, weights=correct.astype(np.float64), minlength=bins)
    confidence_sum = np.bincount(indices, weights=confidence, minlength=bins)
    return counts, correct_sum, confidence_sum


def ece_from_stats(counts: np.ndarray, correct_sum: np.ndarray, confidence_sum: np.ndarray) -> float:
    total = float(counts.sum())
    nonempty = counts > 0
    accuracy = correct_sum[nonempty] / counts[nonempty]
    confidence = confidence_sum[nonempty] / counts[nonempty]
    return float(np.sum((counts[nonempty] / total) * np.abs(accuracy - confidence)))


def transition_stats(
    labels: np.ndarray,
    predictions: np.ndarray,
    active_counts: np.ndarray,
    horizon: int,
) -> dict[str, float | int]:
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    eligible = 0
    delay_sum = 0.0
    missed_sum = 0.0
    for boundary_number, boundary in enumerate(boundaries.tolist()):
        if not (active_counts[boundary - 1] == 1 and active_counts[boundary] == 1):
            continue
        next_boundary = int(boundaries[boundary_number + 1]) if boundary_number + 1 < len(boundaries) else len(labels)
        end = min(len(labels), next_boundary, boundary + horizon)
        if end <= boundary:
            continue
        matches = np.flatnonzero(predictions[boundary:end] == int(labels[boundary]))
        if len(matches):
            delay_sum += float(matches[0])
        else:
            delay_sum += float(horizon)
            missed_sum += 1.0
        eligible += 1
    return {"transition_count": eligible, "delay_sum": delay_sum, "missed_sum": missed_sum}


def sequence_stats(
    sequence: SequencePrediction,
    condition: str,
    protocol: dict[str, object],
) -> dict[str, object]:
    metrics = dict(protocol["metrics"])
    bins = int(metrics["ece_bins"])
    horizon = int(metrics["transition_horizon_feature_timesteps"])
    scores = condition_scores(sequence.scores, condition, protocol)
    predictions = np.argmax(scores, axis=1)
    confidence = np.max(scores, axis=1)
    correct = predictions == sequence.labels
    counts, correct_sum, confidence_sum = ece_stats(correct, confidence, bins)
    transition = transition_stats(sequence.labels, predictions, sequence.active_counts, horizon)
    row: dict[str, object] = {
        "sequence_id": sequence.sequence_id,
        "condition": condition,
        "n_timesteps": len(sequence.labels),
        "correct_sum": int(correct.sum()),
        "tfi": temporal_fragmentation_index(sequence.labels, predictions),
        "predicted_switches": int(np.count_nonzero(predictions[1:] != predictions[:-1])),
        **transition,
    }
    for index in range(bins):
        row[f"ece_bin_{index:02d}_count"] = int(counts[index])
        row[f"ece_bin_{index:02d}_correct_sum"] = float(correct_sum[index])
        row[f"ece_bin_{index:02d}_confidence_sum"] = float(confidence_sum[index])
    return row


METRIC_NAMES = (
    "accuracy",
    "global_ece",
    "mean_tfi",
    "mean_predicted_switches",
    "mean_transition_delay",
    "missed_transition_rate",
)


def aggregate(
    rows: list[dict[str, object]],
    bins: int,
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    if indices is None:
        indices = np.arange(len(rows), dtype=np.int64)
    selected = [rows[int(index)] for index in np.asarray(indices, dtype=np.int64)]
    timesteps = float(sum(int(row["n_timesteps"]) for row in selected))
    transitions = float(sum(int(row["transition_count"]) for row in selected))
    counts = np.asarray(
        [sum(int(row[f"ece_bin_{index:02d}_count"]) for row in selected) for index in range(bins)],
        dtype=np.float64,
    )
    correct_sum = np.asarray(
        [sum(float(row[f"ece_bin_{index:02d}_correct_sum"]) for row in selected) for index in range(bins)],
        dtype=np.float64,
    )
    confidence_sum = np.asarray(
        [sum(float(row[f"ece_bin_{index:02d}_confidence_sum"]) for row in selected) for index in range(bins)],
        dtype=np.float64,
    )
    return {
        "accuracy": float(sum(int(row["correct_sum"]) for row in selected) / timesteps),
        "global_ece": ece_from_stats(counts, correct_sum, confidence_sum),
        "mean_tfi": float(np.mean([float(row["tfi"]) for row in selected])),
        "mean_predicted_switches": float(np.mean([float(row["predicted_switches"]) for row in selected])),
        "mean_transition_delay": float(sum(float(row["delay_sum"]) for row in selected) / transitions),
        "missed_transition_rate": float(sum(float(row["missed_sum"]) for row in selected) / transitions),
        "n_timesteps": timesteps,
        "n_sequences": float(len(selected)),
        "n_transitions": transitions,
    }


def audit_predictions(
    predictions: list[SequencePrediction],
    protocol: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    conditions = list(protocol["smoothing"]["conditions"])
    bins = int(protocol["metrics"]["ece_bins"])
    repetitions = int(protocol["uncertainty"]["bootstrap_repetitions"])
    seed = int(protocol["uncertainty"]["bootstrap_seed"])
    margin = float(protocol["decision_rules"]["accuracy_noninferiority_margin"])
    records = [sequence_stats(sequence, condition, protocol) for sequence in predictions for condition in conditions]
    groups = {
        condition: sorted(
            [row for row in records if row["condition"] == condition],
            key=lambda row: str(row["sequence_id"]),
        )
        for condition in conditions
    }
    ids = [[str(row["sequence_id"]) for row in groups[condition]] for condition in conditions]
    if any(group_ids != ids[0] for group_ids in ids[1:]):
        raise ValueError("Sequence IDs differ across conditions")
    points = {condition: aggregate(groups[condition], bins) for condition in conditions}
    rng = np.random.default_rng(seed)
    samples = {
        f"{condition}__{metric}": np.empty(repetitions, dtype=np.float64)
        for condition in conditions[1:]
        for metric in METRIC_NAMES
    }
    for bootstrap_index in range(repetitions):
        selected = rng.integers(0, len(ids[0]), size=len(ids[0]), dtype=np.int64)
        native = aggregate(groups["native"], bins, selected)
        for condition in conditions[1:]:
            smoothed = aggregate(groups[condition], bins, selected)
            for metric in METRIC_NAMES:
                samples[f"{condition}__{metric}"][bootstrap_index] = smoothed[metric] - native[metric]

    comparison_rows: list[dict[str, object]] = []
    gates: dict[str, dict[str, bool]] = {}
    for condition in conditions[1:]:
        lookup: dict[str, dict[str, object]] = {}
        for metric in METRIC_NAMES:
            values = samples[f"{condition}__{metric}"]
            row = {
                "comparison": f"{condition}_minus_native",
                "metric": metric,
                "native_point": points["native"][metric],
                "smoothed_point": points[condition][metric],
                "delta_point": points[condition][metric] - points["native"][metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
                "resampling_unit": "official EK100 validation video",
            }
            comparison_rows.append(row)
            lookup[metric] = row
        checks = {
            "accuracy_noninferior": float(lookup["accuracy"]["ci_low"]) >= margin,
            "global_ece_improves": float(lookup["global_ece"]["ci_high"]) < 0.0,
            "tfi_improves": float(lookup["mean_tfi"]["ci_high"]) < 0.0,
            "delay_worsens": float(lookup["mean_transition_delay"]["ci_low"]) > 0.0,
            "miss_rate_worsens": float(lookup["missed_transition_rate"]["ci_low"]) > 0.0,
        }
        checks["full_ranking_inversion"] = all(checks.values())
        gates[condition] = checks

    write_csv(output_dir / "per_sequence_statistics.csv", records)
    write_csv(output_dir / "point_estimates.csv", [{"condition": key, **value} for key, value in points.items()])
    write_csv(output_dir / "bootstrap_deltas.csv", comparison_rows)
    np.savez_compressed(output_dir / "bootstrap_samples.npz", **samples)
    summary = {
        "conditions": conditions,
        "points": points,
        "gates": gates,
        "result_is_reported_without_checkpoint_or_parameter_substitution": True,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_report(official: dict[str, object], audit: dict[str, object]) -> str:
    lines = [
        "# TeSTra-EK100 frozen checkpoint audit",
        "",
        "## Official checkpoint integrity gate",
        "",
        f"- Published 1.0 s overall mean top-5 verb recall: {official['published_value_percent']:.1f}%",
        f"- Reproduced horizons: {official['horizon_seconds']}",
        f"- Reproduced values: {[round(value, 3) for value in official['values_percent']]}%",
        f"- Reproduction gate: {'PASS' if official['gate_pass'] else 'FAIL'}",
        "",
        "The paper-facing OAD audit is admissible only when the official checkpoint gate passes.",
        "",
        "## Current-timestep causal postprocessing audit",
        "",
        "| Condition | Delta accuracy | Delta ECE | Delta TFI | Delta delay | Delta miss rate | Full inversion |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    native = audit["points"]["native"]
    for condition in audit["conditions"][1:]:
        point = audit["points"][condition]
        checks = audit["gates"][condition]
        lines.append(
            f"| {condition} | {point['accuracy'] - native['accuracy']:+.6f} | "
            f"{point['global_ece'] - native['global_ece']:+.6f} | "
            f"{point['mean_tfi'] - native['mean_tfi']:+.6f} | "
            f"{point['mean_transition_delay'] - native['mean_transition_delay']:+.6f} | "
            f"{point['missed_transition_rate'] - native['missed_transition_rate']:+.6f} | "
            f"{'PASS' if checks['full_ranking_inversion'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "A failed direction is a boundary result. It does not authorize changing the checkpoint, split, EMA alpha, boxcar window, horizon, or bootstrap seed.",
            "",
        ]
    )
    return "\n".join(lines)


def environment_record(checkpoint: Path, config_path: Path, data_info_path: Path) -> dict[str, object]:
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "official_config": str(config_path),
        "official_config_sha256": sha256_file(config_path),
        "data_info": str(data_info_path),
        "data_info_sha256": sha256_file(data_info_path),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    bundle, checkpoint, protocol_path, output_dir, cache = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    actual_checkpoint_hash = sha256_file(checkpoint).lower()
    permitted_checkpoint_hashes = {
        str(protocol["checkpoint"]["sha256"]).lower(),
        str(protocol["checkpoint"]["inference_artifact"]["sha256"]).lower(),
    }
    if actual_checkpoint_hash not in permitted_checkpoint_hashes:
        raise ValueError("Checkpoint SHA-256 does not match either frozen checkpoint artifact")

    cfg, data_info, config_path, data_info_path = infer_cfg(bundle, args.data_root.resolve(), checkpoint)
    sessions = [str(value) for value in data_info["test_session_set"]]
    if len(sessions) != int(protocol["dataset"]["official_validation_sessions"]):
        raise ValueError("Official validation-session count does not match the frozen protocol")
    inventory = verify_inventory(args.data_root.resolve(), sessions, protocol)
    (output_dir / "data_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    (output_dir / "protocol_snapshot.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    official_path = output_dir / "official_reproduction.json"
    if args.mode == "audit":
        if not cache.is_file() or not official_path.is_file():
            raise FileNotFoundError("Audit mode requires an existing cache and official_reproduction.json")
        predictions = load_cache(cache)
        official = json.loads(official_path.read_text(encoding="utf-8"))
    else:
        model, device = build_model(cfg, checkpoint, args.device)
        segments = load_validation_segments(
            bundle / "vendor" / "TeSTra" / "external" / "rulstm" / "RULSTM" / "data" / "ek100" / "validation.csv"
        )
        predictions, official = export_predictions(
            model,
            device,
            cfg,
            sessions,
            segments,
            args.data_root.resolve(),
            protocol,
            args.batch_size,
            cache,
            output_dir,
        )
        (output_dir / "environment.json").write_text(
            json.dumps(environment_record(checkpoint, config_path, data_info_path), indent=2) + "\n",
            encoding="utf-8",
        )
        if args.mode == "export":
            print(f"Export complete: {cache}", flush=True)
            return

    audit = audit_predictions(predictions, protocol, output_dir)
    (output_dir / "REPORT.md").write_text(build_report(official, audit), encoding="utf-8")
    if not bool(official["gate_pass"]):
        raise RuntimeError("Official checkpoint reproduction gate failed; paper-facing audit is inadmissible")
    print(f"Completed frozen TeSTra-EK100 audit: {output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
