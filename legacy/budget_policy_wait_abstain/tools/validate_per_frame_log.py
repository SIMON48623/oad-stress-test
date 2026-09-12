from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
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


class ValidationError(ValueError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("bool is not a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("non-finite value")
    return numeric


def validate_per_frame_log(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Per-frame log not found: {path}")
    stats = {
        "total_frames": 0,
        "video_ids": set(),
        "decision_counts": {decision: 0 for decision in sorted(DECISIONS)},
        "sum_max_prob": 0.0,
        "sum_entropy": 0.0,
        "sum_top2_margin": 0.0,
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"line {line_no}: invalid JSON: {exc}") from exc
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValidationError(f"line {line_no}: missing required fields: {sorted(missing)}")
            if row["decision"] not in DECISIONS:
                raise ValidationError(f"line {line_no}: invalid decision: {row['decision']!r}")
            for field in ["frame_idx", "gt_label", "pred_label"]:
                if not _is_int(row[field]):
                    raise ValidationError(f"line {line_no}: {field} must be an integer")
            max_prob = _finite_float(row["max_prob"])
            entropy = _finite_float(row["entropy"])
            top2_margin = _finite_float(row["top2_margin"])
            stats["total_frames"] += 1
            stats["video_ids"].add(str(row["video_id"]))
            stats["decision_counts"][row["decision"]] += 1
            stats["sum_max_prob"] += max_prob
            stats["sum_entropy"] += entropy
            stats["sum_top2_margin"] += top2_margin
    if stats["total_frames"] == 0:
        raise ValidationError("per-frame log contains no rows")
    denom = stats["total_frames"]
    return {
        "total_frames": int(stats["total_frames"]),
        "num_videos": int(len(stats["video_ids"])),
        "decision_counts": dict(stats["decision_counts"]),
        "mean_max_prob": stats["sum_max_prob"] / denom,
        "mean_entropy": stats["sum_entropy"] / denom,
        "mean_top2_margin": stats["sum_top2_margin"] / denom,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a v0.8 per-frame JSONL log.")
    parser.add_argument("--input", required=True, help="path to per-frame JSONL log")
    args = parser.parse_args(argv)
    try:
        stats = validate_per_frame_log(args.input)
    except Exception as exc:  # noqa: BLE001 - CLI should report validation failure clearly.
        print(f"validation_failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"total_frames: {stats['total_frames']}")
    print(f"num_videos: {stats['num_videos']}")
    print(f"decision_counts: {stats['decision_counts']}")
    print(f"mean_max_prob: {stats['mean_max_prob']:.6f}")
    print(f"mean_entropy: {stats['mean_entropy']:.6f}")
    print(f"mean_top2_margin: {stats['mean_top2_margin']:.6f}")


if __name__ == "__main__":
    main()
