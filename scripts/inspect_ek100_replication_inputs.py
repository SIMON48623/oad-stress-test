from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_SUFFIXES = {".npy", ".npz"}


def _read_split_ids(split_path: str | Path | None) -> dict[str, Any]:
    if split_path is None:
        return {"status": "MISSING", "path": None, "ids": [], "warnings": ["No split path provided."]}
    path = Path(split_path)
    if not path.exists():
        return {"status": "MISSING", "path": str(path), "ids": [], "warnings": ["Split path does not exist."]}
    if not path.is_file():
        return {"status": "UNSUPPORTED", "path": str(path), "ids": [], "warnings": ["Split path is not a file."]}
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        ids.append(Path(value.split()[0]).stem)
    return {"status": "OK", "path": str(path), "ids": ids, "warnings": []}


def _array_files(path: str | Path | None, label: str) -> dict[str, Any]:
    if path is None:
        return {
            "label": label,
            "status": "MISSING",
            "path": None,
            "files": {},
            "warnings": [f"No {label} path provided."],
        }
    root = Path(path)
    if not root.exists():
        return {
            "label": label,
            "status": "MISSING",
            "path": str(root),
            "files": {},
            "warnings": [f"{label} path does not exist."],
        }
    if root.is_file():
        if root.suffix.lower() not in ARRAY_SUFFIXES:
            return {
                "label": label,
                "status": "UNSUPPORTED",
                "path": str(root),
                "files": {},
                "warnings": [f"{label} path is a file but not .npy/.npz; IDs cannot be inferred safely."],
            }
        return {"label": label, "status": "OK", "path": str(root), "files": {root.stem: root}, "warnings": []}
    if not root.is_dir():
        return {
            "label": label,
            "status": "UNSUPPORTED",
            "path": str(root),
            "files": {},
            "warnings": [f"{label} path is neither a file nor a directory."],
        }
    files: dict[str, Path] = {}
    duplicate_ids: list[str] = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ARRAY_SUFFIXES):
        video_id = file_path.stem
        if video_id in files:
            duplicate_ids.append(video_id)
        files[video_id] = file_path
    warnings = []
    if duplicate_ids:
        warnings.append(f"Duplicate inferred IDs found; last path retained for: {sorted(set(duplicate_ids))[:10]}")
    if not files:
        warnings.append(f"No .npy/.npz files found under {root}; IDs cannot be inferred safely.")
    return {"label": label, "status": "OK" if files else "EMPTY", "path": str(root), "files": files, "warnings": warnings}


def _load_array_shape(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r")
        return {"shape": tuple(int(v) for v in array.shape), "dtype": str(array.dtype), "array_key": None}
    data = np.load(path)
    preferred = ["target", "targets", "labels", "features", "arr_0"]
    key = next((name for name in preferred if name in data.files), data.files[0] if data.files else None)
    if key is None:
        raise ValueError(f"{path} contains no arrays")
    array = data[key]
    return {"shape": tuple(int(v) for v in array.shape), "dtype": str(array.dtype), "array_key": key}


def _sample_shapes(inventory: dict[str, Any], max_files: int) -> list[dict[str, Any]]:
    samples = []
    files: dict[str, Path] = inventory["files"]
    for video_id, path in list(sorted(files.items()))[: max(0, int(max_files))]:
        try:
            info = _load_array_shape(path)
            samples.append({
                "video_id": video_id,
                "path": str(path),
                **info,
            })
        except Exception as exc:  # noqa: BLE001 - probe reports unreadable files.
            samples.append({
                "video_id": video_id,
                "path": str(path),
                "error": str(exc),
            })
    return samples


def _summarize_inventory(inventory: dict[str, Any], max_files: int) -> dict[str, Any]:
    files: dict[str, Path] = inventory["files"]
    ids = sorted(files)
    return {
        "label": inventory["label"],
        "status": inventory["status"],
        "path": inventory["path"],
        "num_array_files": int(len(files)),
        "ids_preview": ids[:10],
        "shape_samples": _sample_shapes(inventory, max_files),
        "warnings": list(inventory["warnings"]),
    }


def _overlap_summary(name_a: str, ids_a: set[str] | None, name_b: str, ids_b: set[str] | None) -> dict[str, Any]:
    key = f"{name_a}_vs_{name_b}"
    if ids_a is None or ids_b is None:
        return {
            key: {
                "status": "UNKNOWN",
                "warning": "IDs could not be inferred safely for at least one side.",
            }
        }
    overlap = ids_a & ids_b
    return {
        key: {
            "status": "OK",
            f"{name_a}_count": int(len(ids_a)),
            f"{name_b}_count": int(len(ids_b)),
            "overlap_count": int(len(overlap)),
            f"{name_a}_missing_in_{name_b}": int(len(ids_a - ids_b)),
            f"{name_b}_missing_in_{name_a}": int(len(ids_b - ids_a)),
            "overlap_preview": sorted(overlap)[:10],
        }
    }


def _frame_count(shape: tuple[int, ...] | list[int] | None) -> int | None:
    if shape is None or len(shape) == 0:
        return None
    return int(shape[0])


def _load_frame_count(path: Path) -> dict[str, Any]:
    info = _load_array_shape(path)
    count = _frame_count(info["shape"])
    if count is None:
        raise ValueError(f"Cannot infer frame count from shape {info['shape']}")
    return {**info, "frame_count": int(count)}


def _compare_frame_counts(
    *,
    target_frames: int,
    feature_frames: int,
    expected_stride: float | None,
    allow_frame_mismatch: bool,
) -> dict[str, Any]:
    if int(target_frames) == int(feature_frames):
        return {"status": "PASS_EXACT", "message": "target and feature frame counts match exactly"}
    if expected_stride is not None:
        expected_target_frames = float(feature_frames) * float(expected_stride)
        tolerance = max(1.0, abs(expected_target_frames) * 0.01)
        delta = abs(float(target_frames) - expected_target_frames)
        if delta <= tolerance:
            return {
                "status": "PASS_EXPECTED_STRIDE",
                "message": (
                    "frame-count mismatch is explained by expected target/features stride "
                    f"{float(expected_stride):g}"
                ),
                "expected_target_frames": expected_target_frames,
                "tolerance": tolerance,
            }
    if allow_frame_mismatch:
        return {"status": "ALLOWED_MISMATCH", "message": "frame-count mismatch allowed by user flag"}
    return {"status": "FAIL_MISMATCH", "message": "frame-count mismatch without documented stride/downsampling rule"}


def _alignment_gate(
    *,
    target_inventory: dict[str, Any],
    rgb_inventory: dict[str, Any],
    flow_inventory: dict[str, Any],
    max_files: int,
    expected_stride: float | None,
    allow_frame_mismatch: bool,
) -> dict[str, Any]:
    required = {
        "target": target_inventory,
        "rgb": rgb_inventory,
        "flow": flow_inventory,
    }
    reasons = []
    for label, inventory in required.items():
        if inventory["status"] != "OK":
            reasons.append(f"{label} input status is {inventory['status']}; alignment gate is not evaluable")
    if reasons:
        return {
            "status": "NOT_EVALUABLE",
            "expected_stride": expected_stride,
            "allow_frame_mismatch": bool(allow_frame_mismatch),
            "sampled_common_ids": 0,
            "reasons": reasons,
            "sampled_frame_counts": [],
        }

    target_ids = set(target_inventory["files"])
    rgb_ids = set(rgb_inventory["files"])
    flow_ids = set(flow_inventory["files"])
    common_ids = sorted(target_ids & rgb_ids & flow_ids)
    if not common_ids:
        return {
            "status": "FAIL",
            "expected_stride": expected_stride,
            "allow_frame_mismatch": bool(allow_frame_mismatch),
            "sampled_common_ids": 0,
            "reasons": ["No common IDs across target, RGB, and Flow inputs."],
            "sampled_frame_counts": [],
        }

    rows = []
    has_failure = False
    has_warning = False
    for video_id in common_ids[: max(1, int(max_files))]:
        row: dict[str, Any] = {"video_id": video_id}
        try:
            target_info = _load_frame_count(target_inventory["files"][video_id])
            rgb_info = _load_frame_count(rgb_inventory["files"][video_id])
            flow_info = _load_frame_count(flow_inventory["files"][video_id])
            row.update({
                "target_frames": int(target_info["frame_count"]),
                "rgb_frames": int(rgb_info["frame_count"]),
                "flow_frames": int(flow_info["frame_count"]),
                "target_shape": target_info["shape"],
                "rgb_shape": rgb_info["shape"],
                "flow_shape": flow_info["shape"],
            })
            row["rgb_alignment"] = _compare_frame_counts(
                target_frames=int(target_info["frame_count"]),
                feature_frames=int(rgb_info["frame_count"]),
                expected_stride=expected_stride,
                allow_frame_mismatch=allow_frame_mismatch,
            )
            row["flow_alignment"] = _compare_frame_counts(
                target_frames=int(target_info["frame_count"]),
                feature_frames=int(flow_info["frame_count"]),
                expected_stride=expected_stride,
                allow_frame_mismatch=allow_frame_mismatch,
            )
            statuses = {row["rgb_alignment"]["status"], row["flow_alignment"]["status"]}
            if any(status.startswith("FAIL") for status in statuses):
                has_failure = True
            if any(status == "ALLOWED_MISMATCH" for status in statuses):
                has_warning = True
        except Exception as exc:  # noqa: BLE001 - probe reports alignment failure clearly.
            row["error"] = str(exc)
            has_failure = True
        rows.append(row)

    if has_failure:
        status = "FAIL"
        reasons = ["At least one sampled target/RGB/Flow frame-count check failed."]
    elif has_warning:
        status = "PASS_WITH_WARNINGS"
        reasons = ["Frame-count mismatch was allowed by user flag; document the alignment rule before v1.9."]
    else:
        status = "PASS"
        reasons = ["Sampled target/RGB/Flow frame counts are aligned or explained by expected stride."]
    return {
        "status": status,
        "expected_stride": expected_stride,
        "allow_frame_mismatch": bool(allow_frame_mismatch),
        "sampled_common_ids": int(len(rows)),
        "common_id_count": int(len(common_ids)),
        "reasons": reasons,
        "sampled_frame_counts": rows,
    }


def inspect_inputs(
    *,
    target_dir: str | Path | None,
    rgb_feature_path: str | Path | None,
    flow_feature_path: str | Path | None,
    split_path: str | Path | None = None,
    max_files: int = 5,
    expected_stride: float | None = None,
    allow_frame_mismatch: bool = False,
) -> dict[str, Any]:
    if expected_stride is not None and float(expected_stride) <= 0:
        raise ValueError("expected_stride must be positive")
    target_inventory = _array_files(target_dir, "target_perframe")
    rgb_inventory = _array_files(rgb_feature_path, "rgb_features")
    flow_inventory = _array_files(flow_feature_path, "flow_features")
    split = _read_split_ids(split_path)

    target = _summarize_inventory(target_inventory, max_files)
    rgb = _summarize_inventory(rgb_inventory, max_files)
    flow = _summarize_inventory(flow_inventory, max_files)
    target["_all_ids"] = sorted(target_inventory["files"])
    rgb["_all_ids"] = sorted(rgb_inventory["files"])
    flow["_all_ids"] = sorted(flow_inventory["files"])

    split_ids = set(split["ids"]) if split["status"] == "OK" else None
    target_ids = set(target["_all_ids"]) if target_inventory["status"] in {"OK", "EMPTY"} else None
    rgb_ids = set(rgb["_all_ids"]) if rgb_inventory["status"] in {"OK", "EMPTY"} else None
    flow_ids = set(flow["_all_ids"]) if flow_inventory["status"] in {"OK", "EMPTY"} else None

    overlaps: dict[str, Any] = {}
    overlaps.update(_overlap_summary("target", target_ids, "rgb", rgb_ids))
    overlaps.update(_overlap_summary("target", target_ids, "flow", flow_ids))
    overlaps.update(_overlap_summary("rgb", rgb_ids, "flow", flow_ids))
    overlaps.update(_overlap_summary("target", target_ids, "split", split_ids))
    overlaps.update(_overlap_summary("rgb", rgb_ids, "split", split_ids))
    overlaps.update(_overlap_summary("flow", flow_ids, "split", split_ids))
    alignment_gate = _alignment_gate(
        target_inventory=target_inventory,
        rgb_inventory=rgb_inventory,
        flow_inventory=flow_inventory,
        max_files=max_files,
        expected_stride=None if expected_stride is None else float(expected_stride),
        allow_frame_mismatch=allow_frame_mismatch,
    )

    for section in [target, rgb, flow]:
        section.pop("_all_ids", None)
    return {
        "target_perframe": target,
        "rgb_features": rgb,
        "flow_features": flow,
        "split": {
            "status": split["status"],
            "path": split["path"],
            "num_ids": int(len(split["ids"])),
            "ids_preview": split["ids"][:10],
            "warnings": split["warnings"],
        },
        "overlaps": overlaps,
        "alignment_gate": alignment_gate,
        "notes": [
            "This is a structural input probe only; it does not train or evaluate a model.",
            "Feature-to-label temporal alignment is a hard gate before formal EK100 replication.",
        ],
    }


def _fmt_shape(shape: tuple[int, ...] | list[int] | None) -> str:
    if shape is None:
        return "unknown"
    return "x".join(str(int(value)) for value in shape)


def format_report(report: dict[str, Any]) -> str:
    lines = ["# EK100 Replication Input Probe", ""]
    for key in ["target_perframe", "rgb_features", "flow_features"]:
        section = report[key]
        lines.extend([
            f"## {section['label']}",
            "",
            f"- status: {section['status']}",
            f"- path: {section['path']}",
            f"- array files: {section['num_array_files']}",
            f"- ids preview: {section['ids_preview']}",
        ])
        if section["warnings"]:
            lines.append(f"- warnings: {section['warnings']}")
        if section["shape_samples"]:
            lines.append("- shape samples:")
            for sample in section["shape_samples"]:
                if "error" in sample:
                    lines.append(f"  - {sample['video_id']}: ERROR {sample['error']}")
                else:
                    lines.append(
                        f"  - {sample['video_id']}: shape={_fmt_shape(sample['shape'])}, "
                        f"dtype={sample['dtype']}, key={sample['array_key']}"
                    )
        lines.append("")
    split = report["split"]
    lines.extend([
        "## split",
        "",
        f"- status: {split['status']}",
        f"- path: {split['path']}",
        f"- ids: {split['num_ids']}",
        f"- ids preview: {split['ids_preview']}",
    ])
    if split["warnings"]:
        lines.append(f"- warnings: {split['warnings']}")
    lines.extend(["", "## ID overlap", ""])
    for name, value in report["overlaps"].items():
        if value["status"] == "UNKNOWN":
            lines.append(f"- {name}: UNKNOWN ({value['warning']})")
        else:
            lines.append(
                f"- {name}: overlap={value['overlap_count']} "
                f"missing_left={value.get(name.split('_vs_')[0] + '_missing_in_' + name.split('_vs_')[1])} "
                f"missing_right={value.get(name.split('_vs_')[1] + '_missing_in_' + name.split('_vs_')[0])}"
            )
    gate = report["alignment_gate"]
    lines.extend(["", "## Alignment gate", ""])
    lines.extend([
        f"- status: {gate['status']}",
        f"- expected_stride: {gate['expected_stride']}",
        f"- allow_frame_mismatch: {gate['allow_frame_mismatch']}",
        f"- common_id_count: {gate.get('common_id_count', 0)}",
        f"- sampled_common_ids: {gate['sampled_common_ids']}",
        f"- reasons: {gate['reasons']}",
    ])
    if gate["sampled_frame_counts"]:
        lines.append("- sampled frame-count checks:")
        for row in gate["sampled_frame_counts"]:
            if "error" in row:
                lines.append(f"  - {row['video_id']}: ERROR {row['error']}")
                continue
            lines.append(
                f"  - {row['video_id']}: target={row['target_frames']} "
                f"rgb={row['rgb_frames']} ({row['rgb_alignment']['status']}) "
                f"flow={row['flow_frames']} ({row['flow_alignment']['status']})"
            )
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect EK100 formal replication inputs without training or evaluation.")
    parser.add_argument("--target-dir", default=None, help="TeSTra EK100 target_perframe directory")
    parser.add_argument("--rgb-feature-path", default=None, help="RGB feature directory or .npy/.npz file")
    parser.add_argument("--flow-feature-path", default=None, help="Flow feature directory or .npy/.npz file")
    parser.add_argument("--split-path", default=None, help="optional split/session list")
    parser.add_argument("--max-files", type=int, default=5, help="maximum files per input to load for shape/dtype samples")
    parser.add_argument(
        "--expected-stride",
        type=float,
        default=None,
        help="expected target frame count divided by feature frame count; documents stride/downsampling mismatches",
    )
    parser.add_argument(
        "--allow-frame-mismatch",
        action="store_true",
        help="do not fail the alignment gate on sampled frame-count mismatches; use only with documented alignment rules",
    )
    args = parser.parse_args()

    report = inspect_inputs(
        target_dir=args.target_dir,
        rgb_feature_path=args.rgb_feature_path,
        flow_feature_path=args.flow_feature_path,
        split_path=args.split_path,
        max_files=args.max_files,
        expected_stride=args.expected_stride,
        allow_frame_mismatch=args.allow_frame_mismatch,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
