from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TVSERIES_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TVSeriesSegment:
    video_id: str
    label_name: str
    label_id: int
    start_time: float
    end_time: float
    attributes: tuple[int, ...] = ()
    comment: str = ""


@dataclass(frozen=True)
class TVSeriesAlignedTargets:
    active_label_sets: tuple[frozenset[int], ...]
    state_ids: np.ndarray
    primary_labels: np.ndarray
    timestamps: np.ndarray


def _read_text(source: str | Path, member: str) -> str:
    path = Path(source)
    if path.is_dir():
        candidates = (path / member, path / "TVSeries_Dataset" / member)
        for candidate in candidates:
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError(f"TVSeries member not found: {member}")
    if not path.exists():
        raise FileNotFoundError(f"TVSeries annotation source not found: {path}")
    if path.suffix.lower() != ".zip":
        raise ValueError("TVSeries annotation source must be a directory or .zip")
    with zipfile.ZipFile(path) as archive:
        candidates = (member, f"TVSeries_Dataset/{member}")
        for candidate in candidates:
            try:
                return archive.read(candidate).decode("utf-8")
            except KeyError:
                continue
    raise FileNotFoundError(f"TVSeries member not found in {path}: {member}")


def read_tvseries_classes(source: str | Path) -> list[str]:
    classes = []
    for line in _read_text(source, "classes.txt").splitlines():
        if not line.strip():
            continue
        classes.append(line.split("\t", 1)[0].strip())
    if not classes:
        raise ValueError("TVSeries classes.txt contains no classes")
    if len(classes) != len(set(classes)):
        raise ValueError("TVSeries classes.txt contains duplicate class names")
    return classes


def tvseries_label_map(class_names: Sequence[str]) -> dict[str, int]:
    """Map background to 0 and official actions to 1..N in classes.txt order."""

    return {str(name): index + 1 for index, name in enumerate(class_names)}


def read_tvseries_segments(
    source: str | Path,
    split: str,
    *,
    label_map: dict[str, int] | None = None,
) -> list[TVSeriesSegment]:
    split_name = str(split).lower()
    if split_name not in TVSERIES_SPLITS:
        raise ValueError(f"Unknown TVSeries split: {split}")
    if label_map is None:
        label_map = tvseries_label_map(read_tvseries_classes(source))
    rows = csv.reader(io.StringIO(_read_text(source, f"GT-{split_name}.txt")), delimiter="\t")
    segments: list[TVSeriesSegment] = []
    for line_no, row in enumerate(rows, start=1):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) < 4:
            raise ValueError(f"GT-{split_name}.txt line {line_no} has fewer than 4 fields")
        video_id, label_name = row[0].strip(), row[1].strip()
        if label_name not in label_map:
            raise KeyError(f"Unknown TVSeries class at line {line_no}: {label_name!r}")
        start_time, end_time = float(row[2]), float(row[3])
        if not np.isfinite(start_time) or not np.isfinite(end_time) or start_time >= end_time:
            raise ValueError(
                f"Invalid TVSeries interval at line {line_no}: {start_time}, {end_time}"
            )
        attribute_values = []
        for raw in row[4:16]:
            value = raw.strip()
            if value:
                attribute_values.append(int(value))
        segments.append(
            TVSeriesSegment(
                video_id=video_id,
                label_name=label_name,
                label_id=int(label_map[label_name]),
                start_time=start_time,
                end_time=end_time,
                attributes=tuple(attribute_values),
                comment="\t".join(row[16:]).strip() if len(row) > 16 else "",
            )
        )
    return segments


def tvseries_split_ids(segments: Iterable[TVSeriesSegment]) -> list[str]:
    return sorted({segment.video_id for segment in segments})


def align_active_label_sets(
    segments: Iterable[TVSeriesSegment],
    *,
    video_id: str,
    length: int,
    timestamps: np.ndarray | None = None,
    feature_fps: float | None = None,
    background_label: int = 0,
) -> TVSeriesAlignedTargets:
    """Align second-level segments to feature-step sample times without losing overlaps.

    Each feature step is treated as a sample at its timestamp. A segment is
    active when start_time <= timestamp < end_time. The empty set denotes
    background/no action; background_label is used only for the compatibility
    primary-label sequence.
    """

    if int(length) <= 0:
        raise ValueError("length must be positive")
    if timestamps is None:
        if feature_fps is None or float(feature_fps) <= 0:
            raise ValueError("timestamps or a positive feature_fps is required")
        sample_times = np.arange(int(length), dtype=np.float64) / float(feature_fps)
    else:
        sample_times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(sample_times) != int(length):
            raise ValueError(
                f"timestamps length mismatch: expected {int(length)}, got {len(sample_times)}"
            )
        if not np.isfinite(sample_times).all() or np.any(np.diff(sample_times) < 0):
            raise ValueError("timestamps must be finite and nondecreasing")

    active = [set() for _ in range(int(length))]
    for segment in segments:
        if segment.video_id != str(video_id):
            continue
        mask = (sample_times >= float(segment.start_time)) & (
            sample_times < float(segment.end_time)
        )
        for index in np.flatnonzero(mask):
            active[int(index)].add(int(segment.label_id))

    active_sets = tuple(frozenset(values) for values in active)
    state_ids = active_label_set_state_ids(active_sets)
    primary = np.asarray(
        [min(values) if values else int(background_label) for values in active_sets],
        dtype=np.int64,
    )
    return TVSeriesAlignedTargets(
        active_label_sets=active_sets,
        state_ids=state_ids,
        primary_labels=primary,
        timestamps=sample_times,
    )


def canonical_active_label_set(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"[]", "()", "{}"}:
            return ()
        text = text.strip("[](){}")
        return tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if isinstance(value, (set, frozenset, list, tuple, np.ndarray)):
        return tuple(sorted({int(item) for item in value}))
    return (int(value),)


def active_label_set_state_id(active_labels: object) -> int:
    """Encode a TVSeries active set losslessly as a 30-bit integer."""

    state_id = 0
    for label in canonical_active_label_set(active_labels):
        if label <= 0 or label > 30:
            raise ValueError(f"TVSeries action label must be in 1..30, got {label}")
        state_id |= 1 << (int(label) - 1)
    return int(state_id)


def active_label_set_state_ids(active_label_sets: Iterable[object]) -> np.ndarray:
    return np.asarray(
        [active_label_set_state_id(labels) for labels in active_label_sets],
        dtype=np.int64,
    )


def tvseries_raw_error(
    pred_label: int | Iterable[int] | np.ndarray,
    active_label_sets: object,
    *,
    background_label: int = 0,
) -> int | np.ndarray:
    """Return top-1 error under TVSeries active-set/background semantics."""

    predictions = np.asarray(pred_label)
    if predictions.ndim == 0:
        active = canonical_active_label_set(active_label_sets)
        pred = int(predictions.item())
        return int(pred not in active) if active else int(pred != int(background_label))

    flat = predictions.astype(np.int64, copy=False).reshape(-1)
    values = list(active_label_sets)
    if len(values) != len(flat):
        raise ValueError(
            f"active_label_sets length mismatch: expected {len(flat)}, got {len(values)}"
        )
    errors = [
        bool(int(pred) not in active)
        if (active := canonical_active_label_set(labels))
        else bool(int(pred) != int(background_label))
        for pred, labels in zip(flat, values)
    ]
    return np.asarray(errors, dtype=np.int8).reshape(predictions.shape)
