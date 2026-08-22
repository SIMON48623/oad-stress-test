from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from .schema import VideoSequence


class FeatureDataset:
    """Loads feature-level OAD sequences from feature files.

    Supported feature formats:
      .npz:
      features: [T, D]
      labels: optional [T]
      timestamps: optional [T]

      .npy:
      raw feature array with shape [T, D]

    If labels are not stored in the feature file, temporal segment annotations
    are converted into timestep labels. Annotation tables must contain
    video_id, label, and either start_idx/end_idx or start_time/end_time.
    """

    FEATURE_SUFFIXES = (".npz", ".npy")
    ANNOTATION_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl"}

    def __init__(
        self,
        feature_dir: str | Path,
        split_file: str | Path,
        annotation_dir: str | Path | None = None,
        annotation_file: str | Path | None = None,
        background_label: int = 0,
        feature_fps: float | None = None,
        label_map: Mapping[str, int] | None = None,
        end_idx_inclusive: bool = True,
    ):
        self.feature_dir = Path(feature_dir)
        self.split_file = Path(split_file)
        self.background_label = int(background_label)
        self.feature_fps = None if feature_fps is None else float(feature_fps)
        self.end_idx_inclusive = bool(end_idx_inclusive)
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {self.feature_dir}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        self.video_ids = self._read_split(self.split_file)
        self.annotation_path = self._resolve_annotation_path(annotation_dir, annotation_file)
        self.annotations = self._load_annotations(self.annotation_path) if self.annotation_path else pd.DataFrame()
        self.label_map = self._build_label_map(label_map)

    @staticmethod
    def _read_split(split_file: Path) -> List[str]:
        ids: List[str] = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
        return ids

    @staticmethod
    def _resolve_annotation_path(annotation_dir: str | Path | None, annotation_file: str | Path | None) -> Path | None:
        if annotation_file is not None:
            return Path(annotation_file)
        if annotation_dir is not None:
            return Path(annotation_dir)
        return None

    @classmethod
    def _annotation_files(cls, path: Path) -> List[Path]:
        if not path.exists():
            raise FileNotFoundError(f"Annotation path not found: {path}")
        if path.is_file():
            return [path]
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in cls.ANNOTATION_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No annotation files found in {path}")
        return files

    @classmethod
    def _load_annotations(cls, path: Path) -> pd.DataFrame:
        frames = [cls._read_annotation_file(p) for p in cls._annotation_files(path)]
        annotations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        annotations = cls._normalize_annotation_columns(annotations)
        if len(annotations) == 0:
            return annotations
        required = {"video_id", "label"}
        missing = required - set(annotations.columns)
        if missing:
            raise KeyError(f"Annotation table is missing required columns: {sorted(missing)}")
        has_idx = {"start_idx", "end_idx"}.issubset(annotations.columns)
        has_time = {"start_time", "end_time"}.issubset(annotations.columns)
        if not has_idx and not has_time:
            raise KeyError("Annotation table must contain start_idx/end_idx or start_time/end_time")
        annotations["video_id"] = annotations["video_id"].astype(str)
        return annotations

    @staticmethod
    def _read_annotation_file(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".txt":
            df = pd.read_csv(path, sep=None, engine="python")
            if df.shape[1] == 1:
                df = pd.read_csv(path, sep=r"\s+", engine="python")
            return df
        return pd.read_csv(path)

    @staticmethod
    def _normalize_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
        aliases = {
            "video": "video_id",
            "video_name": "video_id",
            "vid": "video_id",
            "start": "start_time",
            "end": "end_time",
            "start_frame": "start_idx",
            "end_frame": "end_idx",
            "class": "label",
            "class_name": "label",
            "label_name": "label",
        }
        out = df.copy()
        normalized = []
        for column in out.columns:
            name = str(column).strip()
            key = name.lower()
            normalized.append(aliases.get(key, key))
        out.columns = normalized
        return out

    @staticmethod
    def _is_int_like(value: object) -> bool:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        return numeric.is_integer()

    def _build_label_map(self, label_map: Mapping[str, int] | None) -> Dict[str, int]:
        if label_map is not None:
            return {str(k): int(v) for k, v in label_map.items()}
        if len(self.annotations) == 0:
            return {}
        labels = sorted({str(v) for v in self.annotations["label"].dropna().tolist()})
        if all(self._is_int_like(v) for v in labels):
            return {v: int(float(v)) for v in labels}

        mapping: Dict[str, int] = {}
        next_id = 0
        background_names = {"background", "bg", "none", "no_action"}
        for label in labels:
            if label.strip().lower() in background_names:
                mapping[label] = self.background_label
                continue
            while next_id == self.background_label or next_id in mapping.values():
                next_id += 1
            mapping[label] = next_id
            next_id += 1
        return mapping

    def __len__(self) -> int:
        return len(self.video_ids)

    def __iter__(self) -> Iterable[VideoSequence]:
        for video_id in self.video_ids:
            yield self.load_video(video_id)

    def _feature_path(self, video_id: str) -> Path:
        for suffix in self.FEATURE_SUFFIXES:
            path = self.feature_dir / f"{video_id}{suffix}"
            if path.exists():
                return path
        suffixes = ", ".join(self.FEATURE_SUFFIXES)
        raise FileNotFoundError(f"Feature file not found for {video_id}; expected one of: {suffixes}")

    def load_video(self, video_id: str) -> VideoSequence:
        path = self._feature_path(video_id)
        timestamps = None
        labels = None
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            if "features" not in data:
                raise KeyError(f"{path} must contain an array named 'features'")
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64) if "labels" in data.files else None
            timestamps = data["timestamps"] if "timestamps" in data.files else None
        else:
            features = np.asarray(np.load(path), dtype=np.float32)

        if features.ndim != 2:
            raise ValueError(f"features for {video_id} must have shape [T, D], got {features.shape}")
        if labels is None:
            labels = self.labels_from_annotations(video_id, length=int(features.shape[0]))
        return VideoSequence(
            video_id=video_id,
            features=features,
            labels=np.asarray(labels, dtype=np.int64),
            timestamps=timestamps,
        )

    def _annotation_rows(self, video_id: str) -> pd.DataFrame:
        if len(self.annotations) == 0:
            return pd.DataFrame()
        return self.annotations[self.annotations["video_id"].eq(str(video_id))]

    def _label_id(self, value: object) -> int:
        key = str(value)
        if key in self.label_map:
            return self.label_map[key]
        if self._is_int_like(value):
            return int(float(value))
        raise KeyError(f"Label {value!r} is missing from label_map")

    def _row_interval(self, row: pd.Series) -> tuple[int, int]:
        if "start_idx" in row.index and "end_idx" in row.index and pd.notna(row["start_idx"]) and pd.notna(row["end_idx"]):
            start = int(float(row["start_idx"]))
            raw_end = int(float(row["end_idx"]))
            end = raw_end + 1 if self.end_idx_inclusive else raw_end
            return start, end

        if self.feature_fps is None:
            raise ValueError("feature_fps is required for start_time/end_time annotations")
        start = int(np.floor(float(row["start_time"]) * self.feature_fps))
        end = int(np.ceil(float(row["end_time"]) * self.feature_fps))
        return start, end

    @staticmethod
    def _interval_issue(start: int, end: int, length: int) -> str | None:
        if start < 0 or end < 0:
            return "negative_index"
        if start >= end:
            return "empty_or_reversed_segment"
        if start >= length or end > length:
            return "out_of_bounds"
        return None

    def labels_from_annotations(self, video_id: str, length: int) -> np.ndarray:
        if len(self.annotations) == 0:
            raise ValueError(f"No labels found for {video_id}; provide .npz labels or annotation files")
        labels = np.full(length, self.background_label, dtype=np.int64)
        for _, row in self._annotation_rows(video_id).iterrows():
            start, end = self._row_interval(row)
            issue = self._interval_issue(start, end, length)
            if issue is not None:
                raise ValueError(
                    f"Invalid annotation for {video_id}: start={start}, end={end}, "
                    f"length={length}, issue={issue}"
                )
            labels[start:end] = self._label_id(row["label"])
        return labels

    def feature_length(self, video_id: str) -> int:
        path = self._feature_path(video_id)
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            if "features" not in data:
                raise KeyError(f"{path} must contain an array named 'features'")
            return int(data["features"].shape[0])
        return int(np.load(path, mmap_mode="r").shape[0])

    def annotation_bounds_issues(self) -> List[Dict[str, object]]:
        issues: List[Dict[str, object]] = []
        if len(self.annotations) == 0:
            return issues
        for video_id in self.video_ids:
            length = self.feature_length(video_id)
            for row_index, row in self._annotation_rows(video_id).iterrows():
                try:
                    start, end = self._row_interval(row)
                    issue = self._interval_issue(start, end, length)
                except Exception as exc:  # noqa: BLE001 - report malformed annotation rows.
                    start, end, issue = None, None, str(exc)
                if issue is not None:
                    issues.append({
                        "video_id": video_id,
                        "row_index": int(row_index),
                        "start": start,
                        "end": end,
                        "length": length,
                        "issue": issue,
                    })
        return issues

    def infer_num_classes(self) -> int:
        max_label = 0
        for video in self:
            max_label = max(max_label, int(np.max(video.labels)))
        return max_label + 1
