from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import warnings

import numpy as np

from oad_stress_test.datasets.schema import VideoSequence


@dataclass
class LinearProbeClassifier:
    """CPU linear probe over feature-level samples.

    This is a sanity-check classifier backend. It is not a temporal OAD model:
    it sees individual feature vectors from the train split and predicts one
    timestep at a time during evaluation.
    """

    num_classes: int
    max_train_samples: int | None = None
    class_balanced_sampling: bool = False
    linear_class_weight: str | None = None
    linear_mode: str = "multinomial"
    score_mode: str = "raw_margin"
    background_ratio: float = 1.0
    min_samples_per_class: int = 200
    background_label: int = 0
    seed: int = 13
    max_iter: int = 1000
    tol: float = 1e-3
    alpha: float = 1e-4
    scaler: object | None = None
    model: object | None = None
    observed_classes_: np.ndarray | None = None
    sample_counts_: np.ndarray | None = None
    sampling_seen_counts_: np.ndarray | None = None
    sampling_caps_: np.ndarray | None = None
    label_to_idx_: dict[int, int] | None = None
    idx_to_label_: dict[int, int] | None = None
    convergence_warning_: bool = False
    convergence_messages_: tuple[str, ...] = ()

    def fit(self, videos: Iterable[VideoSequence]) -> "LinearProbeClassifier":
        x_train, y_train = self._collect_samples(videos)
        if len(x_train) == 0:
            raise ValueError("Cannot fit linear_probe with empty dataset")
        if len(np.unique(y_train)) < 2:
            raise ValueError("linear_probe requires at least two classes in training samples")

        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_train)
        self.model = self._make_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model.fit(x_scaled, y_train)
        convergence_warnings = [
            str(warning.message)
            for warning in caught
            if warning.category.__name__ == "ConvergenceWarning"
        ]
        self.convergence_warning_ = bool(convergence_warnings)
        self.convergence_messages_ = tuple(convergence_warnings)
        self.observed_classes_ = np.asarray(self.model.classes_, dtype=np.int64)
        self.sample_counts_ = np.bincount(y_train, minlength=self.num_classes).astype(np.int64)
        self.label_to_idx_ = {class_id: class_id for class_id in range(self.num_classes)}
        self.idx_to_label_ = {class_id: class_id for class_id in range(self.num_classes)}
        return self

    def _make_model(self):
        if self.linear_mode == "multinomial":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                solver="saga",
                penalty="l2",
                C=1.0 / max(float(self.alpha), 1e-12),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                random_state=self.seed,
                class_weight=self.linear_class_weight,
                n_jobs=1,
            )
        if self.linear_mode == "ovr":
            from sklearn.linear_model import SGDClassifier
            from sklearn.multiclass import OneVsRestClassifier

            base = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=float(self.alpha),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                random_state=self.seed,
                class_weight=self.linear_class_weight,
            )
            return OneVsRestClassifier(base, n_jobs=1)
        raise ValueError(f"Unknown linear_mode: {self.linear_mode}")

    def _collect_samples(self, videos: Iterable[VideoSequence]) -> tuple[np.ndarray, np.ndarray]:
        if self.max_train_samples is not None:
            return self._collect_reservoir_samples(videos)
        features = []
        labels = []
        for video in videos:
            features.append(np.asarray(video.features, dtype=np.float32))
            labels.append(np.asarray(video.labels, dtype=np.int64))
        if not features:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        x = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0)
        self.sampling_seen_counts_ = np.bincount(y, minlength=self.num_classes).astype(np.int64)
        self.sampling_caps_ = np.full(self.num_classes, -1, dtype=np.int64)
        return x, y

    def _collect_reservoir_samples(self, videos: Iterable[VideoSequence]) -> tuple[np.ndarray, np.ndarray]:
        if self.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive")
        if self.class_balanced_sampling:
            return self._collect_balanced_reservoir_samples(videos)

        rng = np.random.default_rng(self.seed)
        reservoir_x = None
        reservoir_y = np.empty(self.max_train_samples, dtype=np.int64)
        filled = 0
        seen = 0
        seen_counts = np.zeros(self.num_classes, dtype=np.int64)
        for video in videos:
            features = np.asarray(video.features, dtype=np.float32)
            labels = np.asarray(video.labels, dtype=np.int64)
            if reservoir_x is None:
                reservoir_x = np.empty((self.max_train_samples, features.shape[1]), dtype=np.float32)
            for feature, label in zip(features, labels):
                label = int(label)
                if 0 <= label < self.num_classes:
                    seen_counts[label] += 1
                seen += 1
                if filled < self.max_train_samples:
                    reservoir_x[filled] = feature
                    reservoir_y[filled] = label
                    filled += 1
                    continue
                replace_idx = int(rng.integers(0, seen))
                if replace_idx < self.max_train_samples:
                    reservoir_x[replace_idx] = feature
                    reservoir_y[replace_idx] = label
        self.sampling_seen_counts_ = seen_counts
        self.sampling_caps_ = np.full(self.num_classes, -1, dtype=np.int64)
        if reservoir_x is None:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return reservoir_x[:filled], reservoir_y[:filled]

    def _collect_balanced_reservoir_samples(self, videos: Iterable[VideoSequence]) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        caps = self._balanced_caps()
        reservoirs_x: dict[int, np.ndarray] = {}
        reservoirs_y: dict[int, np.ndarray] = {}
        filled = np.zeros(self.num_classes, dtype=np.int64)
        seen = np.zeros(self.num_classes, dtype=np.int64)

        for video in videos:
            features = np.asarray(video.features, dtype=np.float32)
            labels = np.asarray(video.labels, dtype=np.int64)
            for feature, raw_label in zip(features, labels):
                label = int(raw_label)
                if label < 0 or label >= self.num_classes:
                    continue
                cap = int(caps[label])
                seen[label] += 1
                if label not in reservoirs_x:
                    reservoirs_x[label] = np.empty((cap, features.shape[1]), dtype=np.float32)
                    reservoirs_y[label] = np.full(cap, label, dtype=np.int64)
                if filled[label] < cap:
                    reservoirs_x[label][filled[label]] = feature
                    filled[label] += 1
                    continue
                replace_idx = int(rng.integers(0, seen[label]))
                if replace_idx < cap:
                    reservoirs_x[label][replace_idx] = feature

        xs = []
        ys = []
        for label in sorted(reservoirs_x):
            take = int(filled[label])
            xs.append(reservoirs_x[label][:take])
            ys.append(reservoirs_y[label][:take])
        if not xs:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        if len(y) > self.max_train_samples:
            idx = rng.choice(len(y), size=self.max_train_samples, replace=False)
            idx.sort()
            x = x[idx]
            y = y[idx]
        self.sampling_seen_counts_ = seen
        self.sampling_caps_ = caps.astype(np.int64)
        return x, y

    def _balanced_caps(self) -> np.ndarray:
        if self.background_ratio < 0:
            raise ValueError("background_ratio must be non-negative")
        if self.min_samples_per_class <= 0:
            raise ValueError("min_samples_per_class must be positive")
        action_classes = [class_id for class_id in range(self.num_classes) if class_id != self.background_label]
        denominator = len(action_classes) + float(self.background_ratio)
        per_action = max(self.min_samples_per_class, int(self.max_train_samples // max(1.0, denominator)))
        background_cap = int(round(per_action * self.background_ratio))
        caps = np.full(self.num_classes, per_action, dtype=np.int64)
        if 0 <= self.background_label < self.num_classes:
            caps[self.background_label] = max(1, background_cap)
        total = int(caps.sum())
        if total > self.max_train_samples:
            scale = self.max_train_samples / total
            caps = np.maximum(1, np.floor(caps * scale).astype(np.int64))
            for class_id in action_classes:
                caps[class_id] = max(1, caps[class_id])
        return caps

    def predict_scores(self, feature: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("linear_probe has not been fitted")
        x = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        margins = self._decision_margins(x_scaled)
        partial = self._transform_scores(margins)
        fill_value = -1e9 if self.score_mode == "raw_margin" else 0.0
        scores = np.full(self.num_classes, fill_value, dtype=np.float32)
        classes = np.asarray(self.model.classes_, dtype=np.int64)
        valid = (classes >= 0) & (classes < self.num_classes)
        scores[classes[valid]] = partial[valid]
        total = float(scores.sum())
        if self.score_mode in {"softmax"} and total > 0:
            scores /= total
        return scores

    def _decision_margins(self, x_scaled: np.ndarray) -> np.ndarray:
        margins = np.asarray(self.model.decision_function(x_scaled), dtype=np.float64)
        classes = np.asarray(self.model.classes_, dtype=np.int64)
        margins = margins.reshape(-1)
        if len(classes) == 2 and len(margins) == 1:
            margins = np.asarray([-margins[0], margins[0]], dtype=np.float64)
        if len(margins) != len(classes):
            raise RuntimeError(
                f"decision_function returned {len(margins)} scores for {len(classes)} classes"
            )
        return np.nan_to_num(margins, nan=0.0, posinf=50.0, neginf=-50.0)

    def _transform_scores(self, margins: np.ndarray) -> np.ndarray:
        if self.score_mode == "raw_margin":
            return margins.astype(np.float32)
        if self.score_mode == "sigmoid":
            clipped = np.clip(margins, -50.0, 50.0)
            return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)
        if self.score_mode == "softmax":
            centered = margins - np.max(margins)
            exp_scores = np.exp(np.clip(centered, -50.0, 50.0))
            denom = float(exp_scores.sum())
            if denom <= 0:
                return np.full(len(margins), 1.0 / len(margins), dtype=np.float32)
            return (exp_scores / denom).astype(np.float32)
        raise ValueError(f"Unknown score_mode: {self.score_mode}")

    def predict(self, feature: np.ndarray) -> Tuple[int, float, np.ndarray]:
        scores = self.predict_scores(feature)
        pred = int(np.argmax(scores))
        conf = float(np.max(scores))
        return pred, conf, scores

    def training_diagnostics(self) -> dict[str, object]:
        sample_counts = (
            np.zeros(self.num_classes, dtype=np.int64)
            if self.sample_counts_ is None
            else np.asarray(self.sample_counts_, dtype=np.int64)
        )
        seen_counts = (
            np.zeros(self.num_classes, dtype=np.int64)
            if self.sampling_seen_counts_ is None
            else np.asarray(self.sampling_seen_counts_, dtype=np.int64)
        )
        caps = (
            np.full(self.num_classes, -1, dtype=np.int64)
            if self.sampling_caps_ is None
            else np.asarray(self.sampling_caps_, dtype=np.int64)
        )
        action_classes = [class_id for class_id in range(self.num_classes) if class_id != self.background_label]
        model_classes = [] if self.model is None else [int(v) for v in self.model.classes_]
        score_alignment = {
            f"score_class_{class_id}": (model_classes.index(class_id) if class_id in model_classes else None)
            for class_id in range(self.num_classes)
        }
        return {
            "total_sampled_frames": int(sample_counts.sum()),
            "background_label": int(self.background_label),
            "background_samples": int(sample_counts[self.background_label]) if 0 <= self.background_label < self.num_classes else 0,
            "action_samples": {str(class_id): int(sample_counts[class_id]) for class_id in action_classes},
            "missing_action_classes": [int(class_id) for class_id in action_classes if sample_counts[class_id] == 0],
            "class_balanced_sampling": bool(self.class_balanced_sampling),
            "class_balanced_effective": bool(
                self.class_balanced_sampling
                and len(action_classes) > 0
                and (sample_counts[self.background_label] <= max(sample_counts[action_classes].max(initial=0), 1) * max(self.background_ratio, 1e-9) + 1)
            ),
            "linear_class_weight": self.linear_class_weight,
            "linear_mode": self.linear_mode,
            "score_mode": self.score_mode,
            "linear_max_iter": int(self.max_iter),
            "linear_tol": float(self.tol),
            "linear_alpha": float(self.alpha),
            "convergence_warning": bool(self.convergence_warning_),
            "convergence_messages": list(self.convergence_messages_),
            "background_ratio": float(self.background_ratio),
            "min_samples_per_class": int(self.min_samples_per_class),
            "sampling_caps": {str(class_id): int(caps[class_id]) for class_id in range(self.num_classes)},
            "seen_counts": {str(class_id): int(seen_counts[class_id]) for class_id in range(self.num_classes)},
            "classifier_classes": model_classes,
            "label_to_idx": {str(k): int(v) for k, v in (self.label_to_idx_ or {}).items()},
            "idx_to_label": {str(k): int(v) for k, v in (self.idx_to_label_ or {}).items()},
            "score_class_alignment": score_alignment,
            "score_class_background_excluded_from_frame_mAP": True,
            "standard_scaler": self.scaler is not None,
        }
