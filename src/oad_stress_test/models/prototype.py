from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from oad_stress_test.datasets.schema import VideoSequence


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class PrototypeClassifier:
    """Simple class-mean classifier for first-pass feature-level baselines.

    This is intentionally simple. It is a sanity-check model, not a claimed
    OAD architecture.
    """

    num_classes: int
    temperature: float = 1.0
    class_means: np.ndarray | None = None

    def fit(self, videos: Iterable[VideoSequence]) -> "PrototypeClassifier":
        sums = None
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for video in videos:
            if sums is None:
                sums = np.zeros((self.num_classes, video.features.shape[1]), dtype=np.float64)
            for c in range(self.num_classes):
                mask = video.labels == c
                if np.any(mask):
                    sums[c] += np.sum(video.features[mask], axis=0)
                    counts[c] += int(np.sum(mask))
        if sums is None:
            raise ValueError("Cannot fit classifier with empty dataset")
        global_mean = np.sum(sums, axis=0) / max(1.0, np.sum(counts))
        means = np.zeros_like(sums, dtype=np.float32)
        for c in range(self.num_classes):
            means[c] = sums[c] / counts[c] if counts[c] > 0 else global_mean
        self.class_means = means.astype(np.float32)
        return self

    def predict_scores(self, feature: np.ndarray) -> np.ndarray:
        if self.class_means is None:
            raise RuntimeError("Classifier has not been fitted")
        distances = np.sum((self.class_means - feature[None, :]) ** 2, axis=1)
        logits = -distances / max(1e-6, self.temperature)
        return _softmax(logits)

    def predict(self, feature: np.ndarray) -> Tuple[int, float, np.ndarray]:
        probs = self.predict_scores(feature)
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return pred, conf, probs
