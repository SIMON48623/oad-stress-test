from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Decision:
    action_type: str
    observed: bool
    prediction: int | None
    confidence: float
    scores: np.ndarray | None = None


class DenseObservationPolicy:
    """Minimal test double for the former uniform policy at budget=1."""

    name = "uniform"

    def __init__(self, classifier):
        self.classifier = classifier

    def reset(self, video_id: str, video_length: int, budget: float) -> None:
        if budget != 1.0:
            raise ValueError("DenseObservationPolicy is only defined for budget=1 in active tests")
        if hasattr(self.classifier, "reset_sequence"):
            self.classifier.reset_sequence(video_id=video_id)

    def step(self, t: int, x_t: np.ndarray) -> Decision:
        prediction, confidence, scores = self.classifier.predict(x_t)
        return Decision("predict", True, prediction, confidence, scores)
