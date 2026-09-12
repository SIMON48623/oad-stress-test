from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

ActionType = Literal["predict", "wait", "abstain"]


@dataclass
class Decision:
    action_type: ActionType
    observed: bool
    prediction: Optional[int]
    confidence: float
    scores: Optional[np.ndarray] = None


class StreamingPolicy:
    """Base class for causal streaming policies.

    The evaluator calls this policy sequentially. A policy receives x_t only for
    the current timestep; it never receives future features.
    """

    name = "base"

    def reset(self, video_id: str, video_length: int, budget: float) -> None:
        self.video_id = video_id
        self.video_length = video_length
        self.budget = float(budget)
        self.processed = 0
        self.last_prediction: Optional[int] = None
        self.last_confidence = 0.0
        self.last_scores: Optional[np.ndarray] = None
        classifier = getattr(self, "classifier", None)
        if classifier is not None and hasattr(classifier, "reset_sequence"):
            classifier.reset_sequence(video_id=video_id)

    @property
    def max_processed(self) -> int:
        return max(1, int(round(self.video_length * self.budget)))

    def can_observe(self) -> bool:
        return self.processed < self.max_processed

    def step(self, t: int, x_t: np.ndarray) -> Decision:
        raise NotImplementedError
