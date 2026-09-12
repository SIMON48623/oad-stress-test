from __future__ import annotations

import numpy as np

from .base import Decision, StreamingPolicy


class UniformSamplingPolicy(StreamingPolicy):
    name = "uniform"

    def __init__(self, classifier):
        self.classifier = classifier

    def reset(self, video_id: str, video_length: int, budget: float) -> None:
        super().reset(video_id, video_length, budget)
        self.stride = max(1, int(round(1.0 / max(1e-6, budget))))

    def step(self, t: int, x_t: np.ndarray) -> Decision:
        should_observe = (t % self.stride == 0) and self.can_observe()
        if not should_observe:
            return Decision("wait", False, self.last_prediction, self.last_confidence, self.last_scores)
        self.processed += 1
        pred, conf, scores = self.classifier.predict(x_t)
        self.last_prediction = pred
        self.last_confidence = conf
        self.last_scores = scores
        return Decision("predict", True, pred, conf, scores)


class RandomSamplingPolicy(StreamingPolicy):
    name = "random"

    def __init__(self, classifier, seed: int = 13):
        self.classifier = classifier
        self.seed = seed

    def reset(self, video_id: str, video_length: int, budget: float) -> None:
        super().reset(video_id, video_length, budget)
        seed = (abs(hash(video_id)) + self.seed) % (2**32)
        rng = np.random.default_rng(seed)
        k = self.max_processed
        self.observe_set = set(rng.choice(video_length, size=min(k, video_length), replace=False).tolist())

    def step(self, t: int, x_t: np.ndarray) -> Decision:
        should_observe = (t in self.observe_set) and self.can_observe()
        if not should_observe:
            return Decision("wait", False, self.last_prediction, self.last_confidence, self.last_scores)
        self.processed += 1
        pred, conf, scores = self.classifier.predict(x_t)
        self.last_prediction = pred
        self.last_confidence = conf
        self.last_scores = scores
        return Decision("predict", True, pred, conf, scores)
