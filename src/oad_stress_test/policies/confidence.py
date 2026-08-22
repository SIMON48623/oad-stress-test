from __future__ import annotations

import numpy as np

from .base import Decision, StreamingPolicy


UNCERTAINTY_MODES = {"max_prob", "entropy", "top2_margin"}


def uncertainty_signal(scores: np.ndarray, mode: str) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return 0.0
    if mode == "max_prob":
        return float(np.max(values))
    if mode == "entropy":
        clipped = np.clip(values, 1e-12, 1.0)
        return float(-np.sum(clipped * np.log(clipped)))
    if mode == "top2_margin":
        if len(values) < 2:
            return float(np.max(values))
        top2 = np.partition(values, -2)[-2:]
        return float(np.max(top2) - np.min(top2))
    raise ValueError(f"Unknown uncertainty_mode: {mode}")


def is_confident(scores: np.ndarray, threshold: float, mode: str) -> bool:
    signal = uncertainty_signal(scores, mode)
    if mode == "entropy":
        return signal <= float(threshold)
    return signal >= float(threshold)


def is_confident_decision(confidence: float, scores: np.ndarray | None, threshold: float, mode: str) -> bool:
    if mode == "max_prob":
        return float(confidence) >= float(threshold)
    if scores is None:
        raise ValueError(f"uncertainty_mode={mode} requires class scores")
    return is_confident(scores, threshold, mode)


class ConfidenceThresholdPolicy(StreamingPolicy):
    """Observe at scheduled intervals; predict only if confidence is high.

    Low-confidence observed timesteps become wait decisions. This is a wrapper
    baseline, not a new OAD method.
    """

    name = "confidence"

    def __init__(self, classifier, threshold: float = 0.7, uncertainty_mode: str = "max_prob"):
        if uncertainty_mode not in UNCERTAINTY_MODES:
            raise ValueError(f"Unknown uncertainty_mode: {uncertainty_mode}")
        self.classifier = classifier
        self.threshold = threshold
        self.uncertainty_mode = uncertainty_mode

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
        if is_confident_decision(conf, scores, self.threshold, self.uncertainty_mode):
            return Decision("predict", True, pred, conf, scores)
        return Decision("wait", True, pred, conf, scores)


class ConfidenceThresholdWaitAbstainPolicy(StreamingPolicy):
    """Confidence-threshold reference baseline with bounded wait streaks.

    This is a diagnostic baseline: it predicts only above a confidence
    threshold, waits below it, and abstains after too many consecutive waits.
    It is not intended as a novel OAD method.
    """

    name = "confidence_threshold"

    def __init__(self, classifier, threshold: float = 0.7, max_wait: int = 10, uncertainty_mode: str = "max_prob"):
        if uncertainty_mode not in UNCERTAINTY_MODES:
            raise ValueError(f"Unknown uncertainty_mode: {uncertainty_mode}")
        self.classifier = classifier
        self.threshold = float(threshold)
        self.max_wait = int(max_wait)
        self.uncertainty_mode = uncertainty_mode

    def reset(self, video_id: str, video_length: int, budget: float) -> None:
        super().reset(video_id, video_length, budget)
        self.stride = max(1, int(round(1.0 / max(1e-6, budget))))
        self.consecutive_waits = 0

    def _wait_or_abstain(
        self,
        observed: bool,
        prediction: int | None,
        confidence: float,
        scores: np.ndarray | None,
    ) -> Decision:
        self.consecutive_waits += 1
        if self.consecutive_waits > self.max_wait:
            self.consecutive_waits = 0
            return Decision("abstain", observed, None, confidence, scores)
        return Decision("wait", observed, prediction, confidence, scores)

    def step(self, t: int, x_t: np.ndarray) -> Decision:
        should_observe = (t % self.stride == 0) and self.can_observe()
        if not should_observe:
            return self._wait_or_abstain(False, self.last_prediction, self.last_confidence, self.last_scores)

        self.processed += 1
        pred, conf, scores = self.classifier.predict(x_t)
        self.last_prediction = pred
        self.last_confidence = conf
        self.last_scores = scores
        if is_confident_decision(conf, scores, self.threshold, self.uncertainty_mode):
            self.consecutive_waits = 0
            return Decision("predict", True, pred, conf, scores)
        return self._wait_or_abstain(True, pred, conf, scores)


class UncertaintyWaitAbstainPolicy(StreamingPolicy):
    """Reference wait/abstain policy.

    The policy predicts when confidence is high, waits when confidence is
    intermediate, and abstains after repeated low-confidence observations.
    This is a reference baseline for the benchmark, not the paper's main method.
    """

    name = "uncertainty_wait_abstain"

    def __init__(
        self,
        classifier,
        predict_threshold: float = 0.75,
        abstain_threshold: float = 0.45,
        max_wait_observations: int = 3,
        uncertainty_mode: str = "max_prob",
    ):
        if uncertainty_mode not in UNCERTAINTY_MODES:
            raise ValueError(f"Unknown uncertainty_mode: {uncertainty_mode}")
        self.classifier = classifier
        self.predict_threshold = float(predict_threshold)
        self.abstain_threshold = float(abstain_threshold)
        self.max_wait_observations = int(max_wait_observations)
        self.uncertainty_mode = uncertainty_mode

    def reset(self, video_id: str, video_length: int, budget: float) -> None:
        super().reset(video_id, video_length, budget)
        self.stride = max(1, int(round(1.0 / max(1e-6, budget))))
        self.low_conf_waits = 0

    def step(self, t: int, x_t: np.ndarray) -> Decision:
        should_observe = (t % self.stride == 0) and self.can_observe()
        if not should_observe:
            return Decision("wait", False, self.last_prediction, self.last_confidence, self.last_scores)

        self.processed += 1
        pred, conf, scores = self.classifier.predict(x_t)
        self.last_prediction = pred
        self.last_confidence = conf
        self.last_scores = scores

        if is_confident_decision(conf, scores, self.predict_threshold, self.uncertainty_mode):
            self.low_conf_waits = 0
            return Decision("predict", True, pred, conf, scores)

        signal = uncertainty_signal(scores, self.uncertainty_mode) if scores is not None else float(conf)
        is_low_confidence = signal >= self.abstain_threshold if self.uncertainty_mode == "entropy" else signal <= self.abstain_threshold
        if is_low_confidence:
            self.low_conf_waits += 1
            if self.low_conf_waits >= self.max_wait_observations:
                return Decision("abstain", True, None, conf, scores)

        return Decision("wait", True, pred, conf, scores)
