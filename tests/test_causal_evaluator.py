import numpy as np
import pytest

from _dense_policy import Decision
from oad_stress_test.datasets.schema import VideoSequence
from oad_stress_test.evaluators.streaming import CausalStreamingEvaluator


class SpyPolicy:
    name = "spy"

    def __init__(self):
        self.seen = []

    def reset(self, video_id, video_length, budget):
        self.video_id = video_id
        self.video_length = video_length
        self.budget = budget

    def step(self, t, x_t):
        self.seen.append((t, x_t.copy()))
        return Decision("predict", True, 0, 1.0, np.array([0.8, 0.2], dtype=np.float32))


def test_evaluator_passes_only_current_feature():
    features = np.arange(20, dtype=np.float32).reshape(5, 4)
    labels = np.zeros(5, dtype=np.int64)
    video = VideoSequence("v", features, labels)
    policy = SpyPolicy()
    evaluator = CausalStreamingEvaluator()
    logs = evaluator.evaluate_video(video, policy, budget=1.0)
    assert len(logs) == 5
    assert len(policy.seen) == 5
    for t, x_t in policy.seen:
        np.testing.assert_array_equal(x_t, features[t])


def test_evaluator_logs_class_scores_from_decision():
    features = np.ones((2, 3), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    video = VideoSequence("v", features, labels)
    policy = SpyPolicy()
    evaluator = CausalStreamingEvaluator()

    logs = evaluator.evaluate_video(video, policy, budget=1.0)

    assert "score_class_0" in logs.columns
    assert "score_class_1" in logs.columns
    assert logs["score_class_0"].tolist() == pytest.approx([0.8, 0.8])
    assert logs["score_class_1"].tolist() == pytest.approx([0.2, 0.2])
