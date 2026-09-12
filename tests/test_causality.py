import numpy as np

from _dense_policy import Decision
from oad_stress_test.datasets.schema import VideoSequence
from oad_stress_test.evaluators.streaming import CausalStreamingEvaluator


class FutureProbePolicy:
    name = "future_probe"

    def __init__(self):
        self.future_reads = []
        self.writeable_flags = []

    def reset(self, video_id, video_length, budget):
        self.video_id = video_id
        self.video_length = video_length
        self.budget = budget

    def step(self, t, x_t):
        self.writeable_flags.append(bool(x_t.flags.writeable))

        root = x_t
        while isinstance(getattr(root, "base", None), np.ndarray):
            root = root.base

        future_value = None
        if isinstance(root, np.ndarray) and root.ndim == 2 and t + 1 < root.shape[0]:
            future_value = int(root[t + 1, 1])
        self.future_reads.append(future_value)

        return Decision("predict", True, int(x_t[0]), 1.0)


def make_toy_video_with_future_label_markers():
    labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    features = np.zeros((len(labels), 3), dtype=np.float32)
    features[:, 0] = labels
    features[:, 1] = labels * 1000 + 123
    features[:, 2] = -labels
    return VideoSequence("toy_future_markers", features, labels)


def test_policy_cannot_recover_future_frames_from_current_feature():
    video = make_toy_video_with_future_label_markers()
    policy = FutureProbePolicy()

    logs = CausalStreamingEvaluator().evaluate_video(video, policy, budget=1.0)

    assert len(logs) == video.length
    assert policy.future_reads == [None] * video.length
    assert policy.writeable_flags == [False] * video.length


def test_evaluator_logs_do_not_contain_prediction_inputs():
    video = make_toy_video_with_future_label_markers()
    policy = FutureProbePolicy()

    logs = CausalStreamingEvaluator().evaluate_video(video, policy, budget=1.0)

    forbidden_tokens = ("feature", "input", "x_t", "future", "probs", "logits")
    assert not [
        column
        for column in logs.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]

    for value in logs.to_numpy().ravel():
        assert not isinstance(value, (np.ndarray, list, tuple, dict))
