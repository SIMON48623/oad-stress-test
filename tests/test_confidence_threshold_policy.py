import numpy as np

from oad_stress_test.policies.confidence import ConfidenceThresholdWaitAbstainPolicy
from oad_stress_test.policies.confidence import uncertainty_signal
from oad_stress_test.utils.factory import make_policy


class ToyClassifier:
    def predict(self, x_t):
        return 1, float(x_t[0]), {}


class ScoreToyClassifier:
    def predict(self, x_t):
        if int(x_t[0]) == 0:
            scores = np.array([0.90, 0.05, 0.05], dtype=np.float32)
        elif int(x_t[0]) == 1:
            scores = np.array([0.45, 0.40, 0.15], dtype=np.float32)
        else:
            scores = np.array([0.34, 0.33, 0.33], dtype=np.float32)
        return int(np.argmax(scores)), float(np.max(scores)), scores


def test_confidence_threshold_waits_abstains_then_predicts():
    policy = ConfidenceThresholdWaitAbstainPolicy(ToyClassifier(), threshold=0.7, max_wait=1)
    policy.reset("v", video_length=4, budget=1.0)

    first = policy.step(0, np.array([0.2], dtype=np.float32))
    second = policy.step(1, np.array([0.3], dtype=np.float32))
    third = policy.step(2, np.array([0.8], dtype=np.float32))

    assert first.action_type == "wait"
    assert first.observed is True
    assert second.action_type == "abstain"
    assert second.observed is True
    assert second.prediction is None
    assert third.action_type == "predict"
    assert third.prediction == 1


def test_confidence_threshold_counts_unobserved_waits():
    policy = ConfidenceThresholdWaitAbstainPolicy(ToyClassifier(), threshold=0.7, max_wait=1)
    policy.reset("v", video_length=8, budget=0.25)

    predict = policy.step(0, np.array([0.9], dtype=np.float32))
    wait = policy.step(1, np.array([0.9], dtype=np.float32))
    abstain = policy.step(2, np.array([0.9], dtype=np.float32))

    assert predict.action_type == "predict"
    assert wait.action_type == "wait"
    assert wait.observed is False
    assert abstain.action_type == "abstain"
    assert abstain.observed is False


def test_factory_builds_confidence_threshold_reference_policy():
    policy = make_policy("confidence_threshold", ToyClassifier(), threshold=0.5, max_wait=3)

    assert policy.name == "confidence_threshold"
    assert policy.threshold == 0.5
    assert policy.max_wait == 3


def test_uncertainty_signal_entropy_and_margin_direction():
    confident = np.array([0.90, 0.05, 0.05], dtype=np.float32)
    ambiguous = np.array([0.34, 0.33, 0.33], dtype=np.float32)

    assert uncertainty_signal(confident, "entropy") < uncertainty_signal(ambiguous, "entropy")
    assert uncertainty_signal(confident, "top2_margin") > uncertainty_signal(ambiguous, "top2_margin")


def test_confidence_threshold_entropy_predicts_when_entropy_is_low():
    policy = ConfidenceThresholdWaitAbstainPolicy(
        ScoreToyClassifier(),
        threshold=0.8,
        max_wait=1,
        uncertainty_mode="entropy",
    )
    policy.reset("v", video_length=3, budget=1.0)

    predict = policy.step(0, np.array([0], dtype=np.float32))
    wait = policy.step(1, np.array([2], dtype=np.float32))

    assert predict.action_type == "predict"
    assert wait.action_type == "wait"


def test_confidence_threshold_top2_margin_predicts_when_margin_is_high():
    policy = ConfidenceThresholdWaitAbstainPolicy(
        ScoreToyClassifier(),
        threshold=0.40,
        max_wait=1,
        uncertainty_mode="top2_margin",
    )
    policy.reset("v", video_length=3, budget=1.0)

    predict = policy.step(0, np.array([0], dtype=np.float32))
    wait = policy.step(1, np.array([1], dtype=np.float32))

    assert predict.action_type == "predict"
    assert wait.action_type == "wait"


def test_factory_builds_uncertainty_threshold_alias_with_mode():
    policy = make_policy(
        "uncertainty_threshold",
        ScoreToyClassifier(),
        threshold=0.5,
        max_wait=2,
        uncertainty_mode="top2_margin",
    )

    assert policy.name == "confidence_threshold"
    assert policy.uncertainty_mode == "top2_margin"
    assert policy.max_wait == 2


def test_uncertainty_wait_abstain_factory_uses_threshold_and_uncertainty_mode():
    policy = make_policy(
        "uncertainty_wait_abstain",
        ScoreToyClassifier(),
        threshold=0.8,
        max_wait=2,
        uncertainty_mode="entropy",
    )
    policy.reset("v", video_length=3, budget=1.0)

    predict = policy.step(0, np.array([0], dtype=np.float32))
    wait = policy.step(1, np.array([2], dtype=np.float32))

    assert policy.name == "uncertainty_wait_abstain"
    assert policy.predict_threshold == 0.8
    assert policy.uncertainty_mode == "entropy"
    assert policy.max_wait_observations == 2
    assert predict.action_type == "predict"
    assert wait.action_type == "wait"
