from __future__ import annotations

from ..policies.confidence import (
    ConfidenceThresholdPolicy,
    ConfidenceThresholdWaitAbstainPolicy,
    UncertaintyWaitAbstainPolicy,
)
from ..policies.sampling import RandomSamplingPolicy, UniformSamplingPolicy


def make_policy(name: str, classifier, **kwargs):
    """Build a policy for the archived budget/wait-abstain baseline."""
    if name == "uniform":
        return UniformSamplingPolicy(classifier)
    if name == "random":
        return RandomSamplingPolicy(classifier, seed=int(kwargs.get("seed", 13)))
    uncertainty_mode = str(kwargs.get("uncertainty_mode", "max_prob"))
    if name == "confidence":
        return ConfidenceThresholdPolicy(
            classifier,
            threshold=float(kwargs.get("threshold", 0.7)),
            uncertainty_mode=uncertainty_mode,
        )
    if name in {"confidence_threshold", "uncertainty_threshold"}:
        return ConfidenceThresholdWaitAbstainPolicy(
            classifier,
            threshold=float(kwargs.get("threshold", 0.7)),
            max_wait=int(kwargs.get("max_wait", 10)),
            uncertainty_mode=uncertainty_mode,
        )
    if name == "uncertainty_wait_abstain":
        return UncertaintyWaitAbstainPolicy(
            classifier,
            predict_threshold=float(kwargs.get("threshold", kwargs.get("predict_threshold", 0.75))),
            abstain_threshold=float(kwargs.get("abstain_threshold", 0.45)),
            max_wait_observations=int(kwargs.get("max_wait", kwargs.get("max_wait_observations", 3))),
            uncertainty_mode=uncertainty_mode,
        )
    raise ValueError(f"Unknown policy: {name}")
