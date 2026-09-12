from __future__ import annotations

import json

import numpy as np


def ema_probabilities(probabilities: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) == 0:
        raise ValueError("probabilities must have shape [timesteps, classes]")
    smoothed = np.empty_like(probabilities)
    smoothed[0] = probabilities[0]
    for timestep in range(1, len(probabilities)):
        smoothed[timestep] = alpha * smoothed[timestep - 1] + (1.0 - alpha) * probabilities[timestep]
    return smoothed


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
            )
    return error


def transition_summary(
    labels: np.ndarray | list[int],
    predictions: np.ndarray | list[int],
    horizon: int = 16,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    delays: list[float] = []
    misses = 0
    for boundary_index, boundary in enumerate(boundaries):
        next_boundary = int(boundaries[boundary_index + 1]) if boundary_index + 1 < len(boundaries) else len(labels)
        end = min(len(labels), next_boundary, int(boundary) + int(horizon))
        hits = np.flatnonzero(predictions[boundary:end] == labels[boundary])
        if len(hits):
            delays.append(float(hits[0]))
        else:
            delays.append(float(horizon))
            misses += 1
    return {
        "mean_transition_delay": float(np.mean(delays)),
        "missed_transition_rate": float(misses / len(boundaries)),
    }


def _synthetic_inputs() -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([0] * 8 + [1] * 2 + [0] * 8, dtype=np.int64)
    probabilities = np.asarray(
        [[0.85, 0.15] if label == 0 else [0.45, 0.55] for label in labels],
        dtype=np.float64,
    )
    probabilities[[2, 5, 14]] = [0.40, 0.60]
    return labels, probabilities


def _evaluate(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "ece": expected_calibration_error(labels, probabilities),
        **transition_summary(labels, predictions, horizon=16),
    }


def run_quick_check() -> dict[str, dict[str, float]]:
    labels, native = _synthetic_inputs()
    ema = ema_probabilities(native, alpha=0.5)
    return {
        "native": _evaluate(labels, native),
        "ema_alpha_0.50": _evaluate(labels, ema),
    }


def main() -> None:
    print(json.dumps(run_quick_check(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
