from __future__ import annotations

from collections.abc import Iterable

import numpy as np


DEFAULT_QUANTILES = (0.50, 0.70, 0.85, 0.90, 0.95)


def finite_values(values: Iterable[float] | np.ndarray, *, ignore_nonfinite: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.isfinite(array)
    if not bool(np.all(mask)) and not ignore_nonfinite:
        bad = array[~mask]
        raise ValueError(f"Score array contains non-finite values: {bad[:5].tolist()}")
    clean = array[mask] if ignore_nonfinite else array
    if clean.size == 0:
        raise ValueError("Score array contains no finite values")
    return clean


def quantile_thresholds(
    scores: Iterable[float] | np.ndarray,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    *,
    ignore_nonfinite: bool = True,
) -> list[dict[str, float]]:
    clean = finite_values(scores, ignore_nonfinite=ignore_nonfinite)
    qs = np.asarray(list(quantiles), dtype=np.float64).reshape(-1)
    if qs.size == 0:
        raise ValueError("At least one quantile is required")
    if np.any(~np.isfinite(qs)) or np.any(qs < 0.0) or np.any(qs > 1.0):
        raise ValueError(f"Quantiles must be finite values in [0, 1], got {qs.tolist()}")
    values = np.quantile(clean, qs)
    return [{"quantile": float(q), "threshold": float(v)} for q, v in zip(qs.tolist(), values.tolist())]


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    centered = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(centered, -50.0, 50.0))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def looks_like_probabilities(scores: np.ndarray, *, atol: float = 1e-3) -> bool:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        return False
    if np.any(~np.isfinite(values)) or np.any(values < -atol):
        return False
    row_sums = values.sum(axis=1)
    return bool(np.allclose(row_sums, np.ones_like(row_sums), atol=atol))


def probabilities_from_scores(scores: np.ndarray, input_kind: str = "auto") -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D score/logit array [N, C], got shape {values.shape}")
    if input_kind not in {"auto", "logits", "probabilities"}:
        raise ValueError(f"Unknown input_kind: {input_kind}")
    if input_kind == "probabilities":
        if np.any(values < 0):
            raise ValueError("Probability arrays must be non-negative")
        denom = values.sum(axis=1, keepdims=True)
        if np.any(denom <= 0):
            raise ValueError("Probability rows must have positive sums")
        return values / denom
    if input_kind == "logits":
        return _softmax(values)
    return values if looks_like_probabilities(values) else _softmax(values)


def score_signals(scores: np.ndarray, input_kind: str = "auto") -> dict[str, np.ndarray]:
    probs = probabilities_from_scores(scores, input_kind=input_kind)
    clipped = np.clip(probs, 1e-12, 1.0)
    top2 = np.sort(probs, axis=1)[:, -2:]
    return {
        "max_prob": probs.max(axis=1).astype(np.float64),
        "entropy": (-np.sum(clipped * np.log(clipped), axis=1)).astype(np.float64),
        "top2_margin": (top2[:, 1] - top2[:, 0]).astype(np.float64),
    }


def threshold_grid_from_signals(
    signals: dict[str, Iterable[float] | np.ndarray],
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    *,
    ignore_nonfinite: bool = True,
) -> dict[str, list[dict[str, float]]]:
    return {
        str(name): quantile_thresholds(values, quantiles=quantiles, ignore_nonfinite=ignore_nonfinite)
        for name, values in signals.items()
    }
