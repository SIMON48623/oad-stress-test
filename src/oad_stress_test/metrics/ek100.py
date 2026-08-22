from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _active_set_from_container(active_labels: Any) -> set[int]:
    if isinstance(active_labels, (set, frozenset)):
        return {int(value) for value in active_labels}
    if isinstance(active_labels, (list, tuple)):
        return {int(value) for value in active_labels}
    array = np.asarray(active_labels)
    if array.ndim == 0:
        return {int(array.item())}
    if array.ndim != 1:
        raise ValueError(f"active label container must be one-dimensional for one frame, got shape {array.shape}")
    if array.dtype == bool or _is_binary_vector(array):
        return {int(idx) for idx in np.flatnonzero(array)}
    return {int(value) for value in array.tolist()}


def _is_binary_vector(array: np.ndarray) -> bool:
    if array.size == 0:
        return False
    numeric = np.asarray(array)
    return bool(np.all((numeric == 0) | (numeric == 1)))


def _is_batched_active_labels(active_labels: Any, num_predictions: int) -> bool:
    array = np.asarray(active_labels, dtype=object if isinstance(active_labels, (list, tuple)) else None)
    if array.ndim >= 2:
        return True
    if isinstance(active_labels, (list, tuple)) and len(active_labels) == num_predictions:
        if any(isinstance(value, (set, frozenset, list, tuple, np.ndarray)) for value in active_labels):
            return True
    return False


def top1_in_active_set(top1_pred: int | Iterable[int] | np.ndarray, active_labels: Any) -> bool | np.ndarray:
    """Return whether each top-1 prediction is contained in the active EK100 label set.

    Empty active sets are marked incorrect. EK100 TeSTra action targets should
    not normally contain empty active sets, but treating them as incorrect keeps
    this helper conservative and avoids silently inventing background labels.
    """

    preds = np.asarray(top1_pred)
    if preds.ndim == 0:
        active = _active_set_from_container(active_labels)
        return bool(int(preds.item()) in active) if active else False

    flat_preds = preds.astype(np.int64, copy=False).reshape(-1)
    if not _is_batched_active_labels(active_labels, len(flat_preds)):
        active = _active_set_from_container(active_labels)
        correct = np.asarray([int(pred) in active if active else False for pred in flat_preds], dtype=bool)
        return correct.reshape(preds.shape)

    active_array = list(active_labels)
    if len(active_array) != len(flat_preds):
        raise ValueError(
            f"Number of active-label containers ({len(active_array)}) does not match predictions ({len(flat_preds)})"
        )
    correct = np.asarray(
        [int(pred) in _active_set_from_container(labels) if _active_set_from_container(labels) else False for pred, labels in zip(flat_preds, active_array)],
        dtype=bool,
    )
    return correct.reshape(preds.shape)


def top1_not_in_active_set_error(top1_pred: int | Iterable[int] | np.ndarray, active_labels: Any) -> bool | np.ndarray:
    """Return EK100 multilabel top-1 error under the main v1.8e definition."""

    correct = top1_in_active_set(top1_pred, active_labels)
    if isinstance(correct, bool):
        return not correct
    return np.logical_not(correct)


def set_match_correct(predicted_labels: Any, active_labels: Any) -> bool:
    """Sensitivity-only full set-match correctness.

    This is not the main EK100 metric because exact set equality is stricter
    than THUMOS14-style top-1 correctness and changes the task definition.
    """

    return _active_set_from_container(predicted_labels) == _active_set_from_container(active_labels)
