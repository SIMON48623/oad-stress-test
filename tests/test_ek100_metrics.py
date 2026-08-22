import numpy as np

from oad_stress_test.metrics.ek100 import (
    set_match_correct,
    top1_in_active_set,
    top1_not_in_active_set_error,
)


def test_top1_inside_active_label_set_is_correct():
    assert top1_in_active_set(7, {2, 7, 11}) is True
    assert top1_not_in_active_set_error(7, {2, 7, 11}) is False


def test_top1_outside_active_label_set_is_error():
    assert top1_in_active_set(3, [1, 2, 4]) is False
    assert top1_not_in_active_set_error(3, [1, 2, 4]) is True


def test_multihot_active_label_vector_support():
    active = np.array([0, 1, 0, 1], dtype=np.int64)

    assert top1_in_active_set(1, active) is True
    assert top1_in_active_set(2, active) is False


def test_batched_predictions_with_multilabel_containers():
    preds = np.array([1, 3, 0, 4], dtype=np.int64)
    active = [{1, 2}, [0, 2], np.array([1, 0, 0, 0, 0]), set()]

    correct = top1_in_active_set(preds, active)
    errors = top1_not_in_active_set_error(preds, active)

    np.testing.assert_array_equal(correct, np.array([True, False, True, False]))
    np.testing.assert_array_equal(errors, np.array([False, True, False, True]))


def test_empty_active_set_is_marked_incorrect():
    assert top1_in_active_set(0, set()) is False
    assert top1_not_in_active_set_error(0, set()) is True


def test_set_match_is_sensitivity_only_helper():
    assert set_match_correct({1, 2}, [2, 1]) is True
    assert set_match_correct({1}, [1, 2]) is False
