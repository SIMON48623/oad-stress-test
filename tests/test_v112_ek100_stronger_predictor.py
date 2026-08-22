from __future__ import annotations

import numpy as np
import pytest

from scripts.run_v112_ek100_stronger_predictor import (
    load_train_sequence,
    split_session_ids,
    topk_in_active_set,
)


def test_split_session_ids_matches_existing_lexical_protocol():
    ids = ["P01_01", "P01_02", "P26_21", "P27_01"]
    train, evaluation = split_session_ids(
        ids, max_train_sessions=2, max_eval_sessions=2
    )

    assert train == ["P01_01", "P01_02"]
    assert evaluation == ["P26_21", "P27_01"]


def test_load_train_sequence_fuses_features_and_uses_primary_active_label(tmp_path):
    rgb_dir = tmp_path / "rgb"
    flow_dir = tmp_path / "flow"
    target_dir = tmp_path / "target"
    rgb_dir.mkdir()
    flow_dir.mkdir()
    target_dir.mkdir()
    np.save(rgb_dir / "P01_01.npy", np.ones((4, 2), dtype=np.float32))
    np.save(flow_dir / "P01_01.npy", np.full((4, 3), 2.0, dtype=np.float32))
    target = np.zeros((4, 5), dtype=np.float64)
    target[:, 2] = 1.0
    target[1, 4] = 1.0
    np.save(target_dir / "P01_01.npy", target)

    video = load_train_sequence(
        "P01_01",
        rgb_dir=rgb_dir,
        flow_dir=flow_dir,
        target_dir=target_dir,
        max_frames=3,
    )

    assert video.features.shape == (3, 5)
    assert video.labels.tolist() == [2, 2, 2]


def test_topk_active_set_accuracy_uses_any_active_label():
    scores = np.asarray(
        [
            [0.6, 0.3, 0.1],
            [0.6, 0.3, 0.1],
        ]
    )
    target = np.asarray(
        [
            [False, True, False],
            [False, False, True],
        ]
    )

    assert topk_in_active_set(scores, target, 2).tolist() == [True, False]


def test_formal_split_rejects_missing_sessions():
    with pytest.raises(ValueError, match="Need at least"):
        split_session_ids(["P01_01"], max_train_sessions=1, max_eval_sessions=1)
