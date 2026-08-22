from __future__ import annotations

import argparse

import numpy as np

from scripts.run_v19_ek100_smoke import (
    active_set_state_ids,
    common_session_ids,
    confident_decisions,
    load_ek100_smoke_video,
    primary_labels_from_target,
    resolve_ek100_dirs,
)


def test_primary_labels_and_state_ids_handle_multilabel_targets():
    target = np.asarray(
        [
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=bool,
    )

    labels = primary_labels_from_target(target)
    states = active_set_state_ids(target)

    assert labels.tolist() == [1, 1, 1, 3]
    assert states.tolist() == [0, 1, 1, 2]


def test_load_ek100_smoke_video_fuses_rgb_flow_and_crops(tmp_path):
    rgb_dir = tmp_path / "rgb_kinetics_bninception"
    flow_dir = tmp_path / "flow_kinetics_bninception"
    target_dir = tmp_path / "target_perframe"
    rgb_dir.mkdir()
    flow_dir.mkdir()
    target_dir.mkdir()
    np.save(rgb_dir / "P01_01.npy", np.ones((5, 2), dtype=np.float32))
    np.save(flow_dir / "P01_01.npy", np.full((4, 3), 2.0, dtype=np.float32))
    target = np.zeros((6, 4), dtype=np.float64)
    target[:, 1] = 1.0
    np.save(target_dir / "P01_01.npy", target)

    video = load_ek100_smoke_video(
        "P01_01",
        rgb_dir=rgb_dir,
        flow_dir=flow_dir,
        target_dir=target_dir,
        max_frames=3,
    )

    assert video.features.shape == (3, 5)
    assert video.target.shape == (3, 4)
    assert video.primary_labels.tolist() == [1, 1, 1]


def test_common_session_ids_and_default_dir_resolution(tmp_path):
    for subdir in ["rgb_kinetics_bninception", "flow_kinetics_bninception", "target_perframe"]:
        path = tmp_path / subdir
        path.mkdir()
        np.save(path / "shared.npy", np.zeros((1, 1), dtype=np.float32))
    np.save(tmp_path / "rgb_kinetics_bninception" / "rgb_only.npy", np.zeros((1, 1), dtype=np.float32))

    args = argparse.Namespace(ek100_root=str(tmp_path), rgb_dir=None, flow_dir=None, target_dir=None)
    rgb_dir, flow_dir, target_dir = resolve_ek100_dirs(args)

    assert common_session_ids(rgb_dir, flow_dir, target_dir) == ["shared"]


def test_confident_decisions_use_entropy_in_opposite_direction():
    assert confident_decisions(np.asarray([0.2, 0.8]), 0.5, "max_prob").tolist() == [False, True]
    assert confident_decisions(np.asarray([0.2, 0.8]), 0.5, "top2_margin").tolist() == [False, True]
    assert confident_decisions(np.asarray([0.2, 0.8]), 0.5, "entropy").tolist() == [True, False]
