from __future__ import annotations

import io
import zipfile

import numpy as np

from oad_stress_test.datasets.tvseries import (
    TVSeriesSegment,
    active_label_set_state_id,
    align_active_label_sets,
    read_tvseries_classes,
    read_tvseries_segments,
    tvseries_label_map,
    tvseries_raw_error,
)


def _write_annotation_zip(path):
    classes = "Action A\t2\nAction B\t1\n"
    train = (
        "show_ep1\tAction A\t0.0\t2.0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\n"
        "show_ep1\tAction B\t1.0\t3.0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("TVSeries_Dataset/classes.txt", classes)
        archive.writestr("TVSeries_Dataset/GT-train.txt", train)
        archive.writestr("TVSeries_Dataset/GT-val.txt", "")
        archive.writestr("TVSeries_Dataset/GT-test.txt", "")


def test_tvseries_parser_and_alignment_preserve_overlapping_labels(tmp_path):
    archive_path = tmp_path / "tvseries.zip"
    _write_annotation_zip(archive_path)
    classes = read_tvseries_classes(archive_path)
    label_map = tvseries_label_map(classes)
    segments = read_tvseries_segments(archive_path, "train", label_map=label_map)

    aligned = align_active_label_sets(
        segments,
        video_id="show_ep1",
        length=4,
        feature_fps=1.0,
    )

    assert classes == ["Action A", "Action B"]
    assert aligned.active_label_sets == (
        frozenset({1}),
        frozenset({1, 2}),
        frozenset({2}),
        frozenset(),
    )
    assert aligned.primary_labels.tolist() == [1, 1, 2, 0]
    assert aligned.state_ids.tolist() == [1, 3, 2, 0]


def test_active_label_state_id_is_lossless_for_tvseries_actions():
    assert active_label_set_state_id([]) == 0
    assert active_label_set_state_id([1]) == 1
    assert active_label_set_state_id([1, 3]) == 5
    assert active_label_set_state_id([3, 1, 3]) == 5


def test_tvseries_raw_error_uses_active_set_membership():
    errors = tvseries_raw_error(
        np.asarray([1, 2, 3]),
        [frozenset({1, 2}), frozenset({1, 2}), frozenset({1, 2})],
    )

    assert errors.tolist() == [0, 0, 1]
    assert errors.dtype == np.int8


def test_tvseries_raw_error_uses_explicit_background_for_empty_set():
    errors = tvseries_raw_error(
        np.asarray([0, 2]),
        [frozenset(), frozenset()],
        background_label=0,
    )

    assert errors.tolist() == [0, 1]


def test_alignment_requires_documented_temporal_mapping():
    segments = [
        TVSeriesSegment("show_ep1", "Action A", 1, 0.0, 1.0),
    ]

    try:
        align_active_label_sets(segments, video_id="show_ep1", length=2)
    except ValueError as exc:
        assert "timestamps or a positive feature_fps" in str(exc)
    else:
        raise AssertionError("alignment must not guess feature timing")
