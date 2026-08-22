import numpy as np
import pytest

from oad_stress_test.datasets.feature_dataset import FeatureDataset


def _write_split(path, video_ids):
    path.write_text("\n".join(video_ids) + "\n", encoding="utf-8")


def test_npy_features_with_index_annotations(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    annotation_dir = tmp_path / "annotations"
    feature_dir.mkdir()
    split_dir.mkdir()
    annotation_dir.mkdir()

    np.save(feature_dir / "video_a.npy", np.zeros((6, 4), dtype=np.float32))
    np.save(feature_dir / "video_b.npy", np.ones((4, 4), dtype=np.float32))
    _write_split(split_dir / "test.txt", ["video_a", "video_b"])
    (annotation_dir / "segments.csv").write_text(
        "video_id,start_idx,end_idx,label\n"
        "video_a,1,3,jump\n"
        "video_a,4,5,dive\n"
        "video_b,0,1,dive\n",
        encoding="utf-8",
    )

    dataset = FeatureDataset(
        feature_dir,
        split_dir / "test.txt",
        annotation_dir=annotation_dir,
        background_label=0,
        label_map={"jump": 1, "dive": 2},
    )

    video_a = dataset.load_video("video_a")
    video_b = dataset.load_video("video_b")

    assert video_a.features.shape == (6, 4)
    np.testing.assert_array_equal(video_a.labels, np.array([0, 1, 1, 1, 2, 2]))
    np.testing.assert_array_equal(video_b.labels, np.array([2, 2, 0, 0]))
    assert dataset.infer_num_classes() == 3
    assert dataset.annotation_bounds_issues() == []


def test_time_annotations_require_feature_fps_and_convert_to_steps(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    annotation_dir = tmp_path / "annotations"
    feature_dir.mkdir()
    split_dir.mkdir()
    annotation_dir.mkdir()

    np.save(feature_dir / "video_time.npy", np.zeros((5, 2), dtype=np.float32))
    _write_split(split_dir / "test.txt", ["video_time"])
    (annotation_dir / "segments.csv").write_text(
        "video_id,start_time,end_time,label\n"
        "video_time,0.5,1.5,jump\n",
        encoding="utf-8",
    )

    dataset = FeatureDataset(
        feature_dir,
        split_dir / "test.txt",
        annotation_dir=annotation_dir,
        background_label=0,
        feature_fps=2.0,
        label_map={"jump": 1},
    )

    video = dataset.load_video("video_time")

    np.testing.assert_array_equal(video.labels, np.array([0, 1, 1, 0, 0]))


def test_annotation_bounds_issues_are_reported(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    annotation_dir = tmp_path / "annotations"
    feature_dir.mkdir()
    split_dir.mkdir()
    annotation_dir.mkdir()

    np.save(feature_dir / "video_bad.npy", np.zeros((3, 2), dtype=np.float32))
    _write_split(split_dir / "test.txt", ["video_bad"])
    (annotation_dir / "segments.csv").write_text(
        "video_id,start_idx,end_idx,label\n"
        "video_bad,1,5,jump\n",
        encoding="utf-8",
    )

    dataset = FeatureDataset(
        feature_dir,
        split_dir / "test.txt",
        annotation_dir=annotation_dir,
        label_map={"jump": 1},
    )

    issues = dataset.annotation_bounds_issues()
    assert len(issues) == 1
    assert issues[0]["video_id"] == "video_bad"
    assert issues[0]["issue"] == "out_of_bounds"
    with pytest.raises(ValueError, match="out_of_bounds"):
        dataset.load_video("video_bad")
