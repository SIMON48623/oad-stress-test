import numpy as np
import pytest

from oad_stress_test.config import load_config
from oad_stress_test.utils.factory import make_classifier, make_datasets


def _write_split(path, video_ids):
    path.write_text("\n".join(video_ids) + "\n", encoding="utf-8")


def _write_config(path, feature_dir, train_split, test_split):
    path.write_text(
        "dataset:\n"
        f"  feature_dir: {feature_dir.as_posix()}\n"
        f"  train_split_file: {train_split.as_posix()}\n"
        f"  split_file: {test_split.as_posix()}\n"
        "  num_classes: 3\n"
        "  background_label: 0\n"
        "evaluation:\n"
        "  budgets: [0.10]\n"
        "output:\n"
        "  results_dir: results/toy\n",
        encoding="utf-8",
    )


def test_train_split_is_used_for_fitting_and_test_split_for_evaluation(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    feature_dir.mkdir()
    split_dir.mkdir()

    np.savez_compressed(
        feature_dir / "train_video.npz",
        features=np.full((3, 2), 1.0, dtype=np.float32),
        labels=np.ones(3, dtype=np.int64),
    )
    np.savez_compressed(
        feature_dir / "test_video.npz",
        features=np.full((3, 2), 99.0, dtype=np.float32),
        labels=np.full(3, 2, dtype=np.int64),
    )
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"
    _write_split(train_split, ["train_video"])
    _write_split(test_split, ["test_video"])
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, feature_dir, train_split, test_split)

    cfg = load_config(config_path)
    train, test = make_datasets(cfg)
    clf = make_classifier(cfg, train)

    assert train.video_ids == ["train_video"]
    assert test.video_ids == ["test_video"]
    np.testing.assert_array_equal(clf.class_means[1], np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(clf.class_means[2], np.array([1.0, 1.0], dtype=np.float32))


def test_missing_train_split_errors_unless_test_fit_is_explicit(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    feature_dir.mkdir()
    split_dir.mkdir()
    np.savez_compressed(
        feature_dir / "test_video.npz",
        features=np.ones((2, 2), dtype=np.float32),
        labels=np.ones(2, dtype=np.int64),
    )
    train_split = split_dir / "missing_train.txt"
    test_split = split_dir / "test.txt"
    _write_split(test_split, ["test_video"])
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, feature_dir, train_split, test_split)
    cfg = load_config(config_path)

    with pytest.raises(FileNotFoundError, match="Train split file not found"):
        make_datasets(cfg)

    train, test = make_datasets(cfg, allow_test_fit=True)
    assert train.video_ids == ["test_video"]
    assert test.video_ids == ["test_video"]
