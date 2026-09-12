from pathlib import Path

import numpy as np
import pandas as pd

from _dense_policy import DenseObservationPolicy
from oad_stress_test.config import load_config
from oad_stress_test.datasets.schema import VideoSequence
from oad_stress_test.evaluators.streaming import CausalStreamingEvaluator
from oad_stress_test.metrics.summary import summarize_log
from oad_stress_test.models.linear_probe import LinearProbeClassifier
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
        "  budgets: [1.00]\n"
        "output:\n"
        "  results_dir: results/toy\n",
        encoding="utf-8",
    )


def _build_linear_cache_toy(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    feature_dir.mkdir()
    split_dir.mkdir()
    train_features = np.array(
        [
            [-2.0, 0.0],
            [-1.8, 0.1],
            [-1.5, -0.1],
            [1.5, 0.0],
            [1.8, 0.1],
            [2.0, -0.1],
        ],
        dtype=np.float32,
    )
    train_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    test_features = np.array([[-1.9, 0.0], [1.9, 0.0], [2.1, 0.1]], dtype=np.float32)
    test_labels = np.array([0, 1, 1], dtype=np.int64)
    np.savez_compressed(feature_dir / "train_video.npz", features=train_features, labels=train_labels)
    np.savez_compressed(feature_dir / "test_video.npz", features=test_features, labels=test_labels)
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"
    _write_split(train_split, ["train_video"])
    _write_split(test_split, ["test_video"])
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, feature_dir, train_split, test_split)
    cfg = load_config(config_path)
    return cfg, make_datasets(cfg)


def test_linear_probe_fit_uses_train_and_evaluation_uses_test(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    feature_dir.mkdir()
    split_dir.mkdir()

    np.savez_compressed(
        feature_dir / "train_video.npz",
        features=np.array([[-2.0, 0.0], [-1.5, 0.0], [1.5, 0.0], [2.0, 0.0]], dtype=np.float32),
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
    )
    np.savez_compressed(
        feature_dir / "test_video.npz",
        features=np.array([[-2.0, 0.0], [3.0, 0.0], [3.5, 0.0]], dtype=np.float32),
        labels=np.array([0, 2, 2], dtype=np.int64),
    )
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"
    _write_split(train_split, ["train_video"])
    _write_split(test_split, ["test_video"])
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, feature_dir, train_split, test_split)

    cfg = load_config(config_path)
    train, test = make_datasets(cfg)
    clf = make_classifier(
        cfg,
        train,
        classifier="linear_probe",
        max_train_samples=4,
        class_balanced_sampling=True,
        score_mode="softmax",
        seed=0,
    )

    assert set(clf.observed_classes_.tolist()) == {0, 1}
    assert 2 not in clf.observed_classes_

    video = test.load_video("test_video")
    policy = DenseObservationPolicy(clf)
    logs = CausalStreamingEvaluator().evaluate_video(video, policy=policy, budget=1.0)

    assert logs["video_id"].unique().tolist() == ["test_video"]
    assert "score_class_0" in logs.columns
    assert "score_class_1" in logs.columns
    assert "score_class_2" in logs.columns
    assert logs["score_class_2"].tolist() == [0.0, 0.0, 0.0]
    score_values = logs[["score_class_0", "score_class_1", "score_class_2"]].to_numpy()
    assert np.isfinite(score_values).all()
    np.testing.assert_allclose(score_values.sum(axis=1), np.ones(len(logs)), atol=1e-6)
    assert clf.scaler is not None
    diagnostics = clf.training_diagnostics()
    assert diagnostics["label_to_idx"] == {"0": 0, "1": 1, "2": 2}
    assert diagnostics["idx_to_label"] == {"0": 0, "1": 1, "2": 2}
    assert diagnostics["score_class_alignment"]["score_class_0"] == 0
    assert diagnostics["score_class_alignment"]["score_class_1"] == 1
    assert diagnostics["score_class_alignment"]["score_class_2"] is None
    assert diagnostics["score_class_background_excluded_from_frame_mAP"] is True

    summary = summarize_log(logs, background_label=0)
    assert not pd.isna(summary["frame_mAP"])
    assert "2" in summary["per_class_AP"]


def test_linear_probe_balanced_sampling_limits_background():
    features = np.vstack([
        np.zeros((100, 2), dtype=np.float32),
        np.ones((20, 2), dtype=np.float32),
        np.full((20, 2), 2.0, dtype=np.float32),
    ])
    labels = np.array([0] * 100 + [1] * 20 + [2] * 20, dtype=np.int64)
    video = VideoSequence("train", features, labels)
    clf = LinearProbeClassifier(
        num_classes=3,
        max_train_samples=12,
        class_balanced_sampling=True,
        linear_class_weight="balanced",
        background_ratio=1.0,
        min_samples_per_class=2,
        seed=0,
    )

    clf.fit([video])
    diagnostics = clf.training_diagnostics()

    assert diagnostics["total_sampled_frames"] == 12
    assert diagnostics["background_samples"] == 4
    assert diagnostics["action_samples"]["1"] == 4
    assert diagnostics["action_samples"]["2"] == 4
    assert diagnostics["missing_action_classes"] == []
    assert diagnostics["class_balanced_effective"] is True
    assert diagnostics["linear_class_weight"] == "balanced"


def test_linear_probe_raw_margin_scores_are_not_softmax_normalized():
    video = VideoSequence(
        "train",
        np.array([[-2.0, 0.0], [-1.5, 0.0], [1.5, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.array([0, 0, 1, 1], dtype=np.int64),
    )
    raw = LinearProbeClassifier(num_classes=2, score_mode="raw_margin", seed=0)
    softmax = LinearProbeClassifier(num_classes=2, score_mode="softmax", seed=0)

    raw.fit([video])
    softmax.fit([video])
    feature = np.array([2.0, 0.0], dtype=np.float32)
    raw_scores = raw.predict_scores(feature)
    softmax_scores = softmax.predict_scores(feature)

    assert np.isfinite(raw_scores).all()
    assert not np.allclose(raw_scores, softmax_scores)
    np.testing.assert_allclose(softmax_scores.sum(), 1.0, atol=1e-6)


def test_classifier_cache_hits_on_second_same_configuration(tmp_path):
    cfg, (train, _) = _build_linear_cache_toy(tmp_path)
    cache_dir = tmp_path / "classifier_cache"

    first = make_classifier(
        cfg,
        train,
        classifier="linear_probe",
        max_train_samples=6,
        class_balanced_sampling=True,
        score_mode="softmax",
        seed=0,
        cache_dir=cache_dir,
    )
    second = make_classifier(
        cfg,
        train,
        classifier="linear_probe",
        max_train_samples=6,
        class_balanced_sampling=True,
        score_mode="softmax",
        seed=0,
        cache_dir=cache_dir,
    )

    assert first.cache_hit_ is False
    assert second.cache_hit_ is True
    assert first.cache_path_ == second.cache_path_
    assert (cache_dir / Path(first.cache_path_).name).exists()


def test_classifier_cache_key_changes_with_max_train_samples(tmp_path):
    cfg, (train, _) = _build_linear_cache_toy(tmp_path)
    cache_dir = tmp_path / "classifier_cache"

    first = make_classifier(
        cfg,
        train,
        classifier="linear_probe",
        max_train_samples=4,
        class_balanced_sampling=True,
        score_mode="softmax",
        seed=0,
        cache_dir=cache_dir,
    )
    second = make_classifier(
        cfg,
        train,
        classifier="linear_probe",
        max_train_samples=6,
        class_balanced_sampling=True,
        score_mode="softmax",
        seed=0,
        cache_dir=cache_dir,
    )

    assert first.cache_hit_ is False
    assert second.cache_hit_ is False
    assert first.cache_path_ != second.cache_path_


def test_classifier_cache_preserves_prediction_shape_and_score_columns(tmp_path):
    cfg, (train, test) = _build_linear_cache_toy(tmp_path)
    cache_dir = tmp_path / "classifier_cache"
    kwargs = {
        "classifier": "linear_probe",
        "max_train_samples": 6,
        "class_balanced_sampling": True,
        "score_mode": "softmax",
        "seed": 0,
        "cache_dir": cache_dir,
    }
    first = make_classifier(cfg, train, **kwargs)
    cached = make_classifier(cfg, train, **kwargs)
    video = test.load_video("test_video")
    evaluator = CausalStreamingEvaluator()
    first_logs = evaluator.evaluate_video(video, policy=DenseObservationPolicy(first), budget=1.0)
    cached_logs = evaluator.evaluate_video(video, policy=DenseObservationPolicy(cached), budget=1.0)
    score_cols = [f"score_class_{class_id}" for class_id in range(3)]

    assert cached.cache_hit_ is True
    assert score_cols == [col for col in first_logs.columns if col.startswith("score_class_")]
    assert score_cols == [col for col in cached_logs.columns if col.startswith("score_class_")]
    assert first_logs[score_cols].shape == cached_logs[score_cols].shape
    np.testing.assert_allclose(first_logs[score_cols].to_numpy(), cached_logs[score_cols].to_numpy(), atol=1e-6)
