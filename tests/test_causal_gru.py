import numpy as np
import pytest

from _dense_policy import DenseObservationPolicy
from oad_stress_test.config import load_config
from oad_stress_test.datasets.schema import VideoSequence
from oad_stress_test.evaluators.streaming import CausalStreamingEvaluator
from oad_stress_test.models.causal_gru import CausalGRUClassifier, torch
from oad_stress_test.metrics.summary import summarize_log
from oad_stress_test.utils.factory import make_classifier, make_datasets


pytestmark = pytest.mark.skipif(torch is None, reason="causal_gru requires torch")


def _toy_video(video_id="train"):
    features = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.5, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [1.5, 0.0, 0.1],
            [2.0, 0.0, 0.2],
            [2.2, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    return VideoSequence(video_id, features, labels)


def test_causal_gru_outputs_per_timestep_scores():
    video = _toy_video()
    clf = CausalGRUClassifier(num_classes=3, hidden_dim=4, max_epochs=1, seed=0)

    clf.fit([video])
    scores = clf.predict_scores_sequence(video.features)

    assert scores.shape == (video.length, 3)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=1), np.ones(video.length), atol=1e-6)


def test_causal_gru_eval_is_causal_for_identical_prefix():
    video = _toy_video()
    clf = CausalGRUClassifier(num_classes=3, hidden_dim=4, max_epochs=1, seed=1)
    clf.fit([video])

    seq_a = video.features.copy()
    seq_b = video.features.copy()
    k = 2
    seq_b[k + 1 :] = seq_b[k + 1 :] * -3.0 + 7.0

    logits_a = clf.predict_logits_sequence(seq_a)
    logits_b = clf.predict_logits_sequence(seq_b)

    np.testing.assert_allclose(logits_a[: k + 1], logits_b[: k + 1], atol=1e-6)


def test_factory_constructs_causal_gru_and_evaluator_keeps_scores(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    feature_dir.mkdir()
    split_dir.mkdir()
    train = _toy_video("train_video")
    test = _toy_video("test_video")
    np.savez_compressed(feature_dir / "train_video.npz", features=train.features, labels=train.labels)
    np.savez_compressed(feature_dir / "test_video.npz", features=test.features, labels=test.labels)
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"
    train_split.write_text("train_video\n", encoding="utf-8")
    test_split.write_text("test_video\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
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
    cfg = load_config(config_path)
    train_dataset, test_dataset = make_datasets(cfg)

    clf = make_classifier(
        cfg,
        train_dataset,
        classifier="causal_gru",
        gru_hidden_dim=4,
        gru_max_epochs=1,
        seed=2,
        use_cache=False,
    )
    logs = CausalStreamingEvaluator().evaluate_video(
        test_dataset.load_video("test_video"),
        policy=DenseObservationPolicy(clf),
        budget=1.0,
    )
    summary = summarize_log(logs, background_label=0, train_positive_counts=clf.sampling_seen_counts_)

    assert "score_class_0" in logs.columns
    assert "score_class_1" in logs.columns
    assert "score_class_2" in logs.columns
    assert logs[["score_class_0", "score_class_1", "score_class_2"]].shape == (test.length, 3)
    assert np.isfinite(summary["frame_mAP_20"])
    assert clf.training_diagnostics()["classifier"] == "causal_gru"
