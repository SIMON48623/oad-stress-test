import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oad_stress_test.config import load_config
from oad_stress_test.utils.factory import make_datasets
from scripts.evaluate import aggregate_summary
from scripts.make_shift_set import main as make_shift_set_main
from scripts.run_baseline import main as run_baseline_main


def _write_split(path, video_ids):
    path.write_text("\n".join(video_ids) + "\n", encoding="utf-8")


def _write_config(path, feature_dir, train_split, test_split, results_dir):
    path.write_text(
        "dataset:\n"
        f"  feature_dir: {feature_dir.as_posix()}\n"
        f"  train_split_file: {train_split.as_posix()}\n"
        f"  split_file: {test_split.as_posix()}\n"
        "  num_classes: 2\n"
        "  background_label: 0\n"
        "evaluation:\n"
        "  stable_steps: 2\n"
        "output:\n"
        f"  results_dir: {results_dir.as_posix()}\n",
        encoding="utf-8",
    )


def _toy_config(tmp_path):
    clean_dir = tmp_path / "clean_features"
    eval_dir = tmp_path / "eval_features"
    split_dir = tmp_path / "splits"
    results_dir = tmp_path / "results"
    clean_dir.mkdir()
    eval_dir.mkdir()
    split_dir.mkdir()
    train_features = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0]], dtype=np.float32)
    train_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    clean_test_features = np.array([[0.0, 0.0], [2.0, 0.0], [2.1, 0.0]], dtype=np.float32)
    shifted_test_features = np.array([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]], dtype=np.float32)
    test_labels = np.array([0, 1, 1], dtype=np.int64)
    np.savez_compressed(clean_dir / "train_video.npz", features=train_features, labels=train_labels)
    np.savez_compressed(clean_dir / "test_video.npz", features=clean_test_features, labels=test_labels)
    np.savez_compressed(eval_dir / "test_video.npz", features=shifted_test_features, labels=test_labels)
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"
    _write_split(train_split, ["train_video"])
    _write_split(test_split, ["test_video"])
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, clean_dir, train_split, test_split, results_dir)
    return config_path, clean_dir, eval_dir, results_dir


def test_make_shift_set_shifts_npy_and_preserves_shape(tmp_path, monkeypatch):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(src / "video_a.npy", original)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_shift_set.py",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--kind",
            "noise",
            "--severity",
            "0.1",
            "--seed",
            "3",
        ],
    )

    make_shift_set_main()

    shifted = np.load(dst / "video_a.npy")
    np.testing.assert_array_equal(np.load(src / "video_a.npy"), original)
    assert shifted.shape == original.shape
    assert not np.allclose(shifted, original)


def test_eval_feature_dir_only_changes_test_dataset(tmp_path):
    config_path, clean_dir, eval_dir, _ = _toy_config(tmp_path)
    cfg = load_config(config_path)

    train, test = make_datasets(cfg, eval_feature_dir=eval_dir)
    default_train, default_test = make_datasets(cfg)

    assert train.feature_dir == clean_dir
    assert test.feature_dir == eval_dir
    assert default_train.feature_dir == clean_dir
    assert default_test.feature_dir == clean_dir
    np.testing.assert_array_equal(train.load_video("train_video").features[0], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(test.load_video("test_video").features[0], np.array([9.0, 9.0], dtype=np.float32))


def test_run_baseline_summary_preserves_shift_metadata(tmp_path, monkeypatch):
    config_path, _, eval_dir, results_dir = _toy_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline.py",
            "--config",
            str(config_path),
            "--policy",
            "uniform",
            "--budgets",
            "1.00",
            "--summary-only",
            "--overwrite",
            "--no-cache",
            "--eval-feature-dir",
            str(eval_dir),
            "--shift-name",
            "noise",
            "--shift-severity",
            "0.25",
        ],
    )

    run_baseline_main()

    summary = pd.read_csv(results_dir / "noise_s0.250_uniform_summary.csv")
    assert len(summary) == 1
    assert summary["shift_name"].iloc[0] == "noise"
    assert float(summary["shift_severity"].iloc[0]) == 0.25
    assert str(summary["is_clean"].iloc[0]).lower() == "false"
    assert (results_dir / summary["video_summary_path"].iloc[0]).name.startswith("noise_s0.250_uniform")
    assert eval_dir.resolve() == Path(summary["eval_feature_dir"].iloc[0]).resolve()


def test_run_baseline_default_summary_is_clean(tmp_path, monkeypatch):
    config_path, clean_dir, _, results_dir = _toy_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline.py",
            "--config",
            str(config_path),
            "--policy",
            "uniform",
            "--budgets",
            "1.00",
            "--summary-only",
            "--overwrite",
            "--no-cache",
        ],
    )

    run_baseline_main()

    summary = pd.read_csv(results_dir / "uniform_summary.csv")
    assert summary["shift_name"].iloc[0] == "clean"
    assert float(summary["shift_severity"].iloc[0]) == 0.0
    assert str(summary["is_clean"].iloc[0]).lower() == "true"
    assert clean_dir.resolve() == Path(summary["eval_feature_dir"].iloc[0]).resolve()


def test_aggregate_summary_keeps_clean_and_shift_rows_separate():
    summary = pd.DataFrame({
        "classifier": ["linear_probe", "linear_probe"],
        "policy": ["confidence_threshold", "confidence_threshold"],
        "budget": [0.25, 0.25],
        "threshold": [0.7, 0.7],
        "shift_name": ["clean", "noise"],
        "shift_severity": [0.0, 0.25],
        "is_clean": [True, False],
        "eval_feature_dir": ["clean_features", "noise_features"],
        "selective_risk": [0.1, 0.3],
    })

    aggregated = aggregate_summary(summary)

    assert len(aggregated) == 2
    assert set(aggregated["shift_name"]) == {"clean", "noise"}
    assert "eval_feature_dir" in aggregated.columns
