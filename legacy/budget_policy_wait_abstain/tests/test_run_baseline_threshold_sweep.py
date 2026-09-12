import sys

import numpy as np
import pandas as pd

from scripts.run_baseline import _run_name, _summary_name, _threshold_values, main


def _write_split(path, video_ids):
    path.write_text("\n".join(video_ids) + "\n", encoding="utf-8")


def test_uncertainty_wait_abstain_helpers_preserve_threshold_and_mode():
    assert _threshold_values("uncertainty_wait_abstain", [0.5, 0.9], 0.7) == [0.5, 0.9]
    run_name = _run_name(
        "uncertainty_wait_abstain",
        threshold=0.7,
        classifier="causal_gru",
        uncertainty_mode="entropy",
    )
    summary_name = _summary_name(
        "uncertainty_wait_abstain",
        classifier="causal_gru",
        uncertainty_mode="entropy",
    )

    assert run_name == "causal_gru_uncertainty_wait_abstain_entropy_thr0.70"
    assert summary_name == "causal_gru_uncertainty_wait_abstain_entropy_summary.csv"


def test_uncertainty_wait_abstain_summary_keeps_threshold_rows(tmp_path, monkeypatch):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    results_dir = tmp_path / "results"
    feature_dir.mkdir()
    split_dir.mkdir()
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.2, 0.1],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    np.savez_compressed(feature_dir / "train.npz", features=features, labels=labels)
    np.savez_compressed(feature_dir / "test.npz", features=features, labels=labels)
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"
    _write_split(train_split, ["train"])
    _write_split(test_split, ["test"])
    config = tmp_path / "config.yaml"
    config.write_text(
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline.py",
            "--config",
            str(config),
            "--policy",
            "uncertainty_wait_abstain",
            "--classifier",
            "prototype",
            "--budgets",
            "0.50",
            "1.00",
            "--thresholds",
            "0.50",
            "0.90",
            "--uncertainty-mode",
            "entropy",
            "--summary-only",
            "--overwrite",
        ],
    )

    main()
    summary = pd.read_csv(results_dir / "uncertainty_wait_abstain_entropy_summary.csv")

    assert len(summary) == 4
    assert set(summary["uncertainty_mode"]) == {"entropy"}
    assert set(round(float(value), 2) for value in summary["threshold"]) == {0.50, 0.90}
    assert set(round(float(value), 2) for value in summary["budget"]) == {0.50, 1.00}
    assert summary["video_summary_path"].str.contains("entropy_thr").all()
