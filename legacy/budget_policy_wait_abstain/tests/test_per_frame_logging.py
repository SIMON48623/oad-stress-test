import json
import sys

import numpy as np
import pandas as pd
import pytest

from oad_stress_test.utils.jsonl_logger import (
    iter_per_frame_rows,
    open_jsonl_writer,
    uncertainty_from_probabilities,
    write_jsonl_row,
)
from scripts.run_baseline import main as run_baseline_main
from tools.validate_per_frame_log import ValidationError, validate_per_frame_log


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


def _toy_config(tmp_path, results_name="results"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    results_dir = tmp_path / results_name
    feature_dir.mkdir()
    split_dir.mkdir()
    train_features = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0]], dtype=np.float32)
    train_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    test_features = np.array([[0.0, 0.0], [2.0, 0.0], [2.1, 0.0]], dtype=np.float32)
    test_labels = np.array([0, 1, 1], dtype=np.int64)
    np.savez_compressed(feature_dir / "train_video.npz", features=train_features, labels=train_labels)
    np.savez_compressed(feature_dir / "test_video.npz", features=test_features, labels=test_labels)
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"
    _write_split(train_split, ["train_video"])
    _write_split(test_split, ["test_video"])
    config_path = tmp_path / f"{results_name}.yaml"
    _write_config(config_path, feature_dir, train_split, test_split, results_dir)
    return config_path, results_dir


def test_jsonl_writer_writes_valid_json(tmp_path):
    path = tmp_path / "per_frame.jsonl"
    row = {
        "video_id": "v",
        "frame_idx": 0,
        "gt_label": 1,
        "pred_label": 1,
        "decision": "predict",
        "max_prob": np.float32(0.8),
        "entropy": 0.5,
        "top2_margin": 0.6,
        "budget": 1.0,
        "model": "toy",
        "policy": "confidence_threshold",
        "threshold": 0.7,
        "shift_type": "clean",
        "split": "test",
    }
    with open_jsonl_writer(path) as handle:
        write_jsonl_row(handle, row)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["video_id"] == "v"
    assert loaded["max_prob"] == pytest.approx(0.8)


def test_validation_script_catches_missing_required_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"video_id": "v"}) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="missing required fields"):
        validate_per_frame_log(path)


def test_entropy_and_top2_margin_on_toy_probabilities():
    probs = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    stats = uncertainty_from_probabilities(probs)
    expected_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    assert stats["max_prob"] == pytest.approx(0.7)
    assert stats["entropy"] == pytest.approx(expected_entropy)
    assert stats["top2_margin"] == pytest.approx(0.5)


def test_iter_per_frame_rows_uses_actual_evaluator_decision():
    logs = pd.DataFrame({
        "video_id": ["v", "v"],
        "t": [0, 1],
        "gt_label": [1, 1],
        "prediction": [1, -1],
        "action_type": ["predict", "abstain"],
        "budget": [1.0, 1.0],
        "score_class_0": [0.2, 0.4],
        "score_class_1": [0.8, 0.6],
    })

    rows = list(iter_per_frame_rows(
        logs,
        model="toy",
        policy="confidence_threshold",
        threshold=0.7,
        shift_type="clean",
        split="test",
        run_id="r",
    ))

    assert [row["decision"] for row in rows] == ["predict", "abstain"]
    assert rows[0]["pred_label"] == 1
    assert rows[1]["pred_label"] == -1


def test_dump_per_frame_does_not_change_summary_only_outputs(tmp_path, monkeypatch):
    plain_config, plain_results = _toy_config(tmp_path / "plain", results_name="results")
    dump_config, dump_results = _toy_config(tmp_path / "dump", results_name="results")
    per_frame_path = dump_results / "per_frame.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline.py",
            "--config",
            str(plain_config),
            "--policy",
            "confidence_threshold",
            "--budgets",
            "1.00",
            "--thresholds",
            "0.50",
            "--summary-only",
            "--overwrite",
            "--no-cache",
        ],
    )
    run_baseline_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline.py",
            "--config",
            str(dump_config),
            "--policy",
            "confidence_threshold",
            "--budgets",
            "1.00",
            "--thresholds",
            "0.50",
            "--summary-only",
            "--dump-per-frame",
            "--per-frame-output",
            str(per_frame_path),
            "--overwrite",
            "--no-cache",
        ],
    )
    run_baseline_main()

    plain_summary = pd.read_csv(plain_results / "confidence_threshold_summary.csv")
    dump_summary = pd.read_csv(dump_results / "confidence_threshold_summary.csv")
    assert len(plain_summary) == len(dump_summary) == 1
    assert plain_summary["num_videos"].iloc[0] == dump_summary["num_videos"].iloc[0]
    assert not list(plain_results.glob("*budget*.jsonl"))
    assert not list(dump_results.glob("*budget*.jsonl"))
    assert per_frame_path.exists()
    assert (dump_results / "summary_readable.md").exists()
    stats = validate_per_frame_log(per_frame_path)
    assert stats["total_frames"] == 3
