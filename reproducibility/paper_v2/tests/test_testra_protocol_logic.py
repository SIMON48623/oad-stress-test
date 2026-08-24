#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


def load_runner():
    path = Path(__file__).resolve().parents[1] / "testra_ek100" / "run_testra_ek100.py"
    spec = importlib.util.spec_from_file_location("testra_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_logic() -> None:
    runner = load_runner()
    scores = np.asarray(
        [
            [0.8, 0.2],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    expected_ema = np.asarray(
        [
            [0.8, 0.2],
            [0.5, 0.5],
            [0.65, 0.35],
            [0.425, 0.575],
        ]
    )
    expected_boxcar = np.asarray(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.6, 0.4],
            [0.4, 0.6],
        ]
    )
    np.testing.assert_allclose(runner.ema_probabilities(scores, 0.5), expected_ema)
    np.testing.assert_allclose(runner.boxcar_probabilities(scores, 3), expected_boxcar)

    labels = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    predictions = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    active_counts = np.ones(5, dtype=np.int64)
    transition = runner.transition_stats(labels, predictions, active_counts, horizon=3)
    assert transition == {"transition_count": 1, "delay_sum": 1.0, "missed_sum": 0.0}

    ambiguous = np.asarray([1, 2, 1, 1, 1], dtype=np.int64)
    excluded = runner.transition_stats(labels, predictions, ambiguous, horizon=3)
    assert excluded == {"transition_count": 0, "delay_sum": 0.0, "missed_sum": 0.0}

    logits = np.asarray([[[0.0, 2.0], [2.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]]])
    recalls = runner.mean_top5_recall_multiple_timesteps(logits, np.asarray([0, 1]))
    np.testing.assert_allclose(recalls, np.ones(2))

    bundle = Path(__file__).resolve().parents[1]
    args = Namespace(
        bundle_root=bundle,
        checkpoint=bundle / "checkpoints" / "test_checkpoint.pth",
        protocol=None,
        output_dir=bundle / "_test_output",
        cache=None,
        data_root=bundle / "_test_data",
    )
    _, _, protocol_path, _, _ = runner.resolve_paths(args)
    assert protocol_path == (bundle / "protocols" / "testra_ek100_protocol.json").resolve()
    assert protocol_path.is_file()
    print("protocol logic tests: PASS")


if __name__ == "__main__":
    test_protocol_logic()
