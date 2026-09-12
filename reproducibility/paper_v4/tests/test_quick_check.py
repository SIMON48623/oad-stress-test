from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
QUICK_CHECK_PATH = PACKAGE_ROOT / "quick_check.py"


def _load_quick_check():
    assert QUICK_CHECK_PATH.exists(), "quick_check.py is missing"
    spec = importlib.util.spec_from_file_location("paper_v4_quick_check", QUICK_CHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quick_check_exposes_the_four_native_and_ema_metrics():
    quick_check = _load_quick_check()

    result = quick_check.run_quick_check()

    assert set(result) == {"native", "ema_alpha_0.50"}
    assert set(result["native"]) == {
        "accuracy",
        "ece",
        "mean_transition_delay",
        "missed_transition_rate",
    }
    assert result["ema_alpha_0.50"]["accuracy"] > result["native"]["accuracy"]
    assert result["ema_alpha_0.50"]["ece"] < result["native"]["ece"]
    assert result["ema_alpha_0.50"]["mean_transition_delay"] > result["native"]["mean_transition_delay"]
    assert result["ema_alpha_0.50"]["missed_transition_rate"] > result["native"]["missed_transition_rate"]


def test_transition_summary_truncates_at_next_transition_and_scores_a_miss_as_horizon():
    quick_check = _load_quick_check()
    labels = [0, 0, 1, 1, 0, 0]
    predictions = [0, 0, 0, 0, 0, 0]

    result = quick_check.transition_summary(labels, predictions, horizon=16)

    assert result["mean_transition_delay"] == pytest.approx(8.0)
    assert result["missed_transition_rate"] == pytest.approx(0.5)


def test_cli_prints_json_without_writing_files(tmp_path):
    assert QUICK_CHECK_PATH.exists(), "quick_check.py is missing"

    completed = subprocess.run(
        [sys.executable, str(QUICK_CHECK_PATH)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert set(payload) == {"native", "ema_alpha_0.50"}
    assert list(tmp_path.iterdir()) == []
