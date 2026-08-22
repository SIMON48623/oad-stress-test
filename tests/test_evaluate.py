import pandas as pd
import pytest

from scripts.evaluate import aggregate_summary
from scripts.evaluate import main


def _write_summary(path, policy, budgets):
    rows = [
        {
            "policy": policy,
            "budget": budget,
            "frame_accuracy_on_predicted": 1.0 - budget,
            "log_path": f"{policy}_budget{budget:.2f}.jsonl",
        }
        for budget in budgets
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_evaluate_filters_requested_policies_and_budgets(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_summary(results_dir / "uniform_summary.csv", "uniform", [0.10, 0.25, 0.50])
    _write_summary(results_dir / "random_summary.csv", "random", [0.10, 0.25, 0.50])
    _write_summary(results_dir / "confidence_summary.csv", "confidence", [0.10, 0.25, 0.50])
    _write_summary(results_dir / "summary.csv", "confidence", [1.00])

    main([
        "--results-dir",
        str(results_dir),
        "--policies",
        "uniform",
        "random",
        "--budgets",
        "0.10",
        "0.25",
        "--clean-summary",
    ])

    summary = pd.read_csv(results_dir / "summary.csv")

    assert set(summary["policy"]) == {"uniform", "random"}
    assert set(round(float(v), 2) for v in summary["budget"]) == {0.10, 0.25}
    assert len(summary) == 4


def test_evaluate_ignores_video_summary_files(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_summary(results_dir / "linear_probe_uniform_summary.csv", "uniform", [0.10])
    pd.DataFrame([
        {
            "video_id": "v0",
            "policy": "uniform",
            "budget": 0.10,
            "frame_mAP": 0.99,
        }
    ]).to_csv(results_dir / "linear_probe_uniform_budget0.10_video_summary.csv", index=False)

    main([
        "--results-dir",
        str(results_dir),
        "--policies",
        "uniform",
        "--budgets",
        "0.10",
        "--clean-summary",
    ])

    summary = pd.read_csv(results_dir / "summary.csv")

    assert len(summary) == 1
    assert "video_id" not in summary.columns


def test_aggregate_summary_groups_by_policy_budget_threshold_and_summarizes_seeds():
    summary = pd.DataFrame({
        "policy": ["random", "random", "confidence_threshold"],
        "budget": [0.10, 0.10, 0.10],
        "threshold": [None, None, 0.70],
        "seed": [0, 1, None],
        "frame_mAP": [0.20, 0.40, 0.50],
        "coverage": [0.10, 0.20, 0.30],
        "per_class_AP": ["{}", "{}", "{}"],
    })

    aggregated = aggregate_summary(summary)
    random_row = aggregated[aggregated["policy"].eq("random")].iloc[0]
    confidence_row = aggregated[aggregated["policy"].eq("confidence_threshold")].iloc[0]

    assert random_row["num_runs"] == 2
    assert random_row["seed_count"] == 2
    assert random_row["frame_mAP_mean"] == pytest.approx(0.30)
    assert random_row["frame_mAP_std"] == pytest.approx(0.1414213562)
    assert confidence_row["threshold"] == pytest.approx(0.70)
    assert confidence_row["seed_count"] == 0
    assert pd.isna(confidence_row["frame_mAP_std"])
