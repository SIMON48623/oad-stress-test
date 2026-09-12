import pandas as pd
import pytest

from legacy.budget_policy_wait_abstain.src.oad_stress_test.metrics.budget import (
    degradation_area,
    degradation_slope,
    summarize_budget_curve,
)


def test_budget_degradation_slope_and_auc_on_toy_summary():
    summary = pd.DataFrame(
        {
            "budget": [0.10, 0.25, 0.50, 1.00],
            "frame_accuracy_on_predicted": [0.40, 0.55, 0.70, 1.00],
        }
    )

    expected_auc = (
        (0.25 - 0.10) * (0.40 + 0.55) / 2
        + (0.50 - 0.25) * (0.55 + 0.70) / 2
        + (1.00 - 0.50) * (0.70 + 1.00) / 2
    )
    expected_slope = (1.00 - 0.40) / (1.00 - 0.10)

    assert degradation_area(summary) == pytest.approx(expected_auc)
    assert degradation_slope(summary) == pytest.approx(expected_slope)
    budget_summary = summarize_budget_curve(summary)
    assert budget_summary["frame_accuracy_on_predicted_budget_auc"] == pytest.approx(expected_auc)
    assert budget_summary["frame_accuracy_on_predicted_budget_slope"] == pytest.approx(expected_slope)
