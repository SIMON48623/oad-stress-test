import pandas as pd
import pytest

from oad_stress_test.metrics.basic import action_accuracy, action_precision, action_recall, coverage, summarize_basic
from oad_stress_test.metrics.budget import degradation_area, degradation_slope, summarize_budget_curve
from oad_stress_test.metrics.frame_map import average_precision, frame_map, per_class_ap, summarize_frame_map
from oad_stress_test.metrics.failure import accepted_risk_at_confidence, brier_score, expected_calibration_error, negative_log_likelihood
from oad_stress_test.metrics.selective import mean_decision_latency, selective_risk, summarize_selective
from oad_stress_test.metrics.transition import carried_state_transition_delays
from oad_stress_test.metrics.transition import observed_transition_delays
from oad_stress_test.metrics.transition import transition_count_total
from oad_stress_test.metrics.transition import transition_delays
from oad_stress_test.metrics.transition import summarize_transition_delay


def test_transition_delay_detects_stable_switch():
    logs = pd.DataFrame({
        "video_id": ["v"] * 8,
        "t": list(range(8)),
        "gt_label": [0, 0, 0, 1, 1, 1, 1, 1],
        "action_type": ["predict"] * 8,
        "prediction": [0, 0, 0, 0, 1, 1, 1, 1],
        "confidence": [0.9] * 8,
        "observed": [True] * 8,
    })
    assert transition_delays(logs, stable_steps=2) == [1]


def test_transition_delay_summary_reports_median_and_p90():
    logs = pd.DataFrame({
        "video_id": ["v0"] * 8 + ["v1"] * 8 + ["v2"] * 8,
        "t": list(range(8)) * 3,
        "gt_label": [0, 0, 1, 1, 1, 1, 1, 1] * 3,
        "action_type": ["predict"] * 24,
        "prediction": (
            [0, 0, 1, 1, 1, 1, 1, 1]  # delay 0 at t0=2
            + [0, 0, 0, 1, 1, 1, 1, 1]  # delay 1
            + [0, 0, 0, 0, 1, 1, 1, 1]  # delay 2
        ),
        "confidence": [0.9] * 24,
        "observed": [True] * 24,
    })

    assert transition_delays(logs, stable_steps=2) == [0, 1, 2]
    summary = summarize_transition_delay(logs, stable_steps=2)

    assert summary["transition_delay_median"] == pytest.approx(1.0)
    assert summary["transition_delay_p90"] == pytest.approx(1.8)
    assert summary["transition_delay_mean"] == pytest.approx(1.0)
    assert summary["transition_count_detected"] == 3
    assert summary["transition_count_total"] == 3
    assert summary["stable_transition_delay_median"] == pytest.approx(1.0)
    assert summary["stable_transition_count_detected"] == 3
    assert summary["stable_window_transition_detected_count"] == 3
    assert summary["stable_window_transition_missed_count"] == 0
    assert summary["stable_window_transition_detection_rate"] == pytest.approx(1.0)
    assert summary["stable_window_transition_delay_median"] == pytest.approx(1.0)


def test_sparse_sampling_observed_and_carried_delays_detect_transition():
    logs = pd.DataFrame({
        "video_id": ["v"] * 8,
        "t": list(range(8)),
        "gt_label": [0, 0, 1, 1, 1, 1, 1, 1],
        "action_type": ["predict", "wait", "wait", "predict", "wait", "wait", "predict", "wait"],
        "prediction": [0, 0, 0, 1, 1, 1, 1, 1],
        "confidence": [0.9] * 8,
        "observed": [True, False, False, True, False, False, True, False],
    })

    assert transition_delays(logs, stable_steps=2) == []
    assert transition_count_total(logs) == 1
    assert observed_transition_delays(logs) == [1]
    assert carried_state_transition_delays(logs, stable_steps=2) == [1]
    summary = summarize_transition_delay(logs, stable_steps=2)
    assert pd.isna(summary["transition_delay_median"])
    assert summary["transition_count_detected"] == 0
    assert summary["stable_window_transition_detected_count"] == 0
    assert summary["stable_window_transition_missed_count"] == 1
    assert summary["stable_window_transition_detection_rate"] == pytest.approx(0.0)
    assert summary["observed_transition_delay_median"] == pytest.approx(1.0)
    assert summary["observed_transition_count_detected"] == 1
    assert summary["observed_transition_detected_count"] == 1
    assert summary["carried_transition_delay_median"] == pytest.approx(1.0)
    assert summary["carried_transition_count_detected"] == 1
    assert summary["carried_transition_detected_count"] == 1


def test_dense_predictions_make_stable_observed_and_carried_delays_match():
    logs = pd.DataFrame({
        "video_id": ["v"] * 7,
        "t": list(range(7)),
        "gt_label": [0, 0, 1, 1, 1, 1, 1],
        "action_type": ["predict"] * 7,
        "prediction": [0, 0, 0, 1, 1, 1, 1],
        "confidence": [0.9] * 7,
        "observed": [True] * 7,
    })

    assert transition_delays(logs, stable_steps=2) == [1]
    assert observed_transition_delays(logs) == [1]
    assert carried_state_transition_delays(logs, stable_steps=2) == [1]


def test_budget_degradation_slope_and_auc_on_toy_summary():
    summary = pd.DataFrame({
        "budget": [0.10, 0.25, 0.50, 1.00],
        "frame_accuracy_on_predicted": [0.40, 0.55, 0.70, 1.00],
    })

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


def test_selective_risk_coverage_and_latency_on_toy_logs():
    logs = pd.DataFrame({
        "video_id": ["v"] * 6,
        "t": list(range(6)),
        "gt_label": [0, 0, 1, 1, 2, 2],
        "action_type": ["wait", "predict", "wait", "wait", "predict", "abstain"],
        "prediction": [-1, 0, -1, -1, 0, -1],
        "confidence": [0.0, 0.9, 0.0, 0.0, 0.8, 0.4],
        "observed": [False, True, False, False, True, True],
    })

    assert coverage(logs) == pytest.approx(2 / 6)
    assert action_accuracy(logs) == pytest.approx(0.0)
    assert pd.isna(action_precision(logs))
    assert action_recall(logs) == pytest.approx(0.0)
    assert selective_risk(logs) == pytest.approx(1 / 2)
    assert mean_decision_latency(logs) == pytest.approx((1 + 2 + 0) / 3)
    selective_summary = summarize_selective(logs)
    assert selective_summary["selective_risk"] == pytest.approx(1 / 2)
    assert selective_summary["mean_decision_latency_steps"] == pytest.approx(1.0)
    basic_summary = summarize_basic(logs)
    assert basic_summary["mean_confidence_all"] == pytest.approx((0.0 + 0.9 + 0.0 + 0.0 + 0.8 + 0.4) / 6)
    assert basic_summary["mean_confidence_accepted"] == pytest.approx((0.9 + 0.8) / 2)
    assert basic_summary["mean_confidence_abstained"] == pytest.approx(0.4)
    assert basic_summary["num_candidate_decisions"] == 3
    assert basic_summary["num_accepted_decisions"] == 2
    assert basic_summary["num_abstained_decisions"] == 1
    assert basic_summary["accepted_action_count"] == 1
    assert basic_summary["action_accepted_count"] == 1
    assert basic_summary["background_accepted_count"] == 1
    assert basic_summary["action_only_selective_risk"] == pytest.approx(1.0)
    assert basic_summary["action_only_action_accuracy"] == pytest.approx(0.0)
    assert basic_summary["high_conf_action_count_090"] == 0
    assert pd.isna(basic_summary["high_conf_action_error_rate_090"])


def test_action_focused_reliability_metrics_separate_background_accepts():
    logs = pd.DataFrame({
        "video_id": ["v"] * 5,
        "t": list(range(5)),
        "gt_label": [0, 0, 1, 1, 2],
        "action_type": ["predict", "predict", "predict", "predict", "wait"],
        "prediction": [0, 0, 1, 0, 2],
        "confidence": [0.99, 0.91, 0.96, 0.92, 0.88],
        "observed": [True] * 5,
    })

    summary = summarize_basic(logs, background_label=0)

    assert summary["background_accepted_count"] == 2
    assert summary["accepted_action_count"] == 2
    assert summary["action_accepted_count"] == 2
    assert summary["action_only_action_accuracy"] == pytest.approx(0.5)
    assert summary["action_only_selective_risk"] == pytest.approx(0.5)
    assert summary["high_conf_action_count_090"] == 2
    assert summary["high_conf_action_error_rate_090"] == pytest.approx(0.5)
    assert summary["high_conf_action_count_095"] == 1
    assert summary["high_conf_action_error_rate_095"] == pytest.approx(0.0)


def test_frame_map_treats_background_as_negative_and_excludes_background_ap():
    logs = pd.DataFrame({
        "gt_label": [0, 1, 0, 1, 2, 2],
        "score_class_0": [0.9, 0.1, 0.8, 0.1, 0.1, 0.1],
        "score_class_1": [0.1, 0.9, 0.4, 0.8, 0.2, 0.3],
        "score_class_2": [0.2, 0.1, 0.8, 0.3, 0.7, 0.6],
    })

    aps = per_class_ap(logs, background_label=0)

    assert average_precision(
        logs["gt_label"].eq(1).to_numpy(),
        logs["score_class_1"].to_numpy(),
    ) == pytest.approx(1.0)
    assert aps[1] == pytest.approx(1.0)
    assert aps[2] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert 0 not in aps
    assert frame_map(logs, background_label=0) == pytest.approx((1.0 + (1 / 2 + 2 / 3) / 2) / 2)
    summary = summarize_frame_map(logs, background_label=0)
    assert summary["frame_mAP"] == pytest.approx(0.7916666667)
    assert '"1": 1.0' in summary["per_class_AP"]


def test_frame_map_class_with_no_positive_samples_is_null_and_not_in_mean():
    logs = pd.DataFrame({
        "gt_label": [0, 1, 0, 1],
        "score_class_0": [0.9, 0.1, 0.8, 0.1],
        "score_class_1": [0.2, 0.9, 0.3, 0.8],
        "score_class_2": [0.7, 0.6, 0.5, 0.4],
    })

    aps = per_class_ap(logs, background_label=0)
    summary = summarize_frame_map(logs, background_label=0)

    assert aps[1] == pytest.approx(1.0)
    assert pd.isna(aps[2])
    assert frame_map(logs, background_label=0) == pytest.approx(1.0)
    assert '"2": null' in summary["per_class_AP"]


def test_seen_class_map_excludes_no_train_positive_class_but_map20_keeps_it():
    logs = pd.DataFrame({
        "gt_label": [0, 1, 2, 2],
        "score_class_0": [0.9, 0.1, 0.2, 0.1],
        "score_class_1": [0.1, 0.9, 0.2, 0.1],
        "score_class_2": [0.9, 0.1, 0.2, 0.3],
    })

    summary = summarize_frame_map(
        logs,
        background_label=0,
        train_positive_counts=[10, 5, 0],
    )

    class_2_ap = (1 / 2 + 2 / 3) / 2
    assert summary["frame_mAP"] == pytest.approx((1.0 + class_2_ap) / 2)
    assert summary["frame_mAP_20"] == pytest.approx(summary["frame_mAP"])
    assert summary["frame_mAP_seen_classes"] == pytest.approx(1.0)
    assert summary["num_seen_action_classes"] == 1
    assert summary["no_train_positive_classes"] == "[2]"
    assert '"2":' in summary["per_class_AP"]


def test_calibration_metrics_use_score_columns_and_high_confidence_accepted_risk():
    logs = pd.DataFrame({
        "gt_label": [0, 1, 1, 0],
        "action_type": ["predict", "predict", "wait", "predict"],
        "prediction": [0, 0, 1, 0],
        "confidence": [0.9, 0.95, 0.8, 0.7],
        "score_class_0": [0.9, 0.95, 0.2, 0.7],
        "score_class_1": [0.1, 0.05, 0.8, 0.3],
    })

    assert expected_calibration_error(logs, n_bins=2) >= 0.0
    assert negative_log_likelihood(logs) > 0.0
    assert brier_score(logs) >= 0.0
    assert accepted_risk_at_confidence(logs, threshold=0.90) == pytest.approx(1.0)
