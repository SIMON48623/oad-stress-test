import json

import numpy as np
import pandas as pd
import pytest

from tools.analyze_transition_misalignment import (
    add_transition_features,
    analyze_transition_misalignment,
    distance_bin_label,
    odds_ratio,
    parse_bins,
)


def _required_row(video_id, frame_idx, gt_label, pred_label, decision):
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "gt_label": gt_label,
        "pred_label": pred_label,
        "decision": decision,
        "max_prob": 0.8,
        "entropy": 0.4,
        "top2_margin": 0.5,
        "budget": 1.0,
        "model": "toy",
        "policy": "confidence_threshold",
        "threshold": 0.7,
        "shift_type": "clean",
        "split": "test",
    }


def test_transition_detection_and_distance_on_toy_sequence():
    frame = pd.DataFrame([
        _required_row("v", 0, 0, 0, "predict"),
        _required_row("v", 1, 0, 1, "wait"),
        _required_row("v", 2, 1, 1, "predict"),
        _required_row("v", 3, 1, 0, "abstain"),
        _required_row("v", 4, 1, -1, "wait"),
        _required_row("v", 5, 0, 0, "predict"),
    ])

    out = add_transition_features(frame, tau=8, bins=parse_bins())

    assert out["is_transition_frame"].tolist() == [False, False, True, False, False, True]
    assert out["distance_to_transition"].tolist() == [2.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert out["proximity"].iloc[2] == pytest.approx(1.0)


def test_distance_bin_assignment_default_bins():
    bins = parse_bins("0,4,8,16,32,999999")

    assert distance_bin_label(0, bins) == "0-4"
    assert distance_bin_label(4, bins) == "0-4"
    assert distance_bin_label(5, bins) == "5-8"
    assert distance_bin_label(9, bins) == "9-16"
    assert distance_bin_label(17, bins) == "17-32"
    assert distance_bin_label(33, bins) == ">32"


def test_raw_and_selective_error_definitions():
    frame = pd.DataFrame([
        _required_row("v", 0, 0, 0, "predict"),
        _required_row("v", 1, 0, 1, "wait"),
        _required_row("v", 2, 1, 0, "predict"),
        _required_row("v", 3, 1, -1, "abstain"),
    ])
    out = add_transition_features(frame, tau=8, bins=parse_bins())

    assert out["raw_error"].iloc[0] == pytest.approx(0.0)
    assert out["raw_error"].iloc[1] == pytest.approx(1.0)
    assert out["raw_error"].iloc[2] == pytest.approx(1.0)
    assert pd.isna(out["raw_error"].iloc[3])
    assert out["selective_error"].iloc[0] == pytest.approx(0.0)
    assert pd.isna(out["selective_error"].iloc[1])
    assert out["selective_error"].iloc[2] == pytest.approx(1.0)
    assert pd.isna(out["selective_error"].iloc[3])


def test_provided_error_columns_are_preserved_for_multilabel_logs():
    frame = pd.DataFrame([
        {**_required_row("v", 0, 10, 99, "predict"), "raw_error": 0, "selective_error": 0},
        {**_required_row("v", 1, 11, 99, "predict"), "raw_error": 1, "selective_error": 1},
        {**_required_row("v", 2, 11, 99, "wait"), "raw_error": 0, "selective_error": 0},
        {**_required_row("v", 3, 12, -1, "abstain"), "raw_error": 0, "selective_error": 0},
    ])

    out = add_transition_features(frame, tau=8, bins=parse_bins())

    assert out["raw_error"].iloc[0] == pytest.approx(0.0)
    assert out["raw_error"].iloc[1] == pytest.approx(1.0)
    assert out["raw_error"].iloc[2] == pytest.approx(0.0)
    assert pd.isna(out["raw_error"].iloc[3])
    assert out["selective_error"].iloc[0] == pytest.approx(0.0)
    assert out["selective_error"].iloc[1] == pytest.approx(1.0)
    assert pd.isna(out["selective_error"].iloc[2])
    assert pd.isna(out["selective_error"].iloc[3])


def test_odds_ratio_uses_continuity_correction_for_zero_cells():
    values = pd.Series([True, True, False, False])
    near = pd.Series([True, True, False, False])
    far = pd.Series([False, False, True, True])

    corrected = odds_ratio(values, near, far)

    assert corrected == pytest.approx(((2.5 / 0.5) / (0.5 / 2.5)))


def test_analyzer_handles_zero_predict_frames_and_writes_outputs(tmp_path):
    rows = []
    labels = [0, 0, 1, 1, 0, 0]
    for idx, label in enumerate(labels):
        decision = "abstain" if idx % 3 == 0 else "wait"
        rows.append(_required_row("v1", idx, label, -1, decision))
    rows.extend([
        _required_row("v2", 0, 0, -1, "wait"),
        _required_row("v2", 1, 0, -1, "wait"),
    ])
    input_path = tmp_path / "per_frame.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    outputs = analyze_transition_misalignment(
        input_path=input_path,
        output_dir=tmp_path / "out",
        near_window=1,
        far_window=2,
        tau=2,
        bins="0,1,2,999999",
    )

    for path in outputs.values():
        assert path.exists()
    frame = pd.read_csv(outputs["frame_with_transition_features"])
    curves = pd.read_csv(outputs["transition_curves"])
    atm = pd.read_csv(outputs["atm_summary"])
    logistic_note = outputs["logistic_regression_summary"].read_text(encoding="utf-8")
    summary_note = outputs["summary_readable"].read_text(encoding="utf-8")

    assert {"is_transition_frame", "distance_to_transition", "raw_error", "selective_error"}.issubset(frame.columns)
    assert len(curves) > 0
    assert len(atm) > 0
    assert "zero predict frames" in logistic_note
    assert "zero predict frames" in summary_note


def test_no_figures_produces_no_png_files(tmp_path):
    rows = [
        _required_row("v", 0, 0, 0, "predict"),
        _required_row("v", 1, 1, 1, "predict"),
        _required_row("v", 2, 1, 1, "abstain"),
    ]
    input_path = tmp_path / "per_frame.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "out"
    outputs = analyze_transition_misalignment(input_path, output_dir, no_figures=True, skip_regression=True)

    assert not list(output_dir.glob("*.png"))
    assert not any(name.startswith("figure_") for name in outputs)


def test_skip_regression_writes_clear_summary(tmp_path):
    rows = [
        _required_row("v", 0, 0, 0, "predict"),
        _required_row("v", 1, 1, 0, "predict"),
        _required_row("v", 2, 1, 1, "abstain"),
    ]
    input_path = tmp_path / "per_frame.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    outputs = analyze_transition_misalignment(input_path, tmp_path / "out", skip_regression=True, no_figures=True)
    note = outputs["logistic_regression_summary"].read_text(encoding="utf-8")

    assert "Regression diagnostics skipped" in note
    assert "skipped by user flag" in note


def test_default_behavior_still_writes_figures_and_regression_summary(tmp_path):
    rows = [
        _required_row("v", 0, 0, 0, "predict"),
        _required_row("v", 1, 1, 1, "predict"),
        _required_row("v", 2, 1, 0, "abstain"),
    ]
    input_path = tmp_path / "per_frame.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "out"
    outputs = analyze_transition_misalignment(input_path, output_dir)

    assert outputs["logistic_regression_summary"].exists()
    assert len(list(output_dir.glob("*.png"))) == 3
    assert {"figure_1", "figure_2", "figure_3"}.issubset(outputs)


def test_video_without_transitions_is_excluded_from_conditioned_metrics(tmp_path):
    frame = pd.DataFrame([
        _required_row("flat", 0, 0, 0, "predict"),
        _required_row("flat", 1, 0, 0, "predict"),
        _required_row("switch", 0, 0, 0, "predict"),
        _required_row("switch", 1, 1, 1, "predict"),
    ])

    out = add_transition_features(frame, tau=8, bins=parse_bins())

    assert out[out["video_id"].eq("flat")]["distance_to_transition"].isna().all()
    assert out[out["video_id"].eq("switch")]["distance_to_transition"].notna().all()
