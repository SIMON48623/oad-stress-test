from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest

import tools.reliability_ladder as ladder
from tools.reliability_ladder import (
    LadderError,
    _distance,
    add_oracle_regime,
    apply_threshold_policy,
    causal_transition_regime,
    cluster_bootstrap,
    compute_metrics,
    fit_global_threshold,
    fit_oracle_mondrian_thresholds,
    official_calibration_evaluation,
    read_jsonl_or_tar,
    run_aci_estimated_regime,
    run_aci_global,
)


def _row(video_id: str, frame_idx: int, **updates):
    row = {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "gt_label": 0,
        "pred_label": 0,
        "decision": "abstain",
        "max_prob": 0.8,
        "entropy": 0.3,
        "top2_margin": 0.6,
        "raw_error": 0,
        "budget": 1.0,
        "model": "causal_gru",
        "policy": "confidence_threshold",
        "threshold": 0.5,
        "shift_type": "clean",
        "split": "test",
        "selective_error": 1,
    }
    row.update(updates)
    return row


def test_causal_transition_regime_does_not_read_future():
    frame = pd.DataFrame(
        [
            _row("v", 0, max_prob=0.9, entropy=0.1),
            _row("v", 1, max_prob=0.8, entropy=0.2),
            _row("v", 2, max_prob=0.7, entropy=0.3),
            _row("v", 3, max_prob=0.6, entropy=0.4),
            _row("v", 4, max_prob=0.5, entropy=0.5),
        ]
    )
    changed_future = frame.copy()
    changed_future.loc[3:, "pred_label"] = [7, 9]
    changed_future.loc[3:, "max_prob"] = [0.01, 0.99]
    changed_future.loc[3:, "entropy"] = [9.0, 0.0]

    original = causal_transition_regime(frame)
    modified = causal_transition_regime(changed_future)

    np.testing.assert_array_equal(original[:3], modified[:3])


def test_oracle_transition_distance():
    frame_idx = np.arange(6, dtype=float)
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(_distance(frame_idx, labels), [2, 1, 0, 1, 0, 1])


def test_tvseries_oracle_transition_uses_active_label_set_not_compatibility_label():
    frame = pd.DataFrame(
        [
            _row("v", 0, gt_label=0, active_label_set=[]),
            _row("v", 1, gt_label=0, active_label_set=[1]),
            _row("v", 2, gt_label=0, active_label_set=[1, 2]),
            _row("v", 3, gt_label=0, active_label_set=[1, 2]),
        ]
    )

    output = add_oracle_regime(frame, near=0, far=1, dataset="tvseries")

    assert output["gt_transition_event"].tolist() == [False, True, True, False]
    assert output["distance_to_transition"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_global_quantile_threshold_uses_target_coverage():
    calibration = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4, 0.5]})
    assert fit_global_threshold(calibration, 0.8) == pytest.approx(0.5)


def test_oracle_mondrian_excludes_middle_band():
    calibration = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.9, 0.3, 0.4],
            "oracle_regime": [0, 0, -1, 1, 1],
        }
    )
    thresholds = fit_oracle_mondrian_thresholds(calibration, 0.5, global_tau=0.7)
    assert thresholds[-1] == pytest.approx(0.7)
    assert thresholds[0] == pytest.approx(0.2)
    assert thresholds[1] == pytest.approx(0.4)
    assert thresholds[0] != 0.9
    assert thresholds[1] != 0.9


def test_cluster_bootstrap_output_shape_and_delay():
    outcomes = pd.DataFrame(
        [
            {
                "video_id": "a",
                "frame_idx": 0,
                "raw_error": 0,
                "oracle_regime": 1,
                "gt_transition_event": True,
                "predicted": False,
            },
            {
                "video_id": "a",
                "frame_idx": 1,
                "raw_error": 1,
                "oracle_regime": 1,
                "gt_transition_event": False,
                "predicted": True,
            },
            {
                "video_id": "a",
                "frame_idx": 10,
                "raw_error": 0,
                "oracle_regime": 0,
                "gt_transition_event": False,
                "predicted": True,
            },
            {
                "video_id": "b",
                "frame_idx": 0,
                "raw_error": 0,
                "oracle_regime": 1,
                "gt_transition_event": True,
                "predicted": True,
            },
            {
                "video_id": "b",
                "frame_idx": 10,
                "raw_error": 0,
                "oracle_regime": 0,
                "gt_transition_event": False,
                "predicted": True,
            },
        ]
    )
    first = cluster_bootstrap(
        outcomes, 20, 7, coverage_target=0.8, delay_window=4
    )
    second = cluster_bootstrap(
        outcomes, 20, 7, coverage_target=0.8, delay_window=4
    )
    assert set(first) == {
        "coverage",
        "selective_risk",
        "transition_risk",
        "stable_risk",
        "TEG",
        "Error_OR",
        "miscoverage",
        "delay",
    }
    assert first == second
    assert all(len(interval) == 2 for interval in first.values())


def test_tar_reader_selects_exact_member_and_requires_full_video_count(tmp_path):
    good_rows = [_row("v1", 0), _row("v2", 0)]
    bad_rows = [{**_row("bad", 0), "raw_error": None}]
    archive_path = tmp_path / "logs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, rows in [("raw.jsonl", bad_rows), ("canonical.jsonl", good_rows)]:
            payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    frame = read_jsonl_or_tar(
        [str(archive_path)],
        member="canonical.jsonl",
        expected_videos=2,
    )
    assert frame.video_id.nunique() == 2
    assert frame.attrs["source"].endswith("::canonical.jsonl")

    with pytest.raises(LadderError, match="Expected 3 videos"):
        read_jsonl_or_tar(
            [str(archive_path)],
            member="canonical.jsonl",
            expected_videos=3,
        )


def test_tvseries_official_val_test_inputs_do_not_randomly_split_test(
    tmp_path, monkeypatch
):
    calibration_path = tmp_path / "val.jsonl"
    evaluation_path = tmp_path / "test.jsonl"
    calibration_rows = [
        _row("val_ep", 0, split="val", active_label_set=[]),
        _row("val_ep", 1, split="val", active_label_set=[1], gt_label=1),
    ]
    evaluation_rows = [
        _row("test_ep", 0, split="test", active_label_set=[]),
        _row("test_ep", 1, split="test", active_label_set=[2], gt_label=2),
    ]
    for path, rows in [
        (calibration_path, calibration_rows),
        (evaluation_path, evaluation_rows),
    ]:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def fail_random_split(*_args, **_kwargs):
        raise AssertionError("TVSeries official split must not randomly split test")

    captured = {}

    def capture_outputs(_args, calibration, evaluation):
        captured["calibration_ids"] = set(calibration.video_id)
        captured["evaluation_ids"] = set(evaluation.video_id)
        return captured

    monkeypatch.setattr(ladder, "split_calib_eval_by_video", fail_random_split)
    monkeypatch.setattr(ladder, "write_outputs", capture_outputs)
    args = ladder.parse_args(
        [
            "--calibration-input",
            str(calibration_path),
            "--evaluation-input",
            str(evaluation_path),
            "--dataset",
            "tvseries",
            "--model",
            "causal_gru",
            "--out",
            str(tmp_path / "out"),
            "--expected-calibration-videos",
            "1",
            "--expected-videos",
            "1",
        ]
    )

    result = ladder.run(args)

    assert result["calibration_ids"] == {"val_ep"}
    assert result["evaluation_ids"] == {"test_ep"}


def test_tvseries_official_split_rejects_test_rows_as_calibration():
    calibration = pd.DataFrame([_row("test_a", 0, split="test")])
    evaluation = pd.DataFrame([_row("test_b", 0, split="test")])

    with pytest.raises(LadderError, match="calibration input must contain only val"):
        official_calibration_evaluation(
            calibration,
            evaluation,
            dataset="tvseries",
        )


def test_tvseries_formal_run_rejects_random_single_input(tmp_path):
    input_path = tmp_path / "test.jsonl"
    input_path.write_text(
        json.dumps(_row("test_ep", 0, split="test", active_label_set=[])) + "\n",
        encoding="utf-8",
    )
    args = ladder.parse_args(
        [
            "--input",
            str(input_path),
            "--dataset",
            "tvseries",
            "--model",
            "causal_gru",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(LadderError, match="requires separate official"):
        ladder.run(args)


def test_tvseries_formal_run_requires_active_label_set(tmp_path):
    calibration_path = tmp_path / "val.jsonl"
    evaluation_path = tmp_path / "test.jsonl"
    calibration_path.write_text(
        json.dumps(_row("val_ep", 0, split="val")) + "\n",
        encoding="utf-8",
    )
    evaluation_path.write_text(
        json.dumps(_row("test_ep", 0, split="test")) + "\n",
        encoding="utf-8",
    )
    args = ladder.parse_args(
        [
            "--calibration-input",
            str(calibration_path),
            "--evaluation-input",
            str(evaluation_path),
            "--dataset",
            "tvseries",
            "--model",
            "causal_gru",
            "--out",
            str(tmp_path / "out"),
            "--expected-calibration-videos",
            "1",
            "--expected-videos",
            "1",
        ]
    )

    with pytest.raises(LadderError, match="requires active_label_set"):
        ladder.run(args)


def test_metrics_use_raw_error_not_old_decision_or_selective_error():
    evaluation = pd.DataFrame(
        {
            "video_id": ["v", "v"],
            "frame_idx": [0, 1],
            "raw_error": [1, 0],
            "oracle_regime": [1, 0],
            "gt_transition_event": [True, False],
            "score": [0.1, 0.2],
            "decision": ["abstain", "abstain"],
            "selective_error": [0, 1],
        }
    )
    outcomes = apply_threshold_policy(evaluation, 0.15)
    metrics = compute_metrics(outcomes, coverage_target=0.8, delay_window=4)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["selective_risk"] == pytest.approx(1.0)
    assert metrics["miscoverage"] == pytest.approx(0.8)


def _aci_frames(scores, raw_error=0, regimes=None, prefix="v"):
    regimes = regimes if regimes is not None else [0] * len(scores)
    return pd.DataFrame(
        {
            "video_id": [prefix] * len(scores),
            "frame_idx": np.arange(len(scores)),
            "score": np.asarray(scores, dtype=float),
            "raw_error": np.full(len(scores), raw_error, dtype=int),
            "oracle_regime": np.where(np.arange(len(scores)) % 2, 0, 1),
            "estimated_regime": np.asarray(regimes, dtype=int),
            "gt_transition_event": np.arange(len(scores)) % 20 == 0,
        }
    )


def test_aci_global_feedback_ignores_raw_error_and_tracks_coverage():
    calibration = _aci_frames(np.linspace(0.0, 1.0, 400), prefix="cal")
    evaluation_low_error = _aci_frames(
        np.mod(np.arange(2000) * 0.61803398875, 1.0),
        raw_error=0,
        prefix="eval",
    )
    evaluation_high_error = evaluation_low_error.copy()
    evaluation_high_error["raw_error"] = 1

    low = run_aci_global(calibration, evaluation_low_error, 0.8, 0.01)
    high = run_aci_global(calibration, evaluation_high_error, 0.8, 0.01)

    np.testing.assert_array_equal(low.predicted, high.predicted)
    assert 0.70 <= low.predicted.mean() <= 0.90
    assert low.predicted.mean() != pytest.approx(0.0, abs=0.05)
    assert high.predicted.mean() != pytest.approx(1.0, abs=0.05)


def test_regime_aci_uses_separate_histories_and_ignores_raw_error():
    calibration = pd.concat(
        [
            _aci_frames(np.linspace(0.0, 0.4, 200), regimes=[0] * 200, prefix="c0"),
            _aci_frames(np.linspace(0.6, 1.0, 200), regimes=[1] * 200, prefix="c1"),
        ],
        ignore_index=True,
    )
    regimes = np.tile([0, 1], 500)
    base_scores = np.where(regimes == 0, 0.2, 0.8)
    base = _aci_frames(base_scores, regimes=regimes, prefix="eval")
    changed_other_regime = base.copy()
    changed_other_regime.loc[changed_other_regime.estimated_regime == 1, "score"] = 0.99
    changed_other_regime["raw_error"] = 1

    original = run_aci_estimated_regime(calibration, base, 0.8, 0.01)
    modified = run_aci_estimated_regime(
        calibration, changed_other_regime, 0.8, 0.01
    )
    regime_zero = base.estimated_regime.eq(0).to_numpy()

    np.testing.assert_array_equal(
        original.predicted.to_numpy()[regime_zero],
        modified.predicted.to_numpy()[regime_zero],
    )
    assert 0.65 <= original.predicted.mean() <= 0.95
