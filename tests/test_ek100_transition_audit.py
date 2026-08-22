import numpy as np

from tools.audit_ek100_transitions import (
    active_mask_from_target,
    audit_ek100_paths,
    distance_to_nearest_transition,
    format_audit_markdown,
    recommendation_from_bins,
    required_ek100_schema_note,
    transition_event_counts,
    transition_indices_from_active,
)


def _write_target(path, active_classes, length=80, num_classes=6):
    target = np.zeros((length, num_classes), dtype=np.float32)
    for start, end, class_ids in active_classes:
        for class_id in class_ids:
            target[start:end, class_id] = 1.0
    np.save(path, target)
    return target


def test_sparse_transitions_leave_sufficient_far_frames_for_thumos_bins(tmp_path):
    target_dir = tmp_path / "target_perframe"
    target_dir.mkdir()
    _write_target(target_dir / "P01_01.npy", [(10, 20, [1])], length=90, num_classes=6)

    audit = audit_ek100_paths(
        target_perframe_dir=target_dir,
        min_far_frames=10,
        min_far_ratio=0.05,
    )

    thumos_bins = audit["candidate_bins"]["near_le_8_far_gt_32"]
    assert audit["num_videos"] == 1
    assert audit["background_no_action_convention"] == "all_zero_target_row_inferred_as_no_action"
    assert audit["transition_event_totals"]["action_entry_events"] == 1
    assert audit["transition_event_totals"]["action_exit_events"] == 1
    assert thumos_bins["far_sufficient"] is True
    assert audit["far_frame_recommendation"] == "keep_thumos_style_near_le_8_far_gt_32_for_comparability"


def test_dense_transitions_trigger_continuous_proximity_recommendation(tmp_path):
    target_dir = tmp_path / "target_perframe"
    target_dir.mkdir()
    target = np.zeros((40, 4), dtype=np.float32)
    for idx in range(40):
        target[idx, 1 + (idx // 2) % 3] = 1.0
    np.save(target_dir / "P02_01.npy", target)

    audit = audit_ek100_paths(
        target_perframe_dir=target_dir,
        min_far_frames=1,
        min_far_ratio=0.01,
    )

    assert audit["candidate_bins"]["near_le_8_far_gt_32"]["far_frames"] == 0
    assert audit["candidate_bins"]["near_le_4_far_gt_16"]["far_frames"] == 0
    assert audit["far_frame_recommendation"] == "prefer_continuous_proximity_diagnostic_over_binned_atm_claim"


def test_multilabel_event_counts_and_no_action_frames():
    target = np.zeros((5, 5), dtype=np.float32)
    target[1, 1] = 1.0
    target[2, [1, 2]] = 1.0
    target[3, 2] = 1.0
    active = active_mask_from_target(target)

    assert transition_indices_from_active(active).tolist() == [1, 2, 3, 4]
    assert transition_event_counts(active) == {
        "transition_events": 4,
        "action_entry_events": 1,
        "action_exit_events": 1,
        "action_to_action_events": 2,
        "label_entry_count": 2,
        "label_exit_count": 2,
    }
    np.testing.assert_allclose(distance_to_nearest_transition(active), np.array([1, 0, 0, 0, 0], dtype=np.float32))


def test_optional_split_and_feature_shape_audit(tmp_path):
    target_dir = tmp_path / "target_perframe"
    rgb_dir = tmp_path / "rgb_kinetics_bninception"
    flow_dir = tmp_path / "flow_kinetics_bninception"
    target_dir.mkdir()
    rgb_dir.mkdir()
    flow_dir.mkdir()
    _write_target(target_dir / "P03_01.npy", [(4, 8, [1])], length=20, num_classes=6)
    _write_target(target_dir / "P03_02.npy", [(2, 4, [2])], length=12, num_classes=6)
    np.save(rgb_dir / "P03_01.npy", np.zeros((20, 1024), dtype=np.float32))
    np.save(flow_dir / "P03_01.npy", np.zeros((20, 1024), dtype=np.float32))
    split_file = tmp_path / "test.txt"
    split_file.write_text("P03_01\nmissing_session\n", encoding="utf-8")

    audit = audit_ek100_paths(
        target_perframe_dir=target_dir,
        rgb_feature_dir=rgb_dir,
        flow_feature_dir=flow_dir,
        split_file=split_file,
        feature_stride_seconds=0.25,
        min_far_frames=1,
        min_far_ratio=0.01,
    )
    markdown = format_audit_markdown(audit)

    assert audit["video_ids_audited"] == ["P03_01"]
    assert audit["missing_target_ids"] == ["missing_session"]
    assert audit["feature_stride_seconds"] == 0.25
    assert audit["rgb_feature_audit"]["shape_distribution"] == {"20x1024": 1}
    assert audit["flow_feature_audit"]["matched_files"] == 1
    assert "near window seconds" in markdown


def test_compact_bin_recommendation_and_requirements_note():
    bins = {
        "near_le_8_far_gt_32": {"far_sufficient": False},
        "near_le_4_far_gt_16": {"far_sufficient": True},
    }
    note = required_ek100_schema_note()

    assert recommendation_from_bins(bins) == "use_ek100_specific_near_le_4_far_gt_16_with_caveat"
    assert "target_perframe" in note
    assert "rgb_kinetics_bninception" in note
    assert "L x 3807" in note
