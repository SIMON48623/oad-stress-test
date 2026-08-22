import numpy as np

from scripts.inspect_ek100_replication_inputs import format_report, inspect_inputs


def test_inspect_inputs_reports_missing_paths_without_crashing():
    report = inspect_inputs(
        target_dir=None,
        rgb_feature_path=None,
        flow_feature_path=None,
        split_path=None,
        max_files=2,
    )
    text = format_report(report)

    assert report["target_perframe"]["status"] == "MISSING"
    assert report["rgb_features"]["status"] == "MISSING"
    assert report["flow_features"]["status"] == "MISSING"
    assert report["alignment_gate"]["status"] == "NOT_EVALUABLE"
    assert "MISSING" in text
    assert "Alignment gate" in text
    assert "does not train or evaluate" in text


def test_inspect_inputs_samples_shapes_and_id_overlap(tmp_path):
    target_dir = tmp_path / "target_perframe"
    rgb_dir = tmp_path / "rgb"
    flow_dir = tmp_path / "flow"
    target_dir.mkdir()
    rgb_dir.mkdir()
    flow_dir.mkdir()
    np.save(target_dir / "P01_01.npy", np.zeros((4, 3807), dtype=np.float64))
    np.save(target_dir / "P01_02.npy", np.zeros((5, 3807), dtype=np.float64))
    np.save(rgb_dir / "P01_01.npy", np.zeros((4, 1024), dtype=np.float32))
    np.save(flow_dir / "P01_01.npy", np.zeros((4, 1024), dtype=np.float32))
    split = tmp_path / "test.txt"
    split.write_text("P01_01\nP01_03\n", encoding="utf-8")

    report = inspect_inputs(
        target_dir=target_dir,
        rgb_feature_path=rgb_dir,
        flow_feature_path=flow_dir,
        split_path=split,
        max_files=1,
    )
    text = format_report(report)

    assert report["target_perframe"]["num_array_files"] == 2
    assert report["target_perframe"]["shape_samples"][0]["shape"] == (4, 3807)
    assert report["target_perframe"]["shape_samples"][0]["dtype"] == "float64"
    assert report["rgb_features"]["shape_samples"][0]["shape"] == (4, 1024)
    assert report["overlaps"]["target_vs_rgb"]["overlap_count"] == 1
    assert report["overlaps"]["target_vs_split"]["target_missing_in_split"] == 1
    assert report["overlaps"]["target_vs_split"]["split_missing_in_target"] == 1
    assert report["alignment_gate"]["status"] == "PASS"
    sampled = report["alignment_gate"]["sampled_frame_counts"][0]
    assert sampled["rgb_alignment"]["status"] == "PASS_EXACT"
    assert sampled["flow_alignment"]["status"] == "PASS_EXACT"
    assert "shape=4x3807" in text
    assert "PASS_EXACT" in text


def test_inspect_inputs_fails_alignment_gate_on_unexplained_frame_mismatch(tmp_path):
    target_dir = tmp_path / "target_perframe"
    rgb_dir = tmp_path / "rgb"
    flow_dir = tmp_path / "flow"
    target_dir.mkdir()
    rgb_dir.mkdir()
    flow_dir.mkdir()
    np.save(target_dir / "P03_01.npy", np.zeros((8, 3807), dtype=np.float64))
    np.save(rgb_dir / "P03_01.npy", np.zeros((4, 1024), dtype=np.float32))
    np.save(flow_dir / "P03_01.npy", np.zeros((4, 1024), dtype=np.float32))

    report = inspect_inputs(
        target_dir=target_dir,
        rgb_feature_path=rgb_dir,
        flow_feature_path=flow_dir,
        max_files=1,
    )

    assert report["alignment_gate"]["status"] == "FAIL"
    sampled = report["alignment_gate"]["sampled_frame_counts"][0]
    assert sampled["rgb_alignment"]["status"] == "FAIL_MISMATCH"
    assert sampled["flow_alignment"]["status"] == "FAIL_MISMATCH"


def test_inspect_inputs_accepts_documented_expected_stride(tmp_path):
    target_dir = tmp_path / "target_perframe"
    rgb_dir = tmp_path / "rgb"
    flow_dir = tmp_path / "flow"
    target_dir.mkdir()
    rgb_dir.mkdir()
    flow_dir.mkdir()
    np.save(target_dir / "P04_01.npy", np.zeros((8, 3807), dtype=np.float64))
    np.save(rgb_dir / "P04_01.npy", np.zeros((4, 1024), dtype=np.float32))
    np.save(flow_dir / "P04_01.npy", np.zeros((4, 1024), dtype=np.float32))

    report = inspect_inputs(
        target_dir=target_dir,
        rgb_feature_path=rgb_dir,
        flow_feature_path=flow_dir,
        max_files=1,
        expected_stride=2.0,
    )

    assert report["alignment_gate"]["status"] == "PASS"
    sampled = report["alignment_gate"]["sampled_frame_counts"][0]
    assert sampled["rgb_alignment"]["status"] == "PASS_EXPECTED_STRIDE"
    assert sampled["flow_alignment"]["status"] == "PASS_EXPECTED_STRIDE"


def test_inspect_inputs_can_mark_frame_mismatch_as_user_allowed(tmp_path):
    target_dir = tmp_path / "target_perframe"
    rgb_dir = tmp_path / "rgb"
    flow_dir = tmp_path / "flow"
    target_dir.mkdir()
    rgb_dir.mkdir()
    flow_dir.mkdir()
    np.save(target_dir / "P05_01.npy", np.zeros((9, 3807), dtype=np.float64))
    np.save(rgb_dir / "P05_01.npy", np.zeros((4, 1024), dtype=np.float32))
    np.save(flow_dir / "P05_01.npy", np.zeros((4, 1024), dtype=np.float32))

    report = inspect_inputs(
        target_dir=target_dir,
        rgb_feature_path=rgb_dir,
        flow_feature_path=flow_dir,
        max_files=1,
        allow_frame_mismatch=True,
    )

    assert report["alignment_gate"]["status"] == "PASS_WITH_WARNINGS"
    sampled = report["alignment_gate"]["sampled_frame_counts"][0]
    assert sampled["rgb_alignment"]["status"] == "ALLOWED_MISMATCH"
    assert sampled["flow_alignment"]["status"] == "ALLOWED_MISMATCH"


def test_inspect_inputs_warns_when_feature_ids_cannot_be_inferred(tmp_path):
    target_dir = tmp_path / "target_perframe"
    target_dir.mkdir()
    np.save(target_dir / "P02_01.npy", np.zeros((3, 3807), dtype=np.float64))
    unsupported = tmp_path / "rgb_features.bin"
    unsupported.write_bytes(b"not an array container")

    report = inspect_inputs(
        target_dir=target_dir,
        rgb_feature_path=unsupported,
        flow_feature_path=None,
        max_files=1,
    )
    text = format_report(report)

    assert report["rgb_features"]["status"] == "UNSUPPORTED"
    assert report["overlaps"]["target_vs_rgb"]["status"] == "UNKNOWN"
    assert "IDs could not be inferred safely" in text
