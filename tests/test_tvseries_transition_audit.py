import numpy as np

from oad_stress_test.datasets.feature_dataset import FeatureDataset
from tools.audit_tvseries_transitions import (
    audit_dataset,
    distance_to_nearest_transition,
    format_audit_markdown,
    required_tvseries_schema_note,
    transition_indices,
    transition_type_counts,
)


def test_transition_indices_and_types_ignore_background_stasis():
    labels = np.array([0, 0, 2, 2, 3, 0, 0, 1], dtype=np.int64)

    assert transition_indices(labels).tolist() == [2, 4, 5, 7]
    assert transition_type_counts(labels, background_label=0) == {
        "background_to_action": 2,
        "action_to_background": 1,
        "action_to_action": 1,
    }


def test_distance_to_nearest_transition_for_toy_sequence():
    labels = np.array([0, 0, 1, 1, 0], dtype=np.int64)

    distances = distance_to_nearest_transition(labels)

    np.testing.assert_allclose(distances, np.array([2, 1, 0, 1, 0], dtype=np.float32))


def test_tvseries_audit_reports_transition_distribution_and_seconds(tmp_path):
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    feature_dir.mkdir()
    split_dir.mkdir()
    features = np.arange(24, dtype=np.float32).reshape(8, 3)
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 0], dtype=np.int64)
    timestamps = np.arange(8, dtype=np.float32) * 0.5
    np.savez_compressed(feature_dir / "episode_001.npz", features=features, labels=labels, timestamps=timestamps)
    split_file = split_dir / "test.txt"
    split_file.write_text("episode_001\n", encoding="utf-8")
    dataset = FeatureDataset(feature_dir, split_file, background_label=0)

    audit = audit_dataset(dataset)
    markdown = format_audit_markdown(audit)

    assert audit["num_videos"] == 1
    assert audit["feature_dim_values"] == [3]
    assert audit["feature_step_seconds_estimate"] == 0.5
    assert audit["transition_types"] == {
        "background_to_action": 1,
        "action_to_background": 1,
        "action_to_action": 1,
    }
    assert audit["total_transitions"] == 3
    assert audit["action_ratio"] == 0.5
    assert "Candidate Near/Far Windows" in markdown
    assert audit["candidate_windows"]["near_second_percentiles"]


def test_missing_tvseries_requirements_note_lists_expected_schema():
    note = required_tvseries_schema_note()

    assert "data/tvseries/" in note
    assert "features/" in note
    assert "annotations/" in note
    assert "train.txt" in note
    assert "feature stride" in note
