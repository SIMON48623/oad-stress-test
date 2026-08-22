import numpy as np

from oad_stress_test.datasets.schema import VideoSequence
from oad_stress_test.utils.smoke_selection import select_class_overlap_smoke


class ToyDataset:
    def __init__(self, videos):
        self._videos = {video.video_id: video for video in videos}
        self.video_ids = list(self._videos)

    def load_video(self, video_id):
        return self._videos[video_id]


def _video(video_id, labels):
    labels = np.asarray(labels, dtype=np.int64)
    features = np.zeros((len(labels), 2), dtype=np.float32)
    return VideoSequence(video_id, features, labels)


def test_class_overlap_smoke_selector_writes_audit_and_selects_overlap(tmp_path):
    train = ToyDataset([
        _video("tr1", [0] * 5 + [1] * 8),
        _video("tr2", [0] * 5 + [2] * 8),
        _video("tr3", [0] * 5 + [3] * 8),
        _video("tr4", [0] * 5 + [4] * 8),
    ])
    test = ToyDataset([
        _video("te1", [0] * 4 + [1] * 5),
        _video("te2", [0] * 4 + [2] * 5),
        _video("te3", [0] * 4 + [3] * 5),
        _video("te4", [0] * 4 + [9] * 5),
    ])

    audit = select_class_overlap_smoke(
        train,
        test,
        max_train_videos=3,
        max_test_videos=3,
        background_label=0,
        seed=0,
        target_class_count=3,
        min_action_frames_per_class=1,
        min_overlap_classes=3,
        result_dir=tmp_path,
    )

    assert len(train.video_ids) == 3
    assert len(test.video_ids) == 3
    assert audit["selected_intersection"] == [1, 2, 3]
    assert audit["seen_class_map_should_be_meaningful"] is True
    assert (tmp_path / "label_distribution_audit.md").exists()
    assert (tmp_path / "result_note.md").exists()
    assert "frame_mAP_seen_classes should be meaningful: True" in (tmp_path / "label_distribution_audit.md").read_text(encoding="utf-8")
