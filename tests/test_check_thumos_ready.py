import numpy as np

from scripts.check_thumos_ready import main, missing_required_paths


def _write_config(path, root):
    path.write_text(
        "dataset:\n"
        "  name: thumos14\n"
        f"  root: {root.as_posix()}\n"
        f"  feature_dir: {(root / 'features').as_posix()}\n"
        f"  annotation_dir: {(root / 'annotations').as_posix()}\n"
        f"  split_dir: {(root / 'splits').as_posix()}\n"
        f"  train_split_file: {(root / 'splits' / 'train.txt').as_posix()}\n"
        f"  split_file: {(root / 'splits' / 'test.txt').as_posix()}\n"
        "  train_split: train.txt\n"
        "  test_split: test.txt\n"
        "  background_label: 0\n"
        "  end_idx_inclusive: true\n"
        "evaluation:\n"
        "  budgets: [0.10]\n"
        "output:\n"
        "  results_dir: results/thumos14\n",
        encoding="utf-8",
    )


def test_check_thumos_ready_reports_missing_paths(tmp_path, capsys):
    root = tmp_path / "thumos14"
    config_path = tmp_path / "thumos14.yaml"
    _write_config(config_path, root)

    code = main(["--config", str(config_path)])
    output = capsys.readouterr().out
    missing = missing_required_paths(config_path)

    assert code == 1
    assert len(missing) == 4
    assert "THUMOS14 data is not ready." in output
    assert "Missing feature directory:" in output
    assert "Missing annotation file:" in output
    assert "Missing train split file:" in output
    assert "Missing test split file:" in output


def test_check_thumos_ready_calls_inspection_when_inputs_exist(tmp_path, capsys):
    root = tmp_path / "thumos14"
    feature_dir = root / "features"
    annotation_dir = root / "annotations"
    split_dir = root / "splits"
    feature_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    split_dir.mkdir()

    config_path = tmp_path / "thumos14.yaml"
    _write_config(config_path, root)
    np.save(feature_dir / "video_train.npy", np.ones((4, 3), dtype=np.float32))
    np.save(feature_dir / "video_001.npy", np.zeros((5, 3), dtype=np.float32))
    (annotation_dir / "thumos14.csv").write_text(
        "video_id,start_idx,end_idx,label\n"
        "video_train,0,1,jump\n"
        "video_001,1,3,jump\n",
        encoding="utf-8",
    )
    (split_dir / "train.txt").write_text("video_train\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("video_001\n", encoding="utf-8")

    code = main(["--config", str(config_path)])
    output = capsys.readouterr().out

    assert code == 0
    assert "THUMOS14 data is ready." in output
    assert "Videos: 2" in output
    assert "Mean video length: 4.50 timesteps" in output
    assert "Num classes: 2" in output
    assert "Annotation out-of-bounds issues: 0" in output
