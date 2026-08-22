from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from oad_stress_test.config import load_config

try:
    from inspect_dataset import inspect_config
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.check_thumos_ready.
    from scripts.inspect_dataset import inspect_config


def required_paths(config_path: str | Path) -> list[tuple[str, Path]]:
    cfg = load_config(config_path)
    d = cfg.dataset
    root = Path(d.get("root", "data/thumos14"))
    feature_dir = Path(d.get("feature_dir", root / "features"))
    annotation_dir = Path(d.get("annotation_dir", root / "annotations"))
    annotation_file = Path(d.get("annotation_file", annotation_dir / "thumos14.csv"))
    split_dir = Path(d.get("split_dir", root / "splits"))
    train_split = Path(d["train_split_file"]) if d.get("train_split_file") else split_dir / d.get("train_split", "train.txt")
    test_split = Path(d["split_file"]) if d.get("split_file") else split_dir / d.get("test_split", "test.txt")
    return [
        ("feature directory", feature_dir),
        ("annotation file", annotation_file),
        ("train split file", train_split),
        ("test split file", test_split),
    ]


def missing_required_paths(config_path: str | Path) -> list[tuple[str, Path]]:
    return [(label, path) for label, path in required_paths(config_path) if not path.exists()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether THUMOS14 feature-level inputs are ready.")
    parser.add_argument("--config", default="configs/thumos14.yaml")
    args = parser.parse_args(argv)

    missing = missing_required_paths(args.config)
    if missing:
        print("THUMOS14 data is not ready.")
        for label, path in missing:
            print(f"Missing {label}: {path}")
        return 1

    print("THUMOS14 data is ready.")
    inspect_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
