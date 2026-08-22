from __future__ import annotations

import argparse

from oad_stress_test.config import load_config
from oad_stress_test.utils.factory import make_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate feature dataset layout.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train, test = make_datasets(cfg)
    print(f"Train videos: {len(train)}")
    print(f"Test videos: {len(test)}")
    first = next(iter(test))
    print(f"First test video: {first.video_id}, features={first.features.shape}, labels={first.labels.shape}")


if __name__ == "__main__":
    main()
