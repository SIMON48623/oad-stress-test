from __future__ import annotations

import argparse

from oad_stress_test.config import load_config
from oad_stress_test.datasets.dummy import generate_dummy_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dummy.yaml")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=4)
    parser.add_argument("--length", type=int, default=140)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg.dataset["root"]
    num_classes = int(cfg.dataset.get("num_classes", 4))
    feature_dir, split_dir = generate_dummy_dataset(
        root=root,
        num_train=args.num_train,
        num_test=args.num_test,
        length=args.length,
        dim=args.dim,
        num_classes=num_classes,
        seed=args.seed,
    )
    print(f"Dummy data written to {feature_dir} and {split_dir}")


if __name__ == "__main__":
    main()
