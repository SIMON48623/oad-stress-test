from __future__ import annotations

import argparse

import pandas as pd

from oad_stress_test.plots.curves import plot_threshold_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/dummy/summary.csv")
    parser.add_argument("--out", default="figures/threshold_curve.png")
    parser.add_argument("--metric", default="abstain_rate")
    parser.add_argument("--policy", default="confidence_threshold")
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    plot_threshold_curve(summary, args.out, metric=args.metric, policy=args.policy)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
