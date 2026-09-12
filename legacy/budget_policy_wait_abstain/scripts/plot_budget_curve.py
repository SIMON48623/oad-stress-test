from __future__ import annotations

import argparse

import pandas as pd

from oad_stress_test.plots.curves import plot_budget_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/dummy/summary.csv")
    parser.add_argument("--out", default="figures/budget_curve.png")
    parser.add_argument("--metric", default="frame_accuracy_on_predicted")
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    plot_budget_curve(summary, args.out, metric=args.metric)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
