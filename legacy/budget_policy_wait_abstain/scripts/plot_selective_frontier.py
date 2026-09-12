from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from oad_stress_test.plots.curves import plot_selective_frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=None)
    parser.add_argument("--results-dir", default="results/dummy")
    parser.add_argument("--out", default="figures/selective_frontier.png")
    parser.add_argument("--x-metric", default="mean_decision_latency_steps")
    parser.add_argument("--y-metric", default="selective_risk")
    args = parser.parse_args()
    summary_path = Path(args.summary) if args.summary else Path(args.results_dir) / "summary.csv"
    summary = pd.read_csv(summary_path)
    plot_selective_frontier(summary, args.out, x_metric=args.x_metric, y_metric=args.y_metric)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
