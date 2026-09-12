from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from oad_stress_test.plots.curves import plot_transition_delay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=None)
    parser.add_argument("--results-dir", default="results/dummy")
    parser.add_argument("--out", default="figures/transition_delay.png")
    parser.add_argument("--metric", default="transition_delay_median")
    args = parser.parse_args()
    summary_path = Path(args.summary) if args.summary else Path(args.results_dir) / "summary.csv"
    summary = pd.read_csv(summary_path)
    plot_transition_delay(summary, args.out, metric=args.metric)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
