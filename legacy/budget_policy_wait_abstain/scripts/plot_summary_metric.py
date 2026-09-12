import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.summary)

    required = {"policy", "budget", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    fig, ax = plt.subplots()

    for policy, group in df.groupby("policy"):
        group = group.sort_values("budget")
        ax.plot(group["budget"], group[args.metric], marker="o", label=policy)

    ax.set_xlabel("Budget")
    ax.set_ylabel(args.metric)
    ax.set_title(args.metric)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
