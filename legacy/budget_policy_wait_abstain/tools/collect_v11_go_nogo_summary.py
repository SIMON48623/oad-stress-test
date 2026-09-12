from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODES = ["max_prob", "entropy", "top2_margin"]


def collect_v11_go_nogo_summary(input_root: str | Path, output: str | Path) -> pd.DataFrame:
    input_root = Path(input_root)
    frames = []
    missing = []
    for mode in MODES:
        path = input_root / f"analysis_{mode}" / "atm_summary.csv"
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "source_mode", mode)
        frames.append(frame)
    if missing:
        missing_text = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing required v1.1 ATM summaries:\n{missing_text}")
    summary = pd.concat(frames, ignore_index=True)
    sort_columns = [column for column in ["source_mode", "threshold", "budget"] if column in summary.columns]
    if sort_columns:
        summary = summary.sort_values(sort_columns).reset_index(drop=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect v1.1 GRU clean go/no-go ATM summaries.")
    parser.add_argument(
        "--input-root",
        default="results/v1.1_gru_clean_go_nogo/full",
        help="root containing analysis_max_prob/analysis_entropy/analysis_top2_margin",
    )
    parser.add_argument(
        "--output",
        default="results/v1.1_gru_clean_go_nogo/gru_clean_go_nogo_summary.csv",
        help="merged output CSV path",
    )
    args = parser.parse_args(argv)
    summary = collect_v11_go_nogo_summary(args.input_root, args.output)
    print(f"Wrote {args.output}")
    print(f"rows: {len(summary)}")


if __name__ == "__main__":
    main()
