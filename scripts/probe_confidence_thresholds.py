from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from oad_stress_test.utils.threshold_calibration import (
    DEFAULT_QUANTILES,
    score_signals,
    threshold_grid_from_signals,
)


def load_score_array(path: str | Path, score_key: str | None = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Score/logit file not found: {path}")
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if score_key is not None:
            if score_key not in data.files:
                raise KeyError(f"{path} does not contain score key {score_key!r}; available keys: {data.files}")
            return np.asarray(data[score_key])
        for key in ["scores", "logits", "probabilities", "probs", "arr_0"]:
            if key in data.files:
                return np.asarray(data[key])
        if not data.files:
            raise ValueError(f"{path} contains no arrays")
        raise KeyError(f"Use --score-key; available keys: {data.files}")
    raise ValueError(f"Unsupported file suffix {path.suffix}; expected .npy or .npz")


def format_threshold_report(grid: dict[str, list[dict[str, float]]]) -> str:
    lines = ["# Confidence Threshold Quantile Probe", ""]
    for signal_name, rows in grid.items():
        lines.append(f"## {signal_name}")
        lines.append("")
        for row in rows:
            lines.append(f"- q={row['quantile']:.3f}: {row['threshold']:.6f}")
        lines.append("")
    lines.append("These thresholds are empirical diagnostics only; they do not run model evaluation.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe empirical uncertainty thresholds from a small score/logit file.")
    parser.add_argument("--input", required=True, help="path to .npy/.npz scores, probabilities, or logits with shape [N, C]")
    parser.add_argument("--score-key", default=None, help="array key for .npz inputs")
    parser.add_argument("--input-kind", choices=["auto", "logits", "probabilities"], default="auto")
    parser.add_argument("--quantiles", nargs="+", type=float, default=list(DEFAULT_QUANTILES))
    args = parser.parse_args()

    scores = load_score_array(args.input, score_key=args.score_key)
    signals = score_signals(scores, input_kind=args.input_kind)
    grid = threshold_grid_from_signals(signals, quantiles=args.quantiles)
    print(format_threshold_report(grid))


if __name__ == "__main__":
    main()
