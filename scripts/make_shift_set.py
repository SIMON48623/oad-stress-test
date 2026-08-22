from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def apply_feature_shift(features: np.ndarray, kind: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Feature-level proxy shifts.

    These are not image-space corruptions. They are quick stress-test proxies
    for the first codebase version. Image-space corruptions should be added only
    if raw frames are available.
    """
    x = features.copy()
    if kind == "noise":
        return x + rng.normal(0, severity, size=x.shape).astype(np.float32)
    if kind == "dropout":
        mask = rng.random(x.shape) > severity
        return x * mask.astype(np.float32)
    if kind == "scale":
        return x * float(1.0 - severity)
    if kind == "temporal_subsample":
        step = max(1, int(round(1.0 / max(1e-6, 1.0 - severity))))
        y = x.copy()
        for t in range(len(y)):
            y[t] = x[(t // step) * step]
        return y
    raise ValueError(f"Unknown shift kind: {kind}")


def shift_feature_file(path: Path, dst: Path, kind: str, severity: float, rng: np.random.Generator) -> Path:
    """Shift one .npy or .npz feature file without changing its filename."""
    out_path = dst / path.name
    suffix = path.suffix.lower()
    if suffix == ".npy":
        features = np.asarray(np.load(path), dtype=np.float32)
        shifted = apply_feature_shift(features, kind, severity, rng)
        np.save(out_path, shifted)
        return out_path

    if suffix == ".npz":
        data = np.load(path)
        if "features" not in data.files:
            raise KeyError(f"{path} must contain an array named 'features'")
        payload = {name: data[name] for name in data.files}
        payload["features"] = apply_feature_shift(
            np.asarray(payload["features"], dtype=np.float32),
            kind,
            severity,
            rng,
        )
        np.savez_compressed(out_path, **payload)
        return out_path

    raise ValueError(f"Unsupported feature file suffix: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create feature-level synthetic shift set from .npy or .npz files. "
            "Severity is shift-specific: noise uses Gaussian std, dropout uses drop probability, "
            "scale multiplies by (1 - severity), and temporal_subsample derives a repeat step from severity."
        )
    )
    parser.add_argument("--src", required=True, help="source feature directory")
    parser.add_argument("--dst", required=True, help="destination feature directory")
    parser.add_argument(
        "--kind",
        choices=["noise", "dropout", "scale", "temporal_subsample"],
        default="noise",
        help="feature-level shift kind; no image-space corruptions are generated",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.5,
        help=(
            "shift strength: noise std; dropout probability; scale reduction where output=x*(1-severity); "
            "temporal_subsample repeat severity"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paths = sorted([*src.glob("*.npy"), *src.glob("*.npz")])
    for path in paths:
        shift_feature_file(path, dst, args.kind, args.severity, rng)
    print(f"Wrote {len(paths)} shifted feature files to {dst}")


if __name__ == "__main__":
    main()
