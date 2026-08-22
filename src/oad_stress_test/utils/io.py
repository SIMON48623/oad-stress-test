from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def read_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, orient="records", lines=True)
