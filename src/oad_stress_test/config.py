from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    raw: Dict[str, Any]
    path: Path

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw.get("dataset", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw.get("output", {})


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ProjectConfig(raw=raw, path=path)
