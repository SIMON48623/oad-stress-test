from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VideoSequence:
    video_id: str
    features: np.ndarray  # [T, D]
    labels: np.ndarray    # [T]
    timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"features for {self.video_id} must have shape [T, D]")
        if self.labels.ndim != 1:
            raise ValueError(f"labels for {self.video_id} must have shape [T]")
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"features and labels length mismatch for {self.video_id}: "
                f"{len(self.features)} vs {len(self.labels)}"
            )
        if self.timestamps is not None and len(self.timestamps) != len(self.labels):
            raise ValueError(f"timestamps length mismatch for {self.video_id}")

    @property
    def length(self) -> int:
        return int(len(self.labels))
