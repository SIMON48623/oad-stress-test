from __future__ import annotations

from typing import Dict

import pandas as pd

from .basic import summarize_basic
from .failure import summarize_failure_modes
from .frame_map import summarize_frame_map
from .selective import summarize_selective
from .transition import summarize_transition_delay


def summarize_log(
    logs: pd.DataFrame,
    stable_steps: int = 3,
    background_label: int = 0,
    train_positive_counts: object = None,
) -> Dict[str, float | str]:
    out: Dict[str, float | str] = {}
    out.update(summarize_basic(logs, background_label=background_label))
    out.update(summarize_frame_map(
        logs,
        background_label=background_label,
        train_positive_counts=train_positive_counts,
    ))
    out.update(summarize_selective(logs))
    out.update(summarize_transition_delay(logs, stable_steps=stable_steps))
    out.update(summarize_failure_modes(logs))
    return out
