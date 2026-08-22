from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from oad_stress_test.datasets.schema import VideoSequence
from oad_stress_test.policies.base import StreamingPolicy


class CausalStreamingEvaluator:
    """Runs a policy sequentially over feature sequences.

    This evaluator is intentionally strict: the policy only receives the current
    feature x_t at step t. It never receives the whole feature matrix or labels.
    """

    @staticmethod
    def _current_feature(video: VideoSequence, t: int) -> np.ndarray:
        """Return an isolated current-timestep feature for policy input."""
        x_t = np.array(video.features[t], copy=True)
        if np.shares_memory(x_t, video.features):
            raise RuntimeError("causal evaluator must not pass a view of the full feature matrix")
        x_t.setflags(write=False)
        return x_t

    def evaluate_video(self, video: VideoSequence, policy: StreamingPolicy, budget: float) -> pd.DataFrame:
        policy.reset(video.video_id, video.length, budget)
        rows: List[Dict] = []
        for t in range(video.length):
            decision = policy.step(t=t, x_t=self._current_feature(video, t))
            row = {
                "video_id": video.video_id,
                "t": t,
                "gt_label": int(video.labels[t]),
                "observed": bool(decision.observed),
                "action_type": decision.action_type,
                "prediction": -1 if decision.prediction is None else int(decision.prediction),
                "confidence": float(decision.confidence),
                "budget": float(budget),
                "policy": policy.name,
            }
            if decision.scores is not None:
                scores = np.asarray(decision.scores, dtype=np.float32).reshape(-1)
                for class_id, score in enumerate(scores):
                    row[f"score_class_{class_id}"] = float(score)
            rows.append(row)
        return pd.DataFrame(rows)

    def evaluate_dataset(self, videos: Iterable[VideoSequence], policy: StreamingPolicy, budget: float) -> pd.DataFrame:
        frames = [self.evaluate_video(video, policy, budget) for video in videos]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
