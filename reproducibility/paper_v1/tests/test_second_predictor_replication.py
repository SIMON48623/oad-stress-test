from __future__ import annotations

import numpy as np

from second_predictor_replication import vectorized_linear_scores


class IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


class MockEstimator:
    classes_ = np.asarray([0, 1, 2])

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[:, :3]


class MockProbe:
    scaler = IdentityScaler()
    model = MockEstimator()
    score_mode = "softmax"
    num_classes = 3


def test_vectorized_linear_scores_are_normalized_and_aligned() -> None:
    features = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    scores = vectorized_linear_scores(MockProbe(), features)
    assert scores.shape == (2, 3)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-7)
    assert int(np.argmax(scores[0])) == 2
    assert int(np.argmax(scores[1])) == 0

