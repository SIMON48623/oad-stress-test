from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Tuple
import random

import numpy as np

from oad_stress_test.datasets.schema import VideoSequence


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only on environments without torch
    torch = None
    nn = None
    F = None


if nn is not None:
    class _CausalConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
            super().__init__()
            self.left_padding = int(dilation) * (int(kernel_size) - 1)
            self.conv = nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                dilation=int(dilation),
                padding=0,
            )

        def forward(self, x):
            if self.left_padding > 0:
                x = F.pad(x, (self.left_padding, 0))
            return self.conv(x)


    class _CausalTCNBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.dropout = nn.Dropout(float(dropout))
            self.proj = nn.Identity() if int(in_channels) == int(out_channels) else nn.Conv1d(in_channels, out_channels, 1)

        def forward(self, x):
            residual = self.proj(x)
            y = self.conv1(x)
            y = F.relu(y)
            y = self.dropout(y)
            y = self.conv2(y)
            y = F.relu(y)
            y = self.dropout(y)
            return F.relu(y + residual)


    class _CausalTCNNet(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int,
            dilations: tuple[int, ...],
            dropout: float,
            num_classes: int,
        ):
            super().__init__()
            blocks = []
            in_channels = int(input_dim)
            for dilation in dilations:
                blocks.append(_CausalTCNBlock(in_channels, int(hidden_dim), int(kernel_size), int(dilation), float(dropout)))
                in_channels = int(hidden_dim)
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Conv1d(int(hidden_dim), int(num_classes), 1)

        def forward(self, x):
            # Input is [B, T, D]; Conv1d expects [B, D, T].
            y = x.transpose(1, 2)
            y = self.blocks(y)
            logits = self.head(y).transpose(1, 2)
            return logits
else:  # pragma: no cover
    class _CausalTCNNet:
        pass


@dataclass
class CausalTCNClassifier:
    """Small causal TCN classifier for diagnostic reference runs.

    The network uses left-padded 1D convolutions only, so logits at timestep t
    depend on x_1..x_t and never on future features. It is a reference temporal
    backend for protocol stress-testing, not an OAD-native SOTA model.
    """

    num_classes: int
    hidden_dim: int = 128
    num_layers: int = 4
    kernel_size: int = 3
    dilations: tuple[int, ...] | None = None
    dropout: float = 0.0
    max_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    chunk_length: int = 512
    device: str = "auto"
    seed: int = 13
    background_label: int = 0
    use_class_weights: bool = True
    max_train_samples: int | None = None
    model: object | None = None
    feature_mean_: np.ndarray | None = None
    feature_std_: np.ndarray | None = None
    sample_counts_: np.ndarray | None = None
    sampling_seen_counts_: np.ndarray | None = None
    input_dim_: int | None = None
    device_used_: str | None = None
    train_seconds_: float = 0.0
    loss_history_: tuple[float, ...] = ()
    train_loss_by_epoch_: tuple[float, ...] = ()
    convergence_warning_: bool = False

    def fit(self, videos: Iterable[VideoSequence]) -> "CausalTCNClassifier":
        if torch is None:
            raise ImportError("causal_tcn requires PyTorch, but torch is not installed")
        train_videos = self._materialize_videos(videos)
        if not train_videos:
            raise ValueError("Cannot fit causal_tcn with empty dataset")
        self._set_seed()
        self.input_dim_ = int(train_videos[0].features.shape[1])
        self.device_used_ = self._resolve_device()
        device = torch.device(self.device_used_)
        self._fit_normalizer(train_videos)
        self.sampling_seen_counts_ = self._count_labels(train_videos)
        self.sample_counts_ = self.sampling_seen_counts_.copy()
        self.model = _CausalTCNNet(
            input_dim=self.input_dim_,
            hidden_dim=int(self.hidden_dim),
            kernel_size=int(self.kernel_size),
            dilations=self._dilations(),
            dropout=float(self.dropout),
            num_classes=int(self.num_classes),
        ).to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        class_weights = self._class_weights(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        rng = np.random.default_rng(self.seed)
        losses: list[float] = []
        epoch_losses: list[float] = []
        start = perf_counter()
        for _epoch in range(int(self.max_epochs)):
            losses_before = len(losses)
            order = np.arange(len(train_videos))
            rng.shuffle(order)
            self.model.train()
            for video_idx in order.tolist():
                video = train_videos[video_idx]
                for x_chunk, y_chunk in self._chunks(video):
                    x_tensor = torch.from_numpy(x_chunk).to(device=device, dtype=torch.float32).unsqueeze(0)
                    y_tensor = torch.from_numpy(y_chunk).to(device=device, dtype=torch.long).reshape(-1)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(x_tensor).reshape(-1, int(self.num_classes))
                    loss = criterion(logits, y_tensor)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))
            epoch_values = losses[losses_before:]
            epoch_losses.append(float(np.mean(epoch_values)) if epoch_values else float("nan"))
        self.train_seconds_ = float(perf_counter() - start)
        self.loss_history_ = tuple(losses[-20:])
        self.train_loss_by_epoch_ = tuple(epoch_losses)
        self.reset_sequence()
        return self

    def _dilations(self) -> tuple[int, ...]:
        if self.dilations is not None:
            values = tuple(int(value) for value in self.dilations)
        else:
            values = tuple(2**idx for idx in range(int(self.num_layers)))
        if not values:
            raise ValueError("causal_tcn requires at least one dilation/layer")
        if any(value <= 0 for value in values):
            raise ValueError(f"causal_tcn dilations must be positive: {values}")
        return values

    def receptive_field(self) -> int:
        return 1 + 2 * sum((int(self.kernel_size) - 1) * int(dilation) for dilation in self._dilations())

    def _materialize_videos(self, videos: Iterable[VideoSequence]) -> list[VideoSequence]:
        out: list[VideoSequence] = []
        remaining = None if self.max_train_samples is None else int(self.max_train_samples)
        if remaining is not None and remaining <= 0:
            raise ValueError("max_train_samples must be positive")
        expected_dim: int | None = None
        for video in videos:
            features = np.asarray(video.features, dtype=np.float32)
            labels = np.asarray(video.labels, dtype=np.int64)
            if features.ndim != 2 or labels.ndim != 1:
                raise ValueError(f"Invalid video shapes for {video.video_id}")
            if features.shape[0] == 0:
                continue
            if expected_dim is None:
                expected_dim = int(features.shape[1])
            elif int(features.shape[1]) != expected_dim:
                raise ValueError(f"Feature dimension mismatch for {video.video_id}")
            labels = self._valid_labels(labels, video.video_id)
            if remaining is None:
                out.append(VideoSequence(video.video_id, features.copy(), labels.copy()))
                continue
            if remaining <= 0:
                break
            take = min(int(remaining), int(len(labels)))
            out.append(VideoSequence(video.video_id, features[:take].copy(), labels[:take].copy()))
            remaining -= take
        return out

    def _valid_labels(self, labels: np.ndarray, video_id: str) -> np.ndarray:
        if np.any(labels < 0) or np.any(labels >= int(self.num_classes)):
            bad = labels[(labels < 0) | (labels >= int(self.num_classes))]
            raise ValueError(f"Labels out of range for {video_id}: {bad[:5].tolist()}")
        return labels.astype(np.int64, copy=False)

    def _set_seed(self) -> None:
        random.seed(int(self.seed))
        np.random.seed(int(self.seed))
        torch.manual_seed(int(self.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.seed))
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    def _resolve_device(self) -> str:
        requested = str(self.device)
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("causal_tcn requested cuda, but CUDA is not available")
        return requested

    def _fit_normalizer(self, videos: list[VideoSequence]) -> None:
        x = np.concatenate([np.asarray(video.features, dtype=np.float32) for video in videos], axis=0)
        self.feature_mean_ = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.feature_std_ = std

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("causal_tcn normalizer is not fitted")
        return ((np.asarray(features, dtype=np.float32) - self.feature_mean_) / self.feature_std_).astype(np.float32)

    def _count_labels(self, videos: list[VideoSequence]) -> np.ndarray:
        counts = np.zeros(int(self.num_classes), dtype=np.int64)
        for video in videos:
            counts += np.bincount(np.asarray(video.labels, dtype=np.int64), minlength=int(self.num_classes))[: int(self.num_classes)]
        return counts

    def _class_weights(self, device):
        if not self.use_class_weights or self.sampling_seen_counts_ is None:
            return None
        counts = np.asarray(self.sampling_seen_counts_, dtype=np.float32)
        weights = np.zeros(int(self.num_classes), dtype=np.float32)
        present = counts > 0
        if not np.any(present):
            return None
        weights[present] = counts[present].sum() / (float(present.sum()) * counts[present])
        weights[present] = weights[present] / max(float(weights[present].mean()), 1e-12)
        return torch.from_numpy(weights).to(device=device, dtype=torch.float32)

    def _chunks(self, video: VideoSequence):
        features = self._normalize(video.features)
        labels = np.asarray(video.labels, dtype=np.int64)
        step = max(1, int(self.chunk_length))
        for start in range(0, len(labels), step):
            end = min(start + step, len(labels))
            yield features[start:end], labels[start:end]

    def reset_sequence(self, video_id: str | None = None) -> None:
        self._history_features: list[np.ndarray] = []
        self._active_video_id = video_id
        if self.model is not None:
            self.model.eval()

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("causal_tcn has not been fitted")

    def predict_logits_sequence(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        device = torch.device(self.device_used_ or "cpu")
        self.model.eval()
        x_np = np.asarray(features, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        x = torch.from_numpy(self._normalize(x_np)).to(device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
        return logits.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def predict_scores_sequence(self, features: np.ndarray) -> np.ndarray:
        logits = self.predict_logits_sequence(features)
        centered = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(centered, -50.0, 50.0))
        denom = exp_scores.sum(axis=1, keepdims=True)
        return (exp_scores / np.maximum(denom, 1e-12)).astype(np.float32)

    def predict_scores(self, feature: np.ndarray) -> np.ndarray:
        self._check_fitted()
        current = np.asarray(feature, dtype=np.float32).reshape(-1).copy()
        self._history_features.append(current)
        limit = max(1, int(self.receptive_field()))
        if len(self._history_features) > limit:
            self._history_features = self._history_features[-limit:]
        history = np.stack(self._history_features, axis=0)
        return self.predict_scores_sequence(history)[-1]

    def predict(self, feature: np.ndarray) -> Tuple[int, float, np.ndarray]:
        scores = self.predict_scores(feature)
        pred = int(np.argmax(scores))
        conf = float(np.max(scores))
        return pred, conf, scores

    def training_diagnostics(self) -> dict[str, object]:
        counts = (
            np.zeros(int(self.num_classes), dtype=np.int64)
            if self.sample_counts_ is None
            else np.asarray(self.sample_counts_, dtype=np.int64)
        )
        action_classes = [class_id for class_id in range(int(self.num_classes)) if class_id != int(self.background_label)]
        return {
            "classifier": "causal_tcn",
            "total_sampled_frames": int(counts.sum()),
            "background_label": int(self.background_label),
            "background_samples": int(counts[self.background_label]) if 0 <= self.background_label < len(counts) else 0,
            "action_samples": {str(class_id): int(counts[class_id]) for class_id in action_classes},
            "missing_action_classes": [int(class_id) for class_id in action_classes if counts[class_id] == 0],
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(len(self._dilations())),
            "kernel_size": int(self.kernel_size),
            "dilations": [int(value) for value in self._dilations()],
            "dropout": float(self.dropout),
            "max_epochs": int(self.max_epochs),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "chunk_length": int(self.chunk_length),
            "receptive_field": int(self.receptive_field()),
            "device_used": self.device_used_,
            "train_seconds": float(self.train_seconds_),
            "train_loss_by_epoch": [float(value) for value in self.train_loss_by_epoch_],
            "loss_tail": [float(value) for value in self.loss_history_],
            "score_class_background_excluded_from_frame_mAP": True,
        }
