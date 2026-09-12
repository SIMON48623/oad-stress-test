from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile

from oad_stress_test.config import ProjectConfig
from oad_stress_test.datasets.feature_dataset import FeatureDataset
from oad_stress_test.models.causal_gru import CausalGRUClassifier
from oad_stress_test.models.causal_tcn import CausalTCNClassifier
from oad_stress_test.models.linear_probe import LinearProbeClassifier
from oad_stress_test.models.prototype import PrototypeClassifier


CLASSIFIER_CACHE_VERSION = 2


def _split_path(raw_path, split_dir: Path, fallback_name: str) -> Path:
    if raw_path is None:
        return split_dir / fallback_name
    return Path(raw_path)


def make_datasets(
    cfg: ProjectConfig,
    allow_test_fit: bool = False,
    eval_feature_dir: str | Path | None = None,
):
    d = cfg.dataset
    feature_dir = Path(d["feature_dir"])
    test_feature_dir = Path(eval_feature_dir) if eval_feature_dir is not None else feature_dir
    split_dir = Path(d.get("split_dir", ""))
    dataset_kwargs = {
        "annotation_dir": d.get("annotation_dir"),
        "annotation_file": d.get("annotation_file"),
        "background_label": int(d.get("background_label", 0)),
        "feature_fps": d.get("feature_fps"),
        "label_map": d.get("label_map"),
        "end_idx_inclusive": bool(d.get("end_idx_inclusive", True)),
    }
    train_path = _split_path(d.get("train_split_file"), split_dir, d.get("train_split", "train.txt"))
    test_path = _split_path(d.get("split_file"), split_dir, d.get("test_split", "test.txt"))
    test = FeatureDataset(test_feature_dir, test_path, **dataset_kwargs)
    if train_path.exists():
        train = FeatureDataset(feature_dir, train_path, **dataset_kwargs)
        return train, test
    if allow_test_fit:
        return test, test
    raise FileNotFoundError(
        (
            f"Train split file not found: {train_path}. Generate train.txt and set "
            "dataset.train_split_file, or pass --allow-test-fit for smoke tests only."
        )
    )


def _path_text(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_ids_sha256(video_ids: list[str]) -> str:
    payload = "\n".join(str(video_id) for video_id in video_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _num_classes_from_config_or_dataset(cfg: ProjectConfig, train_dataset: FeatureDataset) -> int:
    num_classes = cfg.dataset.get("num_classes")
    if num_classes is not None:
        return int(num_classes)
    label_map = getattr(train_dataset, "label_map", None)
    if label_map:
        values = [int(value) for value in label_map.values()]
        values.append(int(cfg.dataset.get("background_label", 0)))
        return max(values) + 1
    return int(train_dataset.infer_num_classes())


def classifier_cache_key(
    cfg: ProjectConfig,
    train_dataset: FeatureDataset,
    classifier: str,
    max_train_samples: int | None = None,
    class_balanced_sampling: bool = False,
    linear_class_weight: str | None = None,
    linear_mode: str = "multinomial",
    score_mode: str = "raw_margin",
    background_ratio: float = 1.0,
    min_samples_per_class: int = 200,
    seed: int = 13,
    linear_max_iter: int = 1000,
    linear_tol: float = 1e-3,
    linear_alpha: float = 1e-4,
    gru_hidden_dim: int = 128,
    gru_num_layers: int = 1,
    gru_dropout: float = 0.0,
    gru_max_epochs: int = 1,
    gru_lr: float = 1e-3,
    gru_weight_decay: float = 0.0,
    gru_chunk_length: int = 512,
    gru_device: str = "auto",
    tcn_hidden_dim: int = 128,
    tcn_num_layers: int = 4,
    tcn_kernel_size: int = 3,
    tcn_dilations: tuple[int, ...] | list[int] | None = None,
    tcn_dropout: float = 0.0,
    tcn_max_epochs: int = 20,
    tcn_lr: float = 1e-3,
    tcn_weight_decay: float = 0.0,
    tcn_chunk_length: int = 512,
    tcn_device: str = "auto",
) -> dict[str, object]:
    num_classes = _num_classes_from_config_or_dataset(cfg, train_dataset)
    split_file = Path(train_dataset.split_file)
    annotation_path = getattr(train_dataset, "annotation_path", None)
    return {
        "cache_version": CLASSIFIER_CACHE_VERSION,
        "classifier": str(classifier),
        "linear_mode": str(linear_mode) if classifier == "linear_probe" else None,
        "score_mode": str(score_mode) if classifier == "linear_probe" else None,
        "train_split_file": _path_text(split_file),
        "train_split_sha256": _file_sha256(split_file),
        "train_num_videos": int(len(getattr(train_dataset, "video_ids", []))),
        "train_video_ids_sha256": _video_ids_sha256(list(getattr(train_dataset, "video_ids", []))),
        "feature_dir": _path_text(Path(train_dataset.feature_dir)),
        "annotation_path": None if annotation_path is None else _path_text(Path(annotation_path)),
        "num_classes": int(num_classes),
        "background_label": int(cfg.dataset.get("background_label", 0)),
        "max_train_samples": None if max_train_samples is None else int(max_train_samples),
        "class_balanced_sampling": bool(class_balanced_sampling),
        "background_ratio": float(background_ratio),
        "min_samples_per_class": int(min_samples_per_class),
        "linear_class_weight": None if linear_class_weight in (None, "none") else str(linear_class_weight),
        "linear_max_iter": int(linear_max_iter),
        "linear_tol": float(linear_tol),
        "linear_alpha": float(linear_alpha),
        "gru_hidden_dim": int(gru_hidden_dim) if classifier == "causal_gru" else None,
        "gru_num_layers": int(gru_num_layers) if classifier == "causal_gru" else None,
        "gru_dropout": float(gru_dropout) if classifier == "causal_gru" else None,
        "gru_max_epochs": int(gru_max_epochs) if classifier == "causal_gru" else None,
        "gru_lr": float(gru_lr) if classifier == "causal_gru" else None,
        "gru_weight_decay": float(gru_weight_decay) if classifier == "causal_gru" else None,
        "gru_chunk_length": int(gru_chunk_length) if classifier == "causal_gru" else None,
        "gru_device": str(gru_device) if classifier == "causal_gru" else None,
        "tcn_hidden_dim": int(tcn_hidden_dim) if classifier == "causal_tcn" else None,
        "tcn_num_layers": int(tcn_num_layers) if classifier == "causal_tcn" else None,
        "tcn_kernel_size": int(tcn_kernel_size) if classifier == "causal_tcn" else None,
        "tcn_dilations": None if classifier != "causal_tcn" or tcn_dilations is None else [int(value) for value in tcn_dilations],
        "tcn_dropout": float(tcn_dropout) if classifier == "causal_tcn" else None,
        "tcn_max_epochs": int(tcn_max_epochs) if classifier == "causal_tcn" else None,
        "tcn_lr": float(tcn_lr) if classifier == "causal_tcn" else None,
        "tcn_weight_decay": float(tcn_weight_decay) if classifier == "causal_tcn" else None,
        "tcn_chunk_length": int(tcn_chunk_length) if classifier == "causal_tcn" else None,
        "tcn_device": str(tcn_device) if classifier == "causal_tcn" else None,
        "seed": int(seed),
    }


def classifier_cache_path(key: dict[str, object], cache_dir: str | Path = ".cache/classifiers") -> Path:
    payload = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return Path(cache_dir) / f"{digest}.pkl"


def _load_cached_classifier(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_cached_classifier(path: Path, classifier) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=path.name, suffix=".tmp") as handle:
        pickle.dump(classifier, handle)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def make_classifier(
    cfg: ProjectConfig,
    train_dataset: FeatureDataset,
    classifier: str = "prototype",
    max_train_samples: int | None = None,
    class_balanced_sampling: bool = False,
    linear_class_weight: str | None = None,
    linear_mode: str = "multinomial",
    score_mode: str = "raw_margin",
    background_ratio: float = 1.0,
    min_samples_per_class: int = 200,
    seed: int = 13,
    linear_max_iter: int = 1000,
    linear_tol: float = 1e-3,
    linear_alpha: float = 1e-4,
    gru_hidden_dim: int = 128,
    gru_num_layers: int = 1,
    gru_dropout: float = 0.0,
    gru_max_epochs: int = 1,
    gru_lr: float = 1e-3,
    gru_weight_decay: float = 0.0,
    gru_chunk_length: int = 512,
    gru_device: str = "auto",
    tcn_hidden_dim: int = 128,
    tcn_num_layers: int = 4,
    tcn_kernel_size: int = 3,
    tcn_dilations: tuple[int, ...] | list[int] | None = None,
    tcn_dropout: float = 0.0,
    tcn_max_epochs: int = 20,
    tcn_lr: float = 1e-3,
    tcn_weight_decay: float = 0.0,
    tcn_chunk_length: int = 512,
    tcn_device: str = "auto",
    use_cache: bool = True,
    cache_dir: str | Path = ".cache/classifiers",
):
    num_classes = _num_classes_from_config_or_dataset(cfg, train_dataset)
    key = classifier_cache_key(
        cfg,
        train_dataset,
        classifier=classifier,
        max_train_samples=max_train_samples,
        class_balanced_sampling=class_balanced_sampling,
        linear_class_weight=linear_class_weight,
        linear_mode=linear_mode,
        score_mode=score_mode,
        background_ratio=background_ratio,
        min_samples_per_class=min_samples_per_class,
        seed=seed,
        linear_max_iter=linear_max_iter,
        linear_tol=linear_tol,
        linear_alpha=linear_alpha,
        gru_hidden_dim=gru_hidden_dim,
        gru_num_layers=gru_num_layers,
        gru_dropout=gru_dropout,
        gru_max_epochs=gru_max_epochs,
        gru_lr=gru_lr,
        gru_weight_decay=gru_weight_decay,
        gru_chunk_length=gru_chunk_length,
        gru_device=gru_device,
        tcn_hidden_dim=tcn_hidden_dim,
        tcn_num_layers=tcn_num_layers,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dilations=tcn_dilations,
        tcn_dropout=tcn_dropout,
        tcn_max_epochs=tcn_max_epochs,
        tcn_lr=tcn_lr,
        tcn_weight_decay=tcn_weight_decay,
        tcn_chunk_length=tcn_chunk_length,
        tcn_device=tcn_device,
    )
    cache_path = classifier_cache_path(key, cache_dir)
    if use_cache and cache_path.exists():
        clf = _load_cached_classifier(cache_path)
        setattr(clf, "cache_hit_", True)
        setattr(clf, "cache_path_", str(cache_path))
        setattr(clf, "cache_key_", key)
        return clf

    if classifier == "prototype":
        clf = PrototypeClassifier(num_classes=int(num_classes))
    elif classifier == "linear_probe":
        clf = LinearProbeClassifier(
            num_classes=int(num_classes),
            max_train_samples=max_train_samples,
            class_balanced_sampling=class_balanced_sampling,
            linear_class_weight=None if linear_class_weight in (None, "none") else str(linear_class_weight),
            linear_mode=str(linear_mode),
            score_mode=str(score_mode),
            background_ratio=float(background_ratio),
            min_samples_per_class=int(min_samples_per_class),
            background_label=int(cfg.dataset.get("background_label", 0)),
            seed=int(seed),
            max_iter=int(linear_max_iter),
            tol=float(linear_tol),
            alpha=float(linear_alpha),
        )
    elif classifier == "causal_gru":
        clf = CausalGRUClassifier(
            num_classes=int(num_classes),
            hidden_dim=int(gru_hidden_dim),
            num_layers=int(gru_num_layers),
            dropout=float(gru_dropout),
            max_epochs=int(gru_max_epochs),
            lr=float(gru_lr),
            weight_decay=float(gru_weight_decay),
            chunk_length=int(gru_chunk_length),
            device=str(gru_device),
            seed=int(seed),
            background_label=int(cfg.dataset.get("background_label", 0)),
            max_train_samples=max_train_samples,
        )
    elif classifier == "causal_tcn":
        clf = CausalTCNClassifier(
            num_classes=int(num_classes),
            hidden_dim=int(tcn_hidden_dim),
            num_layers=int(tcn_num_layers),
            kernel_size=int(tcn_kernel_size),
            dilations=None if tcn_dilations is None else tuple(int(value) for value in tcn_dilations),
            dropout=float(tcn_dropout),
            max_epochs=int(tcn_max_epochs),
            lr=float(tcn_lr),
            weight_decay=float(tcn_weight_decay),
            chunk_length=int(tcn_chunk_length),
            device=str(tcn_device),
            seed=int(seed),
            background_label=int(cfg.dataset.get("background_label", 0)),
            max_train_samples=max_train_samples,
        )
    else:
        raise ValueError(f"Unknown classifier backend: {classifier}")
    clf.fit(train_dataset)
    setattr(clf, "cache_hit_", False)
    setattr(clf, "cache_path_", str(cache_path))
    setattr(clf, "cache_key_", key)
    if use_cache:
        _save_cached_classifier(cache_path, clf)
    return clf
