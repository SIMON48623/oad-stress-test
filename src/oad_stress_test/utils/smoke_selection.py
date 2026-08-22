from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping
import hashlib
import random


def _counts_for_dataset(dataset, background_label: int) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for video_id in dataset.video_ids:
        video = dataset.load_video(video_id)
        counts = Counter(int(label) for label in video.labels.tolist())
        out[str(video_id)] = counts
    return out


def _aggregate(video_counts: Mapping[str, Counter], video_ids: list[str], background_label: int) -> tuple[Counter, int]:
    action_counts: Counter = Counter()
    background_count = 0
    for video_id in video_ids:
        counts = video_counts[str(video_id)]
        background_count += int(counts.get(background_label, 0))
        for class_id, count in counts.items():
            if int(class_id) != int(background_label):
                action_counts[int(class_id)] += int(count)
    return action_counts, int(background_count)


def _stable_noise(seed: int, text: str) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 16**8


def _select_videos_for_classes(
    video_counts: Mapping[str, Counter],
    target_classes: list[int],
    max_videos: int,
    background_label: int,
    seed: int,
    min_frames_per_class: int = 1,
) -> list[str]:
    if max_videos <= 0:
        raise ValueError("max_videos must be positive")
    target = set(int(class_id) for class_id in target_classes)
    selected: list[str] = []
    selected_set: set[str] = set()
    selected_counts: Counter = Counter()
    candidates = [
        str(video_id) for video_id, counts in video_counts.items()
        if any(int(class_id) in target and int(count) > 0 for class_id, count in counts.items())
    ]
    if not candidates:
        return []

    while len(selected) < min(max_videos, len(candidates)):
        best_id = None
        best_score = None
        for video_id in candidates:
            if video_id in selected_set:
                continue
            counts = video_counts[video_id]
            video_classes = {int(class_id) for class_id, count in counts.items() if int(class_id) in target and int(count) > 0}
            if not video_classes:
                continue
            deficit_classes = {
                class_id for class_id in video_classes
                if selected_counts[class_id] < int(min_frames_per_class)
            }
            deficit_reduction = sum(
                min(int(counts.get(class_id, 0)), max(0, int(min_frames_per_class) - int(selected_counts[class_id])))
                for class_id in deficit_classes
            )
            target_frames = sum(int(counts.get(class_id, 0)) for class_id in target)
            background = int(counts.get(background_label, 0))
            coverage_bonus = 10000.0 * len(deficit_classes)
            deficit_bonus = 10.0 * float(deficit_reduction)
            diversity_bonus = 100.0 * len(video_classes)
            frame_bonus = min(float(target_frames), 2000.0) / 10.0
            background_penalty = min(float(background), 5000.0) / 5000.0
            score = coverage_bonus + deficit_bonus + diversity_bonus + frame_bonus - background_penalty + _stable_noise(seed, video_id)
            if best_score is None or score > best_score:
                best_score = score
                best_id = video_id
        if best_id is None:
            break
        selected.append(best_id)
        selected_set.add(best_id)
        for class_id, count in video_counts[best_id].items():
            class_id = int(class_id)
            if class_id in target:
                selected_counts[class_id] += int(count)
    return selected


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    return "{" + ", ".join(f"{int(k)}: {int(counter[k])}" for k in sorted(counter)) + "}"


def _format_ids(ids: list[str]) -> str:
    return "\n".join(f"- {video_id}" for video_id in ids) if ids else "- none"


def _format_overlap_counts(train_counts: Counter, test_counts: Counter, classes: list[int]) -> str:
    if not classes:
        return "| class | train_action_frames | test_action_frames |\n| --- | ---: | ---: |\n"
    rows = ["| class | train_action_frames | test_action_frames |", "| --- | ---: | ---: |"]
    for class_id in classes:
        rows.append(f"| {int(class_id)} | {int(train_counts.get(class_id, 0))} | {int(test_counts.get(class_id, 0))} |")
    return "\n".join(rows)


def _write_audit(result_dir: Path, audit: dict[str, object]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    train_counts = audit["selected_train_action_counts"]
    test_counts = audit["selected_test_action_counts"]
    intersection = audit["selected_intersection"]
    lines = [
        "# v0.5 label distribution audit",
        "",
        f"selected train video count: {len(audit['selected_train_video_ids'])}",
        f"selected test video count: {len(audit['selected_test_video_ids'])}",
        "",
        "## Selected Train Videos",
        _format_ids(audit["selected_train_video_ids"]),
        "",
        "## Selected Test Videos",
        _format_ids(audit["selected_test_video_ids"]),
        "",
        f"action classes present in selected train videos: {audit['selected_train_classes']}",
        f"action classes present in selected test videos: {audit['selected_test_classes']}",
        f"train/test action-class intersection: {intersection}",
        "",
        "## Action Frame Counts Per Overlapped Class",
        _format_overlap_counts(train_counts, test_counts, intersection),
        "",
        f"background-frame count in train: {audit['selected_train_background_count']}",
        f"background-frame count in test: {audit['selected_test_background_count']}",
        f"frame_mAP_seen_classes should be meaningful: {audit['seen_class_map_should_be_meaningful']}",
        f"warning: {audit['warning'] or 'none'}",
        "",
        "## Selected Train Action Counts",
        _format_counter(train_counts),
        "",
        "## Selected Test Action Counts",
        _format_counter(test_counts),
        "",
    ]
    (result_dir / "label_distribution_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _write_result_note(result_dir: Path, audit: dict[str, object]) -> None:
    lines = [
        "# v0.5 causal GRU revised smoke result note",
        "",
        "This is a lightweight causal GRU diagnostic smoke run. It is not a full result, not a SOTA comparison, and not a method contribution.",
        "",
        f"selector: class-overlap-controlled deterministic smoke selector",
        f"seed: {audit['seed']}",
        f"target classes: {audit['target_classes']}",
        f"selected train videos: {len(audit['selected_train_video_ids'])}",
        f"selected test videos: {len(audit['selected_test_video_ids'])}",
        f"train classes: {audit['selected_train_classes']}",
        f"test classes: {audit['selected_test_classes']}",
        f"train/test intersection: {audit['selected_intersection']}",
        f"frame_mAP_seen_classes should be meaningful: {audit['seen_class_map_should_be_meaningful']}",
        f"warning: {audit['warning'] or 'none'}",
        "",
        "## Selected Train Video IDs",
        _format_ids(audit["selected_train_video_ids"]),
        "",
        "## Selected Test Video IDs",
        _format_ids(audit["selected_test_video_ids"]),
        "",
    ]
    (result_dir / "result_note.md").write_text("\n".join(lines), encoding="utf-8")


def select_class_overlap_smoke(
    train_dataset,
    test_dataset,
    max_train_videos: int = 80,
    max_test_videos: int = 15,
    background_label: int = 0,
    seed: int = 13,
    target_class_count: int = 8,
    min_action_frames_per_class: int = 500,
    min_overlap_classes: int = 3,
    result_dir: str | Path | None = None,
) -> dict[str, object]:
    random.seed(int(seed))
    train_video_counts = _counts_for_dataset(train_dataset, background_label)
    test_video_counts = _counts_for_dataset(test_dataset, background_label)
    all_train_counts, _train_bg = _aggregate(train_video_counts, list(train_video_counts), background_label)
    all_test_counts, _test_bg = _aggregate(test_video_counts, list(test_video_counts), background_label)
    candidate_classes = [
        class_id for class_id in sorted(set(all_train_counts) & set(all_test_counts))
        if all_train_counts[class_id] >= int(min_action_frames_per_class)
        and all_test_counts[class_id] >= int(min_action_frames_per_class)
    ]
    if len(candidate_classes) < int(min_overlap_classes):
        candidate_classes = sorted(set(all_train_counts) & set(all_test_counts))
    ranked_classes = sorted(
        candidate_classes,
        key=lambda class_id: (
            min(int(all_train_counts[class_id]), int(all_test_counts[class_id])),
            int(all_train_counts[class_id]) + int(all_test_counts[class_id]),
            _stable_noise(seed, str(class_id)),
        ),
        reverse=True,
    )
    target_classes = ranked_classes[: max(int(min_overlap_classes), int(target_class_count))]
    selected_train = _select_videos_for_classes(
        train_video_counts,
        target_classes=target_classes,
        max_videos=int(max_train_videos),
        background_label=background_label,
        seed=seed,
        min_frames_per_class=int(min_action_frames_per_class),
    )
    selected_test = _select_videos_for_classes(
        test_video_counts,
        target_classes=target_classes,
        max_videos=int(max_test_videos),
        background_label=background_label,
        seed=seed + 997,
        min_frames_per_class=int(min_action_frames_per_class),
    )
    train_counts, train_bg = _aggregate(train_video_counts, selected_train, background_label)
    test_counts, test_bg = _aggregate(test_video_counts, selected_test, background_label)
    train_classes = sorted(train_counts)
    test_classes = sorted(test_counts)
    intersection = sorted(set(train_classes) & set(test_classes))
    meaningful = bool(len(intersection) >= int(min_overlap_classes) and any(test_counts[class_id] > 0 for class_id in intersection))
    warnings = []
    if len(intersection) < int(min_overlap_classes):
        warnings.append(f"overlap too small: {len(intersection)} classes")
    if len(train_classes) <= 2:
        warnings.append(f"train subset has only {len(train_classes)} action classes")
    if not any(class_id in train_counts for class_id in test_classes):
        warnings.append("test subset has no seen action classes")
    undercovered = [
        int(class_id) for class_id in intersection
        if train_counts[class_id] < int(min_action_frames_per_class)
        or test_counts[class_id] < int(min_action_frames_per_class)
    ]
    if undercovered:
        warnings.append(f"selected overlap classes below frame target {int(min_action_frames_per_class)}: {undercovered}")
    audit = {
        "seed": int(seed),
        "target_classes": [int(class_id) for class_id in target_classes],
        "selected_train_video_ids": selected_train,
        "selected_test_video_ids": selected_test,
        "selected_train_classes": [int(class_id) for class_id in train_classes],
        "selected_test_classes": [int(class_id) for class_id in test_classes],
        "selected_intersection": [int(class_id) for class_id in intersection],
        "selected_train_action_counts": train_counts,
        "selected_test_action_counts": test_counts,
        "selected_train_background_count": int(train_bg),
        "selected_test_background_count": int(test_bg),
        "seen_class_map_should_be_meaningful": meaningful,
        "warning": "; ".join(warnings),
    }
    train_dataset.video_ids = selected_train
    test_dataset.video_ids = selected_test
    if result_dir is not None:
        result_path = Path(result_dir)
        _write_audit(result_path, audit)
        _write_result_note(result_path, audit)
    return audit
