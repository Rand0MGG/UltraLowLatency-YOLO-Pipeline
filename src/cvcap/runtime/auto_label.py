from __future__ import annotations

import logging
import queue
import random
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from cvcap.core.detections import DetBox

logger = logging.getLogger(__name__)

BODY_CLASS_IDS = {0, 1}
HEAD_CLASS_IDS = {2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SUFFIXES = {".txt"}
DATA_YAML_NAMES = ("data.yaml", "data.yml")
DATASET_SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"valid": "val"}
STAGING_SPLIT = "staging"
DEFAULT_CLASS_NAMES = ("ct_body", "t_body", "head")
INDEX_RE = re.compile(r"(?:^|_)(\d+)$")


class AutoLabeler:
    def __init__(
        self,
        output_dir: Path,
        both_prob: float = 0.1,
        empty_prob: float = 0.01,
        min_interval_s: float = 1.0,
        queue_size: int = 8,
        incomplete_enabled: bool = True,
        incomplete_prob: float = 1.0,
        complete_enabled: bool = True,
        empty_enabled: bool = True,
        low_conf_enabled: bool = True,
        low_conf_prob: float = 0.8,
        low_conf_min: float = 0.25,
        low_conf_max: float = 0.65,
        conflict_enabled: bool = True,
        conflict_prob: float = 1.0,
        conflict_iou: float = 0.45,
        flip_enabled: bool = True,
        flip_prob: float = 1.0,
        flip_iou: float = 0.45,
        flip_max_age_s: float = 0.5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.capture_target = create_capture_target(self.output_dir)
        self.dataset_root = Path(self.capture_target["dataset_root"])
        self.images_dir = Path(self.capture_target["image_dir"])
        self.complete_prob = _prob(both_prob)
        self.empty_prob = _prob(empty_prob)
        self.incomplete_enabled = bool(incomplete_enabled)
        self.incomplete_prob = _prob(incomplete_prob)
        self.complete_enabled = bool(complete_enabled)
        self.empty_enabled = bool(empty_enabled)
        self.low_conf_enabled = bool(low_conf_enabled)
        self.low_conf_prob = _prob(low_conf_prob)
        self.low_conf_min = _prob(low_conf_min)
        self.low_conf_max = _prob(low_conf_max)
        if self.low_conf_min > self.low_conf_max:
            self.low_conf_min, self.low_conf_max = self.low_conf_max, self.low_conf_min
        self.conflict_enabled = bool(conflict_enabled)
        self.conflict_prob = _prob(conflict_prob)
        self.conflict_iou = _prob(conflict_iou)
        self.flip_enabled = bool(flip_enabled)
        self.flip_prob = _prob(flip_prob)
        self.flip_iou = _prob(flip_iou)
        self.flip_max_age_s = max(0.0, float(flip_max_age_s))
        self.min_interval_s = max(1.0, float(min_interval_s))
        self._queue: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._rng = random.Random()
        self._last_save_t = 0.0
        self._previous_body_boxes: list[DetBox] = []
        self._previous_body_t = 0.0

        self._prepare_image_dir()
        self._next_index = self._find_next_index()
        self._thread = threading.Thread(target=self._worker_loop, name="AutoLabeler", daemon=True)
        self._thread.start()
        logger.info(
            "Auto image capture enabled: dir=%s incomplete=%.2f complete=%.2f empty=%.3f low_conf=%.2f conflict=%.2f flip=%.2f min_interval=%.3fs next=%d",
            self.images_dir,
            self.incomplete_prob if self.incomplete_enabled else 0.0,
            self.complete_prob if self.complete_enabled else 0.0,
            self.empty_prob,
            self.low_conf_prob if self.low_conf_enabled else 0.0,
            self.conflict_prob if self.conflict_enabled else 0.0,
            self.flip_prob if self.flip_enabled else 0.0,
            self.min_interval_s,
            self._next_index,
        )

    def maybe_save(self, frame_bgr: np.ndarray, boxes: Iterable[DetBox], now: Optional[float] = None) -> bool:
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        now_value = float(now) if now is not None else 0.0
        boxes_list = list(boxes)
        reason = self._trigger_reason(boxes_list, now_value)
        if reason is None:
            return False

        with self._lock:
            if now_value > 0.0 and now_value - self._last_save_t < self.min_interval_s:
                return False
            index = self._next_index
            try:
                self._queue.put_nowait((frame_bgr.copy(), index))
            except queue.Full:
                return False
            self._next_index += 1
            self._last_save_t = now_value
        return True

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _trigger_reason(self, boxes: list[DetBox], now: float) -> Optional[str]:
        body_boxes = [box for box in boxes if _is_body_box(box)]
        head_boxes = [box for box in boxes if _is_head_box(box)]
        reason: Optional[str] = None

        if not boxes:
            if self.empty_enabled and self._chance(self.empty_prob):
                reason = "empty"
        elif self.conflict_enabled and _has_t_ct_conflict(body_boxes, self.conflict_iou) and self._chance(self.conflict_prob):
            reason = "t_ct_conflict"
        elif self.flip_enabled and self._has_t_ct_flip(body_boxes, now) and self._chance(self.flip_prob):
            reason = "t_ct_flip"
        elif self.low_conf_enabled and _has_low_conf_body(body_boxes, self.low_conf_min, self.low_conf_max) and self._chance(self.low_conf_prob):
            reason = "low_conf_body"
        elif self.incomplete_enabled and ((head_boxes and not body_boxes) or (body_boxes and not head_boxes)) and self._chance(self.incomplete_prob):
            reason = "incomplete"
        elif self.complete_enabled and body_boxes and head_boxes and self._chance(self.complete_prob):
            reason = "complete"

        self._previous_body_boxes = body_boxes
        self._previous_body_t = now
        return reason

    def _has_t_ct_flip(self, body_boxes: list[DetBox], now: float) -> bool:
        if not body_boxes or not self._previous_body_boxes:
            return False
        if self.flip_max_age_s > 0 and now > 0 and self._previous_body_t > 0 and now - self._previous_body_t > self.flip_max_age_s:
            return False
        for current in body_boxes:
            current_cls = _body_team_id(current)
            if current_cls is None:
                continue
            for previous in self._previous_body_boxes:
                previous_cls = _body_team_id(previous)
                if previous_cls is None or previous_cls == current_cls:
                    continue
                if _box_iou(current.xyxy, previous.xyxy) >= self.flip_iou:
                    return True
        return False

    def _chance(self, probability: float) -> bool:
        return self._rng.random() < probability

    def _prepare_image_dir(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _find_next_index(self) -> int:
        highest = -1
        for path in _list_image_files(self.images_dir):
            index = _numeric_suffix(path)
            if index is not None:
                highest = max(highest, index)
        return highest + 1

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame_bgr, index = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                image_path = self.images_dir / _image_filename(self.images_dir, index)
                if not cv2.imwrite(str(image_path), frame_bgr):
                    raise ValueError(f"Could not write auto-label image: {image_path}")
            except Exception as exc:
                logger.error("Auto image save error: %s", exc)


def resolve_dataset_root(selected_path: Path) -> Path:
    path = Path(selected_path)
    if path.suffix.lower() in IMAGE_SUFFIXES | LABEL_SUFFIXES | {".yaml", ".yml"}:
        path = path.parent

    for candidate in (path, *path.parents):
        if _has_dataset_layout(candidate):
            return candidate
        stripped = _strip_dataset_leaf(candidate)
        if stripped != candidate and _has_dataset_layout(stripped):
            return stripped

    stripped = _strip_dataset_leaf(path)
    if stripped != path:
        return stripped

    if path.name.lower() in {"images", "labels"}:
        return path.parent
    return path


def create_capture_target(selected_path: Path) -> dict:
    dataset_root = resolve_dataset_root(selected_path)
    image_dir = _split_images_dir(dataset_root, STAGING_SPLIT)
    labels_dir = _split_labels_dir(dataset_root, STAGING_SPLIT)
    image_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dataset_root": str(dataset_root),
        "staging_dir": str(image_dir.parent),
        "image_dir": str(image_dir),
        "labels_dir": str(labels_dir),
        "prefix": _image_prefix(image_dir),
    }


def prepare_staged_dataset(
    selected_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: Optional[int] = None,
) -> dict:
    dataset_root = resolve_dataset_root(selected_path)
    staging_image_dir = _resolve_staging_image_dir(dataset_root, selected_path)
    if not staging_image_dir.exists() or not staging_image_dir.is_dir():
        raise ValueError(f"Staging image folder does not exist: {staging_image_dir}")

    images = _list_image_files(staging_image_dir)
    if not images:
        return {
            "dataset_root": str(dataset_root),
            "staging_image_dir": str(staging_image_dir),
            "images": 0,
            "labels": 0,
            "converted": 0,
            "splits": {split: 0 for split in DATASET_SPLITS},
            "prefix": _dataset_prefix(dataset_root),
        }

    staging_labels_dir = _sibling_labels_dir(staging_image_dir)
    labels_by_stem = _label_file_map(staging_labels_dir)
    label_required = bool(labels_by_stem)
    if label_required:
        _raise_if_duplicate_image_stems(images, "staged image")
        _raise_if_missing_labels(images, labels_by_stem, "staged image")
        _raise_if_orphan_labels(labels_by_stem, images, "staging")

    shuffled = images[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    ratios = _normalize_split_ratios(train_ratio, val_ratio, test_ratio)
    split_counts = _split_counts(len(shuffled), ratios)
    assignments: list[tuple[str, Path]] = []
    cursor = 0
    for split in DATASET_SPLITS:
        count = split_counts[split]
        assignments.extend((split, path) for path in shuffled[cursor : cursor + count])
        cursor += count

    restart_split_indices = _should_restart_prepared_split_indices(
        dataset_root,
        selected_path,
        staging_image_dir,
    )
    next_indices = {
        split: 0
        if restart_split_indices
        else _find_next_dataset_split_index(dataset_root, split)
        for split in DATASET_SPLITS
    }
    allowed_existing = (
        _prepare_source_paths(images, labels_by_stem)
        if restart_split_indices
        else None
    )
    planned: list[tuple[str, Path, Optional[Path], Path, Optional[Path]]] = []
    planned_destinations: set[Path] = set()
    for split, image_path in assignments:
        target_index = next_indices[split]
        next_indices[split] += 1
        target_image_dir = _split_images_dir(dataset_root, split)
        target_labels_dir = _split_labels_dir(dataset_root, split)
        target_name = _dataset_image_filename(
            dataset_root,
            split,
            target_index,
            image_path.suffix,
        )
        target_image_path = target_image_dir / target_name
        source_label_path = labels_by_stem.get(image_path.stem)
        target_label_path = (
            target_labels_dir / f"{Path(target_name).stem}.txt"
            if source_label_path
            else None
        )
        _reserve_destination(
            target_image_path,
            planned_destinations,
            allowed_existing=allowed_existing,
        )
        if target_label_path is not None:
            _reserve_destination(
                target_label_path,
                planned_destinations,
                allowed_existing=allowed_existing,
            )
        planned.append(
            (
                split,
                image_path,
                source_label_path,
                target_image_path,
                target_label_path,
            )
        )

    _ensure_dataset_scaffold(dataset_root)
    if restart_split_indices:
        planned = _stage_prepare_sources(planned)

    moved = 0
    moved_labels = 0
    converted = 0
    per_split = {split: 0 for split in DATASET_SPLITS}
    for split, source_image_path, source_label_path, target_image_path, target_label_path in planned:
        _move_file_preserving_format(source_image_path, target_image_path)
        if source_label_path is not None and target_label_path is not None:
            source_label_path.rename(target_label_path)
            moved_labels += 1
        moved += 1
        per_split[split] += 1
    caches_removed = _remove_split_cache_files(dataset_root, per_split.keys())

    return {
        "dataset_root": str(dataset_root),
        "staging_image_dir": str(staging_image_dir),
        "images": moved,
        "labels": moved_labels,
        "converted": converted,
        "caches_removed": caches_removed,
        "splits": per_split,
        "ratios": ratios,
        "prefix": _dataset_prefix(dataset_root),
    }


def auto_annotate_dataset(dataset_path: Path, args, overwrite: bool = False) -> dict:
    dataset_root = resolve_dataset_root(dataset_path)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")
    if not _has_dataset_layout(dataset_root):
        raise ValueError(f"Dataset root does not contain a YOLO train/val/test layout: {dataset_root}")

    detector = _make_detector(args)
    totals = {
        "images": 0,
        "labeled": 0,
        "empty": 0,
        "failed": 0,
        "skipped_existing": 0,
        "boxes": 0,
    }
    per_split: dict[str, dict[str, int]] = {}
    class_counts: dict[int, int] = {}

    for split in DATASET_SPLITS:
        image_dir = _split_images_dir(dataset_root, split, prefer_existing=True)
        labels_dir = _split_labels_dir(dataset_root, split, prefer_existing=True)
        images = _list_image_files(image_dir) if image_dir.exists() else []
        split_result = _annotate_images(images, labels_dir, detector, overwrite=overwrite)
        per_split[split] = {key: int(value) for key, value in split_result.items() if key != "class_counts"}
        for key in totals:
            totals[key] += int(split_result.get(key, 0))
        for cls_id, count in split_result["class_counts"].items():
            class_counts[cls_id] = class_counts.get(cls_id, 0) + count
        if split_result.get("labeled", 0):
            _remove_split_cache_files(dataset_root, (split,))

    return {
        **totals,
        "class_counts": class_counts,
        "splits": per_split,
        "dataset_root": str(dataset_root),
    }


def merge_labeled_dataset(source_dataset_path: Path, target_dataset_path: Path) -> dict:
    source_root = resolve_dataset_root(source_dataset_path)
    target_root = resolve_dataset_root(target_dataset_path)
    if source_root == target_root:
        raise ValueError("Source and target dataset roots must be different.")
    if not source_root.exists() or not source_root.is_dir():
        raise ValueError(f"Source dataset root does not exist: {source_root}")
    if not _has_dataset_layout(source_root):
        raise ValueError(f"Source dataset root does not contain a YOLO train/val/test layout: {source_root}")

    source_class_names = _dataset_class_names(source_root)
    target_class_names = _dataset_class_names(target_root)
    if source_class_names and target_class_names and source_class_names != target_class_names:
        raise ValueError("Source and target dataset class names differ; refusing to merge labels with incompatible class maps.")

    planned: list[tuple[str, Path, Path, Path, Path]] = []
    planned_destinations: set[Path] = set()
    per_split = {split: 0 for split in DATASET_SPLITS}
    orphan_labels = 0

    for split in DATASET_SPLITS:
        pairs, split_orphans = _collect_labeled_split_pairs(source_root, split, require_labels=True)
        orphan_labels += split_orphans
        next_index = _find_next_dataset_split_index(target_root, split)
        for offset, (source_image_path, source_label_path) in enumerate(pairs):
            target_name = _dataset_image_filename(target_root, split, next_index + offset, source_image_path.suffix)
            target_image_path = _split_images_dir(target_root, split) / target_name
            target_label_path = _split_labels_dir(target_root, split) / f"{Path(target_name).stem}.txt"
            _reserve_destination(target_image_path, planned_destinations)
            _reserve_destination(target_label_path, planned_destinations)
            planned.append((split, source_image_path, source_label_path, target_image_path, target_label_path))

    _ensure_dataset_scaffold(target_root, class_names=target_class_names or source_class_names)

    copied = 0
    converted = 0
    for split, source_image_path, source_label_path, target_image_path, target_label_path in planned:
        _copy_file_preserving_format(source_image_path, target_image_path)
        shutil.copy2(source_label_path, target_label_path)
        copied += 1
        per_split[split] += 1
    caches_removed = _remove_split_cache_files(target_root, per_split.keys())

    return {
        "images": copied,
        "labels": copied,
        "converted": converted,
        "caches_removed": caches_removed,
        "orphan_labels": orphan_labels,
        "splits": per_split,
        "source_dataset_root": str(source_root),
        "target_dataset_root": str(target_root),
        "prefix": _dataset_prefix(target_root),
    }


def normalize_dataset_filenames(dataset_path: Path) -> dict:
    dataset_root = resolve_dataset_root(dataset_path)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")
    if not _has_dataset_layout(dataset_root):
        raise ValueError(f"Dataset root does not contain a YOLO train/val/test layout: {dataset_root}")
    _ensure_dataset_scaffold(dataset_root)

    renamed = 0
    renamed_labels = 0
    converted = 0
    per_split = {split: 0 for split in DATASET_SPLITS}
    for split in DATASET_SPLITS:
        pairs, _ = _collect_labeled_split_pairs(dataset_root, split, require_labels=False, fail_on_orphans=True)
        if not pairs:
            continue

        image_dir = _split_images_dir(dataset_root, split, prefer_existing=True)
        labels_dir = _split_labels_dir(dataset_root, split, prefer_existing=True)
        current_paths = {image_path for image_path, _ in pairs}
        current_paths.update(label_path for _, label_path in pairs if label_path is not None)
        final_paths: set[Path] = set()
        for index, (image_path, label_path) in enumerate(pairs):
            final_image_path = image_dir / _dataset_image_filename(dataset_root, split, index, image_path.suffix)
            final_label_path = labels_dir / f"{final_image_path.stem}.txt" if label_path is not None else None
            _reserve_destination(final_image_path, final_paths, allowed_existing=current_paths)
            if final_label_path is not None:
                _reserve_destination(final_label_path, final_paths, allowed_existing=current_paths)

        tmp_tag = f".normalize_tmp_{time.time_ns()}"
        tmp_pairs: list[tuple[Path, Optional[Path]]] = []
        for index, (image_path, label_path) in enumerate(pairs):
            tmp_image_path = image_dir / f"{tmp_tag}_{index}{image_path.suffix.lower()}"
            image_path.rename(tmp_image_path)
            tmp_label_path = None
            if label_path is not None:
                tmp_label_path = labels_dir / f"{tmp_tag}_{index}.txt"
                label_path.rename(tmp_label_path)
            tmp_pairs.append((tmp_image_path, tmp_label_path))

        for index, (tmp_image_path, tmp_label_path) in enumerate(tmp_pairs):
            final_image_path = image_dir / _dataset_image_filename(dataset_root, split, index, tmp_image_path.suffix)
            _move_file_preserving_format(tmp_image_path, final_image_path)
            if tmp_label_path is not None:
                tmp_label_path.rename(labels_dir / f"{final_image_path.stem}.txt")
                renamed_labels += 1
            renamed += 1
            per_split[split] += 1
    caches_removed = _remove_split_cache_files(dataset_root, per_split.keys())

    return {
        "dataset_root": str(dataset_root),
        "images": renamed,
        "labels": renamed_labels,
        "converted": converted,
        "caches_removed": caches_removed,
        "splits": per_split,
        "prefix": _dataset_prefix(dataset_root),
    }


def shuffle_dataset_split(split_path: Path, seed: Optional[int] = None) -> dict:
    dataset_root = resolve_dataset_root(split_path)
    split = _selected_dataset_split(split_path)
    if split is None:
        raise ValueError("Select a train, val, or test folder, or its images/labels folder.")
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")
    if not _has_dataset_layout(dataset_root):
        raise ValueError(f"Dataset root does not contain a YOLO train/val/test layout: {dataset_root}")

    selected = Path(split_path)
    if selected.suffix.lower() in IMAGE_SUFFIXES | LABEL_SUFFIXES:
        selected = selected.parent
    image_dir = _split_context_image_dir(selected)
    labels_dir = _sibling_labels_dir(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Split image folder does not exist: {image_dir}")

    images = _list_image_files(image_dir)
    labels_by_stem = _label_file_map(labels_dir)
    if not images:
        return {
            "dataset_root": str(dataset_root),
            "split": split,
            "images": 0,
            "labels": 0,
            "image_dir": str(image_dir),
            "labels_dir": str(labels_dir),
            "prefix": f"{_dataset_prefix(dataset_root)}_{split}",
        }
    if labels_by_stem:
        _raise_if_duplicate_image_stems(images, f"{split} image")
        _raise_if_missing_labels(images, labels_by_stem, f"{split} image")
        _raise_if_orphan_labels(labels_by_stem, images, split)

    pairs = [(image_path, labels_by_stem.get(image_path.stem)) for image_path in images]
    rng = random.Random(seed)
    rng.shuffle(pairs)

    current_paths = {image_path for image_path, _ in pairs}
    current_paths.update(label_path for _, label_path in pairs if label_path is not None)
    final_paths: set[Path] = set()
    for index, (image_path, label_path) in enumerate(pairs):
        final_image_path = image_dir / _dataset_image_filename(dataset_root, split, index, image_path.suffix)
        final_label_path = labels_dir / f"{final_image_path.stem}.txt" if label_path is not None else None
        _reserve_destination(final_image_path, final_paths, allowed_existing=current_paths)
        if final_label_path is not None:
            _reserve_destination(final_label_path, final_paths, allowed_existing=current_paths)

    tmp_tag = f".shuffle_tmp_{time.time_ns()}"
    tmp_pairs: list[tuple[Path, Optional[Path]]] = []
    moved: list[tuple[Path, Path]] = []
    try:
        for index, (image_path, label_path) in enumerate(pairs):
            tmp_image_path = image_dir / f"{tmp_tag}_{index}{image_path.suffix.lower()}"
            _reserve_destination(tmp_image_path, final_paths)
            tmp_label_path = None
            if label_path is not None:
                tmp_label_path = labels_dir / f"{tmp_tag}_{index}.txt"
                _reserve_destination(tmp_label_path, final_paths)

            image_path.rename(tmp_image_path)
            moved.append((tmp_image_path, image_path))
            if label_path is not None:
                assert tmp_label_path is not None
                label_path.rename(tmp_label_path)
                moved.append((tmp_label_path, label_path))
            tmp_pairs.append((tmp_image_path, tmp_label_path))
    except Exception:
        for tmp_path, original_path in reversed(moved):
            if tmp_path.exists() and not original_path.exists():
                tmp_path.rename(original_path)
        raise

    shuffled = 0
    shuffled_labels = 0
    for index, (tmp_image_path, tmp_label_path) in enumerate(tmp_pairs):
        final_image_path = image_dir / _dataset_image_filename(dataset_root, split, index, tmp_image_path.suffix)
        _move_file_preserving_format(tmp_image_path, final_image_path)
        if tmp_label_path is not None:
            tmp_label_path.rename(labels_dir / f"{final_image_path.stem}.txt")
            shuffled_labels += 1
        shuffled += 1
    caches_removed = _remove_split_cache_files(dataset_root, (split,))

    return {
        "dataset_root": str(dataset_root),
        "split": split,
        "images": shuffled,
        "labels": shuffled_labels,
        "caches_removed": caches_removed,
        "image_dir": str(image_dir),
        "labels_dir": str(labels_dir),
        "prefix": f"{_dataset_prefix(dataset_root)}_{split}",
        "first": _dataset_image_filename(dataset_root, split, 0, tmp_pairs[0][0].suffix) if tmp_pairs else "",
        "last": _dataset_image_filename(dataset_root, split, shuffled - 1, tmp_pairs[-1][0].suffix) if tmp_pairs else "",
    }


def shuffle_rename_image_folder(image_dir: Path) -> dict:
    if _selected_path_implies_dataset(image_dir):
        return prepare_staged_dataset(image_dir)
    return _legacy_shuffle_rename_image_folder(image_dir)


def auto_annotate_image_folder(image_dir: Path, args) -> dict:
    dataset_root = resolve_dataset_root(image_dir)
    if _has_dataset_layout(dataset_root):
        return auto_annotate_dataset(dataset_root, args)
    return _legacy_auto_annotate_image_folder(image_dir, args)


def merge_labeled_image_folder(source_image_dir: Path, target_image_dir: Path) -> dict:
    source_root = resolve_dataset_root(source_image_dir)
    target_root = resolve_dataset_root(target_image_dir)
    if _has_dataset_layout(source_root) or _has_dataset_layout(target_root):
        return merge_labeled_dataset(source_root, target_root)
    return _legacy_merge_labeled_image_folder(source_image_dir, target_image_dir)


def _legacy_shuffle_rename_image_folder(image_dir: Path) -> dict:
    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Image folder does not exist: {image_dir}")

    labels_dir = _sibling_labels_dir(image_dir)
    if labels_dir.exists():
        raise ValueError(f"Refusing to shuffle because labels folder already exists: {labels_dir}")

    images = _list_image_files(image_dir)
    if not images:
        return {"images": 0, "image_dir": str(image_dir), "prefix": _image_prefix(image_dir)}

    shuffled = images[:]
    random.shuffle(shuffled)
    tmp_tag = f".rename_tmp_{time.time_ns()}"
    tmp_paths: list[Path] = []
    for index, path in enumerate(shuffled):
        tmp_path = image_dir / f"{tmp_tag}_{index}{path.suffix.lower()}"
        path.rename(tmp_path)
        tmp_paths.append(tmp_path)

    converted = 0
    for index, tmp_path in enumerate(tmp_paths):
        final_path = image_dir / _image_filename(image_dir, index)
        converted += _move_image_as_jpg(tmp_path, final_path)

    return {
        "images": len(images),
        "converted": converted,
        "image_dir": str(image_dir),
        "prefix": _image_prefix(image_dir),
        "first": _image_filename(image_dir, 0),
        "last": _image_filename(image_dir, len(images) - 1),
    }


def _legacy_auto_annotate_image_folder(image_dir: Path, args) -> dict:
    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Image folder does not exist: {image_dir}")

    images = _list_image_files(image_dir)
    labels_dir = _sibling_labels_dir(image_dir)
    detector = _make_detector(args) if images else None
    result = _annotate_images(images, labels_dir, detector, overwrite=True)
    return {
        **result,
        "image_dir": str(image_dir),
        "labels_dir": str(labels_dir),
    }


def _legacy_merge_labeled_image_folder(source_image_dir: Path, target_image_dir: Path) -> dict:
    source_image_dir = Path(source_image_dir)
    target_image_dir = Path(target_image_dir)
    if not source_image_dir.exists() or not source_image_dir.is_dir():
        raise ValueError(f"Source image folder does not exist: {source_image_dir}")

    source_labels_dir = _sibling_labels_dir(source_image_dir)
    if not source_labels_dir.exists() or not source_labels_dir.is_dir():
        raise ValueError(f"Source labels folder does not exist: {source_labels_dir}")

    source_images = _list_image_files(source_image_dir)
    if not source_images:
        return {"images": 0, "labels": 0, "target_image_dir": str(target_image_dir)}

    missing_labels = [path.name for path in source_images if not (source_labels_dir / f"{path.stem}.txt").exists()]
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(f"Missing label files for {len(missing_labels)} source image(s): {preview}")

    target_labels_dir = _sibling_labels_dir(target_image_dir)
    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    next_index = _find_next_pair_index(target_image_dir, target_labels_dir)
    copied = 0
    converted = 0
    for offset, source_image_path in enumerate(source_images):
        target_index = next_index + offset
        target_name = _image_filename(target_image_dir, target_index)
        target_image_path = target_image_dir / target_name
        target_label_path = target_labels_dir / f"{Path(target_name).stem}.txt"

        converted += _copy_image_as_jpg(source_image_path, target_image_path)
        shutil.copy2(source_labels_dir / f"{source_image_path.stem}.txt", target_label_path)
        copied += 1

    return {
        "images": copied,
        "labels": copied,
        "converted": converted,
        "source_image_dir": str(source_image_dir),
        "source_labels_dir": str(source_labels_dir),
        "target_image_dir": str(target_image_dir),
        "target_labels_dir": str(target_labels_dir),
        "prefix": _image_prefix(target_image_dir),
        "first": _image_filename(target_image_dir, next_index),
        "last": _image_filename(target_image_dir, next_index + copied - 1),
    }


def _selected_path_implies_dataset(selected_path: Path) -> bool:
    path = Path(selected_path)
    if path.suffix.lower() in IMAGE_SUFFIXES | LABEL_SUFFIXES | {".yaml", ".yml"}:
        path = path.parent
    dataset_root = resolve_dataset_root(path)
    return dataset_root != path or _has_dataset_layout(dataset_root)


def _strip_dataset_leaf(path: Path) -> Path:
    path = Path(path)
    name = path.name.lower()
    if name in {"images", "labels"}:
        parent_name = path.parent.name.lower()
        if _is_split_name(parent_name) or parent_name == STAGING_SPLIT:
            return path.parent.parent
    if _is_split_name(name) or name == STAGING_SPLIT:
        return path.parent
    return path


def _is_split_name(value: str) -> bool:
    return _canonical_split(value) in DATASET_SPLITS


def _canonical_split(value: str) -> str:
    lowered = str(value).lower()
    return SPLIT_ALIASES.get(lowered, lowered)


def _has_dataset_layout(dataset_root: Path) -> bool:
    dataset_root = Path(dataset_root)
    if _data_yaml_path(dataset_root) is not None or (dataset_root / "classes.txt").exists():
        return True
    for split in (*DATASET_SPLITS, STAGING_SPLIT):
        if _split_images_dir(dataset_root, split, prefer_existing=True).exists():
            return True
        if _split_labels_dir(dataset_root, split, prefer_existing=True).exists():
            return True
    return False


def _resolve_staging_image_dir(dataset_root: Path, selected_path: Path) -> Path:
    selected = Path(selected_path)
    if selected.suffix.lower() in IMAGE_SUFFIXES | LABEL_SUFFIXES:
        selected = selected.parent
    if selected.name.lower() == "images" and selected.parent.name.lower() == STAGING_SPLIT:
        return selected
    if selected.name.lower() == STAGING_SPLIT:
        return selected / "images"
    if selected.exists() and selected.is_dir() and _list_image_files(selected) and not _is_dataset_split_context(selected):
        return selected
    if selected.exists() and selected.is_dir() and _is_dataset_split_context(selected):
        selected_images = selected if selected.name.lower() in {"images", "image"} else selected / "images"
        if selected_images.exists() and _list_image_files(selected_images):
            return selected_images
    staging_images = _split_images_dir(dataset_root, STAGING_SPLIT)
    if staging_images.exists() and _list_image_files(staging_images):
        return staging_images
    root_images = dataset_root / "images"
    if root_images.exists() and _list_image_files(root_images):
        return root_images
    single_split_images = _single_populated_split_image_dir(dataset_root)
    if single_split_images is not None:
        return single_split_images
    return _split_images_dir(dataset_root, STAGING_SPLIT)


def _is_dataset_split_context(path: Path) -> bool:
    path = Path(path)
    if _is_split_name(path.name):
        return True
    if path.name.lower() in {"images", "image", "labels"} and _is_split_name(path.parent.name):
        return True
    return False


def _selected_dataset_split(path: Path) -> Optional[str]:
    selected = Path(path)
    if selected.suffix.lower() in IMAGE_SUFFIXES | LABEL_SUFFIXES:
        selected = selected.parent
    name = selected.name.lower()
    if name in {"images", "image", "labels"}:
        name = selected.parent.name.lower()
    split = _canonical_split(name)
    return split if split in DATASET_SPLITS else None


def _single_populated_split_image_dir(dataset_root: Path) -> Optional[Path]:
    populated: list[Path] = []
    for split in DATASET_SPLITS:
        image_dir = _split_images_dir(dataset_root, split, prefer_existing=True)
        if image_dir.exists() and _list_image_files(image_dir):
            populated.append(image_dir)
    return populated[0] if len(populated) == 1 else None


def _should_restart_prepared_split_indices(
    dataset_root: Path,
    selected_path: Path,
    staging_image_dir: Path,
) -> bool:
    selected = Path(selected_path)
    if selected.suffix.lower() in IMAGE_SUFFIXES | LABEL_SUFFIXES:
        selected = selected.parent
    if not _is_dataset_split_context(selected):
        return False

    selected_images = _split_context_image_dir(selected)
    return _same_path(_strip_dataset_leaf(selected), dataset_root) and _same_path(
        selected_images,
        staging_image_dir,
    )


def _split_context_image_dir(path: Path) -> Path:
    path = Path(path)
    name = path.name.lower()
    if name in {"images", "image"}:
        return path
    if name == "labels":
        return path.parent / "images"
    return path / "images"


def _same_path(first: Path, second: Path) -> bool:
    try:
        return Path(first).resolve() == Path(second).resolve()
    except OSError:
        return Path(first) == Path(second)


def _prepare_source_paths(
    images: Sequence[Path],
    labels_by_stem: dict[str, Path],
) -> set[Path]:
    paths = set(images)
    for image_path in images:
        label_path = labels_by_stem.get(image_path.stem)
        if label_path is not None:
            paths.add(label_path)
    return paths


def _stage_prepare_sources(
    planned: Sequence[tuple[str, Path, Optional[Path], Path, Optional[Path]]],
) -> list[tuple[str, Path, Optional[Path], Path, Optional[Path]]]:
    tmp_tag = f".prepare_tmp_{time.time_ns()}"
    tmp_destinations: set[Path] = set()
    tmp_plan: list[
        tuple[str, Path, Path, Optional[Path], Optional[Path], Path, Optional[Path]]
    ] = []
    for index, (
        split,
        source_image_path,
        source_label_path,
        target_image_path,
        target_label_path,
    ) in enumerate(planned):
        tmp_image_path = source_image_path.parent / (
            f"{tmp_tag}_{index}{source_image_path.suffix.lower()}"
        )
        _reserve_destination(tmp_image_path, tmp_destinations)
        tmp_label_path = None
        if source_label_path is not None:
            tmp_label_path = source_label_path.parent / f"{tmp_tag}_{index}.txt"
            _reserve_destination(tmp_label_path, tmp_destinations)
        tmp_plan.append(
            (
                split,
                source_image_path,
                tmp_image_path,
                source_label_path,
                tmp_label_path,
                target_image_path,
                target_label_path,
            )
        )

    staged: list[tuple[str, Path, Optional[Path], Path, Optional[Path]]] = []
    moved: list[tuple[Path, Path]] = []
    try:
        for (
            split,
            source_image_path,
            tmp_image_path,
            source_label_path,
            tmp_label_path,
            target_image_path,
            target_label_path,
        ) in tmp_plan:
            source_image_path.rename(tmp_image_path)
            moved.append((tmp_image_path, source_image_path))
            if source_label_path is not None and tmp_label_path is not None:
                source_label_path.rename(tmp_label_path)
                moved.append((tmp_label_path, source_label_path))
            staged.append(
                (
                    split,
                    tmp_image_path,
                    tmp_label_path,
                    target_image_path,
                    target_label_path,
                )
            )
    except Exception:
        for tmp_path, original_path in reversed(moved):
            if tmp_path.exists() and not original_path.exists():
                tmp_path.rename(original_path)
        raise
    return staged


def _normalize_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    raw = {
        "train": max(0.0, float(train_ratio)),
        "val": max(0.0, float(val_ratio)),
        "test": max(0.0, float(test_ratio)),
    }
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("At least one split ratio must be greater than zero.")
    return {split: raw[split] / total for split in DATASET_SPLITS}


def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw_counts = {split: total * ratios[split] for split in DATASET_SPLITS}
    counts = {split: int(raw_counts[split]) for split in DATASET_SPLITS}
    remaining = total - sum(counts.values())
    by_remainder = sorted(
        DATASET_SPLITS,
        key=lambda split: (raw_counts[split] - counts[split], ratios[split]),
        reverse=True,
    )
    for split in by_remainder[:remaining]:
        counts[split] += 1
    return counts


def _ensure_dataset_scaffold(dataset_root: Path, class_names: Optional[Sequence[str]] = None) -> None:
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    for split in DATASET_SPLITS:
        _split_images_dir(dataset_root, split).mkdir(parents=True, exist_ok=True)
        _split_labels_dir(dataset_root, split).mkdir(parents=True, exist_ok=True)

    names = list(class_names or _dataset_class_names(dataset_root) or DEFAULT_CLASS_NAMES)
    classes_path = dataset_root / "classes.txt"
    if not classes_path.exists():
        classes_path.write_text("\n".join(names) + "\n", encoding="utf-8")

    data_yaml_path = _data_yaml_path(dataset_root) or dataset_root / "data.yaml"
    _write_data_yaml(data_yaml_path, dataset_root, names)


def _dataset_class_names(dataset_root: Path) -> list[str]:
    return _read_classes_txt(dataset_root) or _read_data_yaml_class_names(dataset_root)


def _read_classes_txt(dataset_root: Path) -> list[str]:
    classes_path = Path(dataset_root) / "classes.txt"
    if not classes_path.exists():
        return []
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_data_yaml_class_names(dataset_root: Path) -> list[str]:
    data_yaml_path = _data_yaml_path(dataset_root)
    if data_yaml_path is None:
        return []
    try:
        import yaml

        data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=_class_index_sort_key)]
    return []


def _class_index_sort_key(value) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _write_data_yaml(data_yaml_path: Path, dataset_root: Path, class_names: Sequence[str]) -> None:
    data = {
        "path": Path(dataset_root).as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: str(name) for index, name in enumerate(class_names)},
    }
    try:
        import yaml

        text = yaml.safe_dump(data, sort_keys=False)
    except Exception:
        name_lines = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
        text = (
            f"path: {Path(dataset_root).as_posix()}\n"
            "train: train/images\n"
            "val: val/images\n"
            "test: test/images\n"
            f"names:\n{name_lines}\n"
        )
    data_yaml_path.write_text(text, encoding="utf-8")


def _data_yaml_path(dataset_root: Path) -> Optional[Path]:
    for name in DATA_YAML_NAMES:
        path = Path(dataset_root) / name
        if path.exists():
            return path
    return None


def _split_dir(dataset_root: Path, split: str, prefer_existing: bool = False) -> Path:
    canonical = _canonical_split(split)
    dataset_root = Path(dataset_root)
    if prefer_existing and canonical == "val" and not (dataset_root / "val").exists() and (dataset_root / "valid").exists():
        return dataset_root / "valid"
    return dataset_root / canonical


def _split_images_dir(dataset_root: Path, split: str, prefer_existing: bool = False) -> Path:
    split_dir = _split_dir(dataset_root, split, prefer_existing=prefer_existing)
    if prefer_existing and not (split_dir / "images").exists() and (split_dir / "image").exists():
        return split_dir / "image"
    return split_dir / "images"


def _split_labels_dir(dataset_root: Path, split: str, prefer_existing: bool = False) -> Path:
    return _split_dir(dataset_root, split, prefer_existing=prefer_existing) / "labels"


def _dataset_prefix(dataset_root: Path) -> str:
    return _safe_filename_part(Path(dataset_root).name)


def _dataset_image_filename(dataset_root: Path, split: str, index: int, suffix: str = ".jpg") -> str:
    safe_suffix = str(suffix or ".jpg").lower()
    if safe_suffix not in IMAGE_SUFFIXES:
        safe_suffix = ".jpg"
    return f"{_dataset_prefix(dataset_root)}_{_canonical_split(split)}_{index:06d}{safe_suffix}"


def _label_file_map(labels_dir: Path) -> dict[str, Path]:
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        return {}
    if not labels_dir.is_dir():
        raise ValueError(f"Labels path is not a folder: {labels_dir}")
    return {path.stem: path for path in _list_label_files(labels_dir)}


def _raise_if_missing_labels(images: Sequence[Path], labels_by_stem: dict[str, Path], subject: str) -> None:
    missing = [image_path.name for image_path in images if image_path.stem not in labels_by_stem]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing label files for {len(missing)} {subject}(s): {preview}")


def _raise_if_duplicate_image_stems(images: Sequence[Path], subject: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for image_path in images:
        stem = image_path.stem.lower()
        if stem in seen:
            duplicates.append(image_path.stem)
        seen.add(stem)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate image stem(s) make label pairing ambiguous for {len(duplicates)} {subject}(s): {preview}")


def _raise_if_orphan_labels(labels_by_stem: dict[str, Path], images: Sequence[Path], context: str) -> None:
    image_stems = {image_path.stem for image_path in images}
    orphan_labels = [path.name for stem, path in labels_by_stem.items() if stem not in image_stems]
    if orphan_labels:
        preview = ", ".join(orphan_labels[:5])
        raise ValueError(f"Found {len(orphan_labels)} orphan label file(s) in {context}: {preview}")


def _collect_labeled_split_pairs(
    dataset_root: Path,
    split: str,
    require_labels: bool,
    fail_on_orphans: bool = False,
) -> tuple[list[tuple[Path, Optional[Path]]], int]:
    image_dir = _split_images_dir(dataset_root, split, prefer_existing=True)
    labels_dir = _split_labels_dir(dataset_root, split, prefer_existing=True)
    images = _list_image_files(image_dir) if image_dir.exists() else []
    labels_by_stem = _label_file_map(labels_dir)
    if require_labels or labels_by_stem:
        _raise_if_duplicate_image_stems(images, f"{split} image")
        _raise_if_missing_labels(images, labels_by_stem, f"{split} image")

    image_stems = {image_path.stem for image_path in images}
    orphan_count = len([stem for stem in labels_by_stem if stem not in image_stems])
    if orphan_count and fail_on_orphans:
        _raise_if_orphan_labels(labels_by_stem, images, split)

    pairs: list[tuple[Path, Optional[Path]]] = []
    for image_path in images:
        pairs.append((image_path, labels_by_stem.get(image_path.stem)))
    return pairs, orphan_count


def _reserve_destination(path: Path, planned: set[Path], allowed_existing: Optional[set[Path]] = None) -> None:
    path = Path(path)
    if path in planned:
        raise ValueError(f"Duplicate target path planned: {path}")
    if path.exists() and (allowed_existing is None or path not in allowed_existing):
        raise ValueError(f"Refusing to overwrite existing file: {path}")
    planned.add(path)


def _move_file_preserving_format(source_path: Path, target_path: Path) -> None:
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.rename(target_path)


def _copy_file_preserving_format(source_path: Path, target_path: Path) -> None:
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _remove_split_cache_files(dataset_root: Path, splits: Iterable[str]) -> int:
    removed = 0
    for split in splits:
        split_dir = _split_dir(dataset_root, split, prefer_existing=True)
        for cache_path in (split_dir / "labels.cache", _split_labels_dir(dataset_root, split, prefer_existing=True).with_suffix(".cache")):
            if cache_path.exists() and cache_path.is_file():
                cache_path.unlink()
                removed += 1
    return removed


def _move_image_as_jpg(source_path: Path, target_path: Path) -> int:
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() in {".jpg", ".jpeg"}:
        source_path.rename(target_path)
        return 0
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise ValueError(f"Could not read image during rename: {source_path}")
    if not cv2.imwrite(str(target_path), frame):
        raise ValueError(f"Could not write renamed image: {target_path}")
    source_path.unlink()
    return 1


def _copy_image_as_jpg(source_path: Path, target_path: Path) -> int:
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source_path, target_path)
        return 0
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise ValueError(f"Could not read source image: {source_path}")
    if not cv2.imwrite(str(target_path), frame):
        raise ValueError(f"Could not write target image: {target_path}")
    return 1


def _make_detector(args):
    from cvcap.adapters.inference.ultralytics_detector import YoloDetector

    return YoloDetector(
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=args.half,
        end2end=args.end2end,
        classes=args.yolo_classes,
        max_det=args.yolo_max_det,
    )


def _annotate_images(images: Sequence[Path], labels_dir: Path, detector, overwrite: bool) -> dict:
    if detector is None:
        return {
            "images": len(images),
            "labeled": 0,
            "empty": 0,
            "failed": 0,
            "skipped_existing": 0,
            "boxes": 0,
            "class_counts": {},
        }

    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    labeled = 0
    empty = 0
    failed = 0
    skipped_existing = 0
    box_count = 0
    class_counts: dict[int, int] = {}
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            failed += 1
            continue
        boxes, _ = detector.infer(frame_bgr)
        labels = _boxes_to_yolo_labels(boxes, frame_bgr.shape)
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        labeled += 1
        if labels:
            box_count += len(labels)
            for label in labels:
                cls_id = int(label.split(" ", 1)[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        else:
            empty += 1

    return {
        "images": len(images),
        "labeled": labeled,
        "empty": empty,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "boxes": box_count,
        "class_counts": class_counts,
    }


def _boxes_to_yolo_labels(boxes: Iterable[DetBox], frame_shape: tuple[int, ...]) -> list[str]:
    height, width = frame_shape[:2]
    labels: list[str] = []
    for box in boxes:
        if not (_is_body_box(box) or _is_head_box(box)):
            continue
        x1, y1, x2, y2 = box.xyxy
        x1 = max(0.0, min(float(width), float(x1)))
        y1 = max(0.0, min(float(height), float(y1)))
        x2 = max(0.0, min(float(width), float(x2)))
        y2 = max(0.0, min(float(height), float(y2)))
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 1.0 or box_h <= 1.0:
            continue
        xc = (x1 + x2) * 0.5 / float(width)
        yc = (y1 + y2) * 0.5 / float(height)
        bw = box_w / float(width)
        bh = box_h / float(height)
        labels.append(f"{int(box.cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return labels


def _is_body_box(box: DetBox) -> bool:
    name = str(box.cls_name).lower()
    return int(box.cls_id) in BODY_CLASS_IDS or name.endswith("_body") or name == "body"


def _is_head_box(box: DetBox) -> bool:
    name = str(box.cls_name).lower()
    return int(box.cls_id) in HEAD_CLASS_IDS or name == "head" or name.endswith("_head")


def _has_low_conf_body(body_boxes: list[DetBox], min_conf: float, max_conf: float) -> bool:
    return any(min_conf <= float(box.conf) <= max_conf for box in body_boxes)


def _has_t_ct_conflict(body_boxes: list[DetBox], iou_threshold: float) -> bool:
    for index, first in enumerate(body_boxes):
        first_cls = _body_team_id(first)
        if first_cls is None:
            continue
        for second in body_boxes[index + 1 :]:
            second_cls = _body_team_id(second)
            if second_cls is None or second_cls == first_cls:
                continue
            if _box_iou(first.xyxy, second.xyxy) >= iou_threshold:
                return True
    return False


def _body_team_id(box: DetBox) -> Optional[int]:
    cls_id = int(box.cls_id)
    if cls_id in BODY_CLASS_IDS:
        return cls_id
    name = str(box.cls_name).lower()
    if name.startswith("ct"):
        return 0
    if name.startswith("t"):
        return 1
    return None


def _box_iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - inter
    return inter / union if union > 0.0 else 0.0


def _prob(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _list_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _list_label_files(labels_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in Path(labels_dir).iterdir()
        if path.is_file() and path.suffix.lower() in LABEL_SUFFIXES
    )


def _numeric_suffix(path: Path) -> Optional[int]:
    match = INDEX_RE.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def _image_prefix(image_dir: Path) -> str:
    image_dir = Path(image_dir)
    dataset_context = _dataset_image_context(image_dir)
    if dataset_context is not None:
        dataset_root, split = dataset_context
        return f"{_dataset_prefix(dataset_root)}_{_canonical_split(split)}"
    if image_dir.name.lower() == "images":
        parent = image_dir.parent
        return _safe_filename_part(parent.name)
    return _safe_filename_part(image_dir.name)


def _image_filename(image_dir: Path, index: int) -> str:
    return f"{_image_prefix(image_dir)}_{index:06d}.jpg"


def _sibling_labels_dir(image_dir: Path) -> Path:
    return Path(image_dir).parent / "labels"


def _dataset_image_context(image_dir: Path) -> Optional[tuple[Path, str]]:
    image_dir = Path(image_dir)
    if image_dir.name.lower() != "images":
        return None
    split_name = image_dir.parent.name.lower()
    if split_name == STAGING_SPLIT:
        return image_dir.parent.parent, STAGING_SPLIT
    if _is_split_name(split_name):
        return image_dir.parent.parent, _canonical_split(split_name)
    return None


def _find_next_pair_index(image_dir: Path, labels_dir: Path) -> int:
    highest = -1
    for folder, suffixes in ((image_dir, IMAGE_SUFFIXES), (labels_dir, {".txt"})):
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            index = _numeric_suffix(path)
            if index is not None:
                highest = max(highest, index)
    return highest + 1


def _find_next_dataset_split_index(dataset_root: Path, split: str) -> int:
    return _find_next_pair_index(
        _split_images_dir(dataset_root, split, prefer_existing=True),
        _split_labels_dir(dataset_root, split, prefer_existing=True),
    )


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", str(value)).strip("._")
    return cleaned or "images"
