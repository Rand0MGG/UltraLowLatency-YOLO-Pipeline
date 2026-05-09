from __future__ import annotations

from dataclasses import fields
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cvcap.app.cli import parse_args
from cvcap.core.config import RunnerArgs
from cvcap.core.config_store import ConfigStore, DEFAULT_CONFIG_PATH, PROJECT_ROOT
from cvcap.web.forms import (
    DEFAULT_AUTO_LABEL_TEST_RATIO,
    DEFAULT_AUTO_LABEL_TRAIN_RATIO,
    DEFAULT_AUTO_LABEL_VAL_RATIO,
    FIELD_GROUPS,
    config_to_payload,
)
from cvcap.web.process_manager import PipelineProcessManager

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_STORE = ConfigStore(DEFAULT_CONFIG_PATH)
PROCESS_MANAGER = PipelineProcessManager(PROJECT_ROOT, CONFIG_STORE.path)
DATASETS_ROOT = PROJECT_ROOT / "datasets"
AUTO_LABEL_STAGING_DIR = "staging"
AUTO_LABEL_IMAGES_DIR = "images"
AUTO_LABEL_LABELS_DIR = "labels"
_HELPER_MISSING = object()

app = FastAPI(title="cvcap Control Panel")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/bootstrap")
def bootstrap():
    config = CONFIG_STORE.load()
    return {
        "config": config_to_payload(config),
        "defaults": config_to_payload(RunnerArgs()),
        "models": CONFIG_STORE.available_models(),
        "field_groups": FIELD_GROUPS,
        "status": PROCESS_MANAGER.status(),
        "config_path": str(CONFIG_STORE.path),
    }


@app.get("/api/status")
def status():
    return PROCESS_MANAGER.status()


@app.post("/api/config")
def save_config(payload: dict[str, Any]):
    config = payload_to_config(payload)
    CONFIG_STORE.save(config)
    return {"ok": True, "config": _config_response_payload(config, payload), "config_path": str(CONFIG_STORE.path)}


@app.post("/api/run")
def run_pipeline(payload: dict[str, Any]):
    config = payload_to_config(payload)
    CONFIG_STORE.save(config)
    return {"ok": True, "status": PROCESS_MANAGER.start(), "config": _config_response_payload(config, payload)}


@app.post("/api/stop")
def stop_pipeline():
    return {"ok": True, "status": PROCESS_MANAGER.stop()}


@app.post("/api/preview-args")
def preview_args(payload: dict[str, Any]):
    config = payload_to_config(payload)
    cli_args = config_to_cli_args(config)
    parsed = parse_args(cli_args)
    return {"ok": True, "args": config_to_payload(parsed), "cli": cli_args}


@app.post("/api/pick-folder")
def pick_folder(payload: dict[str, Any]):
    initial_dir = _project_path(str(payload.get("initial_dir") or ""))
    title = str(payload.get("title") or "Select Folder")
    try:
        selected = _ask_directory(title=title, initial_dir=initial_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not open folder picker: {exc}") from exc
    return {"ok": True, "path": selected}


@app.post("/api/auto-label/prepare")
def prepare_auto_label_dataset(payload: dict[str, Any]):
    _require_pipeline_idle()
    config = payload_to_config(payload)
    operation_path, dataset_root = _payload_dataset_operation_path(payload, config=config)
    split = _payload_split_ratios(payload)
    try:
        from cvcap.runtime import auto_label as auto_label_ops

        result = _call_auto_label_helper(
            auto_label_ops,
            ("prepare_staged_dataset", "prepare_split_dataset", "prepare_auto_label_dataset", "split_auto_label_dataset", "prepare_dataset", "split_dataset"),
            operation_path,
            train_ratio=split["train"],
            val_ratio=split["val"],
            test_ratio=split["test"],
            split_ratios=split,
        )
        if result is _HELPER_MISSING:
            result = _fallback_prepare_dataset(auto_label_ops, dataset_root, split)
        annotate_result = _call_auto_label_helper(
            auto_label_ops,
            ("auto_annotate_dataset", "annotate_auto_label_dataset", "annotate_dataset", "auto_annotate_dataset_root"),
            dataset_root,
            args=config,
            config=config,
            overwrite=False,
        )
        if annotate_result is not _HELPER_MISSING:
            result = {**_result_dict(result), "annotation": annotate_result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset prepare failed: {exc}") from exc
    return _operation_response(result, dataset_root=dataset_root, split=split)


@app.post("/api/auto-label/shuffle")
def shuffle_auto_label_images(payload: dict[str, Any]):
    return prepare_auto_label_dataset(payload)


@app.post("/api/auto-label/annotate")
def annotate_auto_label_dataset(payload: dict[str, Any]):
    _require_pipeline_idle()
    config = payload_to_config(payload)
    dataset_root = _payload_dataset_root(payload, config=config)
    try:
        from cvcap.runtime import auto_label as auto_label_ops

        result = _call_auto_label_helper(
            auto_label_ops,
            ("auto_annotate_dataset", "annotate_auto_label_dataset", "annotate_dataset", "auto_annotate_dataset_root"),
            dataset_root,
            args=config,
            config=config,
        )
        if result is _HELPER_MISSING:
            result = auto_label_ops.auto_annotate_image_folder(_dataset_operation_image_dir(dataset_root), config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto annotation failed: {exc}") from exc
    return _operation_response(result, dataset_root=dataset_root)


@app.post("/api/auto-label/merge")
def merge_auto_label_datasets(payload: dict[str, Any]):
    _require_pipeline_idle()
    config = payload_to_config(payload)
    source_dataset_root = _payload_dataset_root(
        payload,
        config=config,
        keys=("source_dataset_root", "source_image_dir"),
        label="Source dataset root",
    )
    target_dataset_root = _payload_dataset_root(
        payload,
        keys=("target_dataset_root", "target_image_dir"),
        label="Target dataset root",
    )
    try:
        from cvcap.runtime import auto_label as auto_label_ops

        result = _call_auto_label_helper(
            auto_label_ops,
            ("merge_labeled_dataset", "merge_auto_label_dataset", "merge_dataset", "merge_labeled_dataset_root"),
            source_dataset_root,
            target_dataset_root,
            target_dataset_path=target_dataset_root,
            source_dataset_path=source_dataset_root,
        )
        if result is _HELPER_MISSING:
            result = auto_label_ops.merge_labeled_image_folder(
                source_image_dir=_dataset_operation_image_dir(source_dataset_root),
                target_image_dir=_dataset_operation_image_dir(target_dataset_root),
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset merge failed: {exc}") from exc
    return _operation_response(result, source_dataset_root=source_dataset_root, target_dataset_root=target_dataset_root)


@app.post("/api/auto-label/normalize")
def normalize_auto_label_dataset(payload: dict[str, Any]):
    _require_pipeline_idle()
    config = payload_to_config(payload)
    dataset_root = _payload_dataset_root(payload, config=config)
    try:
        from cvcap.runtime import auto_label as auto_label_ops

        result = _call_auto_label_helper(
            auto_label_ops,
            ("normalize_dataset_filenames", "normalize_auto_label_dataset", "normalize_dataset"),
            dataset_root,
        )
        if result is _HELPER_MISSING:
            raise ValueError("Dataset filename normalization helper is unavailable.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset normalize failed: {exc}") from exc
    return _operation_response(result, dataset_root=dataset_root)


@app.post("/api/auto-label/shuffle-split")
def shuffle_auto_label_split(payload: dict[str, Any]):
    _require_pipeline_idle()
    config = payload_to_config(payload)
    operation_path, dataset_root = _payload_dataset_operation_path(payload, config=config)
    try:
        from cvcap.runtime import auto_label as auto_label_ops

        result = _call_auto_label_helper(
            auto_label_ops,
            ("shuffle_dataset_split", "shuffle_auto_label_split", "shuffle_split"),
            operation_path,
        )
        if result is _HELPER_MISSING:
            raise ValueError("Dataset split shuffle helper is unavailable.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset split shuffle failed: {exc}") from exc
    return _operation_response(result, dataset_root=dataset_root)


def payload_to_config(payload: dict[str, Any]) -> RunnerArgs:
    base = config_to_payload(RunnerArgs())
    for key in list(base):
        if key in payload:
            base[key] = payload[key]

    dataset_root = str(payload.get("auto_label_dataset_root") or "").strip()
    if dataset_root:
        base["auto_label_dir"] = str(_dataset_staging_image_dir(Path(dataset_root)))

    try:
        base["monitor_index"] = int(base["monitor_index"])
        base["capture_hz"] = float(base["capture_hz"])
        base["model"] = str(base["model"])
        base["device"] = str(base["device"])
        base["conf"] = float(base["conf"])
        base["iou"] = float(base["iou"])
        base["imgsz"] = int(base["imgsz"])
        base["half"] = bool(base["half"])
        base["end2end"] = bool(base["end2end"])
        base["yolo_classes"] = normalize_yolo_classes(base["yolo_classes"])
        base["yolo_max_det"] = int(base["yolo_max_det"])
        base["save_queue"] = int(base["save_queue"])
        base["demo_capture"] = bool(base["demo_capture"])
        base["demo_capture_dir"] = str(base["demo_capture_dir"])
        base["demo_capture_interval_s"] = max(0.1, float(base["demo_capture_interval_s"]))
        base["demo_capture_require_boxes"] = bool(base["demo_capture_require_boxes"])
        base["stats_interval"] = float(base["stats_interval"])
        base["max_run_seconds"] = float(base["max_run_seconds"])
        base["auto_label"] = bool(base["auto_label"])
        base["auto_label_dir"] = str(base["auto_label_dir"])
        base["auto_label_incomplete_enabled"] = bool(base["auto_label_incomplete_enabled"])
        base["auto_label_incomplete_prob"] = float(base["auto_label_incomplete_prob"])
        base["auto_label_complete_enabled"] = bool(base["auto_label_complete_enabled"])
        base["auto_label_both_prob"] = float(base["auto_label_both_prob"])
        base["auto_label_empty_enabled"] = bool(base["auto_label_empty_enabled"])
        base["auto_label_empty_prob"] = float(base["auto_label_empty_prob"])
        base["auto_label_low_conf_enabled"] = bool(base["auto_label_low_conf_enabled"])
        base["auto_label_low_conf_prob"] = float(base["auto_label_low_conf_prob"])
        base["auto_label_low_conf_min"] = float(base["auto_label_low_conf_min"])
        base["auto_label_low_conf_max"] = float(base["auto_label_low_conf_max"])
        base["auto_label_conflict_enabled"] = bool(base["auto_label_conflict_enabled"])
        base["auto_label_conflict_prob"] = float(base["auto_label_conflict_prob"])
        base["auto_label_conflict_iou"] = float(base["auto_label_conflict_iou"])
        base["auto_label_flip_enabled"] = bool(base["auto_label_flip_enabled"])
        base["auto_label_flip_prob"] = float(base["auto_label_flip_prob"])
        base["auto_label_flip_iou"] = float(base["auto_label_flip_iou"])
        base["auto_label_flip_max_age_s"] = float(base["auto_label_flip_max_age_s"])
        base["auto_label_min_interval_s"] = max(1.0, float(base["auto_label_min_interval_s"]))
        base["roi_square"] = bool(base["roi_square"])
        base["roi_radius_px"] = int(base["roi_radius_px"])
        base["auto_capture"] = bool(base["auto_capture"])
        base["auto_capture_warmup_s"] = float(base["auto_capture_warmup_s"])
        base["cap_min_hz"] = float(base["cap_min_hz"])
        base["cap_max_hz"] = float(base["cap_max_hz"])
        base["target_drop_fps"] = float(base["target_drop_fps"])
        base["deadband"] = float(base["deadband"])
        base["kp"] = float(base["kp"])
        base["ki"] = float(base["ki"])
        base["integral_limit"] = float(base["integral_limit"])
        base["min_apply_delta_hz"] = float(base["min_apply_delta_hz"])
        base["visualize"] = bool(base["visualize"])
        base["smooth"] = bool(base["smooth"])
        base["smooth_alpha"] = float(base["smooth_alpha"])
        base["jsonl_log"] = bool(base["jsonl_log"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config value: {exc}") from exc

    runner_field_names = {field.name for field in fields(RunnerArgs)}
    return RunnerArgs(**{key: value for key, value in base.items() if key in runner_field_names})


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _config_response_payload(config: RunnerArgs, request_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = config_to_payload(config)
    if not request_payload:
        return payload
    for key in ("auto_label_train_ratio", "auto_label_val_ratio", "auto_label_test_ratio"):
        if key in request_payload:
            payload[key] = request_payload[key]
    return payload


def _dataset_staging_image_dir(dataset_root: Path) -> Path:
    return dataset_root / AUTO_LABEL_STAGING_DIR / AUTO_LABEL_IMAGES_DIR


def _dataset_root_from_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() in {".yaml", ".yml", ".txt"}:
        path = path.parent
    parts = tuple(part.lower() for part in path.parts)
    if len(parts) >= 2 and parts[-2:] == (AUTO_LABEL_STAGING_DIR, AUTO_LABEL_IMAGES_DIR):
        return path.parent.parent
    if path.name.lower() in {AUTO_LABEL_IMAGES_DIR, "image", AUTO_LABEL_LABELS_DIR}:
        parent_name = path.parent.name.lower()
        if parent_name in {AUTO_LABEL_STAGING_DIR, "train", "val", "valid", "test"}:
            return path.parent.parent
        return path.parent
    if path.name.lower() in {AUTO_LABEL_STAGING_DIR, "train", "val", "valid", "test"}:
        return path.parent
    return path


def _payload_dataset_root(
    payload: dict[str, Any],
    config: RunnerArgs | None = None,
    keys: tuple[str, ...] = ("dataset_root", "auto_label_dataset_root", "image_dir"),
    label: str = "Dataset root",
) -> Path:
    raw = ""
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            raw = value
            break
    if not raw and config is not None:
        raw = str(config.auto_label_dir).strip()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label} is required.")
    dataset_root = _project_path(str(_dataset_root_from_path(raw)))
    _require_dataset_root_path(dataset_root, label)
    return dataset_root


def _payload_dataset_operation_path(
    payload: dict[str, Any],
    config: RunnerArgs | None = None,
    keys: tuple[str, ...] = ("dataset_root", "auto_label_dataset_root", "image_dir"),
    label: str = "Dataset root",
) -> tuple[Path, Path]:
    raw = ""
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            raw = value
            break
    if not raw and config is not None:
        raw = str(config.auto_label_dir).strip()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label} is required.")

    operation_path = _project_path(raw)
    dataset_root = _project_path(str(_dataset_root_from_path(raw)))
    _require_dataset_root_path(dataset_root, label)
    _require_dataset_root_path(operation_path, f"{label} operation path")
    return operation_path, dataset_root


def _require_dataset_root_path(dataset_root: Path, label: str) -> None:
    try:
        dataset_root.resolve().relative_to(DATASETS_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must be inside {DATASETS_ROOT}.") from exc


def _dataset_operation_image_dir(dataset_root: Path) -> Path:
    staging_images = _dataset_staging_image_dir(dataset_root)
    legacy_images = dataset_root / AUTO_LABEL_IMAGES_DIR
    if staging_images.exists() or not legacy_images.exists():
        return staging_images
    return legacy_images


def _payload_split_ratios(payload: dict[str, Any]) -> dict[str, float]:
    train = _payload_float(payload, "auto_label_train_ratio", DEFAULT_AUTO_LABEL_TRAIN_RATIO)
    val = _payload_float(payload, "auto_label_val_ratio", DEFAULT_AUTO_LABEL_VAL_RATIO)
    test = _payload_float(payload, "auto_label_test_ratio", DEFAULT_AUTO_LABEL_TEST_RATIO)
    total = train + val + test
    if train < 0 or val < 0 or test < 0:
        raise HTTPException(status_code=400, detail="Split ratios must be non-negative.")
    if total <= 0:
        raise HTTPException(status_code=400, detail="At least one split ratio must be greater than zero.")
    return {"train": train, "val": val, "test": test}


def _payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {key}: {exc}") from exc


def _call_auto_label_helper(auto_label_ops, names: tuple[str, ...], *args, **kwargs):
    for name in names:
        helper = getattr(auto_label_ops, name, None)
        if callable(helper):
            return _call_with_supported_kwargs(helper, *args, **kwargs)
    return _HELPER_MISSING


def _call_with_supported_kwargs(helper, *args, **kwargs):
    try:
        parameters = signature(helper).parameters
    except (TypeError, ValueError):
        return helper(*args, **kwargs)
    if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return helper(*args, **kwargs)
    consumed_by_args = set(list(parameters)[: len(args)])
    supported = {key: value for key, value in kwargs.items() if key in parameters and key not in consumed_by_args}
    return helper(*args, **supported)


def _fallback_prepare_dataset(auto_label_ops, dataset_root: Path, split: dict[str, float]) -> dict[str, Any]:
    image_dir = _dataset_operation_image_dir(dataset_root)
    labels_dir = image_dir.parent / AUTO_LABEL_LABELS_DIR
    if not image_dir.exists():
        image_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        return {
            "images": 0,
            "split": False,
            "fallback": "created_staging",
            "staging_image_dir": str(image_dir),
            "staging_labels_dir": str(labels_dir),
            "train_ratio": split["train"],
            "val_ratio": split["val"],
            "test_ratio": split["test"],
        }
    result = auto_label_ops.shuffle_rename_image_folder(image_dir)
    return {
        **result,
        "split": False,
        "fallback": "shuffle_rename_image_folder",
        "staging_image_dir": str(image_dir),
        "staging_labels_dir": str(labels_dir),
        "train_ratio": split["train"],
        "val_ratio": split["val"],
        "test_ratio": split["test"],
    }


def _operation_response(result: Any, **extra: Any) -> dict[str, Any]:
    payload = _result_dict(result)
    for key, value in extra.items():
        payload.setdefault(key, value)
    return {"ok": True, **_json_ready(payload)}


def _result_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return dict(result)
    return {"result": result}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _require_pipeline_idle() -> None:
    if PROCESS_MANAGER.status().get("running"):
        raise HTTPException(status_code=409, detail="Stop the pipeline before running dataset file operations.")


def _ask_directory(title: str, initial_dir: Path) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        selected = filedialog.askdirectory(
            parent=root,
            title=title,
            initialdir=str(_existing_dir(initial_dir)),
            mustexist=False,
        )
        return str(Path(selected)) if selected else ""
    finally:
        root.destroy()


def _existing_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() and path.is_dir():
        return path
    for parent in path.parents:
        if parent.exists() and parent.is_dir():
            return parent
    return PROJECT_ROOT


def normalize_yolo_classes(value: Any):
    if value in ("", None, "none", "all"):
        return None if value in ("", "none", "all") else value
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(int(part) for part in value)
    return value


def config_to_cli_args(config: RunnerArgs) -> list[str]:
    args = [
        "--monitor-index", str(config.monitor_index),
        "--capture-hz", str(config.capture_hz),
        "--model", config.model,
        "--device", config.device,
        "--conf", str(config.conf),
        "--iou", str(config.iou),
        "--imgsz", str(config.imgsz),
        "--end2end" if config.end2end else "--no-end2end",
        "--yolo-max-det", str(config.yolo_max_det),
        "--stats-interval", str(config.stats_interval),
        "--max-run-seconds", str(config.max_run_seconds),
        "--save-queue", str(config.save_queue),
        "--demo-capture-dir", config.demo_capture_dir,
        "--demo-capture-interval-s", str(config.demo_capture_interval_s),
        "--demo-capture-require-boxes" if config.demo_capture_require_boxes else "--no-demo-capture-require-boxes",
        "--auto-label-dir", config.auto_label_dir,
        "--auto-label-incomplete-prob", str(config.auto_label_incomplete_prob),
        "--auto-label-both-prob", str(config.auto_label_both_prob),
        "--auto-label-empty-prob", str(config.auto_label_empty_prob),
        "--auto-label-low-conf-prob", str(config.auto_label_low_conf_prob),
        "--auto-label-low-conf-min", str(config.auto_label_low_conf_min),
        "--auto-label-low-conf-max", str(config.auto_label_low_conf_max),
        "--auto-label-conflict-prob", str(config.auto_label_conflict_prob),
        "--auto-label-conflict-iou", str(config.auto_label_conflict_iou),
        "--auto-label-flip-prob", str(config.auto_label_flip_prob),
        "--auto-label-flip-iou", str(config.auto_label_flip_iou),
        "--auto-label-flip-max-age-s", str(config.auto_label_flip_max_age_s),
        "--auto-label-min-interval-s", str(config.auto_label_min_interval_s),
        "--roi-radius", str(config.roi_radius_px),
        "--auto-capture-warmup-s", str(config.auto_capture_warmup_s),
        "--cap-min-hz", str(config.cap_min_hz),
        "--cap-max-hz", str(config.cap_max_hz),
        "--target-drop-fps", str(config.target_drop_fps),
        "--deadband", str(config.deadband),
        "--kp", str(config.kp),
        "--ki", str(config.ki),
        "--integral-limit", str(config.integral_limit),
        "--min-apply-delta-hz", str(config.min_apply_delta_hz),
        "--smooth-alpha", str(config.smooth_alpha),
    ]
    if config.half:
        args.append("--half")
    if config.yolo_classes is not None:
        args.extend(["--yolo-classes", ",".join(str(part) for part in config.yolo_classes)])
    if config.roi_square:
        args.append("--roi-square")
    if config.auto_capture:
        args.append("--auto-capture")
    if config.auto_label:
        args.append("--auto-label")
    args.append("--demo-capture" if config.demo_capture else "--no-demo-capture")
    args.append("--auto-label-incomplete-enabled" if config.auto_label_incomplete_enabled else "--no-auto-label-incomplete-enabled")
    args.append("--auto-label-complete-enabled" if config.auto_label_complete_enabled else "--no-auto-label-complete-enabled")
    args.append("--auto-label-empty-enabled" if config.auto_label_empty_enabled else "--no-auto-label-empty-enabled")
    args.append("--auto-label-low-conf-enabled" if config.auto_label_low_conf_enabled else "--no-auto-label-low-conf-enabled")
    args.append("--auto-label-conflict-enabled" if config.auto_label_conflict_enabled else "--no-auto-label-conflict-enabled")
    args.append("--auto-label-flip-enabled" if config.auto_label_flip_enabled else "--no-auto-label-flip-enabled")
    if config.visualize:
        args.append("--vis")
    if config.smooth:
        args.append("--smooth")
    if config.jsonl_log:
        args.append("--jsonl-log")
    return args
