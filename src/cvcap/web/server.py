from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cvcap.app.cli import parse_args
from cvcap.core.config import RunnerArgs
from cvcap.core.config_store import ConfigStore, DEFAULT_CONFIG_PATH, PROJECT_ROOT
from cvcap.web.forms import FIELD_GROUPS, config_to_payload
from cvcap.web.process_manager import PipelineProcessManager

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_STORE = ConfigStore(DEFAULT_CONFIG_PATH)
PROCESS_MANAGER = PipelineProcessManager(PROJECT_ROOT, CONFIG_STORE.path)

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
    return {"ok": True, "config": config_to_payload(config), "config_path": str(CONFIG_STORE.path)}


@app.post("/api/run")
def run_pipeline(payload: dict[str, Any]):
    config = payload_to_config(payload)
    CONFIG_STORE.save(config)
    return {"ok": True, "status": PROCESS_MANAGER.start(), "config": config_to_payload(config)}


@app.post("/api/stop")
def stop_pipeline():
    return {"ok": True, "status": PROCESS_MANAGER.stop()}


@app.post("/api/preview-args")
def preview_args(payload: dict[str, Any]):
    config = payload_to_config(payload)
    cli_args = config_to_cli_args(config)
    parsed = parse_args(cli_args)
    return {"ok": True, "args": config_to_payload(parsed), "cli": cli_args}


def payload_to_config(payload: dict[str, Any]) -> RunnerArgs:
    base = config_to_payload(RunnerArgs())
    for key in list(base):
        if key in payload:
            base[key] = payload[key]

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
        base["save_every"] = float(base["save_every"])
        base["save_queue"] = int(base["save_queue"])
        base["stats_interval"] = float(base["stats_interval"])
        base["max_run_seconds"] = float(base["max_run_seconds"])
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
        "--save-every", str(config.save_every),
        "--stats-interval", str(config.stats_interval),
        "--max-run-seconds", str(config.max_run_seconds),
        "--save-queue", str(config.save_queue),
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
    if config.visualize:
        args.append("--vis")
    if config.smooth:
        args.append("--smooth")
    if config.jsonl_log:
        args.append("--jsonl-log")
    return args
