from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path

from cvcap.core.config import RunnerArgs
from cvcap.core.config_store import ConfigStore, DEFAULT_CONFIG_PATH
from cvcap.core.errors import PipelineError


def build_parser(defaults: RunnerArgs) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-Performance YOLO26 Pose Pipeline", allow_abbrev=False)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the persistent config file")
    parser.add_argument("--save-config", action="store_true", help="Save the resolved arguments back into the config file before running")
    parser.add_argument("--monitor-index", type=int, default=defaults.monitor_index, help="Monitor index (1-based)")
    parser.add_argument("--capture-hz", type=float, default=defaults.capture_hz, help="Target capture FPS")
    parser.add_argument("--model", type=str, default=defaults.model, help="Path to YOLO26 .pt or .engine file")
    parser.add_argument("--device", type=str, default=defaults.device, help="CUDA device or cpu")
    parser.add_argument("--conf", type=float, default=defaults.conf, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=defaults.iou, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz, help="Inference image size")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=defaults.half, help="Use FP16 half precision")
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=defaults.end2end, help="Use YOLO26 end-to-end / NMS-free prediction")
    parser.add_argument("--yolo-classes", type=str, default=defaults.yolo_classes, help="Class filter, for example '0,2,3'")
    parser.add_argument("--yolo-max-det", type=int, default=defaults.yolo_max_det, help="Maximum boxes kept after NMS")
    parser.add_argument("--save-every", type=float, default=defaults.save_every)
    parser.add_argument("--stats-interval", type=float, default=defaults.stats_interval)
    parser.add_argument("--max-run-seconds", type=float, default=defaults.max_run_seconds)
    parser.add_argument("--save-queue", type=int, default=defaults.save_queue)
    parser.add_argument("--auto-label", action=argparse.BooleanOptionalAction, default=defaults.auto_label, help="Save useful ROI images only; labels can be generated later from the Web panel")
    parser.add_argument("--auto-label-dir", type=str, default=defaults.auto_label_dir, help="Dataset root or staging image folder for auto-captured samples")
    parser.add_argument("--auto-label-incomplete-enabled", action=argparse.BooleanOptionalAction, default=defaults.auto_label_incomplete_enabled)
    parser.add_argument("--auto-label-incomplete-prob", type=float, default=defaults.auto_label_incomplete_prob)
    parser.add_argument("--auto-label-complete-enabled", action=argparse.BooleanOptionalAction, default=defaults.auto_label_complete_enabled)
    parser.add_argument("--auto-label-both-prob", type=float, default=defaults.auto_label_both_prob, help="Save probability when both body and head are detected")
    parser.add_argument("--auto-label-empty-enabled", action=argparse.BooleanOptionalAction, default=defaults.auto_label_empty_enabled)
    parser.add_argument("--auto-label-empty-prob", type=float, default=defaults.auto_label_empty_prob, help="Save probability when no boxes are detected")
    parser.add_argument("--auto-label-low-conf-enabled", action=argparse.BooleanOptionalAction, default=defaults.auto_label_low_conf_enabled)
    parser.add_argument("--auto-label-low-conf-prob", type=float, default=defaults.auto_label_low_conf_prob)
    parser.add_argument("--auto-label-low-conf-min", type=float, default=defaults.auto_label_low_conf_min)
    parser.add_argument("--auto-label-low-conf-max", type=float, default=defaults.auto_label_low_conf_max)
    parser.add_argument("--auto-label-conflict-enabled", action=argparse.BooleanOptionalAction, default=defaults.auto_label_conflict_enabled)
    parser.add_argument("--auto-label-conflict-prob", type=float, default=defaults.auto_label_conflict_prob)
    parser.add_argument("--auto-label-conflict-iou", type=float, default=defaults.auto_label_conflict_iou)
    parser.add_argument("--auto-label-flip-enabled", action=argparse.BooleanOptionalAction, default=defaults.auto_label_flip_enabled)
    parser.add_argument("--auto-label-flip-prob", type=float, default=defaults.auto_label_flip_prob)
    parser.add_argument("--auto-label-flip-iou", type=float, default=defaults.auto_label_flip_iou)
    parser.add_argument("--auto-label-flip-max-age-s", type=float, default=defaults.auto_label_flip_max_age_s)
    parser.add_argument("--auto-label-min-interval-s", type=float, default=defaults.auto_label_min_interval_s, help="Minimum seconds between auto-label saves")
    parser.add_argument("--roi-square", action=argparse.BooleanOptionalAction, default=defaults.roi_square, help="Crop center square")
    parser.add_argument("--roi-radius", type=int, default=defaults.roi_radius_px, help="Radius of ROI")
    parser.add_argument("--auto-capture", action=argparse.BooleanOptionalAction, default=defaults.auto_capture, help="Adapt capture FPS based on inference latency")
    parser.add_argument("--auto-capture-warmup-s", type=float, default=defaults.auto_capture_warmup_s)
    parser.add_argument("--cap-min-hz", type=float, default=defaults.cap_min_hz)
    parser.add_argument("--cap-max-hz", type=float, default=defaults.cap_max_hz)
    parser.add_argument("--target-drop-fps", type=float, default=defaults.target_drop_fps)
    parser.add_argument("--deadband", type=float, default=defaults.deadband)
    parser.add_argument("--kp", type=float, default=defaults.kp)
    parser.add_argument("--ki", type=float, default=defaults.ki)
    parser.add_argument("--integral-limit", type=float, default=defaults.integral_limit)
    parser.add_argument("--min-apply-delta-hz", type=float, default=defaults.min_apply_delta_hz)
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=defaults.visualize, help="Enable transparent overlay visualization")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=defaults.smooth, help="Enable Kalman box smoothing")
    parser.add_argument("--smooth-alpha", type=float, default=defaults.smooth_alpha)
    parser.add_argument("--jsonl-log", action=argparse.BooleanOptionalAction, default=defaults.jsonl_log, help="Write per-frame JSONL inference records")
    return parser


@dataclass(frozen=True)
class CliRequest:
    args: RunnerArgs
    config_path: Path
    save_config: bool


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    return parse_request(argv).args


def parse_request(argv: list[str] | None = None) -> CliRequest:
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_ns, _ = pre_parser.parse_known_args(argv)

    config_path = Path(pre_ns.config)
    defaults = ConfigStore(config_path).load()
    ns = build_parser(defaults).parse_args(argv)
    args = RunnerArgs(
        monitor_index=ns.monitor_index,
        capture_hz=ns.capture_hz,
        model=ns.model,
        device=ns.device,
        conf=ns.conf,
        iou=ns.iou,
        imgsz=ns.imgsz,
        half=ns.half,
        end2end=ns.end2end,
        yolo_classes=_parse_yolo_classes(ns.yolo_classes, defaults.yolo_classes),
        yolo_max_det=ns.yolo_max_det,
        save_every=ns.save_every,
        stats_interval=ns.stats_interval,
        max_run_seconds=ns.max_run_seconds,
        save_queue=ns.save_queue,
        auto_label=ns.auto_label,
        auto_label_dir=ns.auto_label_dir,
        auto_label_incomplete_enabled=ns.auto_label_incomplete_enabled,
        auto_label_incomplete_prob=ns.auto_label_incomplete_prob,
        auto_label_complete_enabled=ns.auto_label_complete_enabled,
        auto_label_both_prob=ns.auto_label_both_prob,
        auto_label_empty_enabled=ns.auto_label_empty_enabled,
        auto_label_empty_prob=ns.auto_label_empty_prob,
        auto_label_low_conf_enabled=ns.auto_label_low_conf_enabled,
        auto_label_low_conf_prob=ns.auto_label_low_conf_prob,
        auto_label_low_conf_min=ns.auto_label_low_conf_min,
        auto_label_low_conf_max=ns.auto_label_low_conf_max,
        auto_label_conflict_enabled=ns.auto_label_conflict_enabled,
        auto_label_conflict_prob=ns.auto_label_conflict_prob,
        auto_label_conflict_iou=ns.auto_label_conflict_iou,
        auto_label_flip_enabled=ns.auto_label_flip_enabled,
        auto_label_flip_prob=ns.auto_label_flip_prob,
        auto_label_flip_iou=ns.auto_label_flip_iou,
        auto_label_flip_max_age_s=ns.auto_label_flip_max_age_s,
        auto_label_min_interval_s=ns.auto_label_min_interval_s,
        roi_square=bool(ns.roi_square),
        roi_radius_px=int(ns.roi_radius),
        auto_capture=ns.auto_capture,
        auto_capture_warmup_s=ns.auto_capture_warmup_s,
        cap_min_hz=ns.cap_min_hz,
        cap_max_hz=ns.cap_max_hz,
        target_drop_fps=ns.target_drop_fps,
        deadband=ns.deadband,
        kp=ns.kp,
        ki=ns.ki,
        integral_limit=ns.integral_limit,
        min_apply_delta_hz=ns.min_apply_delta_hz,
        visualize=ns.vis,
        smooth=ns.smooth,
        smooth_alpha=ns.smooth_alpha,
        jsonl_log=ns.jsonl_log,
    )
    return CliRequest(args=args, config_path=Path(ns.config), save_config=bool(ns.save_config))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        request = parse_request(argv)
        if request.save_config:
            ConfigStore(request.config_path).save(request.args)
        from cvcap.app.runtime import run_from_args

        return run_from_args(request.args)
    except PipelineError as exc:
        logging.error("%s", exc)
        return 2


def _parse_yolo_classes(raw_value, fallback):
    if raw_value is None:
        return fallback
    if not isinstance(raw_value, str):
        return fallback
    text = raw_value.strip().lower()
    if text in ("", "none", "all"):
        return None
    return tuple(int(part) for part in text.split(","))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
