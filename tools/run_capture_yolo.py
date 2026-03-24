# -*- coding: utf-8 -*-
"""
tools/run_capture_yolo.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import multiprocessing
from pathlib import Path


def _setup_import_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(src))


_setup_import_path()

from pipeline.args import RunnerArgs
from pipeline.runner_mp import run_from_args


def main() -> int:
    # 1. 实例化默认参数对象，作为“单一事实来源”
    DEF = RunnerArgs()

    parser = argparse.ArgumentParser(description="High-Performance YOLO Pipeline")

    # -------- Capture --------
    parser.add_argument("--monitor-index", type=int, default=DEF.monitor_index, help="Monitor index (1-based)")
    parser.add_argument("--capture-hz", type=float, default=DEF.capture_hz, help="Target capture FPS")

    # -------- YOLO --------
    parser.add_argument("--model", type=str, default=DEF.model, help="Path to .pt or .engine file")
    parser.add_argument("--device", type=str, default=DEF.device, help="Cuda device / cpu")
    parser.add_argument("--conf", type=float, default=DEF.conf, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEF.iou, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=DEF.imgsz, help="Inference image size")
    parser.add_argument("--half", action="store_true", help="Use FP16 half precision")
    parser.add_argument("--yolo-classes",type=str,default=DEF.yolo_classes,help="类别过滤，例如 '0,2,3' 只保留这些 class；")
    parser.add_argument("--yolo-max-det",type=int,default=DEF.yolo_max_det,help="每帧 NMS 后最多保留的检测框数量")

    # -------- Output/Debug --------
    parser.add_argument("--save-every", type=float, default=DEF.save_every)
    parser.add_argument("--stats-interval", type=float, default=DEF.stats_interval)
    parser.add_argument("--max-run-seconds", type=float, default=DEF.max_run_seconds)
    parser.add_argument("--save-queue", type=int, default=DEF.save_queue)

    # -------- ROI --------
    parser.add_argument("--roi-square", action="store_true", help="Crop center square")
    # 注意：CLI 参数名是 --roi-radius，但 args.py 属性是 roi_radius_px
    parser.add_argument("--roi-radius", type=int, default=DEF.roi_radius_px, help="Radius of ROI")

    # -------- PID Auto Capture --------
    parser.add_argument("--auto-capture", action="store_true")
    parser.add_argument("--cap-min-hz", type=float, default=DEF.cap_min_hz)
    parser.add_argument("--cap-max-hz", type=float, default=DEF.cap_max_hz)
    parser.add_argument("--target-drop-fps", type=float, default=DEF.target_drop_fps)
    parser.add_argument("--deadband", type=float, default=DEF.deadband)
    parser.add_argument("--kp", type=float, default=DEF.kp)
    parser.add_argument("--ki", type=float, default=DEF.ki)
    parser.add_argument("--integral-limit", type=float, default=DEF.integral_limit)
    parser.add_argument("--min-apply-delta-hz", type=float, default=DEF.min_apply_delta_hz)

    # -------- Visualization & Smooth --------
    parser.add_argument("--vis", action="store_true", help="Enable transparent overlay visualization")
    parser.add_argument("--smooth", action="store_true", help="Enable Kalman box smoothing")
    parser.add_argument("--smooth-alpha", type=float, default=DEF.smooth_alpha)

    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # 解析 yolo_classes 字符串 -> Optional[Tuple[int, ...]]
    if ns.yolo_classes is None:
        # 没在命令行传这个参数 -> 使用 args.py 中的默认值
        yolo_classes = DEF.yolo_classes
    else:
        txt = ns.yolo_classes.strip().lower()
        if txt in ("", "none", "all"):
            yolo_classes = None   # 显式指定不过滤
        else:
            # 例如 "0,2,3" -> (0, 2, 3)
            yolo_classes = tuple(int(x) for x in txt.split(","))

    # 构建最终的 Args 对象
    args = RunnerArgs(
        monitor_index=ns.monitor_index,
        capture_hz=ns.capture_hz,
        model=ns.model,
        device=ns.device,
        conf=ns.conf,
        iou=ns.iou,
        imgsz=ns.imgsz,
        half=ns.half,
        # 新增字段
        yolo_classes=yolo_classes,
        yolo_max_det=ns.yolo_max_det,
        # 下面保持原有逻辑
        save_every=ns.save_every,
        stats_interval=ns.stats_interval,
        max_run_seconds=ns.max_run_seconds,
        save_queue=ns.save_queue,
        roi_square=bool(ns.roi_square),
        roi_radius_px=int(ns.roi_radius),
        auto_capture=ns.auto_capture,
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
    )

    return run_from_args(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
