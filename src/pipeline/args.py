# src/pipeline/args.py
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class RunnerArgs:
    monitor_index: int = 1
    capture_hz: float = 60.0

    model: str = "yolo11n.pt"
    device: str = "cuda:0"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 736
    half: bool = False
    yolo_classes: Optional[Tuple[int, ...]] = None
    yolo_max_det: int = 20

    queue_size: int = 1

    save_every: float = 0.0
    save_queue: int = 8

    stats_interval: float = 1.0
    max_run_seconds: float = 0.0

    # ROI
    roi_square: bool = False
    roi_radius_px: int = 368

    # PI auto capture
    auto_capture: bool = False
    # [新增] 自动控制热身时间，防止启动初期的波动误导 PID
    auto_capture_warmup_s: float = 5.0  
    
    cap_min_hz: float = 10.0
    cap_max_hz: float = 240.0

    target_drop_fps: float = 4.0
    deadband: float = 1.0

    kp: float = 0.6
    ki: float = 0.03

    integral_limit: float = 15.0
    min_apply_delta_hz: float = 0.5
    
    # Visualization & Smooth
    visualize: bool = False
    smooth: bool = False
    smooth_alpha: float = 0.6