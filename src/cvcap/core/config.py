from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class RunnerArgs:
    monitor_index: int = 1
    capture_hz: float = 60.0

    model: str = "models/yolo26l-pose.engine"
    device: str = "cuda:0"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 736
    half: bool = False
    end2end: bool = True
    yolo_classes: Optional[Tuple[int, ...]] = (0,)
    yolo_max_det: int = 20

    save_queue: int = 8
    demo_capture: bool = False
    demo_capture_dir: str = "debug/demo_frames"
    demo_capture_interval_s: float = 1.0
    demo_capture_require_boxes: bool = True
    stats_interval: float = 1.0
    max_run_seconds: float = 0.0

    auto_label: bool = False
    auto_label_dir: str = "datasets/runtime_autolabel/staging/images"
    auto_label_incomplete_enabled: bool = True
    auto_label_incomplete_prob: float = 1.0
    auto_label_complete_enabled: bool = True
    auto_label_both_prob: float = 0.1
    auto_label_empty_enabled: bool = True
    auto_label_empty_prob: float = 0.01
    auto_label_low_conf_enabled: bool = True
    auto_label_low_conf_prob: float = 0.8
    auto_label_low_conf_min: float = 0.25
    auto_label_low_conf_max: float = 0.65
    auto_label_conflict_enabled: bool = True
    auto_label_conflict_prob: float = 1.0
    auto_label_conflict_iou: float = 0.45
    auto_label_flip_enabled: bool = True
    auto_label_flip_prob: float = 1.0
    auto_label_flip_iou: float = 0.45
    auto_label_flip_max_age_s: float = 0.5
    auto_label_min_interval_s: float = 1.0

    roi_square: bool = False
    roi_radius_px: int = 368

    auto_capture: bool = False
    auto_capture_warmup_s: float = 5.0
    cap_min_hz: float = 10.0
    cap_max_hz: float = 240.0
    target_drop_fps: float = 4.0
    deadband: float = 1.0
    kp: float = 0.6
    ki: float = 0.03
    integral_limit: float = 15.0
    min_apply_delta_hz: float = 0.5

    visualize: bool = False
    smooth: bool = False
    smooth_alpha: float = 0.6
    jsonl_log: bool = False
