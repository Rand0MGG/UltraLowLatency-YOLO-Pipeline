from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from cvcap.core.config import RunnerArgs

DEFAULT_AUTO_LABEL_TRAIN_RATIO = 0.8
DEFAULT_AUTO_LABEL_VAL_RATIO = 0.2
DEFAULT_AUTO_LABEL_TEST_RATIO = 0.0

FIELD_GROUPS = [
    {
        "id": "runtime",
        "label": "Runtime",
        "description": "Basic capture and execution settings.",
        "fields": [
            {"name": "monitor_index", "label": "Monitor Index", "type": "number", "step": 1, "min": 1, "help": "Choose which monitor to capture. In most setups the main display is 1."},
            {"name": "capture_hz", "label": "Capture Hz", "type": "number", "step": 1, "min": 1, "help": "Target screen capture rate. Higher values feel smoother but use more CPU and GPU time."},
            {"name": "model", "label": "Model", "type": "select", "options_source": "models", "help": "YOLO26 pose weight or TensorRT engine file used for inference."},
            {"name": "device", "label": "Device", "type": "text", "placeholder": "cuda:0 or cpu", "help": "Inference device. Use cuda:0 for the first GPU or cpu for CPU inference."},
            {"name": "visualize", "label": "Overlay Visualization", "type": "checkbox", "help": "Show the transparent overlay window with boxes and keypoints on screen."},
            {"name": "half", "label": "FP16 / Half Precision", "type": "checkbox", "help": "Use half precision on supported CUDA devices for lower memory usage and often faster inference."},
            {"name": "end2end", "label": "End-to-End Prediction", "type": "checkbox", "help": "Use YOLO26 end-to-end / NMS-free prediction. Disable to test the traditional NMS path."},
            {"name": "max_run_seconds", "label": "Max Run Seconds", "type": "number", "step": 1, "min": 0, "help": "Auto-stop after this many seconds. Use 0 to keep running until you stop it manually."},
        ],
    },
    {
        "id": "detection",
        "label": "Detection",
        "description": "Model inference parameters and class filters.",
        "fields": [
            {"name": "conf", "label": "Confidence", "type": "number", "step": 0.01, "min": 0, "max": 1, "help": "Minimum confidence threshold. Higher values reduce false positives but may miss targets."},
            {"name": "iou", "label": "IoU", "type": "number", "step": 0.01, "min": 0, "max": 1, "help": "IoU threshold used by NMS to merge overlapping detections."},
            {"name": "imgsz", "label": "Image Size", "type": "number", "step": 1, "min": 32, "help": "Inference input size. Larger sizes can improve accuracy but increase latency."},
            {"name": "yolo_classes", "label": "YOLO Classes", "type": "text", "placeholder": "0 or 0,2,3", "help": "Optional class filter. Leave blank for all classes. The default 0 means person only."},
            {"name": "yolo_max_det", "label": "Max Detections", "type": "number", "step": 1, "min": 1, "help": "Maximum number of detections kept for each frame."},
        ],
    },
    {
        "id": "auto_label",
        "label": "Auto Label",
        "description": "Collect useful ROI images into a dataset staging area, then annotate and prepare splits after screening.",
        "fields": [
            {"name": "auto_label", "label": "Enable Auto Image Capture", "type": "checkbox", "section": "base", "help": "Save clean ROI screenshots only when body/head detections match the collection rules."},
            {"name": "auto_label_dataset_root", "label": "Dataset Root", "type": "text", "section": "base", "placeholder": "datasets/mirage2", "help": "Dataset folder selected in the UI. Runtime capture saves staged images under this root."},
            {"name": "auto_label_min_interval_s", "label": "Min Save Interval", "type": "number", "section": "base", "step": 0.05, "min": 1, "help": "Cooldown between saved auto-label samples. Use at least 1 second to avoid near-duplicate frames."},
            {"name": "auto_label_low_conf_enabled", "label": "Low-Confidence Body", "type": "checkbox", "section": "low_conf", "help": "Capture body detections whose confidence is in the uncertain range."},
            {"name": "auto_label_low_conf_prob", "label": "Low-Conf Probability", "type": "number", "section": "low_conf", "step": 0.01, "min": 0, "max": 1, "help": "Save probability for uncertain body samples."},
            {"name": "auto_label_low_conf_min", "label": "Low-Conf Min", "type": "number", "section": "low_conf", "step": 0.01, "min": 0, "max": 1, "help": "Lower confidence bound. The global inference confidence must be no higher than this to see those boxes."},
            {"name": "auto_label_low_conf_max", "label": "Low-Conf Max", "type": "number", "section": "low_conf", "step": 0.01, "min": 0, "max": 1, "help": "Upper confidence bound for uncertain body samples."},
            {"name": "auto_label_conflict_enabled", "label": "T/CT Conflict", "type": "checkbox", "section": "conflict", "help": "Capture when T and CT body boxes overlap strongly in the same region."},
            {"name": "auto_label_conflict_prob", "label": "Conflict Probability", "type": "number", "section": "conflict", "step": 0.01, "min": 0, "max": 1, "help": "Save probability for overlapping T/CT conflict frames."},
            {"name": "auto_label_conflict_iou", "label": "Conflict IoU", "type": "number", "section": "conflict", "step": 0.01, "min": 0, "max": 1, "help": "Minimum overlap between T and CT boxes to treat as a conflict."},
            {"name": "auto_label_flip_enabled", "label": "T/CT Frame Flip", "type": "checkbox", "section": "flip", "help": "Capture when a body in the same place changes between T and CT across adjacent frames."},
            {"name": "auto_label_flip_prob", "label": "Flip Probability", "type": "number", "section": "flip", "step": 0.01, "min": 0, "max": 1, "help": "Save probability for short-term T/CT class flips."},
            {"name": "auto_label_flip_iou", "label": "Flip IoU", "type": "number", "section": "flip", "step": 0.01, "min": 0, "max": 1, "help": "Minimum overlap with the previous frame body box to treat it as the same target."},
            {"name": "auto_label_flip_max_age_s", "label": "Flip Max Age", "type": "number", "section": "flip", "step": 0.05, "min": 0, "help": "Maximum seconds between frames for class-flip detection."},
            {"name": "auto_label_incomplete_enabled", "label": "Incomplete Detection", "type": "checkbox", "section": "incomplete", "help": "Capture frames where body or head is missing."},
            {"name": "auto_label_incomplete_prob", "label": "Incomplete Probability", "type": "number", "section": "incomplete", "step": 0.01, "min": 0, "max": 1, "help": "Save probability for incomplete body/head detections."},
            {"name": "auto_label_empty_enabled", "label": "Empty Frames", "type": "checkbox", "section": "empty", "help": "Capture negative samples when no boxes are detected."},
            {"name": "auto_label_empty_prob", "label": "Empty Probability", "type": "number", "section": "empty", "step": 0.001, "min": 0, "max": 1, "help": "Save probability for empty negative samples."},
            {"name": "auto_label_complete_enabled", "label": "Complete Samples", "type": "checkbox", "section": "complete", "help": "Capture occasional normal frames where both body and head are detected."},
            {"name": "auto_label_both_prob", "label": "Complete Probability", "type": "number", "section": "complete", "step": 0.01, "min": 0, "max": 1, "help": "Save probability when both body and head are detected."},
            {"name": "auto_label_train_ratio", "label": "Train Split", "type": "number", "section": "dataset_tools", "step": 0.01, "min": 0, "max": 1, "help": "Fraction of prepared labeled samples assigned to train."},
            {"name": "auto_label_val_ratio", "label": "Val Split", "type": "number", "section": "dataset_tools", "step": 0.01, "min": 0, "max": 1, "help": "Fraction of prepared labeled samples assigned to validation."},
            {"name": "auto_label_test_ratio", "label": "Test Split", "type": "number", "section": "dataset_tools", "step": 0.01, "min": 0, "max": 1, "help": "Optional fraction assigned to test. Leave at 0 when you only need train/val."},
        ],
    },
    {
        "id": "roi",
        "label": "ROI",
        "description": "Region-of-interest cropping controls.",
        "fields": [
            {"name": "roi_square", "label": "Enable Center Square ROI", "type": "checkbox", "help": "Capture only the centered square region to reduce work and focus on the middle of the screen."},
            {"name": "roi_radius_px", "label": "ROI Radius", "type": "number", "step": 1, "min": 0, "help": "Half-size of the center ROI in pixels. Smaller values mean a tighter cropped region."},
        ],
    },
    {
        "id": "auto_capture",
        "label": "Auto Capture",
        "description": "PID-based capture-rate adaptation.",
        "fields": [
            {"name": "auto_capture", "label": "Enable Auto Capture", "type": "checkbox", "help": "Automatically tune capture rate based on runtime load to reduce backlog and latency."},
            {"name": "auto_capture_warmup_s", "label": "Warmup Seconds", "type": "number", "step": 0.5, "min": 0, "help": "Delay automatic tuning for a short warmup period after startup."},
            {"name": "cap_min_hz", "label": "Min Capture Hz", "type": "number", "step": 1, "min": 1, "help": "Lowest capture rate auto tuning is allowed to apply."},
            {"name": "cap_max_hz", "label": "Max Capture Hz", "type": "number", "step": 1, "min": 1, "help": "Highest capture rate auto tuning is allowed to apply."},
            {"name": "target_drop_fps", "label": "Target Drop FPS", "type": "number", "step": 0.1, "min": 0, "help": "Target dropped-frame level that the controller tries to maintain."},
            {"name": "deadband", "label": "Deadband", "type": "number", "step": 0.1, "min": 0, "help": "Ignore tiny control errors within this range to avoid jittery tuning."},
            {"name": "kp", "label": "Kp", "type": "number", "step": 0.01, "help": "Proportional gain for the auto-capture PID controller."},
            {"name": "ki", "label": "Ki", "type": "number", "step": 0.01, "help": "Integral gain for the auto-capture PID controller."},
            {"name": "integral_limit", "label": "Integral Limit", "type": "number", "step": 0.1, "min": 0, "help": "Clamp the integral term so the controller does not over-correct."},
            {"name": "min_apply_delta_hz", "label": "Min Apply Delta Hz", "type": "number", "step": 0.1, "min": 0, "help": "Minimum rate change required before a new capture frequency is applied."},
        ],
    },
    {
        "id": "output",
        "label": "Output & Smoothing",
        "description": "Saving, stats, and box smoothing controls.",
        "fields": [
            {"name": "save_every", "label": "Save Every (s)", "type": "number", "step": 0.1, "min": 0, "help": "Save screenshots with detections at a fixed interval. Use 0 to disable automatic saving."},
            {"name": "save_queue", "label": "Save Queue", "type": "number", "step": 1, "min": 1, "help": "Length of the asynchronous save queue."},
            {"name": "stats_interval", "label": "Stats Interval", "type": "number", "step": 0.1, "min": 0.1, "help": "How often runtime statistics are printed to the log."},
            {"name": "smooth", "label": "Enable Smoothing", "type": "checkbox", "help": "Smooth boxes and keypoints to reduce visible jitter."},
            {"name": "smooth_alpha", "label": "Smooth Alpha", "type": "number", "step": 0.01, "min": 0, "max": 1, "help": "Smoothing strength. Higher values are steadier but may feel less responsive."},
            {"name": "jsonl_log", "label": "JSONL Frame Log", "type": "checkbox", "help": "Write detailed per-frame inference records to debug/yolo_log.jsonl for debugging."},
        ],
    },
]


def config_to_payload(config: RunnerArgs) -> dict:
    payload = asdict(config)
    payload["yolo_classes"] = "" if config.yolo_classes is None else ",".join(str(part) for part in config.yolo_classes)
    payload["auto_label_dataset_root"] = auto_label_dataset_root(config.auto_label_dir)
    payload["auto_label_train_ratio"] = DEFAULT_AUTO_LABEL_TRAIN_RATIO
    payload["auto_label_val_ratio"] = DEFAULT_AUTO_LABEL_VAL_RATIO
    payload["auto_label_test_ratio"] = DEFAULT_AUTO_LABEL_TEST_RATIO
    return payload


def auto_label_dataset_root(auto_label_dir: str) -> str:
    path = Path(auto_label_dir)
    if path.suffix.lower() in {".yaml", ".yml", ".txt"}:
        path = path.parent
    parts = tuple(part.lower() for part in path.parts)
    if len(parts) >= 2 and parts[-2:] == ("staging", "images"):
        return str(path.parent.parent)
    if path.name.lower() in {"images", "image", "labels"}:
        if path.parent.name.lower() in {"staging", "train", "val", "valid", "test"}:
            return str(path.parent.parent)
        return str(path.parent)
    if path.name.lower() in {"staging", "train", "val", "valid", "test"}:
        return str(path.parent)
    return str(path)
