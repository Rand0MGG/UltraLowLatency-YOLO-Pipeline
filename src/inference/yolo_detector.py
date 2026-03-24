# -*- coding: utf-8 -*-
"""
src/inference/yolo_detector.py

Ultralytics YOLO 通用推理封装模块。
目标：
1. 兼容 YOLOv5 / YOLOv8 / YOLOv11 等 Ultralytics 系列模型（只要 YOLO(...) 能加载）。
2. 同时支持普通检测模型和 Pose 模型（有关键点就提取，没有就自动降级）。
3. 提供统一的数据结构 DetBox，方便后续绘制 / 平滑等模块直接使用。
4. 暴露详细耗时：pre_ms / gpu_ms / post_ms，便于性能分析。
5. 默认不做类别过滤（classes=None），让模型自己决定检测哪些目标。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ultralytics import YOLO  # Ultralytics 统一入口，内部兼容 v5/v8/v11

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetBox:
    """
    单个检测框的数据结构。
    - cls_id   : 类别 id（int）
    - cls_name : 类别名称（如 "person" / "hero" / "enemy"）
    - conf     : 置信度（0~1）
    - xyxy     : 边界框坐标 (x1, y1, x2, y2)，像素坐标
    - kpts_xy  : 关键点坐标列表 ((x, y), ...)，仅当模型为 Pose 且有关键点时才不为 None
    - kpts_conf: 关键点置信度列表 (c0, c1, ...)，与 kpts_xy 对应；无关键点则为 None
    """
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]
    kpts_xy: Optional[Tuple[Tuple[float, float], ...]] = None
    kpts_conf: Optional[Tuple[float, ...]] = None


class YoloDetector:
    """
    Ultralytics YOLO 通用封装。

    只要模型可以通过 YOLO(model_path) 加载（不管是 yolov5 / yolov8 / yolov11，
    也不管是 COCO 还是你针对某个游戏微调的模型），这个类都可以工作。

    接口约定：
    - 输入：BGR 图像 (np.ndarray, HxWx3, uint8)，与 OpenCV 保持一致
    - 输出：List[DetBox], info(dict)
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: Optional[str] = "cuda:0",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        half: bool = False,
        classes: Optional[Sequence[int]] = None,
        max_det: int = 20,
    ) -> None:
        """
        参数：
        - model_path: YOLO 模型权重路径（.pt），可以是 YOLOv5/8/11 甚至 Pose 版本
        - device    : 推理设备，如 'cuda:0' / 'cpu' / None
        - conf      : 置信度阈值
        - iou       : NMS IoU 阈值
        - imgsz     : 推理输入尺寸（短边）
        - half      : 是否使用半精度（仅在 CUDA 下生效）
        - classes   : 想保留的类别 id 集合
                      * None 默认 -> 不做类别过滤，模型检测到什么就返回什么
                      * (0,), (0,2,3) 等 -> 只保留指定类别
        - max_det   : NMS 阶段最多保留的检测框数量
        """
        self.model_path = model_path
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.max_det = int(max_det)

        # 用户指定的 classes（可能是 None，表示不过滤）
        self.classes: Optional[Tuple[int, ...]] = (
            tuple(int(c) for c in classes) if classes is not None else None
        )

        logger.info(
            "Torch version: %s | CUDA available=%s | device_count=%d",
            torch.__version__,
            torch.cuda.is_available(),
            torch.cuda.device_count(),
        )
        if torch.cuda.is_available() and self.device and "cuda" in self.device:
            try:
                idx = 0
                if ":" in self.device:
                    idx = int(self.device.split(":", 1)[1])
                logger.info("CUDA device[%d] = %s", idx, torch.cuda.get_device_name(idx))
            except Exception:
                pass

        # 1. 加载模型（Ultralytics 会自动识别是 v5/v8/v11 / Pose / Detect 等）
        logger.info("Loading YOLO model from %r ...", self.model_path)
        self.model = YOLO(self.model_path)

        # 2. 解析 model.names，构造 {id: name}，后续用于 cls_name
        raw_names = getattr(self.model, "names", None)
        if isinstance(raw_names, dict):
            self._names: Dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)):
            self._names = {i: str(n) for i, n in enumerate(raw_names)}
        else:
            self._names = {}

        logger.info("Model loaded: %s | num_classes=%d", self.model_path, len(self._names))

        # 3. 预构建 Ultralytics predict 参数
        self._predict_kwargs: Dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
            "max_det": self.max_det,
        }
        if self.classes is not None:
            # 只有用户主动传了 classes 时才做过滤
            self._predict_kwargs["classes"] = self.classes
        if self.device is not None:
            self._predict_kwargs["device"] = self.device
        if self.half and self.device and "cuda" in self.device:
            # half 精度只在 CUDA 设备上生效，CPU 上强开会报错
            self._predict_kwargs["half"] = True

        logger.info(
            "YOLO init done: model=%s | device=%s | imgsz=%d | half=%s | classes=%s | max_det=%d",
            self.model_path,
            self.device,
            self.imgsz,
            self.half,
            self.classes if self.classes is not None else "ALL",
            self.max_det,
        )

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[DetBox], Dict[str, Any]]:
        """
        执行一次 YOLO 推理。

        输入：
        - frame_bgr: np.ndarray, HxWx3, uint8, BGR（OpenCV 风格）

        输出：
        - boxes: List[DetBox]
        - info : dict，包含：
            * pre_ms       : 预处理时间（ms）
            * gpu_ms       : 模型前向推理时间（ms）
            * post_ms      : NMS / 后处理时间（ms）
            * imgsz        : 推理使用的输入尺寸
            * conf         : 置信度阈值
            * has_keypoints: 是否存在关键点（Pose 模型）
        """
        if frame_bgr is None or frame_bgr.size == 0:
            info = {
                "pre_ms": 0.0,
                "gpu_ms": 0.0,
                "post_ms": 0.0,
                "imgsz": self.imgsz,
                "conf": self.conf,
                "has_keypoints": False,
            }
            return [], info

        # Ultralytics 统一推理接口：内部兼容 v5/v8/v11
        results = self.model.predict(frame_bgr, **self._predict_kwargs)

        if not results:
            info = {
                "pre_ms": 0.0,
                "gpu_ms": 0.0,
                "post_ms": 0.0,
                "imgsz": self.imgsz,
                "conf": self.conf,
                "has_keypoints": False,
            }
            return [], info

        r0 = results[0]

        # 提取详细耗时信息
        speed = getattr(r0, "speed", {}) or {}
        pre_ms = float(speed.get("preprocess", 0.0))
        gpu_ms = float(speed.get("inference", 0.0))
        post_ms = float(speed.get("postprocess", 0.0))

        # 提取关键点（如果有）
        kpts_xy_all: Optional[np.ndarray] = None
        kpts_conf_all: Optional[np.ndarray] = None
        kp_obj = getattr(r0, "keypoints", None)

        if kp_obj is not None:
            try:
                if getattr(kp_obj, "xy", None) is not None:
                    kpts_xy_all = kp_obj.xy
                    if hasattr(kpts_xy_all, "cpu"):
                        kpts_xy_all = kpts_xy_all.cpu().numpy()
                    else:
                        kpts_xy_all = np.asarray(kpts_xy_all)

                if getattr(kp_obj, "conf", None) is not None:
                    kpts_conf_all = kp_obj.conf
                    if hasattr(kpts_conf_all, "cpu"):
                        kpts_conf_all = kpts_conf_all.cpu().numpy()
                    else:
                        kpts_conf_all = np.asarray(kpts_conf_all)
            except Exception:
                kpts_xy_all = None
                kpts_conf_all = None

        boxes: List[DetBox] = []
        boxes_obj = getattr(r0, "boxes", None)

        if boxes_obj is not None and len(boxes_obj) > 0:
            names = getattr(r0, "names", None)
            if isinstance(names, dict):
                names_map = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                names_map = {i: str(n) for i, n in enumerate(names)}
            else:
                names_map = self._names

            for i, b in enumerate(boxes_obj):
                xyxy = b.xyxy[0].tolist()
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])

                kpts_xy: Optional[Tuple[Tuple[float, float], ...]] = None
                kpts_conf: Optional[Tuple[float, ...]] = None

                if kpts_xy_all is not None and i < kpts_xy_all.shape[0]:
                    pts = kpts_xy_all[i]
                    kpts_xy = tuple((float(p[0]), float(p[1])) for p in pts)

                    if kpts_conf_all is not None and i < kpts_conf_all.shape[0]:
                        confs = kpts_conf_all[i]
                        kpts_conf = tuple(float(c) for c in confs)

                boxes.append(
                    DetBox(
                        cls_id=cls_id,
                        cls_name=str(names_map.get(cls_id, f"class_{cls_id}")),
                        conf=conf,
                        xyxy=(
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ),
                        kpts_xy=kpts_xy,
                        kpts_conf=kpts_conf,
                    )
                )

        info: Dict[str, Any] = {
            "pre_ms": pre_ms,
            "gpu_ms": gpu_ms,
            "post_ms": post_ms,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "has_keypoints": bool(kpts_xy_all is not None),
        }

        return boxes, info
