from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from cvcap.core.detections import DetBox

logger = logging.getLogger(__name__)


class YoloDetector:
    def __init__(
        self,
        model_path: str = "models/yolo11n.pt",
        device: Optional[str] = "cuda:0",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        half: bool = False,
        classes: Optional[Sequence[int]] = None,
        max_det: int = 20,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.max_det = int(max_det)
        self.classes: Optional[Tuple[int, ...]] = tuple(int(c) for c in classes) if classes is not None else None

        logger.info("Torch=%s CUDA=%s devices=%d", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
        logger.info("Loading YOLO model from %r", self.model_path)
        self.model = YOLO(self.model_path)
        raw_names = getattr(self.model, "names", None)
        if isinstance(raw_names, dict):
            self._names = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)):
            self._names = {i: str(name) for i, name in enumerate(raw_names)}
        else:
            self._names = {}

        self._predict_kwargs: Dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
            "max_det": self.max_det,
        }
        if self.classes is not None:
            self._predict_kwargs["classes"] = self.classes
        if self.device is not None:
            self._predict_kwargs["device"] = self.device
        if self.half and self.device and "cuda" in self.device:
            self._predict_kwargs["half"] = True

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[DetBox], Dict[str, Any]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return [], self._empty_info()
        results = self.model.predict(frame_bgr, **self._predict_kwargs)
        if not results:
            return [], self._empty_info()

        result = results[0]
        speed = getattr(result, "speed", {}) or {}
        keypoints_xy = None
        keypoints_conf = None
        keypoint_obj = getattr(result, "keypoints", None)
        if keypoint_obj is not None:
            try:
                if getattr(keypoint_obj, "xy", None) is not None:
                    keypoints_xy = keypoint_obj.xy
                    keypoints_xy = keypoints_xy.cpu().numpy() if hasattr(keypoints_xy, "cpu") else np.asarray(keypoints_xy)
                if getattr(keypoint_obj, "conf", None) is not None:
                    keypoints_conf = keypoint_obj.conf
                    keypoints_conf = keypoints_conf.cpu().numpy() if hasattr(keypoints_conf, "cpu") else np.asarray(keypoints_conf)
            except Exception:
                keypoints_xy = None
                keypoints_conf = None

        boxes: List[DetBox] = []
        box_obj = getattr(result, "boxes", None)
        if box_obj is not None and len(box_obj) > 0:
            names_map = self._resolve_names(result)
            for index, box in enumerate(box_obj):
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                kpts_xy = None
                kpts_conf = None
                if keypoints_xy is not None and index < keypoints_xy.shape[0]:
                    kpts_xy = tuple((float(point[0]), float(point[1])) for point in keypoints_xy[index])
                    if keypoints_conf is not None and index < keypoints_conf.shape[0]:
                        kpts_conf = tuple(float(value) for value in keypoints_conf[index])
                boxes.append(
                    DetBox(
                        cls_id=cls_id,
                        cls_name=str(names_map.get(cls_id, f"class_{cls_id}")),
                        conf=conf,
                        xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        kpts_xy=kpts_xy,
                        kpts_conf=kpts_conf,
                    )
                )

        info = self._empty_info()
        info["pre_ms"] = float(speed.get("preprocess", 0.0))
        info["gpu_ms"] = float(speed.get("inference", 0.0))
        info["post_ms"] = float(speed.get("postprocess", 0.0))
        info["has_keypoints"] = bool(keypoints_xy is not None)
        return boxes, info

    def _resolve_names(self, result) -> Dict[int, str]:
        names = getattr(result, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(name) for i, name in enumerate(names)}
        return self._names

    def _empty_info(self) -> Dict[str, Any]:
        return {
            "pre_ms": 0.0,
            "gpu_ms": 0.0,
            "post_ms": 0.0,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "has_keypoints": False,
        }
