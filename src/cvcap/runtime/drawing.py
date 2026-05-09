from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from cvcap.core.detections import DetBox


def _box_color(box: DetBox) -> tuple[int, int, int]:
    cls_name = str(box.cls_name).lower()
    cls_id = int(box.cls_id)
    if cls_id == 2 or "head" in cls_name:
        return (0, 0, 255)
    if cls_id == 0 or cls_name.startswith("ct"):
        return (255, 120, 0)
    if cls_id == 1 or cls_name.startswith("t"):
        return (255, 255, 255)
    return (0, 255, 0)


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: list[DetBox],
    roi_square: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    output = frame_bgr.copy()
    if roi_square is not None:
        cx, cy, radius = roi_square
        cv2.rectangle(
            output,
            (int(cx - radius), int(cy - radius)),
            (int(cx + radius), int(cy + radius)),
            (0, 255, 0),
            2,
        )

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy)
        color = _box_color(box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{box.cls_name} {box.conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        if box.kpts_xy:
            min_conf = 0.05
            for index, (px, py) in enumerate(box.kpts_xy):
                if box.kpts_conf is not None and index < len(box.kpts_conf) and float(box.kpts_conf[index]) < min_conf:
                    continue
                cv2.circle(output, (int(px), int(py)), 3, (0, 255, 255), -1)

    return output
