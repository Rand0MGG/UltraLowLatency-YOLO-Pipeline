from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from cvcap.core.detections import DetBox


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
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{box.cls_name} {box.conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
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
