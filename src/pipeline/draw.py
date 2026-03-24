# -*- coding: utf-8 -*-
"""
src/pipeline/draw.py

绘制检测框与 ROI 可视化。

增强：若 DetBox 带关键点（pose 模型），在保存帧上叠加可视化关键点。
不改变其他功能：原有 ROI 正方形与 bbox/置信度文本绘制保持不变。
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from inference.yolo_detector import DetBox


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: list[DetBox],
    roi_square: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    out = frame_bgr.copy()

    # ---------- 画 ROI 正方形 ----------
    if roi_square is not None:
        cx, cy, r = roi_square
        x0 = int(cx - r)
        y0 = int(cy - r)
        x1 = int(cx + r)
        y1 = int(cy + r)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # ---------- 画检测框 + 关键点 ----------
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{b.conf:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Pose 关键点可视化（若存在）
        if b.kpts_xy is not None and len(b.kpts_xy) > 0:
            # 可选：设置一个极低的阈值，避免把明显无效点画上去；不想过滤可设为 0.0
            min_kpt_conf = 0.05
            confs = b.kpts_conf

            for j, (px, py) in enumerate(b.kpts_xy):
                if confs is not None and j < len(confs):
                    if float(confs[j]) < min_kpt_conf:
                        continue

                cv2.circle(out, (int(px), int(py)), 3, (0, 255, 255), -1)

    return out
