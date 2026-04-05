from __future__ import annotations

import queue
import sys
import time
from multiprocessing import Process

from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget


class OverlayWindow(QWidget):
    def __init__(self, shared_queue, width, height):
        super().__init__()
        self.data_queue = shared_queue
        self.current_boxes = []
        self.roi_rect = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setGeometry(0, 0, width, height)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_state)
        self.timer.start(1)

    def update_state(self):
        try:
            latest = None
            while True:
                latest = self.data_queue.get_nowait()
        except queue.Empty:
            pass
        if latest:
            _, boxes, roi_param = latest
            self.current_boxes = boxes
            self.roi_rect = roi_param
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen_box = QPen(QColor(0, 255, 0), 2)
        pen_text = QPen(QColor(0, 255, 0), 1)
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = now
        painter.setPen(QPen(QColor(0, 255, 255), 1))
        painter.drawText(10, 30, f"Overlay FPS: {self.current_fps}")

        if self.roi_rect:
            cx, cy, radius = self.roi_rect
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))

        painter.setPen(pen_box)
        painter.setBrush(Qt.NoBrush)
        for box in self.current_boxes:
            x1, y1, x2, y2 = box.xyxy
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            painter.setPen(pen_text)
            painter.drawText(int(x1), int(y1) - 5, f"{box.cls_name} {box.conf:.2f}")
            painter.setPen(pen_box)
            if getattr(box, "kpts_xy", None):
                skeleton = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
                painter.setBrush(QBrush(QColor(0, 255, 255)))
                painter.setPen(Qt.NoPen)
                for kx, ky in box.kpts_xy:
                    if kx == 0 and ky == 0:
                        continue
                    painter.drawEllipse(QRectF(kx - 2, ky - 2, 4, 4))
                painter.setPen(QPen(QColor(0, 255, 255, 150), 1))
                num_keypoints = len(box.kpts_xy)
                for p1, p2 in skeleton:
                    if p1 < num_keypoints and p2 < num_keypoints:
                        xa, ya = box.kpts_xy[p1]
                        xb, yb = box.kpts_xy[p2]
                        if xa > 0 and xb > 0:
                            painter.drawLine(int(xa), int(ya), int(xb), int(yb))
                painter.setPen(pen_box)
                painter.setBrush(Qt.NoBrush)


class VisualizerProcess(Process):
    def __init__(self, shared_queue, window_name="YOLO Overlay", width=1920, height=1080):
        super().__init__(name="Proc-Visualizer", daemon=True)
        self._queue = shared_queue
        self._w = width
        self._h = height
        self._window_name = window_name

    def run(self):
        app = QApplication(sys.argv)
        window = OverlayWindow(self._queue, self._w, self._h)
        window.setWindowTitle(self._window_name)
        window.show()
        sys.exit(app.exec_())

    def stop(self):
        pass
