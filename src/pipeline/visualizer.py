# src/pipeline/visualizer.py
import sys
import multiprocessing
import queue
import time
import ctypes
from multiprocessing import Process, Event
import numpy as np

# 引入 PyQt5
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont

class OverlayWindow(QWidget):
    def __init__(self, queue, width, height):
        super().__init__()
        self.data_queue = queue
        self.screen_w = width
        self.screen_h = height
        self.current_boxes = []
        self.roi_rect = None
        
        # --- [新增] FPS 统计变量 ---
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # --- 核心配置：透明 + 置顶 + 穿透 ---
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.WindowTransparentForInput |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)

        # 设置全屏
        self.setGeometry(0, 0, width, height)
        
        # --- [优化] 定时器 ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_state)
        # 设为 1ms，最大化轮询速度，消除等待延迟
        self.timer.start(1)

    def update_state(self):
        """
        从队列获取最新的推理结果。
        策略：Drain Queue (排空队列) —— 丢弃所有旧帧，只取最后一个！
        """
        try:
            latest = None
            # 循环读取，直到队列为空
            while True:
                latest = self.data_queue.get_nowait()
        except queue.Empty:
            pass

        if latest:
            # latest 结构: (None, boxes, roi_param)
            _, boxes, roi_param = latest
            self.current_boxes = boxes
            self.roi_rect = roi_param
            self.update() # 触发 paintEvent

    def paintEvent(self, event):
        """
        绘图逻辑：只画框和点，背景自动透明
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 定义画笔
        pen_box = QPen(QColor(0, 255, 0), 2)  # 绿色框
        pen_text = QPen(QColor(0, 255, 0), 1)
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)

        # --- [新增] 计算并绘制 Overlay FPS ---
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = now
        
        # 在左上角显示 FPS (青色)
        painter.setPen(QPen(QColor(0, 255, 255), 1)) 
        painter.drawText(10, 30, f"Overlay FPS: {self.current_fps}")

        # --- 1. 画 ROI 区域 (亮黄色实线) ---
        if self.roi_rect:
            cx, cy, r = self.roi_rect
            
            # 使用全不透明的亮黄色，线宽 2
            pen_roi = QPen(QColor(255, 255, 0), 2)
            painter.setPen(pen_roi)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        # --- 2. 画检测框 ---
        painter.setPen(pen_box)
        painter.setBrush(Qt.NoBrush)

        for b in self.current_boxes:
            x1, y1, x2, y2 = b.xyxy
            w = x2 - x1
            h = y2 - y1
            
            # 画矩形
            painter.drawRect(QRectF(x1, y1, w, h))

            # 画文字
            painter.setPen(pen_text)
            painter.drawText(int(x1), int(y1) - 5, f"{b.cls_name} {b.conf:.2f}")
            painter.setPen(pen_box) # 还原画笔

            # --- 3. 画关键点 (骨骼) ---
            if getattr(b, "kpts_xy", None):
                skeleton = [
                    (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 6), (5, 11), (6, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                kpts = b.kpts_xy
                # 画点
                painter.setBrush(QBrush(QColor(0, 255, 255))) # 青色点
                painter.setPen(Qt.NoPen)
                for i, (kx, ky) in enumerate(kpts):
                    if kx == 0 and ky == 0: continue
                    painter.drawEllipse(QRectF(kx - 2, ky - 2, 4, 4))
                
                # 画线
                painter.setPen(QPen(QColor(0, 255, 255, 150), 1))
                num_kpts = len(kpts)
                for p1, p2 in skeleton:
                    if p1 < num_kpts and p2 < num_kpts:
                        x_a, y_a = kpts[p1]
                        x_b, y_b = kpts[p2]
                        if x_a > 0 and x_b > 0:
                            painter.drawLine(int(x_a), int(y_a), int(x_b), int(y_b))
                
                painter.setPen(pen_box) # 还原
                painter.setBrush(Qt.NoBrush)


class VisualizerProcess(Process):
    def __init__(self, shared_queue, window_name="YOLO Overlay", width=1920, height=1080):
        super().__init__(name="Proc-Visualizer", daemon=True)
        self._queue = shared_queue
        self._w = width
        self._h = height
        self._window_name = window_name

    def run(self):
        # PyQt 必须在进程内部创建 QApplication
        app = QApplication(sys.argv)
        window = OverlayWindow(self._queue, self._w, self._h)
        window.setWindowTitle(self._window_name)
        window.show()
        sys.exit(app.exec_())

    def stop(self):
        pass