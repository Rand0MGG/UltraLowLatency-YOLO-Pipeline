# src/pipeline/async_saver.py
import queue
import threading
import cv2
import logging
from typing import Optional, Tuple, Any
from pathlib import Path
import numpy as np

from pipeline.draw import draw_boxes

class AsyncSaver:
    """
    异步保存器：将 [画框] 和 [写盘] 两个耗时操作全部移出主推理线程。
    """
    def __init__(self, save_queue_size: int, roi_square: bool):
        self._queue = queue.Queue(maxsize=save_queue_size)
        self._stop_event = threading.Event()
        self._roi_square = roi_square
        
        self._thread = threading.Thread(target=self._worker_loop, name="AsyncSaver", daemon=True)
        self._thread.start()

    def push(self, frame_raw: np.ndarray, boxes: list, roi_param: Any, out_path: Path):
        """
        [非阻塞] 将原始数据推入队列。若队列满则丢弃，保证不卡主线程。
        """
        try:
            self._queue.put_nowait((frame_raw, boxes, roi_param, out_path))
        except queue.Full:
            pass

    def close(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
                frame_raw, boxes, roi_param, path = item
                
                # --- 耗时操作 (后台执行) ---
                vis = draw_boxes(frame_raw, boxes, roi_square=roi_param)
                
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), vis)
                
                del frame_raw, vis
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"AsyncSaver error: {e}")