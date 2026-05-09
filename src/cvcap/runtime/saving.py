from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cvcap.runtime.drawing import draw_boxes


class AsyncSaver:
    def __init__(self, save_queue_size: int):
        self._queue = queue.Queue(maxsize=save_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, name="AsyncSaver", daemon=True)
        self._thread.start()

    def push(self, frame_raw: np.ndarray, boxes: list, roi_param: Any, out_path: Path):
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
                frame_raw, boxes, roi_param, path = self._queue.get(timeout=0.2)
                visualized = draw_boxes(frame_raw, boxes, roi_square=roi_param)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), visualized):
                    raise ValueError(f"Could not write annotated frame: {path}")
            except queue.Empty:
                continue
            except Exception as exc:
                logging.error("AsyncSaver error: %s", exc)
