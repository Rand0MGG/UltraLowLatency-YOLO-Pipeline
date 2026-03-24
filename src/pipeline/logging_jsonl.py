# -*- coding: utf-8 -*-
"""
src/pipeline/logging_jsonl.py

JSONL 日志写入封装（异步高性能版）。
职责：
- 提供非阻塞的 append_record() 接口。
- 后台线程负责写盘，避免阻塞推理流水线。
"""

from __future__ import annotations

import json
import time
import threading
import queue
from pathlib import Path
from typing import Any, Dict, Iterable


class JsonlLogger:
    """
    异步 JSONL 追加写入器。
    原理：主线程 -> Queue -> 后台线程 -> 磁盘文件
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # 无界队列 (通常日志生成速度不会超过磁盘写入速度)
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        
        # 启动后台写入线程
        self._thread = threading.Thread(target=self._worker_loop, name="LogWriter", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """
        关闭日志：等待队列排空后安全退出。
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def append_record(self, rec: Dict[str, Any]) -> None:
        """
        [非阻塞] 将记录放入队列即返回。耗时微秒级。
        """
        self._queue.put(rec)

    def _worker_loop(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", buffering=8192) as f:
                while True:
                    try:
                        # 阻塞等待数据，每 0.5 秒检查一次退出标志
                        rec = self._queue.get(timeout=0.5)
                        
                        line = json.dumps(rec, ensure_ascii=False) + "\n"
                        f.write(line)
                        f.flush() # 实时刷新，防止 Crash 丢失数据
                        
                        self._queue.task_done()
                        
                    except queue.Empty:
                        if self._stop_event.is_set():
                            break
                        continue
                    except Exception as e:
                        print(f"[LogWriter Error] {e}")
                        
        except Exception as e:
            print(f"[LogWriter Critical] Failed to open log file: {e}")

    @staticmethod
    def make_infer_record(
        *,
        timestamp_ns: int,
        monitor_rect: Iterable[int],
        frame_shape: Iterable[int],
        infer_ms: float,
        boxes: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "event": "yolo_infer",
            "local_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "timestamp_ns": int(timestamp_ns),
            "monitor_rect": list(monitor_rect),
            "frame_shape": list(frame_shape),
            "infer_ms": float(infer_ms),
            "num_boxes": int(len(boxes)),
            "boxes": boxes,
        }