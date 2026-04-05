from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, name="LogWriter", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def append_record(self, record: Dict[str, Any]) -> None:
        self._queue.put(record)

    def _worker_loop(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", buffering=8192) as handle:
                while True:
                    try:
                        record = self._queue.get(timeout=0.5)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        self._queue.task_done()
                    except queue.Empty:
                        if self._stop_event.is_set():
                            break
                    except Exception as exc:
                        print(f"[LogWriter Error] {exc}")
        except Exception as exc:
            print(f"[LogWriter Critical] Failed to open log file: {exc}")

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
