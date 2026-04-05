from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path


class PipelineProcessManager:
    def __init__(self, project_root: Path, config_path: Path) -> None:
        self.project_root = Path(project_root)
        self.config_path = Path(config_path)
        self._process: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._last_exit_code: int | None = None
        self._log_lines: deque[str] = deque(maxlen=400)
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def start(self) -> dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.status()

            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            cmd = [sys.executable, "-m", "cvcap", "--config", str(self.config_path)]
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
            self._started_at = time.time()
            self._last_exit_code = None
            self._log_lines.clear()
            self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self._reader_thread.start()
            self._log_lines.append(f"$ {' '.join(cmd)}")
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.status()

            try:
                if sys.platform.startswith("win"):
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGINT)
                process.wait(timeout=6)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except Exception:
                    process.kill()
            self._last_exit_code = process.poll()
        return self.status()

    def status(self) -> dict:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            pid = self._process.pid if running and self._process is not None else None
            exit_code = self._last_exit_code
            if self._process is not None and self._process.poll() is not None:
                exit_code = self._process.poll()
                self._last_exit_code = exit_code
            return {
                "running": running,
                "pid": pid,
                "started_at": self._started_at,
                "last_exit_code": exit_code,
                "logs": list(self._log_lines),
                "config_path": str(self.config_path),
            }

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                with self._lock:
                    self._log_lines.append(line.rstrip())
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass
