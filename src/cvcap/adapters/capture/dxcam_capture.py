from __future__ import annotations

import ctypes
import platform
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import dxcam
import numpy as np

from cvcap.core.errors import CaptureAccessError

try:
    from dxcam.dxcam import DXCamera as _DXCamera

    def _safe_dxcamera_del(self):
        try:
            if hasattr(self, "is_capturing"):
                self.release()
        except Exception:
            pass

    _DXCamera.__del__ = _safe_dxcamera_del
except Exception:
    pass


@dataclass(frozen=True)
class FrameMetadata:
    timestamp_ns: int
    monitor_rect: Tuple[int, int, int, int]


class _TimerResolutionEnabler:
    _lock = threading.Lock()
    _ref_count = 0
    _enabled = False

    def __init__(self, period_ms: int = 1) -> None:
        self._period_ms = int(period_ms)
        self._is_windows = platform.system().lower().startswith("win")
        self._winmm = None
        if self._is_windows:
            try:
                self._winmm = ctypes.WinDLL("winmm")
            except Exception:
                self._winmm = None

    def enable(self) -> None:
        if not self._is_windows or self._winmm is None:
            return
        with self.__class__._lock:
            self.__class__._ref_count += 1
            if self.__class__._enabled:
                return
            self._winmm.timeBeginPeriod(self._period_ms)
            self.__class__._enabled = True

    def disable(self) -> None:
        if not self._is_windows or self._winmm is None:
            return
        with self.__class__._lock:
            if self.__class__._ref_count <= 0:
                self.__class__._ref_count = 0
                return
            self.__class__._ref_count -= 1
            if self.__class__._ref_count == 0 and self.__class__._enabled:
                self._winmm.timeEndPeriod(self._period_ms)
                self.__class__._enabled = False


class FrameCapturer:
    def __init__(
        self,
        monitor_index: int = 1,
        capture_hz: float = 60.0,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.monitor_index = int(monitor_index)
        self.capture_hz = float(capture_hz)
        self.region = region
        self._lock = threading.Lock()
        self._started = False
        self._video_mode = True
        self._last_frame_bgr: np.ndarray | None = None
        self._timer_res = _TimerResolutionEnabler(period_ms=1)
        self._none_retry_times = 2
        self._none_retry_sleep_s = 0.001
        self._camera = self._create_camera()
        full_rect = self._detect_monitor_rect()
        if self.region is not None:
            left, top, right, bottom = self.region
            self._monitor_rect = (0, 0, right - left, bottom - top)
        else:
            self._monitor_rect = full_rect

    @property
    def monitor_rect(self) -> Tuple[int, int, int, int]:
        return self._monitor_rect

    def set_capture_hz(self, capture_hz: float) -> None:
        with self._lock:
            self.capture_hz = float(capture_hz)
            self._stop_async_capture()
            self._start_async_capture()

    def grab(self) -> Tuple[np.ndarray, FrameMetadata]:
        with self._lock:
            if not self._started:
                self._start_async_capture()

            frame = None
            for _ in range(self._none_retry_times):
                try:
                    frame = self._camera.get_latest_frame()
                except Exception:
                    frame = None
                if frame is not None:
                    break
                time.sleep(self._none_retry_sleep_s)

        if frame is None:
            return self._fallback_frame()

        frame = frame.astype(np.uint8, copy=False)
        if frame.ndim != 3:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        channels = frame.shape[2]
        if channels == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unexpected channel count: {channels}")

        frame_bgr = np.ascontiguousarray(frame_bgr)
        self._last_frame_bgr = frame_bgr
        return frame_bgr, FrameMetadata(time.time_ns(), self._monitor_rect)

    def close(self) -> None:
        with self._lock:
            self._stop_async_capture()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _create_camera(self):
        output_idx = max(0, self.monitor_index - 1)
        try:
            return dxcam.create(output_idx=output_idx)
        except Exception as exc:
            raise CaptureAccessError(
                "dxcam could not access the desktop duplication API. "
                "This usually means the session is restricted or desktop capture is blocked."
            ) from exc

    def _fallback_frame(self) -> Tuple[np.ndarray, FrameMetadata]:
        if self._last_frame_bgr is not None:
            return self._last_frame_bgr, FrameMetadata(time.time_ns(), self._monitor_rect)
        _, _, width, height = self._monitor_rect
        black = np.zeros((height, width, 3), dtype=np.uint8)
        self._last_frame_bgr = black
        return black, FrameMetadata(time.time_ns(), self._monitor_rect)

    def _start_async_capture(self) -> None:
        if self._started:
            return
        self._timer_res.enable()
        target_fps = max(1, int(round(self.capture_hz))) if self.capture_hz > 0 else 1000
        try:
            self._camera.start(target_fps=target_fps, video_mode=self._video_mode, region=self.region)
        except TypeError:
            self._camera.start(target_fps=target_fps)
        self._started = True

    def _stop_async_capture(self) -> None:
        if not self._started:
            return
        try:
            self._camera.stop()
        except Exception:
            pass
        try:
            self._timer_res.disable()
        except Exception:
            pass
        self._started = False

    def _detect_monitor_rect(self) -> Tuple[int, int, int, int]:
        try:
            try:
                self._camera.start(target_fps=30, video_mode=True)
            except TypeError:
                self._camera.start(target_fps=30)
            deadline = time.time() + 1.0
            while time.time() < deadline:
                try:
                    frame = self._camera.get_latest_frame()
                except Exception:
                    frame = None
                if frame is not None:
                    height, width = frame.shape[:2]
                    return 0, 0, int(width), int(height)
                time.sleep(0.01)
        except Exception:
            pass
        finally:
            try:
                self._camera.stop()
            except Exception:
                pass
        return 0, 0, 1920, 1080
