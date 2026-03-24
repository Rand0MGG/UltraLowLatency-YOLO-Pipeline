# -*- coding: utf-8 -*-
"""
tools/test_capture_frame.py (minimal, single-thread capture)

目标：
- 只测试 capture_frame.py 是否能稳定抓屏（特别是全屏游戏场景）。
- 不显示实时视频。
- 主线程持续抓帧，按 S 保存当前最新帧到磁盘，并记录时间/帧率/元数据等信息。
- 按 Q 或 ESC 退出。

重要说明（为什么不用后台抓帧线程）：
- 你的 capture_frame.py 内部使用了线程局部对象保存 GDI/DC 句柄（如 srcdc）。
- 这些句柄只存在于创建它们的线程中，换线程调用 grab() 会报：
  '_thread._local' object has no attribute 'srcdc'
- 因此：抓屏必须在同一个线程中完成。这里采用“主线程抓帧 + 非阻塞读键”的方式。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# 1) 兼容导入：自动配置 sys.path
# -----------------------------
def _setup_import_path() -> None:
    """
    允许你直接运行：
        python tools/test_capture_frame.py
    通过把 project_root 和可选 src/ 加入 sys.path。
    """
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent
    src_dir = project_root / "src"

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if src_dir.exists() and src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _import_frame_capturer():
    """
    尝试导入 FrameCapturer（兼容常见工程结构）。
    """
    _setup_import_path()

    candidates = [
        ("capture.capture_frame", "FrameCapturer"),
        ("src.capture.capture_frame", "FrameCapturer"),
        ("capture_frame", "FrameCapturer"),
        ("src.capture_frame", "FrameCapturer"),
    ]

    last_err = None
    for mod_name, attr in candidates:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            return getattr(mod, attr)
        except Exception as e:
            last_err = e

    raise ModuleNotFoundError(
        "无法导入 FrameCapturer。请确认 capture_frame.py 的实际位置。\n"
        "已尝试：capture.capture_frame / src.capture.capture_frame / capture_frame / src.capture_frame\n"
        f"最后一次错误：{last_err}"
    )


# -----------------------------
# 2) Windows 控制台：非阻塞读键
# -----------------------------
def _get_key_nonblocking_windows() -> Optional[str]:
    """
    非阻塞读取控制台按键（Windows）。
    返回：
        - str：按下的字符（已转小写）
        - None：当前无按键
    """
    try:
        import msvcrt  # Windows only
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch:
                return ch.lower()
    except Exception:
        pass
    return None


# -----------------------------
# 3) 近似 fps 统计（滑动窗口）
# -----------------------------
class FpsMeter:
    """
    用 perf_counter 记录最近 window_s 秒内的帧时间戳，估算抓帧有效fps。
    这是“抓帧循环视角”的近似值，用于验证 capture_hz 是否大致生效。
    """

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = max(0.2, float(window_s))
        self.ts = []

    def tick(self) -> float:
        now = time.perf_counter()
        self.ts.append(now)
        cutoff = now - self.window_s
        while self.ts and self.ts[0] < cutoff:
            self.ts.pop(0)
        return len(self.ts) / self.window_s


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal capture_frame tester: press S to save current frame.")
    parser.add_argument("--monitor-index", type=int, default=1, help="mss monitor index (1..N, 0=virtual all).")
    parser.add_argument("--capture-hz", type=float, default=60.0, help="Target capture Hz. <=0 means unlimited.")
    parser.add_argument("--out-dir", type=str, default=str(Path("debug") / "test_snapshots"),
                        help="Output directory for snapshots and logs.")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (1~100).")
    parser.add_argument("--stats-window", type=float, default=1.0, help="Seconds for fps moving window.")

    # 可选：如果你担心全屏时控制台收不到按键，可开启自动定时保存
    parser.add_argument("--autosave-every", type=float, default=0.0,
                        help="If >0, auto-save one frame every N seconds (useful when hotkey cannot be received).")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    FrameCapturer = _import_frame_capturer()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    capturer = FrameCapturer(monitor_index=args.monitor_index, capture_hz=args.capture_hz)

    fps_meter = FpsMeter(window_s=float(args.stats_window))
    last_frame: Optional[np.ndarray] = None
    last_meta = None

    last_autosave_t = 0.0

    logging.info("Running. Press 'S' to save current frame. Press 'Q' or 'ESC' to quit.")
    logging.info("Output dir: %s", out_dir)
    if args.autosave_every and args.autosave_every > 0:
        logging.info("Auto-save enabled: every %.2f seconds", float(args.autosave_every))

    try:
        while True:
            # 1) 抓帧（必须在同一线程内执行，避免 thread-local/DC 问题）
            try:
                frame_bgr, meta = capturer.grab()
            except Exception as e:
                logging.error("grab() failed: %s", e)
                time.sleep(0.05)
                continue

            if frame_bgr is None:
                continue

            # 2) 统一 OpenCV 兼容性：uint8 + contiguous
            if frame_bgr.dtype != np.uint8:
                frame_bgr = frame_bgr.astype(np.uint8, copy=False)
            frame_bgr = np.ascontiguousarray(frame_bgr)

            last_frame = frame_bgr
            last_meta = meta

            # 3) 统计有效fps（近似）
            eff_fps = fps_meter.tick()

            # 4) 处理按键（非阻塞）
            key = _get_key_nonblocking_windows()
            if key in ("q", "\x1b"):  # Q 或 ESC
                break

            # 5) 自动保存（可选）
            now_wall = time.time()
            do_autosave = False
            if args.autosave_every and args.autosave_every > 0:
                if now_wall - last_autosave_t >= float(args.autosave_every):
                    do_autosave = True
                    last_autosave_t = now_wall

            # 6) 手动保存：S
            do_manual_save = (key == "s")

            if (do_manual_save or do_autosave) and last_frame is not None:
                # 文件名：含毫秒避免冲突
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                ms = int((now_wall * 1000) % 1000)
                tag = "auto" if do_autosave and not do_manual_save else "manual"
                img_name = f"cap_{tag}_{ts}_{ms:03d}.jpg"
                img_path = out_dir / img_name

                # 写盘耗时
                t_write0 = time.perf_counter()
                ok = cv2.imwrite(
                    str(img_path),
                    last_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                )
                t_write1 = time.perf_counter()

                if not ok:
                    logging.error("Failed to write snapshot: %s", img_path)
                    continue

                # 从 meta 尝试提取常用字段（不存在则 None）
                timestamp_ns = getattr(last_meta, "timestamp_ns", None) if last_meta is not None else None
                monitor_rect = getattr(last_meta, "monitor_rect", None) if last_meta is not None else None

                rec = {
                    "event": "snapshot",
                    "mode": tag,
                    "saved_image": str(img_path),
                    "saved_local_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "saved_unix_ms": int(now_wall * 1000),

                    "capture_target_hz": float(args.capture_hz),
                    "capture_effective_fps_approx": float(eff_fps),
                    "capture_stats_window_s": float(fps_meter.window_s),

                    "frame_shape": [int(x) for x in last_frame.shape],
                    "frame_dtype": str(last_frame.dtype),

                    "meta_timestamp_ns": int(timestamp_ns) if isinstance(timestamp_ns, (int, np.integer)) else timestamp_ns,
                    "meta_monitor_rect": monitor_rect,

                    "write_ok": bool(ok),
                    "write_time_ms": float((t_write1 - t_write0) * 1000.0),
                }

                # 追加写入 JSONL
                try:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception as e:
                    logging.warning("Failed to write log.jsonl: %s", e)

                logging.info("Saved: %s | eff_fps≈%.2f | write=%.1fms",
                             img_path.name, eff_fps, rec["write_time_ms"])

            # 7) 给主线程一点喘息，避免空转；不影响 capture_hz 的节拍控制（grab 内部已限速）
            time.sleep(0.001)

    finally:
        logging.info("Exited.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
