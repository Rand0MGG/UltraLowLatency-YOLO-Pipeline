from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import queue
import time
import traceback
from ctypes import c_double, c_longlong, wintypes
from multiprocessing import Array, Condition, Event, Lock, Process, Queue, Value
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import psutil

from cvcap.adapters.capture.dxcam_capture import FrameCapturer
from cvcap.adapters.inference.ultralytics_detector import YoloDetector
from cvcap.core.config import RunnerArgs
from cvcap.core.errors import CaptureAccessError, PipelineError
from cvcap.runtime.logging_jsonl import JsonlLogger
from cvcap.runtime.auto_label import AutoLabeler
from cvcap.runtime.metrics import PerfStats
from cvcap.runtime.overlay import VisualizerProcess
from cvcap.runtime.saving import AsyncSaver
from cvcap.runtime.shared_buffer import SharedTripleBuffer
from cvcap.runtime.smoothing import BoxSmoother


def enable_mmcss() -> bool:
    try:
        avrt = ctypes.windll.avrt
        avrt.AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
        avrt.AvSetMmThreadPriority.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        avrt.AvSetMmThreadPriority.restype = wintypes.BOOL
        task_index = wintypes.DWORD(0)
        handle = avrt.AvSetMmThreadCharacteristicsW("Games", ctypes.byref(task_index))
        if handle:
            avrt.AvSetMmThreadPriority(handle, 2)
            print(f"[MMCSS] Main Capture Thread Registered! Handle: {handle}")
            return True
        print("[MMCSS] Failed. (Run as Admin?)")
        return False
    except Exception as exc:
        print(f"[MMCSS] Error: {exc}")
        return False


def inference_process_target(
    args: RunnerArgs,
    stop_evt: Event,
    shm_name: str,
    frame_shape: Tuple[int, int, int],
    lock: Lock,
    cond: Condition,
    shared_read_count: Value,
    shared_timings: Array,
    model_ready_evt: Event,
    init_status_queue: Queue,
    viz_queue: Optional[Queue],
    offset_x: int,
    offset_y: int,
):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[InferProc-{multiprocessing.current_process().pid}] %(asctime)s %(levelname)s: %(message)s",
    )

    saver = None
    auto_labeler = None
    buffer = None
    json_logger = None
    try:
        detector = YoloDetector(
            model_path=args.model,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            half=args.half,
            end2end=args.end2end,
            classes=args.yolo_classes,
            max_det=args.yolo_max_det,
        )
        buffer = SharedTripleBuffer(shape=frame_shape, name=shm_name, create=False)
        json_logger = JsonlLogger(Path("debug") / "yolo_log.jsonl") if args.jsonl_log else None
        saver = AsyncSaver(save_queue_size=args.save_queue, roi_square=args.roi_square) if args.save_every > 0 else None
        auto_labeler = (
            AutoLabeler(
                output_dir=Path(args.auto_label_dir),
                both_prob=args.auto_label_both_prob,
                empty_prob=args.auto_label_empty_prob,
                min_interval_s=args.auto_label_min_interval_s,
                queue_size=args.save_queue,
                incomplete_enabled=args.auto_label_incomplete_enabled,
                incomplete_prob=args.auto_label_incomplete_prob,
                complete_enabled=args.auto_label_complete_enabled,
                empty_enabled=args.auto_label_empty_enabled,
                low_conf_enabled=args.auto_label_low_conf_enabled,
                low_conf_prob=args.auto_label_low_conf_prob,
                low_conf_min=args.auto_label_low_conf_min,
                low_conf_max=args.auto_label_low_conf_max,
                conflict_enabled=args.auto_label_conflict_enabled,
                conflict_prob=args.auto_label_conflict_prob,
                conflict_iou=args.auto_label_conflict_iou,
                flip_enabled=args.auto_label_flip_enabled,
                flip_prob=args.auto_label_flip_prob,
                flip_iou=args.auto_label_flip_iou,
                flip_max_age_s=args.auto_label_flip_max_age_s,
            )
            if args.auto_label
            else None
        )
        stats = PerfStats(shared_timings)
        smoother = BoxSmoother(alpha=args.smooth_alpha, iou_thresh=0.5) if args.smooth else None

        detector.infer(np.zeros(frame_shape, dtype=np.uint8))
        init_status_queue.put({"ok": True, "message": f"Loaded model: {args.model}"})
        model_ready_evt.set()

        last_save_t = 0.0
        previous_end = time.perf_counter()
        while not stop_evt.is_set():
            payload = buffer.get(lock, cond, timeout=0.5)
            start = time.perf_counter()
            wait_ms = (start - previous_end) * 1000.0
            if payload is None:
                previous_end = time.perf_counter()
                continue

            frame_bgr, ts_ns, monitor_rect = payload
            with shared_read_count.get_lock():
                shared_read_count.value += 1

            boxes, info = detector.infer(frame_bgr)
            raw_boxes = boxes
            if auto_labeler is not None:
                label_frame, label_boxes = _auto_label_input(frame_bgr, raw_boxes, args)
                auto_labeler.maybe_save(label_frame, label_boxes, now=start)
            if (offset_x > 0 or offset_y > 0) and boxes:
                boxes = _offset_boxes(boxes, offset_x, offset_y)
            if smoother is not None:
                boxes = smoother.update(boxes)

            roi_param = None
            if args.roi_square:
                roi_h, roi_w = frame_shape[:2]
                roi_param = (offset_x + roi_w // 2, offset_y + roi_h // 2, roi_h // 2)
            if viz_queue is not None:
                try:
                    viz_queue.put_nowait((None, boxes, roi_param))
                except queue.Full:
                    pass

            if json_logger is not None:
                json_logger.append_record(
                    json_logger.make_infer_record(
                        timestamp_ns=int(ts_ns),
                        monitor_rect=monitor_rect,
                        frame_shape=frame_bgr.shape,
                        infer_ms=info.get("gpu_ms", 0.0),
                        boxes=[
                            {
                                "cls_id": int(box.cls_id),
                                "cls_name": str(box.cls_name),
                                "conf": float(box.conf),
                                "xyxy": list(box.xyxy),
                                "kpts_xy": [[float(x), float(y)] for (x, y) in box.kpts_xy] if box.kpts_xy is not None else None,
                            }
                            for box in boxes
                        ],
                    )
                )

            if args.save_every > 0:
                now = time.perf_counter()
                if now - last_save_t >= float(args.save_every):
                    last_save_t = now
                    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                    out_path = Path("debug") / "yolo_samples" / f"sample_{stamp}_{int(now * 1000) % 1000:03d}.jpg"
                    if saver is not None:
                        saver.push(frame_bgr.copy(), boxes, roi_param, out_path)

            end = time.perf_counter()
            stats.update(
                t_pre=info.get("pre_ms", 0.0),
                t_gpu=info.get("gpu_ms", 0.0),
                t_post=info.get("post_ms", 0.0),
                t_total_ms=(end - start) * 1000.0,
                t_wait=wait_ms,
            )
            previous_end = end
    except Exception:
        error_text = traceback.format_exc()
        logging.critical("Inference process crashed.")
        logging.error(error_text)
        if not model_ready_evt.is_set():
            try:
                init_status_queue.put_nowait({"ok": False, "message": error_text})
            except Exception:
                pass
            model_ready_evt.set()
        else:
            try:
                init_status_queue.put_nowait({"ok": False, "message": error_text})
            except Exception:
                pass
        stop_evt.set()
    finally:
        try:
            if buffer:
                buffer.close()
            if json_logger:
                json_logger.close()
            if saver:
                saver.close()
            if auto_labeler:
                auto_labeler.close()
        except Exception:
            pass


def run_from_args(args: RunnerArgs) -> int:
    _set_process_priority()
    enable_mmcss()

    try:
        screen_rect = _probe_monitor_rect(args.monitor_index)
    except CaptureAccessError as exc:
        raise PipelineError(
            f"Screen capture initialization failed: {exc}. "
            "Try running from a normal Windows desktop session instead of a restricted or remote capture context."
        ) from exc

    screen_w, screen_h = screen_rect[2], screen_rect[3]
    capture_region, final_shape, offset_x, offset_y = _resolve_capture_region(screen_rect, args)

    lock = multiprocessing.Lock()
    cond = multiprocessing.Condition(lock)
    stop_evt = multiprocessing.Event()
    model_ready_evt = multiprocessing.Event()
    init_status_queue = Queue(maxsize=4)
    shared_read_count = Value(c_longlong, 0)
    shared_timings = Array(c_double, [0.0] * 6)
    buffer = None
    viz_queue = None
    viz_proc = None
    infer_proc = None
    capturer = None
    try:
        buffer = SharedTripleBuffer(shape=final_shape, name="yolo_shm_triple_region", create=True)

        if args.visualize:
            viz_queue = Queue(maxsize=2)
            viz_proc = VisualizerProcess(shared_queue=viz_queue, window_name="YOLO Overlay", width=screen_w, height=screen_h)
            viz_proc.start()

        infer_proc = Process(
            target=inference_process_target,
            args=(args, stop_evt, buffer.name, final_shape, lock, cond, shared_read_count, shared_timings, model_ready_evt, init_status_queue, viz_queue, offset_x, offset_y),
            name="Proc-Inference",
            daemon=True,
        )
        infer_proc.start()

        logging.info("Waiting for YOLO model to load...")
        if not model_ready_evt.wait(timeout=60.0):
            stop_evt.set()
            raise PipelineError("Timed out while waiting for the YOLO worker to initialize.")

        init_result = None
        try:
            init_result = init_status_queue.get_nowait()
        except queue.Empty:
            init_result = None

        if init_result and not init_result.get("ok", False):
            raise PipelineError(
                "YOLO worker failed during initialization. "
                f"Model path: {args.model}. "
                "The loaded file may be corrupted or incompatible."
            )
        if stop_evt.is_set() and infer_proc is not None and not infer_proc.is_alive():
            raise PipelineError("YOLO worker failed during initialization. Check the logs above.")

        logging.info("Pipeline ready. Starting capture...")
        capturer = FrameCapturer(monitor_index=args.monitor_index, capture_hz=args.capture_hz, region=capture_region)
        total_writes = 0
        last_writes = 0
        last_reads = 0
        start_t = time.perf_counter()
        stats_t = time.perf_counter()
        integral = 0.0
        last_apply_hz = float(args.capture_hz)

        while not stop_evt.is_set():
            worker_error = _pop_worker_error(init_status_queue)
            if worker_error is not None:
                raise PipelineError(f"YOLO worker crashed during runtime. {worker_error}")
            if infer_proc is not None and not infer_proc.is_alive():
                raise PipelineError(f"YOLO worker exited unexpectedly with code {infer_proc.exitcode}.")
            if args.max_run_seconds > 0 and (time.perf_counter() - start_t) > args.max_run_seconds:
                break
            frame_bgr, meta = capturer.grab()
            buffer.put(frame_bgr, float(meta.timestamp_ns), meta.monitor_rect, lock, cond)
            total_writes += 1

            now = time.perf_counter()
            elapsed = now - stats_t
            if elapsed >= args.stats_interval:
                current_reads = shared_read_count.value
                fps_grab = (total_writes - last_writes) / elapsed
                fps_proc = (current_reads - last_reads) / elapsed
                fps_drop = max(0.0, ((total_writes - last_writes) - (current_reads - last_reads)) / elapsed)
                last_writes = total_writes
                last_reads = current_reads
                stats_t = now
                with shared_timings.get_lock():
                    t_pre, t_gpu, t_post, t_ovhd, t_total, t_wait = shared_timings[:]
                logging.info(
                    "FLOW: Grab=%4.1f Proc=%4.1f Drop=%4.1f Cmd=%4.1fHz | TIME(ms): Wait=%4.1f + Total=%4.1f (Pre=%.1f GPU=%.1f Post=%.1f Ovhd=%.1f)",
                    fps_grab,
                    fps_proc,
                    fps_drop,
                    last_apply_hz,
                    t_wait,
                    t_total,
                    t_pre,
                    t_gpu,
                    t_post,
                    t_ovhd,
                )
                if args.auto_capture and (now - start_t) > args.auto_capture_warmup_s:
                    last_apply_hz, integral = _apply_auto_capture(args, capturer, last_apply_hz, integral, fps_drop, t_total, elapsed)

        worker_error = _pop_worker_error(init_status_queue)
        if worker_error is not None:
            raise PipelineError(f"YOLO worker crashed during runtime. {worker_error}")
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received.")
    finally:
        stop_evt.set()
        with cond:
            cond.notify_all()
        if infer_proc is not None:
            infer_proc.join(timeout=2.0)
        if infer_proc is not None and infer_proc.is_alive():
            infer_proc.terminate()
            infer_proc.join(timeout=1.0)
        if viz_proc:
            viz_proc.stop()
            viz_proc.join(timeout=1.0)
            if viz_proc.is_alive():
                viz_proc.terminate()
                viz_proc.join(timeout=1.0)
        if capturer is not None:
            capturer.close()
        if buffer is not None:
            try:
                buffer.close()
            finally:
                buffer.unlink()
        logging.info("Clean shutdown completed.")

    return 0


def _pop_worker_error(status_queue: Queue) -> Optional[str]:
    try:
        status = status_queue.get_nowait()
    except queue.Empty:
        return None
    if isinstance(status, dict) and not status.get("ok", False):
        message = str(status.get("message") or "").strip()
        if message:
            last_line = message.splitlines()[-1]
            return f"Check logs above. Last error: {last_line}"
        return "Check logs above."
    return None


def _set_process_priority() -> None:
    try:
        process = psutil.Process(os.getpid())
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        print(f"[MainProc] Priority set to HIGH. PID={os.getpid()}")
    except Exception as exc:
        print(f"[MainProc] Failed to set priority: {exc}")


def _probe_monitor_rect(monitor_index: int) -> Tuple[int, int, int, int]:
    logging.info("Probing monitor size...")
    capturer = FrameCapturer(monitor_index)
    try:
        return capturer.monitor_rect
    finally:
        capturer.close()


def _resolve_capture_region(screen_rect: Tuple[int, int, int, int], args: RunnerArgs):
    screen_w, screen_h = screen_rect[2], screen_rect[3]
    capture_region = None
    offset_x, offset_y = 0, 0
    final_shape = (screen_h, screen_w, 3)
    if args.roi_square:
        radius = args.roi_radius_px if args.roi_radius_px > 0 else min(screen_w, screen_h) // 3
        cx, cy = screen_w // 2, screen_h // 2
        left = max(0, cx - radius)
        top = max(0, cy - radius)
        right = min(screen_w, cx + radius)
        bottom = min(screen_h, cy + radius)
        capture_region = (left, top, right, bottom)
        offset_x, offset_y = left, top
        final_shape = (bottom - top, right - left, 3)
        logging.info("ROI mode enabled: region=%s shape=%s", capture_region, final_shape)
    else:
        logging.info("Full screen mode enabled.")
    return capture_region, final_shape, offset_x, offset_y


def _offset_boxes(boxes, offset_x: int, offset_y: int):
    shifted = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        kpts_xy = None
        if box.kpts_xy is not None:
            kpts_xy = tuple((kx + offset_x, ky + offset_y) for kx, ky in box.kpts_xy)
        shifted.append(
            type(box)(
                cls_id=box.cls_id,
                cls_name=box.cls_name,
                conf=box.conf,
                xyxy=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                kpts_xy=kpts_xy,
                kpts_conf=box.kpts_conf,
            )
        )
    return shifted


def _auto_label_input(frame_bgr: np.ndarray, boxes, args: RunnerArgs):
    if args.roi_square:
        return frame_bgr, boxes

    frame_h, frame_w = frame_bgr.shape[:2]
    radius = args.roi_radius_px if args.roi_radius_px > 0 else min(frame_w, frame_h) // 3
    cx, cy = frame_w // 2, frame_h // 2
    left = max(0, cx - radius)
    top = max(0, cy - radius)
    right = min(frame_w, cx + radius)
    bottom = min(frame_h, cy + radius)
    cropped = frame_bgr[top:bottom, left:right]
    return cropped, _clip_boxes_to_region(boxes, left, top, right - left, bottom - top)


def _clip_boxes_to_region(boxes, left: int, top: int, width: int, height: int):
    clipped = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        nx1 = max(0.0, min(float(width), float(x1) - left))
        ny1 = max(0.0, min(float(height), float(y1) - top))
        nx2 = max(0.0, min(float(width), float(x2) - left))
        ny2 = max(0.0, min(float(height), float(y2) - top))
        if nx2 - nx1 <= 1.0 or ny2 - ny1 <= 1.0:
            continue
        kpts_xy = None
        if box.kpts_xy is not None:
            kpts_xy = tuple((kx - left, ky - top) for kx, ky in box.kpts_xy)
        clipped.append(
            type(box)(
                cls_id=box.cls_id,
                cls_name=box.cls_name,
                conf=box.conf,
                xyxy=(nx1, ny1, nx2, ny2),
                kpts_xy=kpts_xy,
                kpts_conf=box.kpts_conf,
            )
        )
    return clipped


def _apply_auto_capture(args, capturer, last_apply_hz: float, integral: float, fps_drop: float, t_total: float, elapsed: float):
    safe_total_ms = max(1.0, t_total)
    theoretical_max_fps = 1000.0 / safe_total_ms
    dynamic_max_hz = theoretical_max_fps * 1.2
    final_max_hz = min(float(args.cap_max_hz), dynamic_max_hz)
    err = fps_drop - float(args.target_drop_fps)
    if err < 0 and last_apply_hz > theoretical_max_fps * 1.1:
        err = 0.0
    if abs(err) < float(args.deadband):
        err = 0.0
    integral += err * elapsed
    integral = max(-float(args.integral_limit), min(float(args.integral_limit), integral))
    delta = float(args.kp) * err + float(args.ki) * integral
    if abs(delta) >= float(args.min_apply_delta_hz):
        raw_new_hz = last_apply_hz - delta
        new_hz = max(float(args.cap_min_hz), min(final_max_hz, raw_new_hz))
        if abs(new_hz - last_apply_hz) >= 0.5:
            capturer.set_capture_hz(new_hz)
            last_apply_hz = new_hz
    return last_apply_hz, integral
