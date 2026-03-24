# src/pipeline/runner_mp.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import time
import multiprocessing
import queue
import traceback
import os
import psutil
import ctypes
from ctypes import wintypes

from ctypes import c_double, c_longlong
from multiprocessing import Process, Event, Value, Array, Lock, Condition, Queue
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# --- 核心模块 ---
from capture.capture_frame import FrameCapturer
from inference.yolo_detector import YoloDetector
from pipeline.shared_buffer import SharedTripleBuffer
from pipeline.logging_jsonl import JsonlLogger
from pipeline.args import RunnerArgs

# --- 辅助模块 ---
from pipeline.async_saver import AsyncSaver
from pipeline.perf_stats import PerfStats
from pipeline.visualizer import VisualizerProcess
from pipeline.box_filter import BoxSmoother


# --------------------------------------------------------------------------
# Windows MMCSS 调度提权 (仅用于主采集线程)
# --------------------------------------------------------------------------
def enable_mmcss():
    """
    仅为主进程开启游戏模式调度。
    """
    try:
        avrt = ctypes.windll.avrt
        avrt.AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
        avrt.AvSetMmThreadPriority.argtypes = [wintypes.HANDLE, wintypes.DWORD] # type: ignore
        avrt.AvSetMmThreadPriority.restype = wintypes.BOOL # type: ignore
        
        task_name = "Games"
        task_index = wintypes.DWORD(0)
        
        handle = avrt.AvSetMmThreadCharacteristicsW(task_name, ctypes.byref(task_index))
        
        if handle:
            avrt.AvSetMmThreadPriority(handle, 2) # Critical Priority
            print(f"[MMCSS] Main Capture Thread Registered! Handle: {handle}")
            return True
        else:
            print("[MMCSS] Failed. (Run as Admin?)")
            return False
    except Exception as e:
        print(f"[MMCSS] Error: {e}")
        return False


# ---------------------------------------------------------
# 子进程：推理核心逻辑 (已移除所有提权，降噪降耗)
# ---------------------------------------------------------
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
    viz_queue: Optional[Queue],
    offset_x: int,
    offset_y: int
):
    # [修改] 移除了 enable_mmcss() 和 psutil 提权
    # 让推理进程作为普通进程运行，避免抢占 CPU 资源
    # 因为它是消费者，只要队列有数据它就会跑，不需要实时调度权
    
    logging.basicConfig(
        level=logging.INFO,
        format=f"[InferProc-{multiprocessing.current_process().pid}] %(asctime)s %(levelname)s: %(message)s",
    )

    saver = None
    buffer = None
    jl = None
    
    try:
        logging.info(f"Init YOLO: {args.model} on {args.device}")
        detector = YoloDetector(
            model_path=args.model,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            half=args.half,
            classes=args.yolo_classes,
            max_det=args.yolo_max_det,
        )


        buffer = SharedTripleBuffer(shape=frame_shape, name=shm_name, create=False)
        jl = JsonlLogger(Path("debug") / "yolo_log.jsonl") 
        saver = AsyncSaver(save_queue_size=args.save_queue, roi_square=args.roi_square)
        stats = PerfStats(shared_timings)

        box_smoother = None
        if args.smooth:
            logging.info(f"Box Smoothing ENABLED (alpha={args.smooth_alpha})")
            box_smoother = BoxSmoother(alpha=args.smooth_alpha, iou_thresh=0.5)
        else:
            logging.info("Box Smoothing DISABLED (Raw Output)")

        logging.info("Warming up YOLO engine...")
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        detector.infer(dummy_frame)
        logging.info("YOLO Model ready. Signaling main process...")
        model_ready_evt.set()

        last_save_t = 0.0
        save_every_s = float(args.save_every)
        
        t_end_prev = time.perf_counter()
        logging.info("Loop started.")

        while not stop_evt.is_set():
            # --- Step 1: 获取数据 ---
            # 普通进程等待 Cond 变量非常高效，不吃 CPU
            data = buffer.get(lock, cond, timeout=0.5)
            t_start = time.perf_counter()
            t_wait = (t_start - t_end_prev) * 1000.0 

            if data is None:
                t_end_prev = time.perf_counter()
                continue

            frame_bgr, ts_ns, monitor_rect = data
            
            with shared_read_count.get_lock():
                shared_read_count.value += 1

            # --- Step 2: 推理 ---
            boxes, info = detector.infer(frame_bgr)
            
            # --- 坐标还原 ---
            if (offset_x > 0 or offset_y > 0) and boxes:
                shifted = []
                for b in boxes:
                    x1, y1, x2, y2 = b.xyxy
                    nx1, ny1, nx2, ny2 = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                    nkpts_xy = None
                    if b.kpts_xy is not None:
                        nkpts_xy = tuple((kx + offset_x, ky + offset_y) for kx, ky in b.kpts_xy)
                    shifted.append(type(b)(
                        cls_id=b.cls_id, cls_name=b.cls_name, conf=b.conf,
                        xyxy=(nx1, ny1, nx2, ny2), kpts_xy=nkpts_xy, kpts_conf=b.kpts_conf
                    ))
                boxes = shifted

            # --- 平滑 ---
            if box_smoother is not None:
                boxes = box_smoother.update(boxes)

            # --- Step 3: 可视化 ---
            if viz_queue is not None:
                roi_param_to_pass = None
                if args.roi_square:
                    roi_h, roi_w = frame_shape[:2]
                    cx_abs = offset_x + roi_w // 2
                    cy_abs = offset_y + roi_h // 2
                    r_abs = roi_h // 2 
                    roi_param_to_pass = (cx_abs, cy_abs, r_abs)
                try:
                    viz_queue.put_nowait((None, boxes, roi_param_to_pass))
                except queue.Full:
                    pass 

            # --- Step 4: 日志 ---
            box_list = []
            for b in boxes:
                kpts_data = None
                if getattr(b, "kpts_xy", None) is not None:
                    kpts_data = [[float(x), float(y)] for (x, y) in b.kpts_xy]
                box_list.append({
                    "cls_id": int(b.cls_id), "cls_name": str(b.cls_name),
                    "conf": float(b.conf), "xyxy": list(b.xyxy), "kpts_xy": kpts_data
                })
            
            jl.append_record(jl.make_infer_record(
                timestamp_ns=int(ts_ns), monitor_rect=monitor_rect,
                frame_shape=frame_bgr.shape, infer_ms=info.get("gpu_ms", 0),
                boxes=box_list
            ))

            # --- Step 5: 保存 ---
            if save_every_s > 0:
                now_t = time.perf_counter()
                if now_t - last_save_t >= save_every_s:
                    last_save_t = now_t
                    ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                    out_path = Path("debug") / "yolo_samples" / f"sample_{ts_str}_{int(now_t*1000)%1000:03d}.jpg"
                    saver.push(frame_bgr.copy(), boxes, roi_param_to_pass, out_path)

            # --- Step 6: 统计 ---
            t_end = time.perf_counter()
            t_total_ms = (t_end - t_start) * 1000.0
            stats.update(
                t_pre=info.get("pre_ms", 0.0), t_gpu=info.get("gpu_ms", 0.0),
                t_post=info.get("post_ms", 0.0), t_total_ms=t_total_ms, t_wait=t_wait
            )
            t_end_prev = t_end
    
    except Exception:
        logging.critical("Inference Process Crashed!")
        logging.error(traceback.format_exc())
    finally:
        try:
            if buffer: buffer.close()
            if jl: jl.close()
            if saver: saver.close()
        except Exception:
            pass
        logging.info("Exited.")


# ---------------------------------------------------------
# 主进程：采集与控制 (保留提权，但加回 Sleep)
# ---------------------------------------------------------
def run_from_args(args: RunnerArgs) -> int:
    # 1. 进程级提权 (仅主进程)
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print(f"[MainProc] Priority set to HIGH. PID={os.getpid()}")
    except Exception as e:
        print(f"[MainProc] Failed to set priority: {e}")

    # 2. 线程级提权 (仅主线程)
    # 这让我们可以放心地使用 sleep 而不被降级
    enable_mmcss()

    logging.info("Probing monitor size...")
    tmp_cap = FrameCapturer(args.monitor_index)
    full_rect = tmp_cap.monitor_rect 
    tmp_cap.close()
    
    screen_w, screen_h = full_rect[2], full_rect[3]
    logging.info(f"Full Screen: {screen_w}x{screen_h}")

    capture_region = None
    offset_x, offset_y = 0, 0
    final_shape = (screen_h, screen_w, 3) 

    if args.roi_square:
        radius = args.roi_radius_px
        if radius <= 0:
             radius = min(screen_w, screen_h) // 3 
        cx, cy = screen_w // 2, screen_h // 2
        left = max(0, cx - radius)
        top = max(0, cy - radius)
        right = min(screen_w, cx + radius)
        bottom = min(screen_h, cy + radius)
        capture_region = (left, top, right, bottom)
        offset_x, offset_y = left, top
        roi_w = right - left
        roi_h = bottom - top
        final_shape = (roi_h, roi_w, 3)
        logging.info(f"ROI Mode Enabled: Region={capture_region}, Shape={final_shape}")
    else:
        logging.info("Full Screen Mode Enabled")

    lock = multiprocessing.Lock()
    cond = multiprocessing.Condition(lock)
    stop_evt = multiprocessing.Event()
    model_ready_evt = multiprocessing.Event()
    shared_read_count = Value(c_longlong, 0)
    shared_timings = Array(c_double, [0.0] * 6)
    shm_name = "yolo_shm_triple_region"
    buffer = SharedTripleBuffer(shape=final_shape, name=shm_name, create=True)

    viz_queue = None
    viz_proc = None
    if args.visualize:
        logging.info("Visualizer enabled. Starting PyQt overlay...")
        viz_queue = Queue(maxsize=2)
        viz_proc = VisualizerProcess(
            shared_queue=viz_queue, 
            window_name="YOLO Overlay", 
            width=screen_w, 
            height=screen_h
        )
        viz_proc.start()
    else:
        logging.info("Visualizer disabled (Headless mode).")

    p_infer = Process(
        target=inference_process_target,
        args=(args, stop_evt, shm_name, final_shape, lock, cond, shared_read_count, shared_timings, model_ready_evt, viz_queue, offset_x, offset_y),
        name="Proc-Inference",
        daemon=True
    )
    p_infer.start()

    logging.info("Waiting for YOLO model to load...")
    model_ready_evt.wait()
    logging.info("Pipeline ready! Starting capture...")

    capturer = FrameCapturer(monitor_index=args.monitor_index, capture_hz=args.capture_hz, region=capture_region)
    
    total_writes = 0
    last_writes = 0
    last_reads = 0
    t_start = time.perf_counter()
    t_stats = time.perf_counter()
    integ = 0.0
    last_apply_hz = float(args.capture_hz)
    
    logging.info(f"Main Pipeline Started. PID={multiprocessing.current_process().pid}")

    try:
        while not stop_evt.is_set():
            if args.max_run_seconds > 0 and (time.perf_counter() - t_start) > args.max_run_seconds:
                break

            # --- 采集 ---
            frame_bgr, meta = capturer.grab()
            buffer.put(frame_bgr, float(meta.timestamp_ns), meta.monitor_rect, lock, cond)
            total_writes += 1
            
            # --- 统计与PID控制 ---
            now = time.perf_counter()
            dt = now - t_stats
            if dt >= args.stats_interval:
                current_reads = shared_read_count.value
                fps_grab = (total_writes - last_writes) / dt
                fps_proc = (current_reads - last_reads) / dt
                delta_writes = total_writes - last_writes
                delta_reads = current_reads - last_reads
                fps_drop = max(0.0, (delta_writes - delta_reads) / dt)

                last_writes = total_writes
                last_reads = current_reads
                t_stats = now

                with shared_timings.get_lock():
                    t_pre, t_gpu, t_post, t_ovhd, t_total, t_wait = shared_timings[:]

                logging.info(
                    f"FLOW: Grab={fps_grab:4.1f} Proc={fps_proc:4.1f} Drop={fps_drop:4.1f} Cmd={last_apply_hz:4.1f}Hz | "
                    f"TIME(ms): Wait={t_wait:4.1f} + Total={t_total:4.1f} (Pre={t_pre:.1f} GPU={t_gpu:.1f} Post={t_post:.1f} Ovhd={t_ovhd:.1f})"
                )

                if args.auto_capture and (now - t_start) > args.auto_capture_warmup_s:
                    safe_total_ms = max(1.0, t_total) 
                    theoretical_max_fps = 1000.0 / safe_total_ms
                    dynamic_max_hz = theoretical_max_fps * 1.2
                    final_max_hz = min(float(args.cap_max_hz), dynamic_max_hz)

                    err = fps_drop - float(args.target_drop_fps)
                    if err < 0 and (last_apply_hz > theoretical_max_fps * 1.1):
                        err = 0.0
                    if abs(err) < float(args.deadband): err = 0.0
                    
                    integ += err * dt
                    integ = max(-float(args.integral_limit), min(float(args.integral_limit), integ))
                    
                    delta = float(args.kp) * err + float(args.ki) * integ
                    if abs(delta) >= float(args.min_apply_delta_hz):
                        raw_new_hz = last_apply_hz - delta
                        new_hz = max(float(args.cap_min_hz), min(final_max_hz, raw_new_hz))
                        if abs(new_hz - last_apply_hz) >= 0.5:
                            capturer.set_capture_hz(new_hz)
                            last_apply_hz = new_hz
            
            #不在此处加sleep。保证项目的性能。
            #time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("Keyboard Interrupt received.")
    finally:
        stop_evt.set()
        with cond:
            cond.notify_all()
        
        p_infer.join(timeout=2.0)
        if p_infer.is_alive():
            p_infer.terminate()
        if viz_proc:
            viz_proc.stop()
            viz_proc.join(timeout=1.0)
            if viz_proc.is_alive():
                viz_proc.terminate()
        capturer.close()
        buffer.unlink()
        logging.info("Clean shutdown completed.")

    return 0