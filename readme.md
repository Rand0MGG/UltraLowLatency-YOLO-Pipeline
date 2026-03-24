***
# UltraLowLatency-YOLO-Pipeline

[中文文档往下看 / Scroll down for Chinese version](#中文版文档)

<p align="center">
  <img src="media/1.gif" width="70%" title="Demo 1" />
  <br><br>
  <img src="media/2.gif" width="70%" title="Demo 2" />
  <br><br>
  <img src="media/3.png" width="70%" title="Screenshot" />
</p>

**Important Disclaimer**: This project is purely a cross-disciplinary computer vision infrastructure project, aimed at exploring pipeline design under extreme performance constraints. Applicable scenarios include, but are not limited to, automated monitoring, digital ergonomics evaluation, and motion capture. It is strictly prohibited to use this project for any form of game cheating or illegal software that disrupts the software ecosystem.

## Overview
This project is an ultra-low latency computer vision system developed in Python. It is specifically designed for real-time identification of human entities and posture structures from monitor output, optimized for both local environments and ultra-low latency LAN streaming setups (based on Sunshine and Moonlight).

## Core Features
* **Extreme Low-Latency Hardware Capture**: Utilizes `dxcam` based on the Windows DXGI Desktop Duplication API, achieving 240+ FPS zero-copy screen pixel grabbing at 1080p, completely breaking the performance bottlenecks of traditional GDI capture.
* **Fully Decoupled Multi-Process Architecture**: The system completely bypasses the Python Global Interpreter Lock (GIL), splitting screen capture, YOLO tensor inference, and frontend graphic rendering into independent, parallel OS processes.
* **Lock-Free Triple-Buffered Shared Memory**: Implements a zero-copy data bus based on `multiprocessing.shared_memory`. This eliminates serialization and deep-copy overhead when transferring massive HD image matrices between processes, ensuring nanosecond-level IPC (Inter-Process Communication).
* **PID-Based Adaptive Framerate Control**: A built-in PID controller dynamically and smoothly adjusts the frontend capture frequency based on current GPU inference latency. This effectively prevents Buffer Bloat, maintaining optimal supply-demand balance under any hardware load.
* **Kalman Filtering & Temporal Smoothing**: Integrates a discrete Kalman Filter specifically to eliminate the inherent high-frequency coordinate jumps and jitter of YOLO object detection models, outputting extremely smooth bounding boxes and skeletal keypoints.
* **Non-Intrusive Transparent Overlay**: Built a borderless, globally top-level transparent rendering layer using PyQt5 that supports "mouse passthrough". This visual overlay runs completely independently and does not intercept or interfere with any physical inputs to the underlying host application.

## LAN Distributed Running Guide
The system works perfectly for both local direct-run and LAN distributed architectures. For distributed setups, the following configuration is recommended for sub-millisecond latency:
* **Host**: Deploy a Sunshine streaming server, utilizing dedicated GPU hardware encoders (NVENC, AMF, or QuickSync) for extreme low-latency video stream compression.
* **Client (Inference Workstation)**: Run the Moonlight client for hardware decoding, and mount this system for screen grabbing and inference.
* **Physical Link**: It is highly recommended to use CAT5e or higher Gigabit wired Ethernet for full-duplex direct connection between the host and client. Wired connections eliminate packet jitter and high-frequency latency spikes caused by wireless channels like Wi-Fi.

## Project Structure
```text
cv_capture_detect/
├── src/
│   ├── capture/
│   │   └── capture_frame.py    # DXCAM wrapper, supports Video Mode non-blocking capture
│   ├── inference/
│   │   └── yolo_detector.py    # YOLOv11 inference wrapper (perf stats & keypoints)
│   ├── pipeline/
│   │   ├── runner_mp.py        # Core multiprocess scheduler (Process/Lock/Event)
│   │   ├── shared_buffer.py    # Ring/Single frame buffer via SharedMemory
│   │   ├── visualizer.py       # PyQt5 transparent overlay (Queue receiver)
│   │   ├── box_filter.py       # Kalman filter & IOU matching
│   │   ├── async_saver.py      # Async Disk IO for image saving
│   │   ├── logging_jsonl.py    # Async JSONL structured logging
│   │   ├── perf_stats.py       # Performance statistics (EMA smoothing)
│   │   └── args.py             # Dataclass for configuration arguments
│   └── pipeline/draw.py        # OpenCV drawing utility
├── tools/
│   └── run_capture_yolo.py     # CLI entry point
├── debug/                      # (Auto-generated) Logs and screenshots
└── README.md
```

## Quick Start
All execution logic is managed through the `tools/run_capture_yolo.py` entry point.

**Basic Run (Headless Mode)**
Only performs capture and inference, outputting FPS stats to the console without rendering graphics. Ideal for performance testing.
```bash
python tools/run_capture_yolo.py --model yolo11n.pt --capture-hz 60
```

**Enable Visualization (Overlay)**
Opens a transparent, always-on-top window to draw bounding boxes and skeletal keypoints on the screen in real-time.
```bash
python tools/run_capture_yolo.py --vis --conf 0.4
```

**Enable Smooth Filtering**
Enables the Kalman filter to reduce visual jitter of the detection boxes.
```bash
python tools/run_capture_yolo.py --vis --smooth --smooth-alpha 0.6
```

**Auto Framerate Control (PID)**
Applies the PID controller to automatically adjust the capture speed based on GPU load, maintaining low system latency.
```bash
python tools/run_capture_yolo.py --vis --auto-capture --target-drop-fps 4.0
```

## Performance Tuning
If you experience high latency or frame drops, check the console logs:
1.  **High `Wait` Time**: If wait time > 2ms, the capture end is lagging. Check `--capture-hz` or `dxcam` compatibility.
2.  **High `GPU` Time**: If GPU inference > 10ms, your GPU is the bottleneck. Try reducing `--imgsz` (e.g., 480), using a smaller model, or exporting to a TensorRT `.engine` format.
3.  **High `Ovhd` (Overhead)**: Likely a Python process scheduling issue. Try closing unnecessary background applications.

---

<span id="中文版文档"></span>
# 高性能实时 YOLO 目标检测流水线

**重要声明**：本项目纯粹为一个跨学科的计算机视觉基础设施项目，旨在探索极限性能下的视觉流水线设计。应用场景包括但不限于：自动化监控、数字人体工学评估以及行为捕捉等。严禁将本项目与任何形式的游戏外挂或破坏软件生态平衡的非法程序混为一谈。

## 核心特性
* **极致低延迟的硬件级采集**：采用基于 Windows DXGI Desktop Duplication API 的 `dxcam` 库，在 1080p 分辨率下可实现 240+ FPS 的极速零拷贝屏幕像素抓取，彻底突破传统 GDI 采集的性能瓶颈。
* **全解耦多进程架构**：系统彻底绕过了 Python 的全局解释器锁（GIL），将屏幕采集、YOLO 张量推理与前端图形渲染拆分为相互独立、并行运行的系统进程。
* **无锁三重缓冲共享内存**：底层实现了基于 `multiprocessing.shared_memory` 的零拷贝（Zero-copy）数据总线。该结构消除了海量高清图像矩阵在进程间传递时的序列化与深拷贝开销，确保了纳秒级的进程间通信。
* **基于 PID 的自适应帧率控制**：内置 PID 控制器能够根据 GPU 当前的推理耗时，动态且平滑地调节前端画面的捕获频率。这有效避免了数据积压（Buffer Bloat），在任何硬件负载下都能保持流水线的最佳供需平衡。
* **卡尔曼滤波与时序平滑**：集成了离散卡尔曼滤波器（Kalman Filter），专门用于消除 YOLO 目标检测模型固有的高频坐标跳变与毛刺，输出极度平滑的检测边界框与骨骼关键点。
* **非侵入式透明渲染蒙层**：通过 PyQt5 构建了一个无边框、全局置顶且支持“鼠标穿透”的透明渲染层。该视觉蒙层完全独立运行，不会拦截或干扰用户对底层宿主程序的任何物理输入控制。

## 局域网分布式运行建议
系统完美兼容本地直连运行与局域网分布式运行。在分布式架构中，建议采用以下配置以获得极致的亚毫秒级延迟体验：
* **主机端**：部署 Sunshine 串流服务端，利用专用硬件编码器（如 NVENC、AMF 或 QuickSync）进行极低延迟的视频流压缩。
* **客户端（推理工作站）**：运行 Moonlight 客户端进行硬件解码，并挂载本系统进行屏幕抓取与推理。
* **物理链路**：强烈建议串流主机与推理客户端均使用 CAT5e 及以上规格的千兆有线以太网进行全双工直连。有线连接能从根本上消除 Wi-Fi 等无线信道带来的数据包抖动与高频延迟突波。

## 项目结构
```text
cv_capture_detect/
├── src/
│   ├── capture/
│   │   └── capture_frame.py    # DXCAM 封装，支持 Video Mode 非阻塞采集
│   ├── inference/
│   │   └── yolo_detector.py    # YOLOv11 推理封装，包含性能统计与关键点处理
│   ├── pipeline/
│   │   ├── runner_mp.py        # 多进程核心调度器 (Process/Lock/Event 管理)
│   │   ├── shared_buffer.py    # 基于 SharedMemory 的环形/单帧缓冲区管理
│   │   ├── visualizer.py       # PyQt5 透明覆盖层，通过 Queue 接收推理结果
│   │   ├── box_filter.py       # 卡尔曼滤波与 IOU 匹配算法
│   │   ├── async_saver.py      # 异步磁盘 IO (保存图片)
│   │   ├── logging_jsonl.py    # 异步 JSONL 结构化日志记录
│   │   ├── perf_stats.py       # 性能统计 (EMA 平滑)
│   │   └── args.py             # 配置参数 Dataclass 定义
│   └── pipeline/draw.py        # OpenCV 绘图工具库
├── tools/
│   └── run_capture_yolo.py     # CLI 启动入口
├── debug/                      # (自动生成) 存放日志和截图
└── README.md
```

## 快速开始
所有的运行逻辑都通过 `tools/run_capture_yolo.py` 入口进行管理。

**基础运行（Headless模式）**
仅进行采集和推理，控制台输出 FPS 统计，不显示画面。适合性能测试。
```bash
python tools/run_capture_yolo.py --model yolo11n.pt --capture-hz 60
```

**开启可视化（Overlay）**
开启透明置顶窗口，在屏幕上实时绘制检测框和骨骼关键点。
```bash
python tools/run_capture_yolo.py --vis --conf 0.4
```

**开启平滑滤波**
启用卡尔曼滤波，减少检测框的视觉抖动。
```bash
python tools/run_capture_yolo.py --vis --smooth --smooth-alpha 0.6
```

**自动帧率控制（PID）**
启用 PID 控制器，根据 GPU 负载自动调整采集速度，保持系统低延迟。
```bash
python tools/run_capture_yolo.py --vis --auto-capture --target-drop-fps 4.0
```

## 调试与性能调优指南
如果遇到延迟高或掉帧的情况，请检查控制台日志：
1.  **Wait 延迟**：如果日志显示等待时间 > 2ms，通常意味着采集端的抓取循环出现了滞后。请检查 `--capture-hz` 设置或 `dxcam` 兼容性。
2.  **GPU 耗时**：如果单帧 GPU 推理时间持续 > 10ms，表明显卡算力已成为流水线瓶颈。建议附加 `--half` 参数开启 FP16 半精度推理优化，缩小 `--imgsz`（例如 480），或将 YOLO 模型导出为 TensorRT 的 `.engine` 格式。
3.  **Overhead (Ovhd)**：如果 Ovhd 过高，可能是 Python 进程调度问题，尝试关闭不必要的后台程序以释放 CPU 资源。

***
