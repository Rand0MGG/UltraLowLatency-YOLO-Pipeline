高性能实时 YOLO 目标检测流水线 (High-Performance Real-Time YOLO Detection Pipeline)
**重要声明**：本项目纯粹为一个跨学科的计算机视觉基础设施项目，旨在探索极限性能下的视觉流水线设计。应用场景包括但不限于：自动化监控、数字人体工学评估以及行为捕捉等。严禁将本项目与任何形式的游戏外挂或破坏软件生态平衡的非法程序混为一谈。

项目简介 本项目是一个基于 Python 开发的极低延迟计算机视觉系统，专门用于在本地电脑或局域网极低延迟串流环境（基于 Sunshine 和 Moonlight）下，实时识别显示器输出画面中的人物实体与人体姿态结构。

## 核心特性
*   **极致低延迟的硬件级采集**：采用基于 Windows DXGI Desktop Duplication API 的 `dxcam` 库，在 1080p 分辨率下可实现 240+ FPS 的极速零拷贝屏幕像素抓取，彻底突破传统 GDI 采集的性能瓶颈 [2, 1]。
*   **全解耦多进程架构**：系统彻底绕过了 Python 的全局解释器锁（GIL），将屏幕采集、YOLO 张量推理与前端图形渲染拆分为相互独立、并行运行的系统进程 [1]。
*   **无锁三重缓冲共享内存**：底层实现了基于 `multiprocessing.shared_memory` 的零拷贝（Zero-copy）数据总线 [1]。该结构消除了海量高清图像矩阵在进程间传递时的序列化与深拷贝开销，确保了纳秒级的进程间通信。
*   **基于 PID 的自适应帧率控制**：内置 PID 控制器能够根据 GPU 当前的推理耗时，动态且平滑地调节前端画面的捕获频率 [1]。这有效避免了数据积压（Buffer Bloat），在任何硬件负载下都能保持流水线的最佳供需平衡。
*   **卡尔曼滤波与时序平滑**：集成了离散卡尔曼滤波器（Kalman Filter），专门用于消除 YOLO 目标检测模型固有的高频坐标跳变与毛刺，输出极度平滑的检测边界框与骨骼关键点 [1]。
*   **非侵入式透明渲染蒙层**：通过 PyQt5 构建了一个无边框、全局置顶且支持“鼠标穿透”的透明渲染层。该视觉蒙层完全独立运行，不会拦截或干扰用户对底层宿主程序的任何物理输入控制 [1]。

## 局域网分布式运行建议
系统完美兼容本地直连运行与局域网分布式运行。在分布式架构中，建议采用以下配置以获得极致的亚毫秒级延迟体验：
*   **主机端**：部署 Sunshine 串流服务端，Sunshine 将直接调用现代 GPU 的专用硬件编码器（如 NVENC、AMF 或 QuickSync）进行极低延迟的视频流压缩 [3]。
*   **客户端（推理工作站）**：运行 Moonlight 客户端进行硬件解码，并挂载本系统进行屏幕抓取与推理。
*   **物理链路**：强烈建议串流主机与推理客户端均使用 CAT5e 及以上规格的千兆有线以太网进行全双工直连。有线连接能从根本上消除 Wi-Fi 等无线信道带来的数据包抖动与高频延迟突波 [3]。

## 启动模式与参数指南
系统入口脚本为 `tools/run_capture_yolo.py`，支持高度灵活的参数配置：

*   **实时视觉反馈模式（Overlay）**：
    使用 `--vis` 参数启动，系统将在屏幕上方以全透明蒙层的形式实时绘制检测框与骨架节点 [1]。
    *注意：为了确保坐标映射正确，请将操作系统的显示器缩放比例设置为 100%。如果用于捕获全屏程序，建议将其设置为“无边框窗口化（Borderless Windowed）”模式，以防 DXGI 采集被独占全屏阻塞* [1]。
*   **平滑追踪模式**：
    搭配 `--smooth` 参数激活卡尔曼滤波算法，可通过修改 `--smooth-alpha`（默认 0.6）来控制坐标滤波的强度，兼顾平滑度与跟手性 [1]。
*   **极限性能测试（Headless 模式）**：
    若不添加 `--vis` 参数（例如 `python tools/run_capture_yolo.py --model yolo11n.pt --capture-hz 60`），系统将关闭所有 UI 渲染开销，仅执行后台采集与张量推理 [1]。此模式非常适合用于纯数据监控和系统极限吞吐量测试。
*   **局部 ROI 加速**：
    利用 `--roi-square` 参数，可强制系统仅截取屏幕正中心的方形兴趣区域（Region of Interest）传递给模型，能成倍降低 GPU 运算量 [1]。

## 调试与性能调优
建议用户在终端中观察系统打印的实时性能日志：
1.  **Wait 延迟**：如果日志显示等待时间 > 2ms，通常意味着采集端的抓取循环出现了滞后 [1]。
2.  **GPU 耗时**：如果单帧 GPU 推理时间持续 > 10ms，表明显卡算力已成为流水线瓶颈 [1]。
    *   *调优方案*：建议附加 `--half` 参数开启 FP16 半精度推理优化，或将 YOLO 模型导出为 TensorRT 的 `.engine` 格式以彻底释放运算潜能 [1]。
3.  **数据落盘记录**：使用 `--save-every` 参数可将含有检测结果的图像与结构化的 JSONL 日志异步保存至 `debug/` 目录下，用于后期的数据分析与模型微调 [1]。

项目结构 
cv_capture_detect_v12/
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

快速开始 所有的运行逻辑都通过tools/run_capture_yolo.py入口进行管理。

基础运行（Headless模式） 仅进行采集和推理，控制台输出FPS统计，不显示画面。适合性能测试。 命令：python tools/run_capture_yolo.py --model yolo11n.pt --capture-hz 60

开启可视化（Overlay） 开启透明置顶窗口，在屏幕上实时绘制检测框和骨骼关键点。 命令：python tools/run_capture_yolo.py --vis --conf 0.4

开启平滑滤波 启用卡尔曼滤波，减少检测框的视觉抖动。 命令：python tools/run_capture_yolo.py --vis --smooth --smooth-alpha 0.6

自动帧率控制（PID） 启用PID控制器，根据GPU负载自动调整采集速度，保持系统低延迟。 命令：python tools/run_capture_yolo.py --vis --auto-capture --target-drop-fps 4.0

参数详解 --monitor-index：监视器索引，1为主屏。 --model：模型路径，默认为yolo11n.pt，支持.engine格式。 --imgsz：推理图片尺寸，默认为640。 --vis：开启PyQt透明覆盖层可视化的开关。 --smooth：开启卡尔曼滤波平滑的开关。 --auto-capture：开启PID动态帧率控制的开关。 --roi-square：仅检测屏幕中心正方形区域的开关，可提升速度。 --save-every：每隔N秒保存一张推理结果图，0表示不保存。 --half：开启FP16半精度推理的开关，需GPU支持。

性能调优指南 如果遇到延迟高或掉帧的情况，请检查控制台日志。 如果Wait时间长于2ms，说明采集端跟不上，请检查capture_hz设置或DXCAM兼容性。 如果GPU时间长于10ms，说明显卡瓶颈，尝试减小imgsz（如480），使用更小的模型，或导出TensorRT模型。 如果Overhead（Ovhd）高，可能是Python进程调度问题，尝试关闭不必要的后台程序。

常见问题 如果PyQt报错或黑屏，请确保显示器缩放比例为100%，或在Python属性中覆盖高DPI缩放行为。 如果dxcam无法采集，某些全屏独占游戏会阻止DXGI采集，请尝试将游戏设置为无边框窗口模式。




