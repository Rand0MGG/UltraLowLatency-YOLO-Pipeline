# src/pipeline/shared_buffer.py
import numpy as np
from multiprocessing import shared_memory, Lock, Condition
from typing import Tuple, Optional

# Metadata: timestamp(float64) + rect(4 * int64 -> actually stored as float64 for simplicity)
# 我们统一用 float64 存储 meta，取出来再转 int
META_COUNT = 5  
META_DTYPE = np.float64
META_BYTES = META_COUNT * 8

# Control Header: [middle_index(int32), has_new_data(int32)]
CTRL_SIZE = 2
CTRL_DTYPE = np.int32
CTRL_BYTES = CTRL_SIZE * 4 # 8 bytes

# Memory Alignment Padding (Optional, kept simple here)
HEADER_OFFSET = 128 # 预留 128 字节头部，足够放 Control

class SharedTripleBuffer:
    """
    基于共享内存的三缓冲实现 (Triple Buffering)。
    Writer(Capture) 和 Reader(Inference) 拥有各自的私有 buffer，
    仅在交换指针时需要极短的锁，数据拷贝完全并行。
    """
    def __init__(self, shape: Tuple[int, int, int], create: bool = True, name: str = "shared_buffer_triple"):
        self.shape = shape
        self.dtype = np.uint8
        self.size = int(np.prod(shape))
        self.name = name
        
        # 计算单帧大小
        self.frame_bytes = self.size * np.dtype(self.dtype).itemsize
        
        # 总大小计算: 
        # Header (128B) + 
        # Meta (3 * META_BYTES) + 
        # Frames (3 * FRAME_BYTES)
        self.meta_section_offset = HEADER_OFFSET
        self.frame_section_offset = self.meta_section_offset + (3 * META_BYTES)
        
        self.total_size = self.frame_section_offset + (3 * self.frame_bytes)

        if create:
            # 【创建者模式】：Capture 进程使用
            try:
                try:
                    old = shared_memory.SharedMemory(name=self.name)
                    old.close()
                    old.unlink()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(create=True, size=self.total_size, name=self.name)
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(name=self.name)
                
            # 初始化控制头
            # middle_index = 2 (0给Writer, 1给Reader, 2做Middle)
            # has_new_data = 0 (False)
            ctrl_arr = np.ndarray((2,), dtype=CTRL_DTYPE, buffer=self.shm.buf[:8])
            ctrl_arr[0] = 2 
            ctrl_arr[1] = 0
            
        else:
            # 【连接者模式】：Inference 进程使用
            self.shm = shared_memory.SharedMemory(name=self.name)

        # --- 映射视图 ---
        
        # 1. 控制区 [middle_index, has_new_data]
        self.ctrl_array = np.ndarray((2,), dtype=CTRL_DTYPE, buffer=self.shm.buf[:8])

        # 2. 元数据区 (3 份)
        self.meta_arrays = []
        for i in range(3):
            offset = self.meta_section_offset + (i * META_BYTES)
            self.meta_arrays.append(
                np.ndarray(META_COUNT, dtype=META_DTYPE, buffer=self.shm.buf[offset : offset + META_BYTES])
            )

        # 3. 图像帧区 (3 份)
        self.frame_arrays = []
        for i in range(3):
            offset = self.frame_section_offset + (i * self.frame_bytes)
            self.frame_arrays.append(
                np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf[offset : offset + self.frame_bytes])
            )

        # --- 私有状态 (无需共享) ---
        if create:
            self.my_index = 0 # Capture 初始持有 0
        else:
            self.my_index = 1 # Inference 初始持有 1
            
        # Middle 初始持有 2 (在 ctrl_array[0] 中)

    def put(self, frame: np.ndarray, timestamp: float, rect: Tuple[int, int, int, int], lock: Lock, cond: Condition):
        """
        Capture 进程调用：写入数据（无锁），然后交换指针（极短锁）
        """
        idx = self.my_index
        
        # 1. 【无锁操作】写入私有 Buffer (耗时操作在这里！)
        # 直接使用 copyto 加速
        np.copyto(self.frame_arrays[idx], frame)
        
        # 写入元数据
        ma = self.meta_arrays[idx]
        ma[0] = timestamp
        ma[1] = rect[0]
        ma[2] = rect[1]
        ma[3] = rect[2]
        ma[4] = rect[3]

        # 2. 【临界区】交换指针 (极快)
        with lock:
            # 交换 my_index <-> middle_index
            middle_idx = self.ctrl_array[0]
            
            self.ctrl_array[0] = idx     # Middle 拿走我的 (最新的)
            self.my_index = middle_idx   # 我拿走 Middle (旧的/废弃的)
            
            # 标记有新数据
            self.ctrl_array[1] = 1
            
            # 通知消费者
            cond.notify_all()
            
        # put 结束，现在 self.my_index 指向了刚才的 middle buffer，
        # Capture 进程下次直接往里写即可，无需等待 Reader。

    def get(self, lock: Lock, cond: Condition, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """
        Inference 进程调用：检查是否有新数据，若有则交换指针，然后无锁读取
        """
        swapped = False
        
        # 1. 【临界区】检查并交换指针
        with cond:
            # 检查标志位
            if self.ctrl_array[1] == 0:
                # 如果没有新数据，等待
                if not cond.wait(timeout=timeout):
                    return None # 超时
            
            # 再次检查 (防止惊群效应，虽然这里是一对一)
            if self.ctrl_array[1] == 1:
                # 交换 middle_index <-> my_index
                middle_idx = self.ctrl_array[0]
                
                temp = self.my_index
                self.my_index = middle_idx  # 我拿到最新的
                self.ctrl_array[0] = temp   # Middle 拿回我刚才读完的(旧的)
                
                # 重置标志位
                self.ctrl_array[1] = 0
                swapped = True
            else:
                # 即使被唤醒，如果标志位是 0 (极其罕见)，说明可能被抢了或者伪唤醒
                # 此时为了不返回 None 导致逻辑中断，我们可以返回上一帧，或者继续 wait
                # 为了简单，这里如果没抢到新数据，就返回 None 或继续循环
                # 这里选择返回 None 让外层 loop 重试
                pass

        if not swapped:
            # 没拿到新数据 (超时)
            return None

        # 2. 【无锁操作】读取私有 Buffer
        # 现在 self.my_index 指向的是刚才 Writer 放入的那块内存
        # 我们可以安全地读，Writer 绝对不会碰这块内存，因为它在 Reader 手里。
        
        idx = self.my_index
        
        # Zero-Copy: 直接返回视图！
        # 注意：只要 Inference 进程不调用下一次 get()，这块内存就是安全的。
        # 如果 Inference 需要长期保存该图，请在外部 .copy()。
        # 但对于 YOLO 推理，直接传这个 view 进去通常是安全的（只要推理是同步的）。
        frame_view = self.frame_arrays[idx]
        
        ma = self.meta_arrays[idx]
        ts = float(ma[0])
        rect = (int(ma[1]), int(ma[2]), int(ma[3]), int(ma[4]))
        
        return frame_view, ts, rect
            
    def close(self):
        self.shm.close()
        
    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass