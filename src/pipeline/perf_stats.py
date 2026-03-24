# src/pipeline/perf_stats.py
from multiprocessing import Array
from typing import List

class PerfStats:
    """
    性能统计器：封装 EMA 平滑计算与共享内存更新。
    """
    def __init__(self, shared_timings: Array):
        self._shared = shared_timings # [Pre, GPU, Post, Ovhd, Total, Wait]
        self._local = [0.0] * 6 
        self._alpha = 0.1 # 平滑系数

    def update(self, t_pre: float, t_gpu: float, t_post: float, t_total_ms: float, t_wait: float):
        t_model_sum = t_pre + t_gpu + t_post
        t_ovhd = max(0.0, t_total_ms - t_model_sum)

        new_values = [t_pre, t_gpu, t_post, t_ovhd, t_total_ms, t_wait]

        for i in range(6):
            self._local[i] = self._local[i] * (1 - self._alpha) + new_values[i] * self._alpha

        # 加锁写入共享内存
        with self._shared.get_lock():
            self._shared[:] = self._local[:]

    @property
    def current_total_ms(self) -> float:
        return self._local[4]