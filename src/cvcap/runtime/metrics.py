from multiprocessing import Array


class PerfStats:
    def __init__(self, shared_timings: Array):
        self._shared = shared_timings
        self._local = [0.0] * 6
        self._alpha = 0.1

    def update(self, t_pre: float, t_gpu: float, t_post: float, t_total_ms: float, t_wait: float):
        t_model_sum = t_pre + t_gpu + t_post
        t_ovhd = max(0.0, t_total_ms - t_model_sum)
        new_values = [t_pre, t_gpu, t_post, t_ovhd, t_total_ms, t_wait]
        for index in range(6):
            self._local[index] = self._local[index] * (1 - self._alpha) + new_values[index] * self._alpha
        with self._shared.get_lock():
            self._shared[:] = self._local[:]
