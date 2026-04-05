from __future__ import annotations

from multiprocessing import Condition, Lock, shared_memory
from typing import Optional, Tuple

import numpy as np

META_COUNT = 5
META_DTYPE = np.float64
META_BYTES = META_COUNT * 8
CTRL_DTYPE = np.int32
HEADER_OFFSET = 128


class SharedTripleBuffer:
    def __init__(self, shape: Tuple[int, int, int], create: bool = True, name: str = "shared_buffer_triple"):
        self.shape = shape
        self.dtype = np.uint8
        self.size = int(np.prod(shape))
        self.name = name
        self.frame_bytes = self.size * np.dtype(self.dtype).itemsize
        self.meta_section_offset = HEADER_OFFSET
        self.frame_section_offset = self.meta_section_offset + (3 * META_BYTES)
        self.total_size = self.frame_section_offset + (3 * self.frame_bytes)

        if create:
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
            ctrl_arr = np.ndarray((2,), dtype=CTRL_DTYPE, buffer=self.shm.buf[:8])
            ctrl_arr[0] = 2
            ctrl_arr[1] = 0
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)

        self.ctrl_array = np.ndarray((2,), dtype=CTRL_DTYPE, buffer=self.shm.buf[:8])
        self.meta_arrays = []
        for index in range(3):
            offset = self.meta_section_offset + (index * META_BYTES)
            self.meta_arrays.append(np.ndarray(META_COUNT, dtype=META_DTYPE, buffer=self.shm.buf[offset: offset + META_BYTES]))
        self.frame_arrays = []
        for index in range(3):
            offset = self.frame_section_offset + (index * self.frame_bytes)
            self.frame_arrays.append(np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf[offset: offset + self.frame_bytes]))
        self.my_index = 0 if create else 1

    def put(self, frame: np.ndarray, timestamp: float, rect: Tuple[int, int, int, int], lock: Lock, cond: Condition):
        index = self.my_index
        np.copyto(self.frame_arrays[index], frame)
        metadata = self.meta_arrays[index]
        metadata[0] = timestamp
        metadata[1] = rect[0]
        metadata[2] = rect[1]
        metadata[3] = rect[2]
        metadata[4] = rect[3]
        with lock:
            middle_index = self.ctrl_array[0]
            self.ctrl_array[0] = index
            self.my_index = middle_index
            self.ctrl_array[1] = 1
            cond.notify_all()

    def get(self, lock: Lock, cond: Condition, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        swapped = False
        with cond:
            if self.ctrl_array[1] == 0 and not cond.wait(timeout=timeout):
                return None
            if self.ctrl_array[1] == 1:
                middle_index = self.ctrl_array[0]
                temp = self.my_index
                self.my_index = middle_index
                self.ctrl_array[0] = temp
                self.ctrl_array[1] = 0
                swapped = True
        if not swapped:
            return None

        index = self.my_index
        metadata = self.meta_arrays[index]
        return (
            self.frame_arrays[index],
            float(metadata[0]),
            (int(metadata[1]), int(metadata[2]), int(metadata[3]), int(metadata[4])),
        )

    def close(self):
        self.shm.close()

    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
