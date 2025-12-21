import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2


class RandomEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, num_batches=20, seed=42):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.num_batches = num_batches
        self.current_index = 0
        self.rng = np.random.default_rng(seed)

        self.shape_xyz = (1024, 3)
        self.shape_sun_dir = (1024, 3)
        self.shape_t = (1024, 4)

        # self.shape_rays = (40960, 11)
        # self.shape_ts = (40960,)

        # Allocate device memory
        self.device_input_xyz = cuda.mem_alloc(np.zeros(self.shape_xyz, dtype=np.float32).nbytes)
        self.device_input_sun_dir = cuda.mem_alloc(np.zeros(self.shape_sun_dir, dtype=np.float32).nbytes)
        self.device_input_t = cuda.mem_alloc(np.zeros(self.shape_t, dtype=np.float32).nbytes)

        # self.device_input_rays = cuda.mem_alloc(np.zeros(self.shape_rays, dtype=np.float32).nbytes)
        # self.device_input_ts = cuda.mem_alloc(np.zeros(self.shape_ts, dtype=np.int64).nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_index >= self.num_batches:
            return None

        print(f"[RandomCalibrator] Calibrating batch {self.current_index + 1}/{self.num_batches}")

        xyz = self.rng.uniform(-1.0, 1.0, size=self.shape_xyz).astype(np.float32)
        sun_dir = self.rng.uniform(0.0, 1.0, size=self.shape_sun_dir).astype(np.float32)
        t = self.rng.integers(0, 29, size=self.shape_t).astype(np.float32)

        # rays = self.rng.uniform(-1.0, 1.0, size=self.shape_rays).astype(np.float32)
        # ts = self.rng.integers(0, 29, size=self.shape_ts).astype(np.float32)

        cuda.memcpy_htod(self.device_input_xyz, xyz)
        cuda.memcpy_htod(self.device_input_sun_dir, sun_dir)
        cuda.memcpy_htod(self.device_input_t, t)

        self.current_index += 1

        # Return device pointers in correct order
        return [self.device_input_xyz, self.device_input_sun_dir, self.device_input_t]
        # return [self.device_input_rays, self.device_input_ts]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
