"""
Real system resource measurement around a callable.
Uses psutil to capture actual CPU% and RSS memory usage of this process.
"""
import os
import time
import psutil


class ResourceMonitor:
    """Context manager that measures real CPU% and RSS memory around a block."""

    def __init__(self, label: str = ""):
        self.label = label
        self.process = psutil.Process(os.getpid())
        self.cpu_samples = []
        self.mem_samples = []
        self.start_time = 0.0
        self.end_time = 0.0
        self.cpu_start = 0.0
        self.mem_start_mb = 0.0
        self.cpu_end = 0.0
        self.mem_end_mb = 0.0

    def __enter__(self):
        self.process.cpu_percent(interval=None)  # prime
        self.cpu_start = self.process.cpu_percent(interval=None)
        self.mem_start_mb = self.process.memory_info().rss / (1024 * 1024)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.cpu_end = self.process.cpu_percent(interval=None)
        self.mem_end_mb = self.process.memory_info().rss / (1024 * 1024)

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def cpu_percent(self) -> float:
        return max(0.0, self.cpu_end)

    @property
    def rss_mb(self) -> float:
        return max(0.0, self.mem_end_mb)

    @property
    def rss_delta_mb(self) -> float:
        return self.mem_end_mb - self.mem_start_mb

    def to_dict(self) -> dict:
        return {
            "wall_time_s": round(self.duration_s, 6),
            "cpu_percent": round(self.cpu_percent, 2),
            "rss_mb":      round(self.rss_mb, 2),
            "rss_delta_mb": round(self.rss_delta_mb, 2),
        }