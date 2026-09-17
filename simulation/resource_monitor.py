"""
Real system resource monitor using psutil.
Measures CPU time consumed by this process and RSS memory during a block.
"""
import os
import time
import psutil


class ResourceMonitor:
    """Context manager: measures wall time, CPU time, and RSS memory of a block."""

    def __init__(self, label: str = ""):
        self.label = label
        self.process = psutil.Process(os.getpid())
        self.start_time = 0.0
        self.end_time = 0.0
        self.cpu_start_s = 0.0
        self.cpu_end_s = 0.0
        self.mem_start_mb = 0.0
        self.mem_end_mb = 0.0

    def __enter__(self):
        # Prime the CPU counter, then record baseline
        self.process.cpu_percent(interval=None)
        t = self.process.cpu_times()
        self.cpu_start_s = t.user + t.system
        self.mem_start_mb = self.process.memory_info().rss / (1024 * 1024)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        t = self.process.cpu_times()
        self.cpu_end_s = t.user + t.system
        self.mem_end_mb = self.process.memory_info().rss / (1024 * 1024)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def cpu_percent(self) -> float:
        """CPU usage as % of elapsed wall time. Can exceed 100% on multi-core."""
        d = self.duration_s
        if d <= 0.0:
            return 0.0
        cpu_used_s = max(0.0, self.cpu_end_s - self.cpu_start_s)
        return 100.0 * cpu_used_s / d

    @property
    def rss_mb(self) -> float:
        return self.mem_end_mb

    @property
    def rss_delta_mb(self) -> float:
        return self.mem_end_mb - self.mem_start_mb

    def to_dict(self) -> dict:
        return {
            "wall_time_s":  round(self.duration_s, 6),
            "cpu_percent":  round(self.cpu_percent, 2),
            "rss_mb":       round(self.rss_mb, 2),
            "rss_delta_mb": round(self.rss_delta_mb, 2),
        }