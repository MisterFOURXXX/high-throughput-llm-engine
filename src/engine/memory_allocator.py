"""
Memory Allocator Profiler - works with or without torch/CUDA.
"""
import time
from collections import deque

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


class MemoryAllocatorProfiler:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.history = deque(maxlen=1000)

    def snapshot(self):
        if _HAS_TORCH and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved  = torch.cuda.memory_reserved(self.device)
        else:
            allocated = reserved = 0
        snap = {
            "timestamp": time.time(),
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
        }
        self.history.append(snap)
        return snap

    def fragmentation_ratio(self) -> float:
        """Uses the latest snapshot only. Returns 0.0 if reserved is 0."""
        if not self.history:
            return 0.0
        latest = self.history[-1]
        if latest["reserved_bytes"] == 0:
            return 0.0
        return (latest["reserved_bytes"] - latest["allocated_bytes"]) / latest["reserved_bytes"]