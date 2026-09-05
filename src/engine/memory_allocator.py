import torch
import time
from collections import deque

class MemoryAllocatorProfiler:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.history = deque(maxlen=1000)

    def snapshot(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
        else:
            allocated = reserved = 0
        self.history.append({
            "timestamp": time.time(),
            "allocated_bytes": allocated,
            "reserved_bytes": reserved
        })
        return self.history[-1]

    def fragmentation_ratio(self) -> float:
        if len(self.history) < 2:
            return 0.0
        # Simple metric: (reserved - allocated) / reserved
        latest = self.history[-1]
        if latest["reserved_bytes"] == 0:
            return 0.0
        return (latest["reserved_bytes"] - latest["allocated_bytes"]) / latest["reserved_bytes"]