import time
from typing import Dict, Any

class PagedAttentionMemoryTracker:
    def __init__(self, block_size: int, total_gpu_memory_bytes: int):
        self.block_size = block_size
        self.total_gpu_memory = total_gpu_memory_bytes
        self.snapshots = []

    def compute_fragmentation(
        self,
        num_allocated_blocks: int,
        num_active_tokens: int,
        total_reserved_blocks: int
    ) -> Dict[str, float]:
        total_allocated_capacity_tokens = num_allocated_blocks * self.block_size
        internal_frag = 0.0
        if total_allocated_capacity_tokens > 0:
            internal_frag = (total_allocated_capacity_tokens - num_active_tokens) / total_allocated_capacity_tokens

        # External fragmentation: unused reserved blocks
        max_blocks = self.total_gpu_memory // (self.block_size * 2 * 16 * 8)  # heuristic
        external_frag = 0.0
        if max_blocks > 0:
            external_frag = 1.0 - (total_reserved_blocks / max_blocks)

        metrics = {
            "timestamp": time.perf_counter(),
            "internal_fragmentation_pct": internal_frag * 100.0,
            "external_fragmentation_pct": external_frag * 100.0,
            "allocated_blocks": num_allocated_blocks,
            "active_tokens": num_active_tokens,
            "reserved_blocks": total_reserved_blocks,
            "kv_cache_utilization_pct": (num_allocated_blocks / max(1, total_reserved_blocks)) * 100.0
        }
        self.snapshots.append(metrics)
        return metrics