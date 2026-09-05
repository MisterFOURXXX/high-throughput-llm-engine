import pytest
from src.engine.memory_allocator import MemoryAllocatorProfiler

def test_memory_allocator_snapshot():
    profiler = MemoryAllocatorProfiler(device="cpu")
    snap = profiler.snapshot()
    assert "timestamp" in snap
    assert "allocated_bytes" in snap

def test_fragmentation_ratio():
    profiler = MemoryAllocatorProfiler(device="cpu")
    profiler.snapshot()
    # Simulate memory usage
    profiler.history[-1]["reserved_bytes"] = 1000
    profiler.history[-1]["allocated_bytes"] = 600
    ratio = profiler.fragmentation_ratio()
    assert ratio == 0.4