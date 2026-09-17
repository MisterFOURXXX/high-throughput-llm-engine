from src.engine.memory_allocator import MemoryAllocatorProfiler


def test_memory_allocator_snapshot():
    p = MemoryAllocatorProfiler(device="cpu")
    snap = p.snapshot()
    assert "timestamp" in snap
    assert "allocated_bytes" in snap
    assert "reserved_bytes" in snap


def test_fragmentation_ratio_single_snapshot():
    p = MemoryAllocatorProfiler(device="cpu")
    p.snapshot()
    p.history[-1]["reserved_bytes"] = 1000
    p.history[-1]["allocated_bytes"] = 600
    ratio = p.fragmentation_ratio()
    assert abs(ratio - 0.4) < 1e-6


def test_fragmentation_no_history():
    p = MemoryAllocatorProfiler(device="cpu")
    assert p.fragmentation_ratio() == 0.0


def test_fragmentation_zero_reserved():
    p = MemoryAllocatorProfiler(device="cpu")
    p.snapshot()
    p.history[-1]["reserved_bytes"] = 0
    p.history[-1]["allocated_bytes"] = 0
    assert p.fragmentation_ratio() == 0.0