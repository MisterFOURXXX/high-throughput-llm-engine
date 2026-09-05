#!/usr/bin/env python
"""Orchestrator for benchmarking sweeps."""
import os
import subprocess
import json
import time
import yaml
from src.engine import PagedAttentionMemoryTracker, MemoryAllocatorProfiler

def load_config():
    with open("configs/trace_profiles.yaml", "r") as f:
        return yaml.safe_load(f)

def run_benchmark(concurrency, block_size, duration):
    # Build and run C++ loadgen
    cmd = ["./benchmarks/build/loadgen", "http://localhost:8000/generate",
           str(concurrency), "10", str(duration)]  # rate=10 req/s fixed
    subprocess.run(cmd, check=True)

    # Collect memory snapshots
    profiler = MemoryAllocatorProfiler()
    tracker = PagedAttentionMemoryTracker(block_size=block_size, total_gpu_memory=16*1024**3)
    # In real integration, these values would come from vLLM's block manager.
    metrics = tracker.compute_fragmentation(100, 512, 200)
    print(f"Fragmentation: {metrics}")

def main():
    config = load_config()
    sweeps = config.get("sweeps", [])
    for sweep in sweeps:
        concurrency = sweep["concurrency"]
        block_size = sweep["block_size"]
        duration = sweep.get("duration", 30)
        print(f"Running benchmark: concurrency={concurrency}, block_size={block_size}")
        run_benchmark(concurrency, block_size, duration)

if __name__ == "__main__":
    main()