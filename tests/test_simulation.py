"""
Unit tests for the collective communication simulators.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation import (
    NCCLSimulator, NCCLConfig,
    XDPSimulator,  XDPConfig,
    GlooSimulator, GlooConfig,
)


def test_nccl_backend_returns_trace():
    sim = NCCLSimulator(NCCLConfig(num_nodes=2, gpus_per_node=4))
    trace = sim.generate_trace(num_iterations=5, model_size_mb=1000,
                               gradient_size_mb=256, seed=42)
    assert trace["backend"] == "nccl"
    assert trace["num_iterations"] == 5
    assert trace["total_time_ms"] > 0
    assert len(trace["trace"]) == 5 * 4  # 4 collectives per iteration


def test_xdp_backend_returns_trace():
    sim = XDPSimulator(XDPConfig(num_nodes=2, gpus_per_node=4))
    trace = sim.generate_trace(num_iterations=5, model_size_mb=1000,
                               gradient_size_mb=256, seed=42)
    assert trace["backend"] == "xdp"
    assert trace["total_time_ms"] > 0


def test_gloo_backend_returns_trace():
    sim = GlooSimulator(GlooConfig(num_nodes=2, gpus_per_node=4))
    trace = sim.generate_trace(num_iterations=5, model_size_mb=1000,
                               gradient_size_mb=256, seed=42)
    assert trace["backend"] == "gloo"
    assert trace["total_time_ms"] > 0


def test_nccl_faster_than_gloo():
    """NCCL uses RDMA; Gloo uses TCP. NCCL must be significantly faster."""
    nccl = NCCLSimulator(NCCLConfig()).generate_trace(
        10, 1000, 256, seed=42)
    gloo = GlooSimulator(GlooConfig()).generate_trace(
        10, 1000, 256, seed=42)
    assert nccl["total_time_ms"] < gloo["total_time_ms"]


def test_determinism_with_seed():
    """Same seed must produce identical results."""
    a = NCCLSimulator(NCCLConfig()).generate_trace(10, 1000, 256, seed=7)
    b = NCCLSimulator(NCCLConfig()).generate_trace(10, 1000, 256, seed=7)
    assert abs(a["total_time_ms"] - b["total_time_ms"]) < 1e-9