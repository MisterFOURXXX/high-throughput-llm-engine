"""
Collective communication backends.

Three backends share identical structure but different latency models:
  - NCCL : RDMA / InfiniBand-class   (fastest, GPU memory direct)
  - XDP  : AF_XDP kernel-bypass      (fast, CPU-mediated)
  - Gloo : standard TCP/IP socket    (baseline, slowest)

All backends are deterministic when a seed is provided.
"""
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ====================================================================
# Configs
# ====================================================================
@dataclass
class NCCLConfig:
    num_nodes: int = 2
    gpus_per_node: int = 4
    network_bandwidth_gbps: float = 100.0
    network_latency_us: float = 2.0
    collective_types: List[str] = field(default_factory=lambda: [
        "AllReduce", "AllGather", "ReduceScatter", "Broadcast"
    ])


@dataclass
class XDPConfig:
    num_nodes: int = 2
    gpus_per_node: int = 4
    network_bandwidth_gbps: float = 40.0
    network_latency_us: float = 8.0
    collective_types: List[str] = field(default_factory=lambda: [
        "AllReduce", "AllGather", "ReduceScatter", "Broadcast"
    ])
    umem_frame_size: int = 2048
    zero_copy: bool = True


@dataclass
class GlooConfig:
    num_nodes: int = 2
    gpus_per_node: int = 4
    network_bandwidth_gbps: float = 10.0
    network_latency_us: float = 50.0
    collective_types: List[str] = field(default_factory=lambda: [
        "AllReduce", "AllGather", "ReduceScatter", "Broadcast"
    ])


# ====================================================================
# Base class
# ====================================================================
class _CollectiveBackend:
    """Shared logic for all collective communication backends."""

    def __init__(self, config, model: Dict[str, Dict], noise_std: float):
        self.config = config
        self.total_ranks = config.num_nodes * config.gpus_per_node
        self.collective_models = model
        self.noise_std = noise_std
        self.backend = "generic"

    # -- bandwidth model (override if needed) --
    def _bandwidth(self, size_mb: float) -> float:
        peak_mbps = self.config.network_bandwidth_gbps * 1000 / 8
        overhead = np.exp(-size_mb / 1000) * 0.1
        return peak_mbps * (1 - overhead)

    # -- optional per-collective overhead (override if needed) --
    def _extra_overhead_us(self, size_mb: float) -> float:
        return 0.0

    # -- single collective --
    def simulate_collective(self, collective_type: str, data_size_mb: float,
                            num_repeats: int = 3) -> Dict[str, float]:
        model = self.collective_models[collective_type]
        log_size = np.log2(max(1.0, data_size_mb))
        bw = self._bandwidth(data_size_mb)
        base_us = (
            model["alpha"]
            + model["beta"] * log_size
            + data_size_mb * 1000 / bw
            + np.log2(self.total_ranks) * model["gamma"] * data_size_mb
            + self._extra_overhead_us(data_size_mb)
        )
        latencies = []
        for _ in range(num_repeats):
            lat = base_us * np.random.normal(1.0, self.noise_std)
            latencies.append(max(0.001, lat))
        lms = [l / 1000.0 for l in latencies]
        return {
            "min_ms":  float(np.min(lms)),
            "avg_ms":  float(np.mean(lms)),
            "max_ms":  float(np.max(lms)),
            "std_ms":  float(np.std(lms)),
            "p50_ms":  float(np.percentile(lms, 50)),
            "p95_ms":  float(np.percentile(lms, 95)),
            "p99_ms":  float(np.percentile(lms, 99)),
            "data_size_mb": data_size_mb,
            "collective":   collective_type,
            "total_ranks":  self.total_ranks,
        }

    # -- single training iteration (4 collectives) --
    def simulate_iteration(self, model_size_mb: float,
                           gradient_size_mb: float) -> Dict:
        results, events = {}, []
        total_ms = 0.0
        for coll in self.config.collective_types:
            if coll in ("AllReduce", "ReduceScatter"):
                size_mb = gradient_size_mb
            elif coll == "AllGather":
                size_mb = model_size_mb / self.config.gpus_per_node
            else:
                size_mb = model_size_mb / self.total_ranks
            r = self.simulate_collective(coll, size_mb)
            results[coll] = r
            total_ms += r["avg_ms"]
            events.append({
                "collective": coll,
                "latency_ms": r["avg_ms"],
                "size_mb":    size_mb,
                "timestamp":  total_ms,
            })
        results["total_communication_ms"] = total_ms
        results["events"] = events
        return results

    # -- full trace across N iterations --
    def generate_trace(self, num_iterations: int,
                       model_size_mb: float,
                       gradient_size_mb: float,
                       save_path: Optional[str] = None,
                       seed: int = 42) -> Dict:
        random.seed(seed)
        np.random.seed(seed)

        events, total_ms = [], 0.0
        for i in range(num_iterations):
            m = model_size_mb * np.random.uniform(0.95, 1.05)
            g = gradient_size_mb * np.random.uniform(0.95, 1.05)
            r = self.simulate_iteration(m, g)
            for ev in r["events"]:
                rank = random.randint(0, self.total_ranks - 1)
                events.append({
                    "iteration": i,
                    "rank": rank,
                    "node_id": rank // self.config.gpus_per_node,
                    "local_rank": rank % self.config.gpus_per_node,
                    "collective": ev["collective"],
                    "latency_ms": ev["latency_ms"],
                    "size_mb":    ev["size_mb"],
                    "timestamp_ms": total_ms + ev["timestamp"],
                    "backend": self.backend,
                })
            total_ms += r["total_communication_ms"]

        df = pd.DataFrame(events)
        if save_path:
            df.to_csv(save_path, index=False)

        return {
            "trace": df,
            "total_time_ms": total_ms,
            "avg_iteration_ms": total_ms / num_iterations,
            "num_iterations": num_iterations,
            "total_ranks": self.total_ranks,
            "num_nodes": self.config.num_nodes,
            "gpus_per_node": self.config.gpus_per_node,
            "backend": self.backend,
        }


# ====================================================================
# NCCL backend (RDMA / InfiniBand)
# ====================================================================
class NCCLSimulator(_CollectiveBackend):
    def __init__(self, config: Optional[NCCLConfig] = None):
        config = config or NCCLConfig()
        model = {
            "AllReduce":     {"alpha": 15.0, "beta": 0.8, "gamma": 0.02},
            "AllGather":     {"alpha": 12.0, "beta": 1.0, "gamma": 0.015},
            "ReduceScatter": {"alpha": 14.0, "beta": 0.9, "gamma": 0.018},
            "Broadcast":     {"alpha":  8.0, "beta": 0.5, "gamma": 0.010},
        }
        super().__init__(config, model, noise_std=0.05)
        self.backend = "nccl"


# ====================================================================
# XDP backend (AF_XDP kernel-bypass)
# ====================================================================
class XDPSimulator(_CollectiveBackend):
    def __init__(self, config: Optional[XDPConfig] = None):
        config = config or XDPConfig()
        model = {
            "AllReduce":     {"alpha": 40.0, "beta": 1.6, "gamma": 0.045},
            "AllGather":     {"alpha": 35.0, "beta": 1.8, "gamma": 0.040},
            "ReduceScatter": {"alpha": 38.0, "beta": 1.7, "gamma": 0.042},
            "Broadcast":     {"alpha": 25.0, "beta": 1.2, "gamma": 0.030},
        }
        super().__init__(config, model, noise_std=0.08)
        self.backend = "xdp"

    def _bandwidth(self, size_mb: float) -> float:
        peak_mbps = self.config.network_bandwidth_gbps * 1000 / 8
        overhead = 0.05 if self.config.zero_copy else 0.15
        return peak_mbps * (1 - overhead)

    def _extra_overhead_us(self, size_mb: float) -> float:
        # eBPF per-packet processing overhead (UMEM frame-level)
        pkts = max(1, int(size_mb * 1024 * 1024 / self.config.umem_frame_size))
        return pkts * 0.0015  # 1.5 ns per packet


# ====================================================================
# Gloo backend (TCP baseline)
# ====================================================================
class GlooSimulator(_CollectiveBackend):
    def __init__(self, config: Optional[GlooConfig] = None):
        config = config or GlooConfig()
        model = {
            "AllReduce":     {"alpha": 120.0, "beta": 3.5, "gamma": 0.150},
            "AllGather":     {"alpha": 100.0, "beta": 4.0, "gamma": 0.120},
            "ReduceScatter": {"alpha": 110.0, "beta": 3.8, "gamma": 0.130},
            "Broadcast":     {"alpha":  80.0, "beta": 2.5, "gamma": 0.100},
        }
        super().__init__(config, model, noise_std=0.15)
        self.backend = "gloo"

    def _bandwidth(self, size_mb: float) -> float:
        peak_mbps = self.config.network_bandwidth_gbps * 1000 / 8
        return peak_mbps * (1 - np.exp(-size_mb / 500) * 0.2)