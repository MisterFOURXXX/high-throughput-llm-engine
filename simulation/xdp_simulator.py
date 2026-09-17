"""
XDP (AF_XDP kernel-bypass) collective communication model.
Latency between NCCL and Gloo. Deterministic when seeded.
"""
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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


class XDPSimulator:
    def __init__(self, config: Optional[XDPConfig] = None):
        self.config = config or XDPConfig()
        self.total_ranks = self.config.num_nodes * self.config.gpus_per_node
        self.collective_models = {
            "AllReduce":     {"alpha": 40.0, "beta": 1.6, "gamma": 0.045},
            "AllGather":     {"alpha": 35.0, "beta": 1.8, "gamma": 0.040},
            "ReduceScatter": {"alpha": 38.0, "beta": 1.7, "gamma": 0.042},
            "Broadcast":     {"alpha": 25.0, "beta": 1.2, "gamma": 0.030},
        }
        self.noise_std = 0.08

    def _bandwidth(self, size_mb: float) -> float:
        peak_mbps = self.config.network_bandwidth_gbps * 1000 / 8
        overhead = 0.05 if self.config.zero_copy else 0.15
        return peak_mbps * (1 - overhead)

    def simulate_collective(self, collective_type: str, data_size_mb: float,
                            num_repeats: int = 3) -> Dict[str, float]:
        model = self.collective_models[collective_type]
        log_size = np.log2(max(1, data_size_mb))
        bw = self._bandwidth(data_size_mb)
        pkts = max(1, int(data_size_mb * 1024 * 1024 / self.config.umem_frame_size))
        ebpf_overhead_us = pkts * 0.0015
        base_us = (model["alpha"]
                   + model["beta"] * log_size
                   + data_size_mb * 1000 / bw
                   + np.log2(self.total_ranks) * model["gamma"] * data_size_mb
                   + ebpf_overhead_us)
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
            "packets_processed": pkts,
        }

    def simulate_iteration(self, model_size_mb: float,
                           gradient_size_mb: float) -> Dict[str, any]:
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
                "collective": coll, "latency_ms": r["avg_ms"],
                "size_mb": size_mb, "timestamp": total_ms,
            })
        results["total_communication_ms"] = total_ms
        results["events"] = events
        return results

    def generate_trace(self, num_iterations: int, model_size_mb: float,
                       gradient_size_mb: float,
                       save_path: Optional[str] = None,
                       seed: Optional[int] = 42) -> Dict[str, any]:
        import pandas as pd
        if seed is not None:
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
                    "iteration": i, "rank": rank,
                    "node_id": rank // self.config.gpus_per_node,
                    "local_rank": rank % self.config.gpus_per_node,
                    "collective": ev["collective"],
                    "latency_ms": ev["latency_ms"],
                    "size_mb": ev["size_mb"],
                    "timestamp_ms": total_ms + ev["timestamp"],
                    "backend": "xdp",
                })
            total_ms += r["total_communication_ms"]
        df = pd.DataFrame(events)
        if save_path:
            df.to_csv(save_path, index=False)
        return {
            "trace": df, "total_time_ms": total_ms,
            "avg_iteration_ms": total_ms / num_iterations,
            "num_iterations": num_iterations,
            "total_ranks": self.total_ranks,
            "num_nodes": self.config.num_nodes,
            "gpus_per_node": self.config.gpus_per_node,
            "backend": "xdp",
        }