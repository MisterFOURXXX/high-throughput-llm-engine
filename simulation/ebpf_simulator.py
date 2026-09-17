"""
eBPF / bpfd network monitoring simulator.
Emits kernel-level tracing events (packet send/drop/retransmit).
Deterministic given a seed.
"""
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BPFEvent:
    timestamp_ns: int
    event_type: str
    rank_src: int
    rank_dst: int
    size_bytes: int
    latency_ns: int
    protocol: str = "RDMA"
    extra_info: Dict = field(default_factory=dict)


class eBPFSimulator:
    def __init__(self, num_nodes: int = 2, gpus_per_node: int = 4,
                 seed: int = 42):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_ranks = num_nodes * gpus_per_node
        self.events: List[BPFEvent] = []
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.network_params = {
            "base_latency_ns": 2000,
            "jitter_ns": 500,
            "drop_probability": 0.001,
            "retransmit_prob": 0.005,
            "bandwidth_gbps": 100,
        }
        self.metrics = {
            "packet_count": 0, "packet_drops": 0, "retransmits": 0,
            "avg_latency_ns": 0.0, "p95_latency_ns": 0.0, "p99_latency_ns": 0.0,
        }

    def generate_event(self, event_type: str, rank_src: int,
                       rank_dst: int, size_mb: float) -> BPFEvent:
        size_bytes = size_mb * 1024 * 1024
        bw_bps = self.network_params["bandwidth_gbps"] * 1e9 / 8
        tx_ns = (size_bytes / bw_bps) * 1e9
        lat_ns = (self.network_params["base_latency_ns"] + tx_ns
                  + float(self._np_rng.normal(0, self.network_params["jitter_ns"])))
        if event_type == "packet_drop":
            lat_ns = 0.0
            self.metrics["packet_drops"] += 1
        if event_type == "retransmit":
            lat_ns *= 2.0
            self.metrics["retransmits"] += 1

        ev = BPFEvent(
            timestamp_ns=1_000_000 + len(self.events) * 1000,
            event_type=event_type,
            rank_src=rank_src, rank_dst=rank_dst,
            size_bytes=int(size_bytes), latency_ns=int(lat_ns),
            extra_info={
                "node_src": rank_src // self.gpus_per_node,
                "node_dst": rank_dst // self.gpus_per_node,
            },
        )
        self.events.append(ev)
        self.metrics["packet_count"] += 1
        return ev

    def generate_nccl_communication_events(
        self, collective_type: str, num_messages: int = 100,
        size_range_mb: Tuple[float, float] = (10, 500),
    ) -> List[BPFEvent]:
        events = []
        for _ in range(num_messages):
            rank_src = self._rng.randint(0, self.total_ranks - 1)
            rank_dst = self._rng.randint(0, self.total_ranks - 1)
            while rank_dst == rank_src:
                rank_dst = self._rng.randint(0, self.total_ranks - 1)
            size_mb = self._rng.uniform(*size_range_mb)

            events.append(self.generate_event("packet_send", rank_src, rank_dst, size_mb))
            if self._rng.random() < self.network_params["drop_probability"]:
                events.append(self.generate_event("packet_drop", rank_src, rank_dst, size_mb))
            if self._rng.random() < self.network_params["retransmit_prob"]:
                events.append(self.generate_event("retransmit", rank_src, rank_dst, size_mb))
            events.append(self.generate_event("packet_recv", rank_dst, rank_src, size_mb))
        return events

    def collect_statistics(self) -> Dict:
        if not self.events:
            return self.metrics
        lats = [e.latency_ns for e in self.events if e.latency_ns > 0]
        if lats:
            self.metrics["avg_latency_ns"] = float(np.mean(lats))
            self.metrics["p95_latency_ns"] = float(np.percentile(lats, 95))
            self.metrics["p99_latency_ns"] = float(np.percentile(lats, 99))
        total = max(1, self.metrics["packet_count"])
        self.metrics["packet_drop_rate"] = self.metrics["packet_drops"] / total
        self.metrics["retransmit_rate"]  = self.metrics["retransmits"] / total
        return self.metrics

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self.events:
            rows.append({
                "timestamp_ns": e.timestamp_ns,
                "event_type": e.event_type,
                "rank_src": e.rank_src, "rank_dst": e.rank_dst,
                "size_mb": e.size_bytes / (1024 * 1024),
                "latency_ns": e.latency_ns,
                "protocol": e.protocol,
                **e.extra_info,
            })
        return pd.DataFrame(rows)