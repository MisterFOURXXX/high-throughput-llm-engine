"""
eBPF/bpfd Simulator
Simulates eBPF probes for NCCL network tracing.
"""

import numpy as np
import pandas as pd
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BPFEvent:
    """Simulated eBPF event for NCCL traffic."""
    timestamp_ns: int
    event_type: str  # 'packet_send', 'packet_recv', 'packet_drop', 'retransmit'
    rank_src: int
    rank_dst: int
    size_bytes: int
    latency_ns: int
    protocol: str = "RDMA"
    extra_info: Dict = field(default_factory=dict)


class eBPFSimulator:
    """
    Simulates eBPF probes monitoring NCCL communication.
    Generates realistic network tracing data.
    """
    
    def __init__(self, num_nodes: int = 2, gpus_per_node: int = 4):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_ranks = num_nodes * gpus_per_node
        self.events: List[BPFEvent] = []
        
        self.network_params = {
            'base_latency_ns': 2000,
            'jitter_ns': 500,
            'drop_probability': 0.001,
            'retransmit_prob': 0.005,
            'bandwidth_gbps': 100,
            'congestion_threshold': 0.7,
        }
        
        self.metrics = {
            'packet_count': 0,
            'packet_drops': 0,
            'retransmits': 0,
            'avg_latency_ns': 0,
            'p95_latency_ns': 0,
            'p99_latency_ns': 0,
        }
    
    def generate_event(
        self,
        event_type: str,
        rank_src: int,
        rank_dst: int,
        size_mb: float
    ) -> BPFEvent:
        """Generate a simulated eBPF event."""
        size_bytes = size_mb * 1024 * 1024
        bandwidth_bps = self.network_params['bandwidth_gbps'] * 1e9 / 8
        
        transmission_ns = (size_bytes / bandwidth_bps) * 1e9
        latency_ns = (
            self.network_params['base_latency_ns'] +
            transmission_ns +
            np.random.normal(0, self.network_params['jitter_ns'])
        )
        
        if random.random() < self.network_params['congestion_threshold']:
            latency_ns *= random.uniform(1.2, 1.8)
        
        if event_type == 'packet_drop':
            latency_ns = 0
            self.metrics['packet_drops'] += 1
        
        if event_type == 'retransmit':
            latency_ns *= 2
            self.metrics['retransmits'] += 1
        
        event = BPFEvent(
            timestamp_ns=time.time_ns(),
            event_type=event_type,
            rank_src=rank_src,
            rank_dst=rank_dst,
            size_bytes=int(size_bytes),
            latency_ns=int(latency_ns),
            extra_info={
                'bandwidth_utilized': random.uniform(0.3, 0.95),
                'queue_depth': random.randint(0, 100),
                'node_src': rank_src // self.gpus_per_node,
                'node_dst': rank_dst // self.gpus_per_node,
            }
        )
        
        self.events.append(event)
        self.metrics['packet_count'] += 1
        return event
    
    def generate_nccl_communication_events(
        self,
        collective_type: str,
        num_messages: int = 100,
        size_range_mb: Tuple[float, float] = (10, 500)
    ) -> List[BPFEvent]:
        """Generate events for a specific NCCL collective operation."""
        events = []
        
        for _ in range(num_messages):
            rank_src = random.randint(0, self.total_ranks - 1)
            rank_dst = random.randint(0, self.total_ranks - 1)
            while rank_dst == rank_src:
                rank_dst = random.randint(0, self.total_ranks - 1)
            
            size_mb = random.uniform(*size_range_mb)
            
            send_event = self.generate_event('packet_send', rank_src, rank_dst, size_mb)
            events.append(send_event)
            
            if random.random() < self.network_params['drop_probability']:
                drop_event = self.generate_event('packet_drop', rank_src, rank_dst, size_mb)
                events.append(drop_event)
            
            if random.random() < self.network_params['retransmit_prob']:
                retransmit_event = self.generate_event('retransmit', rank_src, rank_dst, size_mb)
                events.append(retransmit_event)
            
            if random.random() > self.network_params['drop_probability']:
                recv_event = self.generate_event('packet_recv', rank_dst, rank_src, size_mb)
                events.append(recv_event)
        
        return events
    
    def collect_statistics(self) -> Dict:
        """Compute statistics from all generated events."""
        if not self.events:
            return self.metrics
        
        latencies = [e.latency_ns for e in self.events if e.latency_ns > 0]
        
        self.metrics['avg_latency_ns'] = np.mean(latencies) if latencies else 0
        self.metrics['p95_latency_ns'] = np.percentile(latencies, 95) if latencies else 0
        self.metrics['p99_latency_ns'] = np.percentile(latencies, 99) if latencies else 0
        self.metrics['packet_drop_rate'] = self.metrics['packet_drops'] / max(1, self.metrics['packet_count'])
        self.metrics['retransmit_rate'] = self.metrics['retransmits'] / max(1, self.metrics['packet_count'])
        
        return self.metrics
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert events to DataFrame for analysis."""
        data = []
        for e in self.events:
            data.append({
                'timestamp_ns': e.timestamp_ns,
                'event_type': e.event_type,
                'rank_src': e.rank_src,
                'rank_dst': e.rank_dst,
                'size_mb': e.size_bytes / (1024 * 1024),
                'latency_ns': e.latency_ns,
                'protocol': e.protocol,
                **e.extra_info
            })
        return pd.DataFrame(data)