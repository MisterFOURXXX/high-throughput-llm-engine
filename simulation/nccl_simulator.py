"""
NCCL Collective Communication Simulator
Simulates AllReduce, AllGather, ReduceScatter operations.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class NCCLConfig:
    """Configuration for NCCL simulation."""
    num_nodes: int = 2
    gpus_per_node: int = 4
    network_bandwidth_gbps: float = 100.0
    network_latency_us: float = 2.0
    collective_types: List[str] = field(default_factory=lambda: [
        "AllReduce", "AllGather", "ReduceScatter", "Broadcast"
    ])


class NCCLSimulator:
    """
    Simulates NCCL collective operations with realistic latency models.
    Based on established NCCL performance benchmarks.
    """
    
    def __init__(self, config: Optional[NCCLConfig] = None):
        self.config = config or NCCLConfig()
        self.total_ranks = self.config.num_nodes * self.config.gpus_per_node
        
        # NCCL latency models (empirically derived)
        self.collective_models = {
            "AllReduce": {"alpha": 15.0, "beta": 0.8, "gamma": 0.02},
            "AllGather": {"alpha": 12.0, "beta": 1.0, "gamma": 0.015},
            "ReduceScatter": {"alpha": 14.0, "beta": 0.9, "gamma": 0.018},
            "Broadcast": {"alpha": 8.0, "beta": 0.5, "gamma": 0.01},
        }
        self.noise_std = 0.05
    
    def _calculate_bandwidth(self, size_mb: float) -> float:
        """Calculate effective bandwidth including overhead."""
        peak_mbps = self.config.network_bandwidth_gbps * 1000 / 8
        overhead = np.exp(-size_mb / 1000) * 0.1
        return peak_mbps * (1 - overhead)
    
    def simulate_collective(
        self,
        collective_type: str,
        data_size_mb: float,
        num_repeats: int = 1,
        add_noise: bool = True
    ) -> Dict[str, float]:
        """Simulate a single collective operation."""
        if collective_type not in self.collective_models:
            raise ValueError(f"Unknown collective: {collective_type}")
        
        model = self.collective_models[collective_type]
        log_size = np.log2(max(1, data_size_mb))
        bandwidth = self._calculate_bandwidth(data_size_mb)
        
        base_latency_us = (
            model["alpha"] + 
            model["beta"] * log_size + 
            data_size_mb * 1000 / bandwidth
        )
        
        rank_factor = np.log2(self.total_ranks) * model["gamma"]
        base_latency_us += rank_factor * data_size_mb
        
        latencies_us = []
        for _ in range(max(1, num_repeats)):
            latency = base_latency_us
            if add_noise:
                noise_factor = np.random.normal(1.0, self.noise_std)
                latency *= noise_factor
                if random.random() < 0.02:
                    latency += random.uniform(50, 500)
            latencies_us.append(max(0, latency))
        
        latencies_ms = [l / 1000.0 for l in latencies_us]
        
        return {
            "min_ms": np.min(latencies_ms),
            "avg_ms": np.mean(latencies_ms),
            "max_ms": np.max(latencies_ms),
            "std_ms": np.std(latencies_ms),
            "p50_ms": np.percentile(latencies_ms, 50),
            "p95_ms": np.percentile(latencies_ms, 95),
            "p99_ms": np.percentile(latencies_ms, 99),
            "num_repeats": len(latencies_ms),
            "data_size_mb": data_size_mb,
            "collective": collective_type,
            "total_ranks": self.total_ranks,
        }
    
    def simulate_training_iteration(
        self,
        model_size_mb: float = 500.0,
        gradient_size_mb: float = 256.0,
        include_all_collectives: bool = True
    ) -> Dict[str, any]:
        """Simulate a full training iteration."""
        collectives = self.config.collective_types if include_all_collectives else ["AllReduce", "AllGather"]
        
        results = {}
        total_latency_ms = 0
        all_events = []
        
        for coll_type in collectives:
            if coll_type in ["AllReduce", "ReduceScatter"]:
                size_mb = gradient_size_mb
            elif coll_type == "AllGather":
                size_mb = model_size_mb / self.config.gpus_per_node
            else:
                size_mb = model_size_mb / self.total_ranks
            
            result = self.simulate_collective(coll_type, size_mb, num_repeats=3)
            results[coll_type] = result
            total_latency_ms += result["avg_ms"]
            
            all_events.append({
                "collective": coll_type,
                "latency_ms": result["avg_ms"],
                "size_mb": size_mb,
                "timestamp": total_latency_ms,
            })
        
        results["total_communication_ms"] = total_latency_ms
        results["events"] = all_events
        results["timestamp"] = time.time()
        
        return results
    
    def generate_nccl_trace(
        self,
        num_iterations: int = 100,
        model_size_mb: float = 500.0,
        gradient_size_mb: float = 256.0,
        save_path: Optional[str] = None
    ) -> Dict[str, any]:
        """Generate a full NCCL trace."""
        import pandas as pd
        
        events = []
        total_time_ms = 0
        
        for i in range(num_iterations):
            model_var = model_size_mb * np.random.uniform(0.9, 1.1)
            grad_var = gradient_size_mb * np.random.uniform(0.8, 1.2)
            
            result = self.simulate_training_iteration(
                model_size_mb=model_var,
                gradient_size_mb=grad_var
            )
            
            for event in result["events"]:
                rank = random.randint(0, self.total_ranks - 1)
                node_id = rank // self.config.gpus_per_node
                local_rank = rank % self.config.gpus_per_node
                
                events.append({
                    "iteration": i,
                    "rank": rank,
                    "node_id": node_id,
                    "local_rank": local_rank,
                    "collective": event["collective"],
                    "latency_ms": event["latency_ms"],
                    "size_mb": event["size_mb"],
                    "timestamp_ms": total_time_ms + event["timestamp"],
                })
            
            total_time_ms += result["total_communication_ms"]
        
        df = pd.DataFrame(events)
        
        result = {
            "trace": df,
            "total_time_ms": total_time_ms,
            "avg_iteration_ms": total_time_ms / num_iterations,
            "num_iterations": num_iterations,
            "total_ranks": self.total_ranks,
            "num_nodes": self.config.num_nodes,
            "gpus_per_node": self.config.gpus_per_node,
        }
        
        if save_path:
            df.to_csv(save_path, index=False)
        
        return result