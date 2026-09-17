"""
Collective communication simulation: NCCL (RDMA), XDP (AF_XDP), Gloo (TCP).
"""
from .backends import (
    NCCLSimulator, NCCLConfig,
    XDPSimulator,  XDPConfig,
    GlooSimulator, GlooConfig,
)
from .ebpf_simulator import eBPFSimulator, BPFEvent
from .k8s_operator_simulator import (
    KubernetesOperatorSimulator, Pod, NCCLJob, PodStatus,
)
from .terraform_simulator import (
    TerraformSimulator, TerraformResource, TerraformState, Cluster,
)
from .resource_monitor import ResourceMonitor
from .orchestrator import run_simulation, SimulationConfig

__all__ = [
    "NCCLSimulator", "NCCLConfig",
    "XDPSimulator",  "XDPConfig",
    "GlooSimulator", "GlooConfig",
    "eBPFSimulator", "BPFEvent",
    "KubernetesOperatorSimulator", "Pod", "NCCLJob", "PodStatus",
    "TerraformSimulator", "TerraformResource", "TerraformState", "Cluster",
    "ResourceMonitor",
    "run_simulation", "SimulationConfig",
]