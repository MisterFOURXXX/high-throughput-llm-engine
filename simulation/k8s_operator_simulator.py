"""
Kubernetes Operator Simulator
Simulates an NCCL Operator managing distributed training jobs.
"""

import time
import random
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class PodStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


@dataclass
class Pod:
    """Simulated Kubernetes Pod."""
    name: str
    namespace: str = "default"
    status: PodStatus = PodStatus.PENDING
    node_name: str = ""
    rank: int = -1
    start_time: float = 0
    end_time: float = 0
    restart_count: int = 0
    exit_code: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class NCCLJob:
    """Simulated NCCL job (like PyTorchJob CRD)."""
    name: str
    namespace: str = "default"
    num_nodes: int = 2
    gpus_per_node: int = 4
    total_ranks: int = 8
    status: str = "Pending"
    pods: List[Pod] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0
    completion_time: float = 0


class KubernetesOperatorSimulator:
    """
    Simulates a Kubernetes Operator managing NCCL jobs.
    Emulates the behavior of the NCCL Operator or Kubeflow PyTorch Operator.
    """
    
    def __init__(self):
        self.jobs: Dict[str, NCCLJob] = {}
        self.pods: Dict[str, Pod] = {}
        self.nodes = [f"node-{i}" for i in range(10)]
    
    def create_nccl_job(
        self,
        job_name: str,
        num_nodes: int = 2,
        gpus_per_node: int = 4,
        namespace: str = "default"
    ) -> NCCLJob:
        """Create a new NCCL job."""
        job = NCCLJob(
            name=job_name,
            namespace=namespace,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            total_ranks=num_nodes * gpus_per_node,
            status="Pending",
            start_time=time.time()
        )
        
        total_ranks = job.total_ranks
        for rank in range(total_ranks):
            node_name = random.choice(self.nodes)
            pod_name = f"{job_name}-rank-{rank}"
            
            pod = Pod(
                name=pod_name,
                namespace=namespace,
                status=PodStatus.PENDING,
                node_name=node_name,
                rank=rank,
                labels={
                    "job-name": job_name,
                    "role": "worker",
                    "rank": str(rank),
                    "app": "nccl-job"
                },
                annotations={
                    "nccl.io/rank": str(rank),
                    "nccl.io/world-size": str(total_ranks),
                    "nccl.io/master-addr": f"{job_name}-rank-0.{namespace}.svc",
                }
            )
            job.pods.append(pod)
            self.pods[pod_name] = pod
        
        self.jobs[job_name] = job
        return job
    
    def reconcile(self, job_name: str) -> Dict[str, Any]:
        """Simulate the reconciliation loop of a Kubernetes Operator."""
        if job_name not in self.jobs:
            return {"status": "error", "message": "Job not found"}
        
        job = self.jobs[job_name]
        all_running = True
        all_succeeded = True
        any_failed = False
        
        for pod in job.pods:
            if pod.status == PodStatus.PENDING:
                if random.random() < 0.2:
                    pod.status = PodStatus.RUNNING
                    pod.start_time = time.time()
            elif pod.status == PodStatus.RUNNING:
                if random.random() < 0.3:
                    pod.status = PodStatus.SUCCEEDED
                    pod.end_time = time.time()
                elif random.random() < 0.02:
                    pod.status = PodStatus.FAILED
                    pod.exit_code = random.randint(1, 255)
                    pod.end_time = time.time()
                    any_failed = True
            elif pod.status == PodStatus.FAILED:
                if pod.restart_count < 3:
                    pod.restart_count += 1
                    pod.status = PodStatus.PENDING
                    pod.exit_code = 0
                else:
                    all_succeeded = False
        
        if any_failed:
            job.status = "Failed"
        elif all(p.status == PodStatus.SUCCEEDED for p in job.pods):
            job.status = "Succeeded"
            job.end_time = time.time()
            job.completion_time = job.end_time - job.start_time
        elif any(p.status == PodStatus.RUNNING for p in job.pods) and not any_failed:
            job.status = "Running"
        else:
            job.status = "Pending"
        
        return {
            "job_name": job_name,
            "status": job.status,
            "total_ranks": job.total_ranks,
            "running_pods": sum(1 for p in job.pods if p.status == PodStatus.RUNNING),
            "succeeded_pods": sum(1 for p in job.pods if p.status == PodStatus.SUCCEEDED),
            "failed_pods": sum(1 for p in job.pods if p.status == PodStatus.FAILED),
            "pending_pods": sum(1 for p in job.pods if p.status == PodStatus.PENDING),
            "completion_time_sec": job.completion_time if job.completion_time else None,
        }
    
    def run_job_to_completion(
        self,
        job_name: str,
        max_attempts: int = 100,
        sleep_interval: float = 0.5
    ) -> Dict[str, Any]:
        """Simulate running a job until completion."""
        attempts = 0
        while attempts < max_attempts:
            status = self.reconcile(job_name)
            if status["status"] in ["Succeeded", "Failed"]:
                return status
            time.sleep(sleep_interval)
            attempts += 1
        return {"status": "Timeout", "job_name": job_name}
    
    def get_job_metrics(self, job_name: str) -> Dict[str, Any]:
        """Get detailed metrics about a job."""
        if job_name not in self.jobs:
            return {"error": "Job not found"}
        
        job = self.jobs[job_name]
        schedule_times = []
        runtime_times = []
        
        for pod in job.pods:
            if pod.start_time > 0:
                schedule_times.append(pod.start_time - job.start_time)
                if pod.end_time > 0:
                    runtime_times.append(pod.end_time - pod.start_time)
        
        return {
            "job_name": job_name,
            "status": job.status,
            "total_ranks": job.total_ranks,
            "num_nodes": job.num_nodes,
            "gpus_per_node": job.gpus_per_node,
            "scheduling_avg_sec": np.mean(schedule_times) if schedule_times else 0,
            "scheduling_p95_sec": np.percentile(schedule_times, 95) if schedule_times else 0,
            "runtime_avg_sec": np.mean(runtime_times) if runtime_times else 0,
            "runtime_p95_sec": np.percentile(runtime_times, 95) if runtime_times else 0,
            "restart_count": sum(p.restart_count for p in job.pods),
            "completion_time_sec": job.completion_time if job.completion_time else 0,
        }
    
    def generate_k8s_manifest(self, job_name: str) -> Dict:
        """Generate a simulated Kubernetes manifest (PyTorchJob CRD)."""
        if job_name not in self.jobs:
            return {"error": "Job not found"}
        
        job = self.jobs[job_name]
        
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": job_name,
                "namespace": job.namespace,
                "creationTimestamp": datetime.now().isoformat(),
            },
            "spec": {
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": 1,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "pytorch",
                                    "image": "pytorch/pytorch:latest",
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": str(job.gpus_per_node)}
                                    }
                                }]
                            }
                        }
                    },
                    "Worker": {
                        "replicas": job.num_nodes - 1,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "pytorch",
                                    "image": "pytorch/pytorch:latest",
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": str(job.gpus_per_node)}
                                    }
                                }]
                            }
                        }
                    }
                }
            }
        }


# Import numpy for metrics
import numpy as np