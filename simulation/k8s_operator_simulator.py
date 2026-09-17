"""
Kubernetes Operator Simulator - deterministic.
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum


class PodStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"


@dataclass
class Pod:
    name: str
    namespace: str = "default"
    status: PodStatus = PodStatus.PENDING
    node_name: str = ""
    rank: int = -1
    start_time: float = 0.0
    end_time: float = 0.0
    restart_count: int = 0
    exit_code: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class NCCLJob:
    name: str
    namespace: str = "default"
    num_nodes: int = 2
    gpus_per_node: int = 4
    total_ranks: int = 8
    status: str = "Pending"
    pods: List[Pod] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    completion_time: float = 0.0


class KubernetesOperatorSimulator:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self.jobs: Dict[str, NCCLJob] = {}
        self.pods: Dict[str, Pod] = {}
        self.nodes = [f"node-{i}" for i in range(10)]

    def create_nccl_job(self, job_name: str, num_nodes: int = 2,
                        gpus_per_node: int = 4,
                        namespace: str = "default") -> NCCLJob:
        job = NCCLJob(
            name=job_name, namespace=namespace,
            num_nodes=num_nodes, gpus_per_node=gpus_per_node,
            total_ranks=num_nodes * gpus_per_node,
            status="Pending", start_time=0.0,
        )
        for rank in range(job.total_ranks):
            node_name = self._rng.choice(self.nodes)
            pod_name = f"{job_name}-rank-{rank}"
            pod = Pod(
                name=pod_name, namespace=namespace,
                status=PodStatus.PENDING, node_name=node_name,
                rank=rank,
                labels={"job-name": job_name, "role": "worker", "rank": str(rank)},
                annotations={
                    "nccl.io/rank": str(rank),
                    "nccl.io/world-size": str(job.total_ranks),
                },
            )
            job.pods.append(pod)
            self.pods[pod_name] = pod
        self.jobs[job_name] = job
        return job

    def reconcile(self, job_name: str) -> Dict[str, Any]:
        """
        Single reconciliation pass. Moves pods PENDING -> RUNNING -> SUCCEEDED
        deterministically across successive calls.
        """
        if job_name not in self.jobs:
            return {"status": "error", "message": "job not found"}

        job = self.jobs[job_name]

        for pod in job.pods:
            if pod.status == PodStatus.PENDING:
                pod.status = PodStatus.RUNNING
                pod.start_time = 1.0
            elif pod.status == PodStatus.RUNNING:
                pod.status = PodStatus.SUCCEEDED
                pod.end_time = 2.0

        all_succeeded = all(p.status == PodStatus.SUCCEEDED for p in job.pods)
        any_failed    = any(p.status == PodStatus.FAILED for p in job.pods)

        if any_failed:
            job.status = "Failed"
        elif all_succeeded:
            job.status = "Succeeded"
            job.end_time = 2.0
            job.completion_time = 2.0
        elif any(p.status == PodStatus.RUNNING for p in job.pods):
            job.status = "Running"
        else:
            job.status = "Pending"

        return {
            "job_name": job_name,
            "status": job.status,
            "total_ranks": job.total_ranks,
            "running_pods":   sum(1 for p in job.pods if p.status == PodStatus.RUNNING),
            "succeeded_pods": sum(1 for p in job.pods if p.status == PodStatus.SUCCEEDED),
            "failed_pods":    sum(1 for p in job.pods if p.status == PodStatus.FAILED),
            "pending_pods":   sum(1 for p in job.pods if p.status == PodStatus.PENDING),
            "completion_time_sec": job.completion_time,
        }

    def run_job_to_completion(self, job_name: str,
                              max_attempts: int = 20,
                              sleep_interval: float = 0.0) -> Dict[str, Any]:
        """
        Loop reconcile until the job reaches a terminal state.
        Fixes the bug where a single reconcile call only moved pods to RUNNING.
        """
        result = self.reconcile(job_name)
        attempt = 0
        while result.get("status") not in ("Succeeded", "Failed") and attempt < max_attempts:
            result = self.reconcile(job_name)
            attempt += 1
        return result

    def get_job_metrics(self, job_name: str) -> Dict[str, Any]:
        if job_name not in self.jobs:
            return {"error": "Job not found"}
        job = self.jobs[job_name]
        return {
            "job_name": job_name,
            "status": job.status,
            "total_ranks": job.total_ranks,
            "num_nodes": job.num_nodes,
            "gpus_per_node": job.gpus_per_node,
            "completion_time_sec": job.completion_time,
        }