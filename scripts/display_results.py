#!/usr/bin/env python3
"""Print a formatted summary of the simulation report."""
import json
import sys
from pathlib import Path


def line(width=78, char="="):
    print(char * width)


def row(label, value, unit="", width=42):
    if unit:
        print(f"  {label:<{width}s} : {value} {unit}")
    else:
        print(f"  {label:<{width}s} : {value}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/simulation_report.json"
    with open(path) as f:
        r = json.load(f)

    line()
    print(" MULTI-NODE COLLECTIVE COMMUNICATION - RESULTS SUMMARY")
    print(f" Generated: {r['timestamp']}")
    print(f" Cloud    : {r['cloud']}")
    print(f" Seed     : {r['seed']}")
    line()

    # CONFIG
    line(78, "-")
    print(" CONFIGURATION")
    line(78, "-")
    c = r["config"]
    row("Nodes", c["num_nodes"])
    row("GPUs per node", c["gpus_per_node"])
    row("Total GPUs", c["total_gpus"])
    row("Iterations", c["num_iterations"])
    row("Model size", c["model_size_mb"], "MB")
    row("Gradient size", c["gradient_size_mb"], "MB")
    row("Network bandwidth", c["network_bandwidth_gbps"], "Gbps")
    row("Region", c["region"])
    row("Instance type", c["instance_type"])

    # BACKEND RESULTS
    line(78, "-")
    print(" BACKEND TOTALS")
    line(78, "-")
    row("NCCL total", r["nccl"]["total_communication_ms"], "ms")
    row("XDP  total", r["xdp"]["total_communication_ms"], "ms")
    row("Gloo total", r["gloo"]["total_communication_ms"], "ms")

    line(78, "-")
    print(" SPEEDUP")
    line(78, "-")
    cmp = r["comparison"]
    row("XDP  speedup vs Gloo", f"{cmp['xdp_speedup_vs_gloo']}x")
    row("NCCL speedup vs XDP", f"{cmp['nccl_speedup_vs_xdp']}x")
    row("NCCL speedup vs Gloo", f"{cmp['nccl_speedup_vs_gloo']}x")

    # K8S
    line(78, "-")
    print(" KUBERNETES OPERATOR")
    line(78, "-")
    k = r["kubernetes"]
    row("Job name", k["job_name"])
    row("Status", k["status"])
    row("Total ranks", k["total_ranks"])
    row("Completion time", k["completion_time_sec"], "s")

    # eBPF
    line(78, "-")
    print(" eBPF / bpfd MONITORING")
    line(78, "-")
    e = r["ebpf"]
    row("Events captured", e["events_count"])
    row("Avg latency", e["avg_latency_us"], "us")
    row("P95 latency", e["p95_latency_us"], "us")
    row("P99 latency", e["p99_latency_us"], "us")
    row("Packet drop rate", f"{e['packet_drop_rate']*100:.4f}", "%")
    line()


if __name__ == "__main__":
    main()