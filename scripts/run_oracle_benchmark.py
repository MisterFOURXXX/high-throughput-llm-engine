#!/usr/bin/env python
"""
Oracle Cloud Multi-Node Benchmark Runner

Fetches instance IPs from Terraform output and orchestrates
benchmarks across all OCI nodes via SSH.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def run_terraform_output(terraform_dir: str, output_name: str):
    """Fetch a Terraform output value."""
    result = subprocess.run(
        ["terraform", f"-chdir={terraform_dir}", "output", "-json", output_name],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"✗ Failed to get Terraform output: {output_name}")
        print(result.stderr)
        return None
    try:
        return json.loads(result.stdout).get("value")
    except json.JSONDecodeError:
        return None


def run_on_node(
    ip: str,
    ssh_key: str,
    node_index: int,
    node_rank: int,
    total_nodes: int,
    concurrency: int,
    duration: int,
    output_dir: str,
):
    """Execute benchmark on a single OCI node via SSH."""
    print(f"[Node {node_index}] Starting benchmark on {ip}")

    # Build SSH command to run container
    remote_cmd = (
        f"docker exec {os.environ.get('CLUSTER_NAME', 'nccl-cluster')}-node-{node_rank} "
        f"python3 scripts/run_simulation.py "
        f"--nodes {total_nodes} "
        f"--iterations 50"
    )

    ssh_cmd = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        f"opc@{ip}",
        remote_cmd,
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=600)
        status = "success" if result.returncode == 0 else "failed"

        # Copy artifacts back
        scp_cmd = [
            "scp",
            "-i", ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-r",
            f"opc@{ip}:/home/opc/artifacts/",
            f"{output_dir}/node_{node_index}/",
        ]
        subprocess.run(scp_cmd, capture_output=True, timeout=120)

        return {
            "node_index": node_index,
            "ip": ip,
            "status": status,
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"node_index": node_index, "ip": ip, "status": "timeout"}


def main():
    parser = argparse.ArgumentParser(description="Run multi-node benchmark on Oracle Cloud")
    parser.add_argument("--terraform-dir", default="configs", help="Terraform directory")
    parser.add_argument("--output-dir", default="artifacts", help="Local output directory")
    parser.add_argument("--concurrency", type=int, default=100, help="Benchmark concurrency")
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration (s)")
    parser.add_argument("--parallel", action="store_true", help="Run benchmarks in parallel")
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Fetch Terraform outputs
    print("Fetching Terraform outputs...")
    ips = run_terraform_output(args.terraform_dir, "instance_ips")
    ssh_key = run_terraform_output(args.terraform_dir, "ssh_private_key_path")

    if not ips:
        print("✗ No instance IPs found. Run `terraform apply` first.")
        sys.exit(1)

    if not ssh_key or not Path(ssh_key).exists():
        print(f"✗ SSH key not found: {ssh_key}")
        sys.exit(1)

    print(f"Found {len(ips)} nodes: {ips}")
    print(f"SSH key: {ssh_key}")

    # Run benchmarks
    print(f"\nStarting benchmarks on {len(ips)} nodes...")
    start_time = time.time()

    if args.parallel:
        with ThreadPoolExecutor(max_workers=len(ips)) as executor:
            futures = [
                executor.submit(
                    run_on_node,
                    ip, ssh_key, i, i, len(ips),
                    args.concurrency, args.duration, args.output_dir
                )
                for i, ip in enumerate(ips)
            ]
            results = [f.result() for f in as_completed(futures)]
    else:
        results = []
        for i, ip in enumerate(ips):
            r = run_on_node(
                ip, ssh_key, i, i, len(ips),
                args.concurrency, args.duration, args.output_dir
            )
            results.append(r)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("ORACLE CLOUD BENCHMARK SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  Node {r['node_index']} ({r['ip']}): {r['status']}")
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Artifacts: {args.output_dir}/")

    # Save report
    report_path = Path(args.output_dir) / "oracle_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "nodes": len(ips),
            "ips": ips,
            "results": results,
            "elapsed_sec": elapsed,
        }, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    import os
    main()