#!/usr/bin/env python
"""
Orchestrate multi‑node benchmarks using Terraform output.
Run after `terraform apply` to get instance IPs.
"""
import subprocess
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

def get_instance_ips():
    result = subprocess.run(
        ["terraform", "-chdir=configs", "output", "-json", "instance_ips"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Failed to get instance IPs from Terraform")
        sys.exit(1)
    try:
        data = json.loads(result.stdout)
        return data["value"]
    except:
        print("Invalid JSON output")
        sys.exit(1)

def run_on_node(ip, node_id, concurrency, duration):
    # Execute the load generator remotely via SSH
    cmd = (
        f"ssh -o StrictHostKeyChecking=no ubuntu@{ip} "
        f"'cd /app && ./benchmarks/build/loadgen http://{ip}:8000/generate {concurrency} 10 {duration}'"
    )
    subprocess.run(cmd, shell=True, check=True)
    # Scp back the results
    subprocess.run(
        f"scp ubuntu@{ip}:/app/benchmark_results.csv ./artifacts/benchmark_node{node_id}.csv",
        shell=True, check=True
    )

def main():
    ips = get_instance_ips()
    if len(ips) < 1:
        print("No instances found.")
        sys.exit(1)
    # Example: run with concurrency=100, duration=60 on each node
    with ThreadPoolExecutor(max_workers=len(ips)) as executor:
        futures = []
        for i, ip in enumerate(ips):
            futures.append(executor.submit(run_on_node, ip, i, 100, 60))
        for f in futures:
            f.result()
    print("All node benchmarks complete. Results in artifacts/")

if __name__ == "__main__":
    main()