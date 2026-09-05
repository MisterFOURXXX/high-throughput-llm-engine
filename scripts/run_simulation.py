#!/usr/bin/env python
"""
Main entry point for running the Multi-Node NCCL simulation.
"""

import sys
import os
import argparse
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import run_simulation, SimulationConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run Multi-Node NCCL Simulation"
    )
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per node")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--model-size", type=float, default=1000, help="Model size in MB")
    parser.add_argument("--gradient-size", type=float, default=256, help="Gradient size in MB")
    parser.add_argument("--bandwidth", type=float, default=100, help="Network bandwidth in Gbps")
    parser.add_argument("--region", type=str, default="eu-central-1", help="AWS region")
    parser.add_argument("--instance-type", type=str, default="g5.12xlarge", help="Instance type")
    parser.add_argument("--use-real-terraform", action="store_true", help="Use real Terraform")
    parser.add_argument("--terraform-dir", type=str, default="configs", help="Terraform directory")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    config = SimulationConfig(
        num_nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        num_iterations=args.iterations,
        model_size_mb=args.model_size,
        gradient_size_mb=args.gradient_size,
        network_bandwidth_gbps=args.bandwidth,
        region=args.region,
        instance_type=args.instance_type,
        use_real_terraform=args.use_real_terraform,
        terraform_dir=args.terraform_dir,
    )
    
    results = run_simulation(config, output_dir=args.output_dir)
    
    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()