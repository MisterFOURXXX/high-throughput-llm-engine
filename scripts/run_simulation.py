#!/usr/bin/env python3
"""CLI entry point for the collective communication simulator."""
import sys
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation import run_simulation, SimulationConfig


def main():
    p = argparse.ArgumentParser(description="NCCL vs XDP vs Gloo simulation")
    p.add_argument("--nodes", type=int, default=2)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--model-size", type=float, default=1000)
    p.add_argument("--gradient-size", type=float, default=256)
    p.add_argument("--bandwidth", type=float, default=100)
    p.add_argument("--region", type=str, default="eu-frankfurt-1")
    p.add_argument("--instance-type", type=str, default="VM.GPU.A10.1")
    p.add_argument("--use-real-terraform", action="store_true")
    p.add_argument("--terraform-dir", type=str, default="configs")
    p.add_argument("--output-dir", type=str, default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    cfg = SimulationConfig(
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
        seed=args.seed,
    )
    r = run_simulation(cfg, output_dir=args.output_dir)
    if args.json:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()