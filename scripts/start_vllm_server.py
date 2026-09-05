#!/usr/bin/env python
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cmd = [
        "python", "-m", "vllm.entrypoints.api_server",
        "--model", args.model,
        "--block-size", str(args.block_size),
        "--tensor-parallel-size", str(args.tp_size),
        "--port", str(args.port),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9"
    ]
    print("Starting vLLM server with command:")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()