#!/usr/bin/env python
"""Parse NCCL debug logs for collective call latencies."""
import re
import json
import os

def parse_nccl_log(log_path: str):
    pattern = re.compile(r"\[(\d+)\]\s+NCCL.*?(\w+)\s+(\d+\.\d+)")
    events = []
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                events.append({
                    "rank": int(match.group(1)),
                    "collective": match.group(2),
                    "latency_ms": float(match.group(3))
                })
    return events

def main():
    log_dir = "./artifacts/nccl_traces"
    output = []
    for file in os.listdir(log_dir):
        if file.endswith(".log"):
            path = os.path.join(log_dir, file)
            events = parse_nccl_log(path)
            output.extend(events)
    with open("artifacts/nccl_parsed.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()