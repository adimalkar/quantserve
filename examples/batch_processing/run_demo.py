"""Scenario 3 Demo: Batch Document Processing Deployment Optimization."""
import os
import subprocess
import sys


def main():
    print("=" * 60)
    print("DEMO 3: Batch Document Processing")
    print("Workload: High burst volume, maximum throughput, loose latency")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m", "quantserve",
        "recommend",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--hardware", "auto",
        "--trace", "examples/batch_processing/trace.jsonl",
        "--p95-ttft", "2500ms",
        "--p95-tpot", "80ms",
        "--quality-retention", "0.97",
        "--objective", "maximize_throughput",
        "--export", "examples/batch_processing/recommendation.yaml",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
