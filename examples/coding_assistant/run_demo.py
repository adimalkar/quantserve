"""Scenario 1 Demo: Interactive Coding Assistant Deployment Optimization."""
import os
import subprocess
import sys


def main():
    print("=" * 60)
    print("DEMO 1: Interactive Coding Assistant (Qwen2.5-Coder)")
    print("Workload: Short prompts, low TTFT/TPOT, high responsiveness")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m", "quantserve",
        "recommend",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--hardware", "auto",
        "--trace", "examples/coding_assistant/trace.jsonl",
        "--p95-ttft", "650ms",
        "--p95-tpot", "40ms",
        "--quality-retention", "0.98",
        "--objective", "minimize_latency",
        "--export", "examples/coding_assistant/recommendation.yaml",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
