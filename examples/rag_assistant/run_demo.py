"""Scenario 2 Demo: RAG Knowledge Assistant Deployment Optimization."""
import os
import subprocess
import sys


def main():
    print("=" * 60)
    print("DEMO 2: RAG Knowledge Assistant")
    print("Workload: Long retrieved contexts (2K-6K), expensive prefill")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m", "quantserve",
        "recommend",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--hardware", "auto",
        "--trace", "examples/rag_assistant/trace.jsonl",
        "--p95-ttft", "1200ms",
        "--p95-tpot", "50ms",
        "--quality-retention", "0.98",
        "--objective", "minimize_cost",
        "--export", "examples/rag_assistant/recommendation.yaml",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
