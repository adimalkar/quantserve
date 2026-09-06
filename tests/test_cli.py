"""Unit tests for the unified QuantServe CLI interface."""
import os
import subprocess
import sys
import pytest


def run_cli_command(args_list):
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    cmd = [sys.executable, "-m", "quantserve"] + args_list
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return res


def test_cli_hardware():
    res = run_cli_command(["hardware"])
    assert res.returncode == 0
    assert "QuantServe: Hardware Inspection" in res.stdout


def test_cli_analyze():
    res = run_cli_command(["analyze", "examples/coding_assistant/trace.jsonl"])
    assert res.returncode == 0
    assert "Workload Fingerprint Summary" in res.stdout


def test_cli_recommend():
    os.makedirs("outputs", exist_ok=True)
    res = run_cli_command([
        "recommend",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--hardware", "ada_4050",
        "--trace", "examples/coding_assistant/trace.jsonl",
        "--p95-ttft", "650ms",
        "--p95-tpot", "40ms",
        "--export", "outputs/test_recommendation.yaml",
    ])
    assert res.returncode == 0
    assert "RECOMMENDED DEPLOYMENT CONFIGURATION" in res.stdout
    assert "vllm serve" in res.stdout


def test_cli_export():
    os.makedirs("outputs", exist_ok=True)
    res = run_cli_command([
        "export",
        "--recommendation", "outputs/test_recommendation.yaml",
        "--format", "vllm",
        "--output", "outputs/test_vllm.yaml",
    ])
    assert res.returncode == 0
    assert "Successfully generated VLLM deployment file" in res.stdout
