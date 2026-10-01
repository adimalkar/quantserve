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


def test_cli_recommend(tmp_path):
    output_path = tmp_path / "recommendation.yaml"
    res = run_cli_command([
        "recommend",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--hardware", "ada_4050",
        "--trace", "examples/coding_assistant/trace.jsonl",
        "--p95-ttft", "650ms",
        "--p95-tpot", "40ms",
        "--export", str(output_path),
    ])
    assert res.returncode == 0
    assert "RECOMMENDED DEPLOYMENT CONFIGURATION" in res.stdout
    assert "vllm serve" in res.stdout
    assert output_path.exists()


def test_cli_export(tmp_path):
    recommendation_path = tmp_path / "recommendation.yaml"
    recommendation = run_cli_command([
        "recommend",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--hardware", "ada_4050",
        "--export", str(recommendation_path),
    ])
    assert recommendation.returncode == 0
    res = run_cli_command([
        "export",
        "--recommendation", str(recommendation_path),
        "--format", "vllm",
        "--output", str(tmp_path / "vllm.yaml"),
    ])
    assert res.returncode == 0
    assert "Successfully generated VLLM deployment file" in res.stdout
