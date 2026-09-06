"""Unit tests for configuration space pruning and deployment optimization."""
import pytest
from quantserve.advisor.config_space import ConfigurationSpace, DeploymentConfig
from quantserve.advisor.hardware_probe import HARDWARE_PRESETS
from quantserve.advisor.optimizer import DeploymentOptimizer


def test_config_space_generation_and_pruning():
    space = ConfigurationSpace()
    all_cands = space.generate_all_candidates()
    assert len(all_cands) >= 100

    ada_hw = HARDWARE_PRESETS["ada_4050"]  # 6GB VRAM
    # For a 3B model (~3.21e9 params): 3B in FP16 takes ~6.4GB, which exceeds 6GB VRAM!
    valid_cands = space.prune_for_hardware(all_cands, ada_hw, param_count=3.21e9)

    # All FP16 configs should be pruned for a 3B model on 6GB VRAM
    for c in valid_cands:
        assert c.quantization != "fp16"
        assert c.quantization in ("int4_awq", "int4_gptq", "fp8_e4m3", "int8_smoothquant")


def test_deployment_optimizer_recommendation():
    optimizer = DeploymentOptimizer()
    ada_hw = HARDWARE_PRESETS["ada_4050"]

    workload = {
        "avg_prompt_tokens": 300.0,
        "avg_output_tokens": 64.0,
        "concurrency": 8,
        "request_rate": 5.0,
        "slo_ttft_ms": 650.0,
        "slo_tpot_ms": 40.0,
    }

    rec = optimizer.recommend(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        hardware=ada_hw,
        workload=workload,
        p95_ttft_max_ms=650.0,
        p95_tpot_max_ms=40.0,
        quality_retention_min=0.98,
        objective="minimize_latency",
    )

    assert rec.recommended_config is not None
    assert rec.total_candidates > 0
    assert rec.prediction.ttft.point > 0.0
    assert rec.prediction.tpot.point > 0.0
    assert rec.quality_retention_pct >= 98.0
    assert "vllm serve" in rec.generate_vllm_command()
    assert len(rec.explanation) > 20
