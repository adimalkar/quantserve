"""Unit tests for ML surrogate models and prediction intervals."""
import pytest
from quantserve.advisor.surrogate import PerformanceSurrogate
from quantserve.advisor.predictor import PerformancePredictor
from quantserve.advisor.config_space import DeploymentConfig


def test_surrogate_pretrain_and_predict():
    surrogate = PerformanceSurrogate()
    surrogate.pretrain_on_synthetic_physics(samples=100)
    assert surrogate.is_fitted is True

    cfg = {
        "quantization": "int4_awq",
        "kv_cache_precision": "int8",
        "max_num_seqs": 16,
        "max_num_batched_tokens": 2048,
        "chunked_prefill": True,
    }
    hw = {
        "memory_bandwidth_gbs": 192.0,
        "fp16_tflops": 36.0,
        "total_vram_gb": 6.0,
    }
    wl = {
        "avg_prompt_tokens": 256.0,
        "avg_output_tokens": 64.0,
        "concurrency": 8,
        "request_rate": 5.0,
        "slo_ttft_ms": 500.0,
        "slo_tpot_ms": 40.0,
    }
    meta = {
        "params_billion": 1.23,
        "num_layers": 16,
        "hidden_size": 2048,
        "num_kv_heads": 8,
        "num_heads": 32,
    }

    pred = surrogate.predict(cfg, hw, wl, meta)
    assert pred.ttft_ms > 0.0
    assert pred.tpot_ms > 0.0
    assert pred.throughput_tokens_s > 0.0
    assert pred.peak_vram_gb > 0.0
    assert 0.0 <= pred.slo_compliance_prob <= 1.0


def test_predictor_uncertainty_intervals():
    surrogate = PerformanceSurrogate()
    surrogate.pretrain_on_synthetic_physics(samples=100)
    predictor = PerformancePredictor(surrogate)

    cfg = {"quantization": "fp16", "kv_cache_precision": "fp16", "max_num_seqs": 8, "max_num_batched_tokens": 1024, "chunked_prefill": True}
    hw = {"memory_bandwidth_gbs": 192.0, "fp16_tflops": 36.0, "total_vram_gb": 6.0}
    wl = {"avg_prompt_tokens": 128.0, "avg_output_tokens": 32.0, "concurrency": 4, "request_rate": 2.0}
    meta = {"params_billion": 0.5, "num_layers": 16, "hidden_size": 2048, "num_kv_heads": 8, "num_heads": 32}

    cal_pred = predictor.predict_with_uncertainty(cfg, hw, wl, meta)

    # Verify 90% prediction interval bounds: lower <= point <= upper
    assert cal_pred.ttft.lower_90 <= cal_pred.ttft.point <= cal_pred.ttft.upper_90
    assert cal_pred.tpot.lower_90 <= cal_pred.tpot.point <= cal_pred.tpot.upper_90
    assert cal_pred.throughput.lower_90 <= cal_pred.throughput.point <= cal_pred.throughput.upper_90
    assert 0.0 <= cal_pred.slo_probability_pct <= 100.0
