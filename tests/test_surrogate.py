"""Unit tests for ML surrogate models and prediction intervals."""
import numpy as np
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


def test_build_interval_preserves_wide_lower_tail_and_skewed_invariant():
    # Wide lower tail: p5=20.0 is well below 0.7 * mean (~64.4), so lower_90 must stay 20.0.
    wide_tail = np.array([20.0] * 10 + [100.0] * 90)
    wide_iv = PerformancePredictor._build_interval(wide_tail)
    assert wide_iv.point == pytest.approx(92.0)
    assert wide_iv.lower_90 == pytest.approx(20.0)
    assert wide_iv.upper_90 == pytest.approx(92.0 * 1.3)
    assert wide_iv.lower_90 <= wide_iv.point <= wide_iv.upper_90

    # Tight ensemble agreement: enforces minimum +-30% interval around point.
    tight = np.array([98.0, 99.0, 100.0, 101.0, 102.0])
    tight_iv = PerformancePredictor._build_interval(tight)
    assert tight_iv.point == pytest.approx(100.0)
    assert tight_iv.lower_90 == pytest.approx(70.0)
    assert tight_iv.upper_90 == pytest.approx(130.0)

    # Left-skewed ensemble where p5 (100.0) > mean (96.0): lower_90 must not exceed point.
    left_skewed = np.array([0.0] * 2 + [100.0] * 48)
    skewed_iv = PerformancePredictor._build_interval(left_skewed)
    assert skewed_iv.point == pytest.approx(96.0)
    assert skewed_iv.lower_90 <= skewed_iv.point <= skewed_iv.upper_90
    assert skewed_iv.lower_90 == pytest.approx(96.0 * 0.7)

    # Negative lower percentile with non-negative point clamps lower_90 at 0.0.
    neg_tail = np.array([-20.0] * 5 + [100.0] * 45)
    neg_iv = PerformancePredictor._build_interval(neg_tail)
    assert neg_iv.point == pytest.approx(88.0)
    assert neg_iv.lower_90 == pytest.approx(0.0)
    assert neg_iv.lower_90 <= neg_iv.point <= neg_iv.upper_90


def test_predict_batch_with_uncertainty_wide_and_skewed_trees():
    class _FixedTree:
        def __init__(self, value: float):
            self._value = value

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(len(X), self._value, dtype=np.float64)

    class _FixedForest:
        def __init__(self, values):
            self.estimators_ = [_FixedTree(v) for v in values]

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(len(X), 2.5, dtype=np.float64)

    class _StubScaler:
        def transform(self, X: np.ndarray) -> np.ndarray:
            return X

    surrogate = PerformanceSurrogate()
    surrogate.is_fitted = True
    surrogate.scaler = _StubScaler()
    # Wide lower tail for TTFT, left-skewed for TPOT, tight for throughput
    surrogate.model_ttft = _FixedForest([20.0] * 10 + [100.0] * 40)
    surrogate.model_tpot = _FixedForest([0.0] * 2 + [100.0] * 48)
    surrogate.model_throughput = _FixedForest([100.0] * 50)
    surrogate.model_vram = _FixedForest([2.5])

    predictor = PerformancePredictor(surrogate)
    cfg = {"quantization": "fp16", "kv_cache_precision": "fp16", "max_num_seqs": 8, "max_num_batched_tokens": 1024, "chunked_prefill": True}
    hw = {"memory_bandwidth_gbs": 192.0, "fp16_tflops": 36.0, "total_vram_gb": 6.0}
    wl = {"avg_prompt_tokens": 128.0, "avg_output_tokens": 32.0, "concurrency": 4, "request_rate": 2.0}
    meta = {"params_billion": 0.5, "num_layers": 16, "hidden_size": 2048, "num_kv_heads": 8, "num_heads": 32}

    preds = predictor.predict_batch_with_uncertainty([cfg], hw, wl, meta)
    assert len(preds) == 1
    pred = preds[0]
    assert pred.ttft.lower_90 == pytest.approx(20.0)
    assert pred.ttft.lower_90 <= pred.ttft.point <= pred.ttft.upper_90
    assert pred.tpot.lower_90 <= pred.tpot.point <= pred.tpot.upper_90
    assert pred.throughput.lower_90 == pytest.approx(70.0)
    assert pred.throughput.upper_90 == pytest.approx(130.0)
