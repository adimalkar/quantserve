"""Unit tests for Roofline model and operational intensity."""
import pytest
from src.analysis.roofline import RooflineModel


def test_roofline_model_ridge_points():
    # Model RTX 4050: 192 GB/s, 36 TFLOPs FP16
    model = RooflineModel(
        memory_bandwidth_gbs=192.0,
        peak_fp16_tflops=36.0,
        peak_int8_tops=72.0,
        peak_fp8_tflops=72.0,
    )

    # Ridge point = 36e12 / 192e9 = 187.5 FLOPs / Byte
    assert pytest.approx(model.ridge_point_fp16, abs=0.1) == 187.5
    # Ridge point for FP8/INT8 = 72e12 / 192e9 = 375.0 FLOPs / Byte
    assert pytest.approx(model.ridge_point_fp8, abs=0.1) == 375.0


def test_decode_operational_intensity_and_prefill_flip():
    model = RooflineModel(memory_bandwidth_gbs=192.0, peak_fp16_tflops=36.0)

    # Decode at batch 1 for 1.23B param model: heavily memory-bandwidth bound
    pt_b1 = model.compute_decode_point(batch_size=1, param_count=1.23e9, precision="fp16")
    assert pt_b1.execution_bound == "Memory Bandwidth Bound"
    assert pt_b1.operational_intensity < 5.0  # ~1.0 FLOP/byte

    # Prefill with 512 tokens: Operational Intensity = 2 * 512 / 2 = 512 FLOPs/Byte > 187.5 Ridge Point
    pt_prefill = model.compute_prefill_point(prompt_len=512, param_count=1.23e9, precision="fp16")
    assert pt_prefill.execution_bound == "Compute Bound"
    assert pt_prefill.operational_intensity > model.ridge_point_fp16
