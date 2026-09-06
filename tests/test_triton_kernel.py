"""Unit tests for Triton fused W4A16 GEMV kernel numerical accuracy."""
import pytest
import torch

from src.internals.triton_dequant import (
    triton_fused_w4a16_gemv,
    _pytorch_unfused_w4a16_gemv,
    is_triton_available,
)
from src.compression.quantize import QuantizedLinear


def test_triton_fused_gemv_numerical_accuracy():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    M, K = 512, 1024
    group_size = 128

    # Synthetic input and weights
    x = torch.randn(K, dtype=torch.float16, device=dev)
    float_w = torch.randn((M, K), dtype=torch.float16, device=dev)

    ql = QuantizedLinear(in_features=K, out_features=M, precision="int4_awq", group_size=group_size)
    ql.pack_from_float(float_w)
    ql.to(dev)

    # Reference PyTorch output
    ref_y = _pytorch_unfused_w4a16_gemv(ql.qweight, ql.scales, ql.zeros, x, group_size=group_size)

    # Triton kernel output
    if is_triton_available():
        triton_y = triton_fused_w4a16_gemv(ql.qweight, ql.scales, ql.zeros, x, group_size=group_size)
        # Verify relative error between Triton and reference PyTorch unpack
        # Because FP16 has limited mantissa and accumulate is done in FP32, tolerance is atol=0.05
        max_diff = torch.max(torch.abs(ref_y - triton_y)).item()
        assert max_diff < 0.1, f"Triton max diff {max_diff} exceeded tolerance"
    else:
        pytest.skip("CUDA/Triton not available for kernel test")
