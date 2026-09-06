"""Unit tests for quantization modules, 2:4 structured sparsity, and KV cache quantizers."""
import pytest
import torch
import torch.nn as nn

from src.compression.quantize import QuantizedLinear, AWQQuantizer, SmoothQuantizer, FP8Quantizer
from src.compression.sparsity import StructuredSparsity24, Sparse24Linear
from src.compression.kv_cache import KVCacheFootprint, KVCacheQuantizer


def test_quantized_linear_int4_pack_and_forward():
    in_feat, out_feat = 256, 128
    layer = QuantizedLinear(in_features=in_feat, out_features=out_feat, precision="int4_awq", group_size=128)

    float_w = torch.randn((out_feat, in_feat), dtype=torch.float16)
    layer.pack_from_float(float_w)

    x = torch.randn((2, in_feat), dtype=torch.float16)
    out = layer(x)

    assert out.shape == (2, out_feat)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_quantized_linear_w8a8():
    in_feat, out_feat = 256, 128
    layer = QuantizedLinear(in_features=in_feat, out_features=out_feat, precision="w8a8")

    float_w = torch.randn((out_feat, in_feat), dtype=torch.float16)
    layer.pack_from_float(float_w)

    x = torch.randn((2, in_feat), dtype=torch.float16)
    out = layer(x)

    assert out.shape == (2, out_feat)
    assert not torch.isnan(out).any()


def test_2_4_structured_sparsity_constraint():
    tensor = torch.randn((64, 128))
    pruned = StructuredSparsity24.prune_to_2_4(tensor)

    # Check 2:4 constraint
    assert StructuredSparsity24.verify_2_4_constraint(pruned) is True

    # Pack into 50% dense and metadata
    packed_vals, metadata = StructuredSparsity24.pack_dense_and_metadata(pruned)
    assert packed_vals.shape == (64, 64)
    assert metadata.shape == (64, 32)


def test_kv_cache_footprint_and_quantizer():
    # Llama-3.2-1B: 16 layers, 8 kv_heads, 64 head_dim
    fp16_footprint = KVCacheFootprint(
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        precision="fp16",
        bytes_per_element=2.0,
    )
    # Total bytes per token across all layers: 2 * 16 * 8 * 64 * 2 = 32,768 bytes
    assert fp16_footprint.bytes_per_token_per_sequence() == 32768

    int8_footprint = KVCacheFootprint(
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        precision="int8",
        bytes_per_element=1.0,
    )
    # Memory should be exactly half of FP16
    assert int8_footprint.bytes_per_token_per_sequence() == 16384

    # Max concurrent sequences with 2GB VRAM budget at 2048 context length
    seqs_fp16 = fp16_footprint.max_concurrent_sequences(available_vram_gb=2.0, context_len=2048)
    seqs_int8 = int8_footprint.max_concurrent_sequences(available_vram_gb=2.0, context_len=2048)

    assert seqs_int8 == seqs_fp16 * 2

    # Test quantizer round-trip
    quantizer = KVCacheQuantizer(precision="int8")
    sample_kv = torch.randn((4, 8, 64), dtype=torch.float16)
    q_kv, scale = quantizer.quantize(sample_kv)
    dequant = quantizer.dequantize(q_kv, scale)

    # Relative L2 error should be small (< 0.05)
    l2_err = torch.norm(sample_kv - dequant) / torch.norm(sample_kv)
    assert l2_err < 0.05
