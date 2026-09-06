"""PyTorch internals and Triton GPU kernels: Paged KV simulator and fused dequant-GEMV."""
from src.internals.paged_cache_sim import PagedKVCacheSimulator, FragmentationStats
from src.internals.triton_dequant import triton_fused_w4a16_gemv, is_triton_available

__all__ = [
    "PagedKVCacheSimulator",
    "FragmentationStats",
    "triton_fused_w4a16_gemv",
    "is_triton_available",
]
