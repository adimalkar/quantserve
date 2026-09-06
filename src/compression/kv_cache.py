"""KV cache memory footprints, scaling equations, and INT8/FP8 quantization."""
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import torch


@dataclass
class KVCacheFootprint:
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    precision: str
    bytes_per_element: float

    def bytes_per_token_per_sequence(self) -> int:
        """Calculates total memory (Key + Value) in bytes for 1 token across all layers."""
        # 2 (Key + Value) * layers * num_kv_heads * head_dim * bytes_per_element
        return int(2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element)

    def total_memory_mb(self, batch_size: int, seq_len: int) -> float:
        """Total memory in Megabytes for a given batch size and context length."""
        total_bytes = batch_size * seq_len * self.bytes_per_token_per_sequence()
        return total_bytes / (1024.0 * 1024.0)

    def max_concurrent_sequences(self, available_vram_gb: float, context_len: int) -> int:
        """Maximum concurrent sequences that fit within allocated VRAM before OOM or swapping."""
        vram_bytes = available_vram_gb * (1024.0 ** 3)
        bytes_per_seq = context_len * self.bytes_per_token_per_sequence()
        if bytes_per_seq == 0:
            return 0
        return int(vram_bytes // bytes_per_seq)


class KVCacheQuantizer:
    """Handles runtime quantization and dequantization of Key and Value tensors."""

    def __init__(self, precision: str = "int8"):
        self.precision = precision.lower()

    def quantize(self, kv_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes Key/Value activations: returns (quantized_tensor, per_token_scale)."""
        if self.precision == "fp16":
            return kv_tensor.to(torch.float16), torch.tensor(1.0, device=kv_tensor.device)

        if self.precision in ("int8", "w8a8"):
            # Per-token / per-channel symmetric int8
            scale = kv_tensor.abs().amax(dim=-1, keepdim=True) / 127.0
            scale = torch.clamp(scale, min=1e-5)
            q = torch.clamp(torch.round(kv_tensor / scale), -128, 127).to(torch.int8)
            return q, scale

        if self.precision in ("fp8", "fp8_e4m3"):
            scale = kv_tensor.abs().amax(dim=-1, keepdim=True) / 448.0
            scale = torch.clamp(scale, min=1e-5)
            if hasattr(torch, "float8_e4m3fn"):
                q = (kv_tensor / scale).to(torch.float8_e4m3fn)
            else:
                q = torch.clamp(torch.round(kv_tensor / scale), -128, 127).to(torch.int8)
            return q, scale

        return kv_tensor, torch.tensor(1.0, device=kv_tensor.device)

    def dequantize(self, q_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantizes Key/Value tensor back to float16 for attention compute."""
        return q_tensor.to(torch.float16) * scale.to(torch.float16)
