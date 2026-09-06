"""Kernel category classification and timing breakdown parser."""
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class KernelBreakdown:
    batch_size: int
    precision: str
    total_cuda_time_us: float
    gemm_time_us: float
    gemm_pct: float
    dequant_time_us: float
    dequant_pct: float
    attention_time_us: float
    attention_pct: float
    norm_activation_time_us: float
    norm_activation_pct: float
    other_time_us: float
    other_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "precision": self.precision,
            "total_cuda_time_us": round(self.total_cuda_time_us, 1),
            "gemm_pct": round(self.gemm_pct, 1),
            "dequant_pct": round(self.dequant_pct, 1),
            "attention_pct": round(self.attention_pct, 1),
            "norm_activation_pct": round(self.norm_activation_pct, 1),
            "other_pct": round(self.other_pct, 1),
        }


def parse_kernel_trace(
    raw_events: List[Dict[str, Any]], batch_size: int, precision: str
) -> KernelBreakdown:
    """Classifies CUDA events into high-level categories (GEMM, Dequant, Attention, Norms)."""
    gemm_us = 0.0
    dequant_us = 0.0
    attn_us = 0.0
    norm_us = 0.0
    other_us = 0.0

    for ev in raw_events:
        name = ev.get("name", "").lower()
        dur = float(ev.get("dur", 0.0))

        if any(k in name for k in ["gemm", "cutlass", "cublas", "matmul", "int_mm"]):
            gemm_us += dur
        elif any(k in name for k in ["dequant", "unpack", "marlin", "awq", "smooth"]):
            dequant_us += dur
        elif any(k in name for k in ["attention", "flash", "sdpa", "fmha"]):
            attn_us += dur
        elif any(k in name for k in ["norm", "silu", "gelu", "swiglu", "bias", "add"]):
            norm_us += dur
        else:
            other_us += dur

    total_us = gemm_us + dequant_us + attn_us + norm_us + other_us
    denom = max(total_us, 1e-5)

    return KernelBreakdown(
        batch_size=batch_size,
        precision=precision,
        total_cuda_time_us=total_us,
        gemm_time_us=gemm_us,
        gemm_pct=(gemm_us / denom) * 100.0,
        dequant_time_us=dequant_us,
        dequant_pct=(dequant_us / denom) * 100.0,
        attention_time_us=attn_us,
        attention_pct=(attn_us / denom) * 100.0,
        norm_activation_time_us=norm_us,
        norm_activation_pct=(norm_us / denom) * 100.0,
        other_time_us=other_us,
        other_pct=(other_us / denom) * 100.0,
    )
