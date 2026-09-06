"""Hardware Roofline Model: Arithmetic Intensity, Ridge Points, and Bound Classification."""
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class OperationalPoint:
    phase: str  # "prefill" or "decode"
    batch_size: int
    precision: str
    flops: float
    bytes_transferred: float
    operational_intensity: float  # FLOPs / Byte
    attainable_performance_tflops: float
    execution_bound: str  # "Memory Bandwidth Bound" or "Compute Bound"


class RooflineModel:
    """Calculates theoretical hardware ceilings and operational intensity for LLM inference."""

    def __init__(
        self,
        memory_bandwidth_gbs: float = 192.0,
        peak_fp16_tflops: float = 36.0,
        peak_int8_tops: float = 72.0,
        peak_fp8_tflops: float = 72.0,
    ):
        self.bw_bytes_per_s = memory_bandwidth_gbs * 1e9
        self.peak_fp16_flops = peak_fp16_tflops * 1e12
        self.peak_int8_ops = peak_int8_tops * 1e12
        self.peak_fp8_flops = peak_fp8_tflops * 1e12

        # Ridge points (FLOPs / Byte) where bound flips
        self.ridge_point_fp16 = self.peak_fp16_flops / self.bw_bytes_per_s
        self.ridge_point_int8 = self.peak_int8_ops / self.bw_bytes_per_s
        self.ridge_point_fp8 = self.peak_fp8_flops / self.bw_bytes_per_s

    def compute_decode_point(
        self,
        batch_size: int,
        param_count: float,
        precision: str,
        context_len: int = 128,
        layers: int = 16,
        hidden_dim: int = 2048,
    ) -> OperationalPoint:
        """Calculates decode operational intensity for a given batch size."""
        prec = precision.lower()
        if prec in ("fp16", "bf16"):
            bytes_per_param = 2.0
            peak_compute = self.peak_fp16_flops
            dequant_factor = 1.0
        elif prec in ("int8", "w8a8", "int8_smoothquant"):
            bytes_per_param = 1.0
            peak_compute = self.peak_int8_ops
            dequant_factor = 1.05
        elif prec in ("fp8", "fp8_e4m3"):
            bytes_per_param = 1.0
            peak_compute = self.peak_fp8_flops
            dequant_factor = 1.02
        elif prec in ("int4", "int4_awq", "int4_gptq", "w4a16"):
            bytes_per_param = 0.5
            peak_compute = self.peak_fp16_flops
            dequant_factor = 1.35
        else:
            bytes_per_param = 2.0
            peak_compute = self.peak_fp16_flops
            dequant_factor = 1.0

        weight_bytes = param_count * bytes_per_param
        # In GQA with 4:1 query:kv ratio, kv heads = hidden_dim // 4
        kv_bytes = batch_size * context_len * layers * 2 * (hidden_dim // 4) * 2.0
        total_bytes = weight_bytes + kv_bytes

        flops = 2.0 * param_count * batch_size * dequant_factor
        operational_intensity = flops / max(total_bytes, 1.0)

        memory_ceiling_flops = operational_intensity * self.bw_bytes_per_s
        attainable_flops = min(peak_compute, memory_ceiling_flops)

        bound = "Memory Bandwidth Bound" if memory_ceiling_flops < peak_compute else "Compute Bound"

        return OperationalPoint(
            phase="decode",
            batch_size=batch_size,
            precision=precision,
            flops=flops,
            bytes_transferred=total_bytes,
            operational_intensity=operational_intensity,
            attainable_performance_tflops=attainable_flops / 1e12,
            execution_bound=bound,
        )

    def compute_prefill_point(
        self,
        prompt_len: int,
        param_count: float,
        precision: str = "fp16",
    ) -> OperationalPoint:
        """Prefill is fundamentally compute-bound: weights read once, FLOPs = 2 * params * prompt_len."""
        bytes_per_param = 2.0 if precision in ("fp16", "bf16") else 1.0
        weight_bytes = param_count * bytes_per_param

        flops = 2.0 * param_count * prompt_len
        operational_intensity = flops / max(weight_bytes, 1.0)

        memory_ceiling_flops = operational_intensity * self.bw_bytes_per_s
        attainable_flops = min(self.peak_fp16_flops, memory_ceiling_flops)

        bound = "Memory Bandwidth Bound" if memory_ceiling_flops < self.peak_fp16_flops else "Compute Bound"

        return OperationalPoint(
            phase="prefill",
            batch_size=1,
            precision=precision,
            flops=flops,
            bytes_transferred=weight_bytes,
            operational_intensity=operational_intensity,
            attainable_performance_tflops=attainable_flops / 1e12,
            execution_bound=bound,
        )
