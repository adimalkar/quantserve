"""Hardware-aware economic cost models for LLM serving."""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class HardwareProfile:
    name: str
    arch: str
    memory_gb: float
    memory_bandwidth_gbs: float
    fp16_tflops: float
    fp8_tflops: float
    int8_tops: float
    sparse_fp16_tflops: float
    hourly_rate_usd: float


HARDWARE_PRESETS: Dict[str, HardwareProfile] = {
    "ada_4050": HardwareProfile(
        name="NVIDIA GeForce RTX 4050 Laptop GPU",
        arch="Ada Lovelace (sm_89)",
        memory_gb=6.0,
        memory_bandwidth_gbs=192.0,
        fp16_tflops=36.0,
        fp8_tflops=72.0,
        int8_tops=72.0,
        sparse_fp16_tflops=72.0,
        hourly_rate_usd=0.0,  # Local workstation
    ),
    "a10g": HardwareProfile(
        name="NVIDIA A10G",
        arch="Ampere (sm_86)",
        memory_gb=24.0,
        memory_bandwidth_gbs=600.0,
        fp16_tflops=70.0,
        fp8_tflops=0.0,
        int8_tops=140.0,
        sparse_fp16_tflops=140.0,
        hourly_rate_usd=0.45,
    ),
    "l4": HardwareProfile(
        name="NVIDIA L4",
        arch="Ada Lovelace (sm_89)",
        memory_gb=24.0,
        memory_bandwidth_gbs=300.0,
        fp16_tflops=60.0,
        fp8_tflops=120.0,
        int8_tops=120.0,
        sparse_fp16_tflops=120.0,
        hourly_rate_usd=0.40,
    ),
    "rtx_4090": HardwareProfile(
        name="NVIDIA GeForce RTX 4090",
        arch="Ada Lovelace (sm_89)",
        memory_gb=24.0,
        memory_bandwidth_gbs=1008.0,
        fp16_tflops=165.0,
        fp8_tflops=330.0,
        int8_tops=330.0,
        sparse_fp16_tflops=330.0,
        hourly_rate_usd=0.55,
    ),
}


class ServingCostModel:
    """Calculates operational $/1M token economics under service-level objectives."""

    def __init__(self, hardware_profile: HardwareProfile):
        self.hw = hardware_profile

    def calculate_cost_per_million_tokens(
        self,
        valid_tokens_per_second: float,
        override_hourly_rate: Optional[float] = None,
    ) -> float:
        """Calculates cost per 1,000,000 valid tokens meeting SLO:
        $/1M = (Hourly Rate / (Tokens/sec * 3600)) * 1,000,000
        """
        hourly_rate = override_hourly_rate if override_hourly_rate is not None else self.hw.hourly_rate_usd
        if hourly_rate <= 0.0:
            return 0.0
        if valid_tokens_per_second <= 0.0:
            return float("inf")

        tokens_per_hour = valid_tokens_per_second * 3600.0
        cost_per_token = hourly_rate / tokens_per_hour
        return cost_per_token * 1_000_000.0
