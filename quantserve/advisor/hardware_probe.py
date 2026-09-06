"""Automated GPU hardware probing and specification detection."""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import torch


@dataclass
class HardwareProfile:
    name: str
    arch: str
    compute_capability: str
    total_vram_gb: float
    memory_bandwidth_gbs: float
    fp16_tflops: float
    fp8_tflops: float
    int8_tops: float
    sparse_fp16_tflops: float
    hourly_rate_usd: float
    supports_fp8: bool
    supports_sparse_tensor_cores: bool
    supported_precisions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arch": self.arch,
            "compute_capability": self.compute_capability,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "memory_bandwidth_gbs": round(self.memory_bandwidth_gbs, 1),
            "fp16_tflops": round(self.fp16_tflops, 1),
            "fp8_tflops": round(self.fp8_tflops, 1),
            "int8_tops": round(self.int8_tops, 1),
            "sparse_fp16_tflops": round(self.sparse_fp16_tflops, 1),
            "hourly_rate_usd": self.hourly_rate_usd,
            "supports_fp8": self.supports_fp8,
            "supports_sparse_tensor_cores": self.supports_sparse_tensor_cores,
            "supported_precisions": self.supported_precisions,
        }


HARDWARE_PRESETS: Dict[str, HardwareProfile] = {
    "ada_4050": HardwareProfile(
        name="NVIDIA GeForce RTX 4050 Laptop GPU",
        arch="Ada Lovelace (sm_89)",
        compute_capability="8.9",
        total_vram_gb=6.0,
        memory_bandwidth_gbs=192.0,
        fp16_tflops=36.0,
        fp8_tflops=72.0,
        int8_tops=72.0,
        sparse_fp16_tflops=72.0,
        hourly_rate_usd=0.0,
        supports_fp8=True,
        supports_sparse_tensor_cores=True,
        supported_precisions=["fp16", "int4_awq", "int4_gptq", "fp8_e4m3", "int8_smoothquant", "sparse_2_4"],
    ),
    "a10g": HardwareProfile(
        name="NVIDIA A10G",
        arch="Ampere (sm_86)",
        compute_capability="8.6",
        total_vram_gb=24.0,
        memory_bandwidth_gbs=600.0,
        fp16_tflops=70.0,
        fp8_tflops=0.0,
        int8_tops=140.0,
        sparse_fp16_tflops=140.0,
        hourly_rate_usd=0.45,
        supports_fp8=False,
        supports_sparse_tensor_cores=True,
        supported_precisions=["fp16", "int4_awq", "int4_gptq", "int8_smoothquant", "sparse_2_4"],
    ),
    "l4": HardwareProfile(
        name="NVIDIA L4",
        arch="Ada Lovelace (sm_89)",
        compute_capability="8.9",
        total_vram_gb=24.0,
        memory_bandwidth_gbs=300.0,
        fp16_tflops=60.0,
        fp8_tflops=120.0,
        int8_tops=120.0,
        sparse_fp16_tflops=120.0,
        hourly_rate_usd=0.40,
        supports_fp8=True,
        supports_sparse_tensor_cores=True,
        supported_precisions=["fp16", "int4_awq", "int4_gptq", "fp8_e4m3", "int8_smoothquant", "sparse_2_4"],
    ),
    "rtx_4090": HardwareProfile(
        name="NVIDIA GeForce RTX 4090",
        arch="Ada Lovelace (sm_89)",
        compute_capability="8.9",
        total_vram_gb=24.0,
        memory_bandwidth_gbs=1008.0,
        fp16_tflops=165.0,
        fp8_tflops=330.0,
        int8_tops=330.0,
        sparse_fp16_tflops=330.0,
        hourly_rate_usd=0.55,
        supports_fp8=True,
        supports_sparse_tensor_cores=True,
        supported_precisions=["fp16", "int4_awq", "int4_gptq", "fp8_e4m3", "int8_smoothquant", "sparse_2_4"],
    ),
    "h100": HardwareProfile(
        name="NVIDIA H100 SXM5",
        arch="Hopper (sm_90)",
        compute_capability="9.0",
        total_vram_gb=80.0,
        memory_bandwidth_gbs=3350.0,
        fp16_tflops=989.0,
        fp8_tflops=1978.0,
        int8_tops=1978.0,
        sparse_fp16_tflops=1978.0,
        hourly_rate_usd=2.50,
        supports_fp8=True,
        supports_sparse_tensor_cores=True,
        supported_precisions=["fp16", "int4_awq", "fp8_e4m3", "int8_smoothquant", "sparse_2_4"],
    ),
}


class HardwareProber:
    """Probes local GPU hardware capabilities or retrieves preset profiles."""

    @staticmethod
    def probe(override_preset: Optional[str] = None) -> HardwareProfile:
        if override_preset and override_preset in HARDWARE_PRESETS:
            return HARDWARE_PRESETS[override_preset]

        if not torch.cuda.is_available():
            # CPU Fallback profile
            return HardwareProfile(
                name="Generic CPU",
                arch="x86_64",
                compute_capability="0.0",
                total_vram_gb=16.0,
                memory_bandwidth_gbs=50.0,
                fp16_tflops=2.0,
                fp8_tflops=0.0,
                int8_tops=4.0,
                sparse_fp16_tflops=0.0,
                hourly_rate_usd=0.0,
                supports_fp8=False,
                supports_sparse_tensor_cores=False,
                supported_precisions=["fp16", "int8_smoothquant"],
            )

        device_idx = 0
        device_name = torch.cuda.get_device_name(device_idx)
        cap_major, cap_minor = torch.cuda.get_device_capability(device_idx)
        compute_cap = f"{cap_major}.{cap_minor}"
        total_vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024.0 ** 3)

        # Match known presets by name keywords
        name_lower = device_name.lower()
        if "4050" in name_lower:
            matched = HARDWARE_PRESETS["ada_4050"]
            matched.name = device_name
            matched.total_vram_gb = total_vram_gb
            return matched
        elif "4090" in name_lower:
            matched = HARDWARE_PRESETS["rtx_4090"]
            matched.name = device_name
            matched.total_vram_gb = total_vram_gb
            return matched
        elif "a10" in name_lower:
            matched = HARDWARE_PRESETS["a10g"]
            matched.name = device_name
            matched.total_vram_gb = total_vram_gb
            return matched
        elif "l4" in name_lower:
            matched = HARDWARE_PRESETS["l4"]
            matched.name = device_name
            matched.total_vram_gb = total_vram_gb
            return matched
        elif "h100" in name_lower:
            matched = HARDWARE_PRESETS["h100"]
            matched.name = device_name
            matched.total_vram_gb = total_vram_gb
            return matched

        # Dynamic heuristic for other GPUs
        supports_fp8 = (cap_major >= 9) or (cap_major == 8 and cap_minor >= 9)
        supports_sparse = cap_major >= 8

        precisions = ["fp16", "int4_awq", "int8_smoothquant"]
        if supports_fp8:
            precisions.append("fp8_e4m3")
        if supports_sparse:
            precisions.append("sparse_2_4")

        # Rough bandwidth estimation by arch
        est_bw = 192.0 if cap_major == 8 else 300.0

        return HardwareProfile(
            name=device_name,
            arch=f"CUDA Compute Capability {compute_cap}",
            compute_capability=compute_cap,
            total_vram_gb=total_vram_gb,
            memory_bandwidth_gbs=est_bw,
            fp16_tflops=40.0,
            fp8_tflops=80.0 if supports_fp8 else 0.0,
            int8_tops=80.0,
            sparse_fp16_tflops=80.0 if supports_sparse else 0.0,
            hourly_rate_usd=0.0,
            supports_fp8=supports_fp8,
            supports_sparse_tensor_cores=supports_sparse,
            supported_precisions=precisions,
        )
