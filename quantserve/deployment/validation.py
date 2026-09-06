"""Validation utilities for deployment configurations against GPU hardware limits."""
from typing import Tuple
from quantserve.advisor.config_space import DeploymentConfig
from quantserve.advisor.hardware_probe import HardwareProfile


class DeploymentValidator:
    """Validates that a deployment configuration is structurally and physically valid for target hardware."""

    @staticmethod
    def validate(
        config: DeploymentConfig,
        hardware: HardwareProfile,
        param_count: float,
        context_len: int = 2048,
    ) -> Tuple[bool, str]:
        # 1. Precision checks
        if config.quantization not in hardware.supported_precisions:
            return False, f"Quantization '{config.quantization}' not supported on {hardware.name}"

        if config.kv_cache_precision == "fp8" and not hardware.supports_fp8:
            return False, f"FP8 KV-cache not supported on {hardware.arch} (requires Ada/Hopper+)"

        # 2. Memory limit check
        b_per_param = 0.5 if "int4" in config.quantization else (1.0 if "8" in config.quantization else 2.0)
        weight_gb = (param_count * b_per_param) / (1024.0 ** 3)

        kv_bytes_per_tok = 1.0 if config.kv_cache_precision in ("int8", "fp8") else 2.0
        kv_gb = (config.max_num_seqs * context_len * 32 * 1024 * (kv_bytes_per_tok / 2.0)) / (1024.0 ** 3)

        est_vram = weight_gb + kv_gb + 0.4
        max_vram = hardware.total_vram_gb * config.gpu_memory_utilization

        if est_vram > max_vram:
            return False, f"Estimated VRAM ({est_vram:.2f} GB) exceeds usable VRAM limit ({max_vram:.2f} GB)"

        return True, "Configuration is physically valid."
