"""Search space definition and constraint-based pruning for deployment configurations."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from quantserve.advisor.hardware_probe import HardwareProfile


@dataclass(frozen=True)
class DeploymentConfig:
    quantization: str
    kv_cache_precision: str
    max_num_seqs: int
    max_num_batched_tokens: int
    chunked_prefill: bool
    gpu_memory_utilization: float = 0.90

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantization": self.quantization,
            "kv_cache_precision": self.kv_cache_precision,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "chunked_prefill": self.chunked_prefill,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }

    @property
    def id_string(self) -> str:
        return f"{self.quantization}-kv_{self.kv_cache_precision}-seq_{self.max_num_seqs}-batch_{self.max_num_batched_tokens}-cp_{int(self.chunked_prefill)}"


class ConfigurationSpace:
    """Generates the multi-dimensional candidate deployment space and prunes physical impossibilities."""

    def __init__(
        self,
        quantizations: Optional[List[str]] = None,
        kv_cache_precisions: Optional[List[str]] = None,
        max_num_seqs_list: Optional[List[int]] = None,
        max_batched_tokens_list: Optional[List[int]] = None,
        chunked_prefill_options: Optional[List[bool]] = None,
    ):
        self.quantizations = quantizations or ["fp16", "int4_awq", "fp8_e4m3", "int8_smoothquant"]
        self.kv_cache_precisions = kv_cache_precisions or ["fp16", "int8", "fp8"]
        self.max_num_seqs_list = max_num_seqs_list or [4, 8, 16, 32, 64]
        self.max_batched_tokens_list = max_batched_tokens_list or [512, 1024, 2048, 4096]
        self.chunked_prefill_options = chunked_prefill_options or [True, False]

    def generate_all_candidates(self) -> List[DeploymentConfig]:
        candidates: List[DeploymentConfig] = []
        for q in self.quantizations:
            for kv in self.kv_cache_precisions:
                for seq in self.max_num_seqs_list:
                    for bt in self.max_batched_tokens_list:
                        for cp in self.chunked_prefill_options:
                            candidates.append(
                                DeploymentConfig(
                                    quantization=q,
                                    kv_cache_precision=kv,
                                    max_num_seqs=seq,
                                    max_num_batched_tokens=bt,
                                    chunked_prefill=cp,
                                )
                            )
        return candidates

    def prune_for_hardware(
        self,
        candidates: List[DeploymentConfig],
        hardware: HardwareProfile,
        param_count: float,
        context_len: int = 2048,
    ) -> List[DeploymentConfig]:
        """Prunes configurations that violate hardware capability or VRAM boundaries."""
        valid: List[DeploymentConfig] = []

        for c in candidates:
            # 1. Hardware precision support check
            if c.quantization not in hardware.supported_precisions:
                continue
            if c.kv_cache_precision == "fp8" and not hardware.supports_fp8:
                continue

            # 2. VRAM footprint estimate
            b_per_param = 0.5 if "int4" in c.quantization else (1.0 if "8" in c.quantization else 2.0)
            weight_mem_gb = (param_count * b_per_param) / (1024.0 ** 3)

            kv_bytes_per_tok = 1.0 if c.kv_cache_precision in ("int8", "fp8") else 2.0
            # Approx GQA 16 layers, 8 kv_heads, 64 dim: ~32KB per tok (fp16) or 16KB (int8)
            kv_mem_gb = (c.max_num_seqs * context_len * 32 * 1024 * (kv_bytes_per_tok / 2.0)) / (1024.0 ** 3)

            est_peak_vram_gb = weight_mem_gb + kv_mem_gb + 0.5  # 500MB runtime buffer

            if est_peak_vram_gb > (hardware.total_vram_gb * c.gpu_memory_utilization):
                continue

            valid.append(c)

        return valid
