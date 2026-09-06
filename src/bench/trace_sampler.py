"""Calibrated trace sampling for realistic prompt and generation length distributions."""
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class RequestSpec:
    request_id: str
    prompt_len: int
    output_len: int
    arrival_offset_s: float = 0.0


class TraceSampler:
    """Samples request prompt and decode token lengths matching real-world serving traces (e.g. ShareGPT / LMSYS)."""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def sample_prompt_length(self) -> int:
        p_cfg = self.config.get("prompt_length_lognormal", {})
        mean = p_cfg.get("mean", 5.4)
        sigma = p_cfg.get("sigma", 1.1)
        min_tok = p_cfg.get("min_tokens", 16)
        max_tok = p_cfg.get("max_tokens", 2048)

        val = int(np.exp(self.rng.normal(mean, sigma)))
        return max(min_tok, min(max_tok, val))

    def sample_output_length(self) -> int:
        d_cfg = self.config.get("decode_length_bimodal", {})
        m1 = d_cfg.get("mode1_tokens", 64)
        w1 = d_cfg.get("mode1_weight", 0.65)
        m2 = d_cfg.get("mode2_tokens", 384)
        noise_std = d_cfg.get("noise_std", 16)
        min_tok = d_cfg.get("min_tokens", 1)
        max_tok = d_cfg.get("max_tokens", 1024)

        if self.rng.random() < w1:
            val = int(self.rng.normal(m1, noise_std))
        else:
            val = int(self.rng.normal(m2, noise_std * 1.5))

        return max(min_tok, min(max_tok, val))

    def sample_request(self, request_id: str, arrival_offset_s: float = 0.0) -> RequestSpec:
        return RequestSpec(
            request_id=request_id,
            prompt_len=self.sample_prompt_length(),
            output_len=self.sample_output_length(),
            arrival_offset_s=arrival_offset_s,
        )

    def sample_batch(self, count: int) -> List[RequestSpec]:
        return [self.sample_request(f"req_{i:06d}") for i in range(count)]
