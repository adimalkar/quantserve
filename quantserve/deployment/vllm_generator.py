"""Deployment generator for vLLM commands and server YAML configurations."""
import yaml
from typing import Dict, Any
from quantserve.advisor.config_space import DeploymentConfig


class VLLMDeploymentGenerator:
    """Generates production vLLM CLI commands and YAML configuration files."""

    @staticmethod
    def generate_cli_command(model_name: str, config: DeploymentConfig) -> str:
        quant_flag = ""
        if "awq" in config.quantization:
            quant_flag = " --quantization awq"
        elif "gptq" in config.quantization:
            quant_flag = " --quantization gptq"
        elif "fp8" in config.quantization:
            quant_flag = " --quantization fp8"

        cp_flag = " --enable-chunked-prefill" if config.chunked_prefill else ""

        return (
            f"vllm serve {model_name}{quant_flag}"
            f" --kv-cache-dtype {config.kv_cache_precision}"
            f" --max-num-seqs {config.max_num_seqs}"
            f" --max-num-batched-tokens {config.max_num_batched_tokens}"
            f" --gpu-memory-utilization {config.gpu_memory_utilization}"
            f"{cp_flag}"
        )

    @staticmethod
    def generate_yaml_config(model_name: str, config: DeploymentConfig) -> str:
        cfg_dict: Dict[str, Any] = {
            "model": model_name,
            "serving": {
                "engine": "vllm",
                "quantization": config.quantization,
                "kv_cache_dtype": config.kv_cache_precision,
                "max_num_seqs": config.max_num_seqs,
                "max_num_batched_tokens": config.max_num_batched_tokens,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "enable_chunked_prefill": config.chunked_prefill,
                "port": 8000,
                "host": "0.0.0.0",
            },
        }
        return yaml.dump(cfg_dict, sort_keys=False)
