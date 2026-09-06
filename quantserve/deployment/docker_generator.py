"""Docker and Docker Compose deployment generator."""
import yaml
from typing import Dict, Any
from quantserve.advisor.config_space import DeploymentConfig


class DockerDeploymentGenerator:
    """Generates Docker Compose definitions for running recommended LLM inference containers."""

    @staticmethod
    def generate_docker_compose(model_name: str, config: DeploymentConfig) -> str:
        cmd_args = [
            f"--model {model_name}",
            f"--kv-cache-dtype {config.kv_cache_precision}",
            f"--max-num-seqs {config.max_num_seqs}",
            f"--max-num-batched-tokens {config.max_num_batched_tokens}",
            f"--gpu-memory-utilization {config.gpu_memory_utilization}",
        ]
        if "awq" in config.quantization:
            cmd_args.append("--quantization awq")
        elif "fp8" in config.quantization:
            cmd_args.append("--quantization fp8")
        if config.chunked_prefill:
            cmd_args.append("--enable-chunked-prefill")

        compose_dict: Dict[str, Any] = {
            "version": "3.8",
            "services": {
                "llm-server": {
                    "image": "vllm/vllm-openai:latest",
                    "runtime": "nvidia",
                    "ports": ["8000:8000"],
                    "environment": [
                        "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}",
                        "NCCL_DEBUG=WARN",
                    ],
                    "volumes": [
                        "/mnt/1TB_Drive/Data/MyFiles/models/huggingface:/root/.cache/huggingface",
                    ],
                    "command": " ".join(cmd_args),
                    "deploy": {
                        "resources": {
                            "reservations": {
                                "devices": [
                                    {
                                        "driver": "nvidia",
                                        "count": 1,
                                        "capabilities": ["gpu"],
                                    }
                                ]
                            }
                        }
                    },
                }
            },
        }
        return yaml.dump(compose_dict, sort_keys=False)
