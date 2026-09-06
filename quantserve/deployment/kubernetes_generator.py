"""Kubernetes Deployment and Service manifest generator."""
import yaml
from typing import Dict, Any
from quantserve.advisor.config_space import DeploymentConfig


class KubernetesDeploymentGenerator:
    """Generates production Kubernetes manifests with GPU resource reservations."""

    @staticmethod
    def generate_manifest(model_name: str, config: DeploymentConfig, app_name: str = "llm-inference") -> str:
        safe_model_id = model_name.split("/")[-1].lower().replace(".", "-")

        cmd_args = [
            f"--model={model_name}",
            f"--kv-cache-dtype={config.kv_cache_precision}",
            f"--max-num-seqs={config.max_num_seqs}",
            f"--max-num-batched-tokens={config.max_num_batched_tokens}",
            f"--gpu-memory-utilization={config.gpu_memory_utilization}",
        ]
        if "awq" in config.quantization:
            cmd_args.append("--quantization=awq")
        elif "fp8" in config.quantization:
            cmd_args.append("--quantization=fp8")
        if config.chunked_prefill:
            cmd_args.append("--enable-chunked-prefill")

        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": f"{app_name}-{safe_model_id}", "labels": {"app": app_name}},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": app_name}},
                "template": {
                    "metadata": {"labels": {"app": app_name}},
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm",
                                "image": "vllm/vllm-openai:latest",
                                "args": cmd_args,
                                "ports": [{"containerPort": 8000, "name": "http"}],
                                "resources": {
                                    "limits": {"nvidia.com/gpu": "1", "memory": "16Gi", "cpu": "4"},
                                    "requests": {"nvidia.com/gpu": "1", "memory": "8Gi", "cpu": "2"},
                                },
                            }
                        ]
                    },
                },
            },
        }

        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{app_name}-svc"},
            "spec": {
                "selector": {"app": app_name},
                "ports": [{"port": 8000, "targetPort": 8000}],
                "type": "ClusterIP",
            },
        }

        return yaml.dump(deployment, sort_keys=False) + "---\n" + yaml.dump(service, sort_keys=False)
