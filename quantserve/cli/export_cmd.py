"""CLI command for exporting deployment artifacts (vLLM YAML, Docker Compose, Kubernetes)."""
import argparse
import os
import yaml
from quantserve.advisor.config_space import DeploymentConfig
from quantserve.deployment.vllm_generator import VLLMDeploymentGenerator
from quantserve.deployment.docker_generator import DockerDeploymentGenerator
from quantserve.deployment.kubernetes_generator import KubernetesDeploymentGenerator


def run_export_cmd(args):
    with open(args.recommendation, "r") as f:
        data = yaml.safe_load(f)

    model_name = data.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    cfg_raw = data.get("recommended_config", {})

    cfg = DeploymentConfig(
        quantization=cfg_raw.get("quantization", "int4_awq"),
        kv_cache_precision=cfg_raw.get("kv_cache_precision", "int8"),
        max_num_seqs=int(cfg_raw.get("max_num_seqs", 16)),
        max_num_batched_tokens=int(cfg_raw.get("max_num_batched_tokens", 2048)),
        chunked_prefill=bool(cfg_raw.get("chunked_prefill", True)),
        gpu_memory_utilization=float(cfg_raw.get("gpu_memory_utilization", 0.90)),
    )

    fmt = args.format.lower()
    if fmt == "vllm":
        content = VLLMDeploymentGenerator.generate_yaml_config(model_name, cfg)
        out_name = args.output or "vllm_deployment.yaml"
    elif fmt == "docker":
        content = DockerDeploymentGenerator.generate_docker_compose(model_name, cfg)
        out_name = args.output or "docker-compose.yml"
    elif fmt in ("k8s", "kubernetes"):
        content = KubernetesDeploymentGenerator.generate_manifest(model_name, cfg)
        out_name = args.output or "k8s_deployment.yaml"
    else:
        print(f"Unknown format: {args.format}. Choose vllm, docker, or k8s.")
        return

    with open(out_name, "w") as f:
        f.write(content)

    print(f"Successfully generated {fmt.upper()} deployment file: {out_name}\n")
