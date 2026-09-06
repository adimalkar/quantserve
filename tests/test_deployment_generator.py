"""Unit tests for deployment artifact generators and validation."""
import pytest
from quantserve.advisor.config_space import DeploymentConfig
from quantserve.advisor.hardware_probe import HARDWARE_PRESETS
from quantserve.deployment.vllm_generator import VLLMDeploymentGenerator
from quantserve.deployment.docker_generator import DockerDeploymentGenerator
from quantserve.deployment.kubernetes_generator import KubernetesDeploymentGenerator
from quantserve.deployment.validation import DeploymentValidator


@pytest.fixture
def sample_config():
    return DeploymentConfig(
        quantization="int4_awq",
        kv_cache_precision="int8",
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        chunked_prefill=True,
    )


def test_vllm_command_and_yaml_generator(sample_config):
    cmd = VLLMDeploymentGenerator.generate_cli_command("Qwen/Qwen2.5-0.5B-Instruct", sample_config)
    assert "vllm serve Qwen/Qwen2.5-0.5B-Instruct" in cmd
    assert "--quantization awq" in cmd
    assert "--kv-cache-dtype int8" in cmd
    assert "--enable-chunked-prefill" in cmd

    yaml_str = VLLMDeploymentGenerator.generate_yaml_config("Qwen/Qwen2.5-0.5B-Instruct", sample_config)
    assert "quantization: int4_awq" in yaml_str
    assert "enable_chunked_prefill: true" in yaml_str


def test_docker_and_kubernetes_generator(sample_config):
    compose = DockerDeploymentGenerator.generate_docker_compose("Qwen/Qwen2.5-0.5B-Instruct", sample_config)
    assert "vllm/vllm-openai" in compose
    assert "runtime: nvidia" in compose

    k8s = KubernetesDeploymentGenerator.generate_manifest("Qwen/Qwen2.5-0.5B-Instruct", sample_config)
    assert "kind: Deployment" in k8s
    assert "nvidia.com/gpu: '1'" in k8s or "nvidia.com/gpu: 1" in k8s
    assert "kind: Service" in k8s


def test_deployment_validator(sample_config):
    ada = HARDWARE_PRESETS["ada_4050"]
    valid, msg = DeploymentValidator.validate(sample_config, ada, param_count=1.23e9)
    assert valid is True

    # Check precision not supported
    invalid_cfg = DeploymentConfig(
        quantization="fp8_e4m3",
        kv_cache_precision="fp8",
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        chunked_prefill=True,
    )
    a10 = HARDWARE_PRESETS["a10g"]  # No native FP8
    valid_a10, msg_a10 = DeploymentValidator.validate(invalid_cfg, a10, param_count=1.23e9)
    assert valid_a10 is False
