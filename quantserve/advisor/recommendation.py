"""Recommendation formatters and explanation builders for QuantServe."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from quantserve.advisor.config_space import DeploymentConfig
from quantserve.advisor.predictor import CalibratedPrediction
from quantserve.advisor.hardware_probe import HardwareProfile


@dataclass
class RecommendationResult:
    model_name: str
    hardware: HardwareProfile
    recommended_config: DeploymentConfig
    prediction: CalibratedPrediction
    quality_retention_pct: float
    total_candidates: int
    benchmarks_executed: int
    candidates_eliminated: int
    objective: str
    explanation: str
    pareto_alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hardware": self.hardware.to_dict(),
            "recommended_config": self.recommended_config.to_dict(),
            "predicted_metrics": {
                "ttft_p95_ms": self.prediction.ttft.point,
                "ttft_p95_ci_90": [self.prediction.ttft.lower_90, self.prediction.ttft.upper_90],
                "tpot_p95_ms": self.prediction.tpot.point,
                "tpot_p95_ci_90": [self.prediction.tpot.lower_90, self.prediction.tpot.upper_90],
                "throughput_tokens_s": self.prediction.throughput.point,
                "peak_vram_gb": self.prediction.peak_vram_gb,
                "slo_probability_pct": self.prediction.slo_probability_pct,
            },
            "quality_retention_pct": round(self.quality_retention_pct, 1),
            "search_statistics": {
                "total_candidates": self.total_candidates,
                "benchmarks_executed": self.benchmarks_executed,
                "candidates_eliminated": self.candidates_eliminated,
            },
            "objective": self.objective,
            "explanation": self.explanation,
            "vllm_command": self.generate_vllm_command(),
        }

    def generate_vllm_command(self) -> str:
        cfg = self.recommended_config
        quant_flag = ""
        if "awq" in cfg.quantization:
            quant_flag = " --quantization awq"
        elif "gptq" in cfg.quantization:
            quant_flag = " --quantization gptq"
        elif "fp8" in cfg.quantization:
            quant_flag = " --quantization fp8"

        cp_flag = " --enable-chunked-prefill" if cfg.chunked_prefill else ""

        return (
            f"vllm serve {self.model_name}{quant_flag}"
            f" --kv-cache-dtype {cfg.kv_cache_precision}"
            f" --max-num-seqs {cfg.max_num_seqs}"
            f" --max-num-batched-tokens {cfg.max_num_batched_tokens}"
            f" --gpu-memory-utilization {cfg.gpu_memory_utilization}"
            f"{cp_flag}"
        )

    def format_cli_summary(self) -> str:
        cfg = self.recommended_config
        pred = self.prediction
        lines = [
            "\n" + "=" * 62,
            "   QuantServe: Hardware-Aware Deployment Recommendation",
            "=" * 62,
            f"Model:                {self.model_name}",
            f"Target Hardware:      {self.hardware.name} ({self.hardware.total_vram_gb:.1f} GB VRAM)",
            f"Optimization Goal:    {self.objective}",
            "-" * 62,
            "RECOMMENDED DEPLOYMENT CONFIGURATION",
            f"  • Quantization:         {cfg.quantization.upper()}",
            f"  • KV-Cache Precision:   {cfg.kv_cache_precision.upper()}",
            f"  • Max Concurrent Seqs:  {cfg.max_num_seqs}",
            f"  • Max Batched Tokens:   {cfg.max_num_batched_tokens}",
            f"  • Chunked Prefill:      {cfg.chunked_prefill}",
            f"  • GPU Memory Util:      {cfg.gpu_memory_utilization}",
            "-" * 62,
            "PERFORMANCE FORECAST (WITH 90% UNCERTAINTY INTERVALS)",
            f"  • Predicted P95 TTFT:   {pred.ttft.point:.1f} ms [{pred.ttft.lower_90:.1f}–{pred.ttft.upper_90:.1f} ms]",
            f"  • Predicted P95 TPOT:   {pred.tpot.point:.1f} ms [{pred.tpot.lower_90:.1f}–{pred.tpot.upper_90:.1f} ms]",
            f"  • Safe Throughput:      {pred.throughput.point:.1f} tokens/s",
            f"  • Peak VRAM Usage:      {pred.peak_vram_gb:.2f} GB / {self.hardware.total_vram_gb:.1f} GB",
            f"  • Quality Retention:    {self.quality_retention_pct:.1f}%",
            f"  • SLO Success Prob:     {pred.slo_probability_pct:.1f}%",
            "-" * 62,
            "SEARCH EFFICIENCY & PRUNING",
            f"  • Total Candidates:     {self.total_candidates}",
            f"  • Benchmarks Executed:  {self.benchmarks_executed} ({self.benchmarks_executed / max(self.total_candidates, 1) * 100:.1f}% of space)",
            f"  • Invalid Pruned:       {self.candidates_eliminated}",
            "-" * 62,
            "WHY THIS CONFIGURATION WON (ROOFLINE EXPLANATION)",
            f"  {self.explanation}",
            "-" * 62,
            "READY-TO-RUN DEPLOYMENT COMMAND",
            f"  $ {self.generate_vllm_command()}",
            "=" * 62 + "\n",
        ]
        return "\n".join(lines)
