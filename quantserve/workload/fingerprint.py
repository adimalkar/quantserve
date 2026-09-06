"""Workload fingerprinting, pattern classification, and bottleneck analysis."""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from quantserve.workload.trace_parser import TraceRecord


@dataclass
class WorkloadFingerprint:
    total_requests: int
    duration_seconds: float
    request_rate: float
    avg_prompt_tokens: float
    p95_prompt_tokens: float
    avg_output_tokens: float
    p95_output_tokens: float
    burst_coefficient: float
    peak_concurrency: int
    workload_class: str
    likely_bottleneck: str
    recommended_search_focus: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "duration_seconds": round(self.duration_seconds, 1),
            "request_rate": round(self.request_rate, 2),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 1),
            "p95_prompt_tokens": round(self.p95_prompt_tokens, 1),
            "avg_output_tokens": round(self.avg_output_tokens, 1),
            "p95_output_tokens": round(self.p95_output_tokens, 1),
            "burst_coefficient": round(self.burst_coefficient, 2),
            "peak_concurrency": self.peak_concurrency,
            "workload_class": self.workload_class,
            "likely_bottleneck": self.likely_bottleneck,
            "recommended_search_focus": self.recommended_search_focus,
        }

    def format_cli_summary(self) -> str:
        return (
            f"\nWorkload Fingerprint Summary:\n"
            f"  • Total Requests:          {self.total_requests}\n"
            f"  • Request Rate:            {self.request_rate:.2f} req/s\n"
            f"  • Average Prompt:          {self.avg_prompt_tokens:.0f} tokens (P95: {self.p95_prompt_tokens:.0f})\n"
            f"  • Average Output:          {self.avg_output_tokens:.0f} tokens (P95: {self.p95_output_tokens:.0f})\n"
            f"  • Burst Coefficient:       {self.burst_coefficient:.2f}\n"
            f"  • Peak Concurrency:        {self.peak_concurrency}\n"
            f"  • Dominant Workload:       {self.workload_class}\n"
            f"  • Likely Bottleneck:       {self.likely_bottleneck}\n"
            f"  • Recommended Strategy:    {self.recommended_search_focus}\n"
        )


class WorkloadFingerprinter:
    """Analyzes request traces to characterize workload distributions and identify deployment bottlenecks."""

    @staticmethod
    def analyze(records: List[TraceRecord]) -> WorkloadFingerprint:
        if not records:
            return WorkloadFingerprint(
                total_requests=0,
                duration_seconds=0.0,
                request_rate=0.0,
                avg_prompt_tokens=256.0,
                p95_prompt_tokens=512.0,
                avg_output_tokens=64.0,
                p95_output_tokens=128.0,
                burst_coefficient=1.0,
                peak_concurrency=1,
                workload_class="Default Interactive",
                likely_bottleneck="Decode Memory Bandwidth",
                recommended_search_focus="AWQ W4A16 + Standard Batching",
            )

        prompts = np.array([r.input_tokens for r in records], dtype=np.float64)
        outputs = np.array([r.output_tokens for r in records], dtype=np.float64)
        timestamps = np.array([r.timestamp_s for r in records], dtype=np.float64)

        duration = float(np.max(timestamps) - np.min(timestamps))
        duration = max(duration, 1.0)
        rate = len(records) / duration

        # Inter-arrival intervals & burstiness (coefficient of variation: std / mean)
        if len(timestamps) > 1:
            inter_arrivals = np.diff(np.sort(timestamps))
            mean_ia = float(np.mean(inter_arrivals))
            std_ia = float(np.std(inter_arrivals))
            burst_coeff = (std_ia / mean_ia) if mean_ia > 0 else 1.0
        else:
            burst_coeff = 1.0

        avg_prompt = float(np.mean(prompts))
        p95_prompt = float(np.percentile(prompts, 95))
        avg_output = float(np.mean(outputs))
        p95_output = float(np.percentile(outputs, 95))

        # Peak concurrency estimate
        concurrencies = [r.concurrency for r in records if r.concurrency is not None]
        peak_conc = int(np.max(concurrencies)) if concurrencies else max(1, int(rate * 2.0))

        # Classify workload
        ratio = avg_prompt / max(avg_output, 1.0)
        if avg_prompt >= 1500 or ratio >= 8.0:
            workload_class = "Long-Prefill Interactive RAG"
            bottleneck = "Prefill Compute & KV-Cache Footprint"
            focus = "FP8 / AWQ + Chunked Prefill + INT8 KV Cache"
        elif rate >= 20.0 or avg_output >= 256:
            workload_class = "Batch Document Processing"
            bottleneck = "Compute Throughput & Batch Saturation"
            focus = "FP8 (Ada Native) + High Concurrency (C>=32)"
        else:
            workload_class = "Interactive Coding / Chat Assistant"
            bottleneck = "Decode Memory Bandwidth (Low Concurrency)"
            focus = "AWQ W4A16 + Moderate Concurrency (C=8-16)"

        return WorkloadFingerprint(
            total_requests=len(records),
            duration_seconds=duration,
            request_rate=rate,
            avg_prompt_tokens=avg_prompt,
            p95_prompt_tokens=p95_prompt,
            avg_output_tokens=avg_output,
            p95_output_tokens=p95_output,
            burst_coefficient=burst_coeff,
            peak_concurrency=peak_conc,
            workload_class=workload_class,
            likely_bottleneck=bottleneck,
            recommended_search_focus=focus,
        )
