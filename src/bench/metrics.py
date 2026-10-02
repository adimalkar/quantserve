"""Serving metrics collector, latency quantiles, and SLO goodput arbiter."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class RequestRecord:
    request_id: str
    prompt_len: int
    output_len: int
    scheduled_time: float
    dispatched_time: float
    prefill_done_time: float
    token_timestamps: List[float] = field(default_factory=list)
    completed_time: float = 0.0
    error: Optional[str] = None
    tpot_override_ms: Optional[float] = None

    @property
    def ttft_ms(self) -> float:
        """Time to First Token in milliseconds (from dispatch to first token completion)."""
        if self.token_timestamps:
            return (self.token_timestamps[0] - self.dispatched_time) * 1000.0
        return (self.prefill_done_time - self.dispatched_time) * 1000.0

    @property
    def tpot_ms(self) -> float:
        """Time Per Output Token (mean inter-token latency during decode phase) in milliseconds."""
        if self.tpot_override_ms is not None:
            return self.tpot_override_ms
        if len(self.token_timestamps) > 1:
            intervals = np.diff(self.token_timestamps)
            return float(np.mean(intervals) * 1000.0)
        return 0.0

    @property
    def e2e_latency_ms(self) -> float:
        """Total end-to-end request duration in milliseconds."""
        return (self.completed_time - self.dispatched_time) * 1000.0

    def meets_slo(self, ttft_max_ms: float, tpot_max_ms: float) -> bool:
        """Returns True if this request met both TTFT and TPOT SLO thresholds without errors."""
        if self.error is not None:
            return False
        return (self.ttft_ms <= ttft_max_ms) and (self.tpot_ms <= tpot_max_ms)


@dataclass
class BenchmarkResult:
    model_name: str
    precision: str
    concurrency_or_rate: float
    total_requests: int
    completed_requests: int
    failed_requests: int
    duration_s: float
    total_prompt_tokens: int
    total_output_tokens: int
    throughput_tokens_per_s: float
    output_throughput_tokens_per_s: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p90_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    e2e_p50_ms: float
    e2e_p99_ms: float
    slo_ttft_threshold_ms: float
    slo_tpot_threshold_ms: float
    slo_goodput_requests_per_s: float
    slo_compliance_rate_pct: float
    effective_cost_per_1m_valid_tokens_usd: Optional[float] = None
    records: List[RequestRecord] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "precision": self.precision,
            "concurrency_or_rate": self.concurrency_or_rate,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "duration_s": round(self.duration_s, 2),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "throughput_tokens_per_s": round(self.throughput_tokens_per_s, 2),
            "output_throughput_tokens_per_s": round(self.output_throughput_tokens_per_s, 2),
            "ttft_p50_ms": round(self.ttft_p50_ms, 2),
            "ttft_p95_ms": round(self.ttft_p95_ms, 2),
            "ttft_p99_ms": round(self.ttft_p99_ms, 2),
            "tpot_p50_ms": round(self.tpot_p50_ms, 2),
            "tpot_p95_ms": round(self.tpot_p95_ms, 2),
            "tpot_p99_ms": round(self.tpot_p99_ms, 2),
            "slo_goodput_requests_per_s": round(self.slo_goodput_requests_per_s, 2),
            "slo_compliance_rate_pct": round(self.slo_compliance_rate_pct, 2),
            "effective_cost_per_1m_valid_tokens_usd": round(self.effective_cost_per_1m_valid_tokens_usd, 4)
            if self.effective_cost_per_1m_valid_tokens_usd is not None
            else None,
        }


class MetricsCollector:
    """Aggregates request execution records and computes latency distributions and SLO Goodput."""

    def __init__(self, ttft_slo_ms: float = 150.0, tpot_slo_ms: float = 35.0):
        self.ttft_slo_ms = ttft_slo_ms
        self.tpot_slo_ms = tpot_slo_ms
        self.records: List[RequestRecord] = []

    def record_request(self, record: RequestRecord) -> None:
        self.records.append(record)

    def compute_summary(
        self,
        model_name: str,
        precision: str,
        concurrency_or_rate: float,
        start_time: float,
        end_time: float,
        hourly_rate_usd: float = 0.0,
    ) -> BenchmarkResult:
        duration_s = max(end_time - start_time, 0.001)
        valid_records = [r for r in self.records if r.error is None]
        failed_count = len(self.records) - len(valid_records)

        if not valid_records:
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                concurrency_or_rate=concurrency_or_rate,
                total_requests=len(self.records),
                completed_requests=0,
                failed_requests=failed_count,
                duration_s=duration_s,
                total_prompt_tokens=0,
                total_output_tokens=0,
                throughput_tokens_per_s=0.0,
                output_throughput_tokens_per_s=0.0,
                ttft_p50_ms=0.0,
                ttft_p90_ms=0.0,
                ttft_p95_ms=0.0,
                ttft_p99_ms=0.0,
                tpot_p50_ms=0.0,
                tpot_p90_ms=0.0,
                tpot_p95_ms=0.0,
                tpot_p99_ms=0.0,
                e2e_p50_ms=0.0,
                e2e_p99_ms=0.0,
                slo_ttft_threshold_ms=self.ttft_slo_ms,
                slo_tpot_threshold_ms=self.tpot_slo_ms,
                slo_goodput_requests_per_s=0.0,
                slo_compliance_rate_pct=0.0,
                effective_cost_per_1m_valid_tokens_usd=None,
            )

        ttfts = [r.ttft_ms for r in valid_records]
        tpots = [r.tpot_ms for r in valid_records if r.output_len > 1]
        if not tpots:
            tpots = [0.0]
        e2es = [r.e2e_latency_ms for r in valid_records]

        total_prompt_tokens = sum(r.prompt_len for r in valid_records)
        total_output_tokens = sum(r.output_len for r in valid_records)
        total_tokens = total_prompt_tokens + total_output_tokens

        throughput_total = total_tokens / duration_s
        output_throughput = total_output_tokens / duration_s

        # Calculate SLO adherence
        slo_compliant_records = [
            r for r in valid_records if r.meets_slo(self.ttft_slo_ms, self.tpot_slo_ms)
        ]
        slo_compliant_count = len(slo_compliant_records)
        slo_goodput = slo_compliant_count / duration_s
        compliance_pct = (slo_compliant_count / len(self.records)) * 100.0

        # Cost model calculation:
        # $/1M tokens = GPU hourly rate / (valid_tokens_per_second * 3600) * 1,000,000
        cost_per_1m_tokens = None
        if hourly_rate_usd > 0.0:
            valid_output_tokens = sum(r.output_len for r in slo_compliant_records)
            valid_tokens_per_s = valid_output_tokens / duration_s
            if valid_tokens_per_s > 0:
                cost_per_1m_tokens = (hourly_rate_usd / (valid_tokens_per_s * 3600.0)) * 1_000_000.0

        return BenchmarkResult(
            model_name=model_name,
            precision=precision,
            concurrency_or_rate=concurrency_or_rate,
            total_requests=len(self.records),
            completed_requests=len(valid_records),
            failed_requests=failed_count,
            duration_s=duration_s,
            total_prompt_tokens=total_prompt_tokens,
            total_output_tokens=total_output_tokens,
            throughput_tokens_per_s=throughput_total,
            output_throughput_tokens_per_s=output_throughput,
            ttft_p50_ms=float(np.percentile(ttfts, 50)),
            ttft_p90_ms=float(np.percentile(ttfts, 90)),
            ttft_p95_ms=float(np.percentile(ttfts, 95)),
            ttft_p99_ms=float(np.percentile(ttfts, 99)),
            tpot_p50_ms=float(np.percentile(tpots, 50)),
            tpot_p90_ms=float(np.percentile(tpots, 90)),
            tpot_p95_ms=float(np.percentile(tpots, 95)),
            tpot_p99_ms=float(np.percentile(tpots, 99)),
            e2e_p50_ms=float(np.percentile(e2es, 50)),
            e2e_p99_ms=float(np.percentile(e2es, 99)),
            slo_ttft_threshold_ms=self.ttft_slo_ms,
            slo_tpot_threshold_ms=self.tpot_slo_ms,
            slo_goodput_requests_per_s=slo_goodput,
            slo_compliance_rate_pct=compliance_pct,
            effective_cost_per_1m_valid_tokens_usd=cost_per_1m_tokens,
            records=self.records,
        )
