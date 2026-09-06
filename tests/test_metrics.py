"""Unit tests for metrics calculations and SLO Goodput evaluation."""
import pytest
from src.bench.metrics import RequestRecord, MetricsCollector


def test_request_record_latency_calculations():
    # Simulate a request:
    # Dispatched at t = 1.0
    # Prefill done at t = 1.05 (TTFT = 50ms)
    # Output tokens generated at t = [1.05, 1.07, 1.09, 1.11] (3 intervals of 20ms -> TPOT = 20ms)
    # Completed at t = 1.11 (E2E = 110ms)
    record = RequestRecord(
        request_id="req_001",
        prompt_len=128,
        output_len=4,
        scheduled_time=0.99,
        dispatched_time=1.0,
        prefill_done_time=1.05,
        token_timestamps=[1.05, 1.07, 1.09, 1.11],
        completed_time=1.11,
        error=None,
    )

    assert pytest.approx(record.ttft_ms, abs=0.1) == 50.0
    assert pytest.approx(record.tpot_ms, abs=0.1) == 20.0
    assert pytest.approx(record.e2e_latency_ms, abs=0.1) == 110.0

    # Check SLO evaluation
    assert record.meets_slo(ttft_max_ms=100.0, tpot_max_ms=25.0) is True
    assert record.meets_slo(ttft_max_ms=40.0, tpot_max_ms=25.0) is False  # TTFT violation
    assert record.meets_slo(ttft_max_ms=100.0, tpot_max_ms=15.0) is False  # TPOT violation


def test_metrics_collector_aggregation():
    collector = MetricsCollector(ttft_slo_ms=100.0, tpot_slo_ms=30.0)

    # Add 2 compliant requests and 1 non-compliant request
    r1 = RequestRecord(
        request_id="r1", prompt_len=100, output_len=10,
        scheduled_time=0.0, dispatched_time=0.0, prefill_done_time=0.05,
        token_timestamps=[0.05 + 0.02 * i for i in range(10)],
        completed_time=0.25,
    )
    r2 = RequestRecord(
        request_id="r2", prompt_len=100, output_len=10,
        scheduled_time=0.1, dispatched_time=0.1, prefill_done_time=0.16,
        token_timestamps=[0.16 + 0.02 * i for i in range(10)],
        completed_time=0.36,
    )
    # Violates TTFT (150ms > 100ms)
    r3 = RequestRecord(
        request_id="r3", prompt_len=100, output_len=10,
        scheduled_time=0.2, dispatched_time=0.2, prefill_done_time=0.35,
        token_timestamps=[0.35 + 0.02 * i for i in range(10)],
        completed_time=0.55,
    )

    collector.record_request(r1)
    collector.record_request(r2)
    collector.record_request(r3)

    summary = collector.compute_summary(
        model_name="llama3_1b",
        precision="fp16",
        concurrency_or_rate=10.0,
        start_time=0.0,
        end_time=1.0,  # 1 second duration
        hourly_rate_usd=0.36,
    )

    assert summary.total_requests == 3
    assert summary.completed_requests == 3
    assert summary.slo_goodput_requests_per_s == 2.0  # 2 requests met SLO in 1 second
    assert pytest.approx(summary.slo_compliance_rate_pct, abs=0.1) == 66.67
    assert summary.effective_cost_per_1m_valid_tokens_usd is not None
    # 2 compliant requests * 10 output tokens = 20 valid tokens / sec = 72,000 valid tokens/hour
    # Hourly rate = $0.36 -> Cost/1M = ($0.36 / 72,000) * 1,000,000 = $5.00 / 1M tokens
    assert pytest.approx(summary.effective_cost_per_1m_valid_tokens_usd, abs=0.05) == 5.00
