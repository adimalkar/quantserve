"""Unit tests for trace sampler and workload generator."""
import pytest
import numpy as np
from scipy import stats

from src.bench.trace_sampler import TraceSampler
from src.bench.workload_generator import WorkloadGenerator


@pytest.fixture
def trace_config():
    return {
        "prompt_length_lognormal": {
            "mean": 5.4,
            "sigma": 1.1,
            "min_tokens": 16,
            "max_tokens": 2048,
        },
        "decode_length_bimodal": {
            "mode1_tokens": 64,
            "mode1_weight": 0.65,
            "mode2_tokens": 384,
            "mode2_weight": 0.35,
            "noise_std": 16,
            "min_tokens": 1,
            "max_tokens": 1024,
        },
    }


def test_trace_sampler_bounds(trace_config):
    sampler = TraceSampler(trace_config, seed=123)
    batch = sampler.sample_batch(100)

    assert len(batch) == 100
    for req in batch:
        assert 16 <= req.prompt_len <= 2048
        assert 1 <= req.output_len <= 1024


def test_poisson_arrival_distribution(trace_config):
    workload_config = {"type": "open_loop_poisson"}
    gen = WorkloadGenerator(workload_config, trace_config, seed=42)

    arrival_rate = 10.0  # 10 req/s -> mean inter-arrival = 0.1s
    duration_s = 50.0
    schedule = gen.generate_poisson_schedule(arrival_rate=arrival_rate, duration_s=duration_s, warmup_count=0)

    assert len(schedule) > 100

    # Extract inter-arrival intervals
    timestamps = [req.scheduled_timestamp_s for req in schedule]
    inter_arrivals = np.diff(timestamps)

    # In a Poisson process, inter-arrivals follow Exponential(scale = 1 / lambda)
    # Expected mean = 1 / 10.0 = 0.10s
    sample_mean = float(np.mean(inter_arrivals))
    assert pytest.approx(0.10, rel=0.15) == sample_mean

    # Kolmogorov-Smirnov test against exponential distribution
    ks_stat, p_value = stats.kstest(inter_arrivals, "expon", args=(0, 1.0 / arrival_rate))
    # KS null hypothesis: sample is drawn from the reference distribution (p > 0.01)
    assert p_value > 0.01
