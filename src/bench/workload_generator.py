"""Open-loop workload generator supporting Poisson and Gamma burst arrival processes."""
import asyncio
import time
from dataclasses import dataclass
from typing import List, AsyncGenerator, Dict, Any
import numpy as np

from src.bench.trace_sampler import TraceSampler, RequestSpec


@dataclass
class ScheduledRequest:
    spec: RequestSpec
    scheduled_timestamp_s: float  # Planned relative arrival offset in seconds


class WorkloadGenerator:
    """Generates open-loop request streams where arrival timing is strictly decoupling from serving speed."""

    def __init__(self, workload_config: Dict[str, Any], trace_config: Dict[str, Any], seed: int = 42):
        self.workload_config = workload_config
        self.trace_sampler = TraceSampler(trace_config, seed=seed)
        self.rng = np.random.default_rng(seed)

    def generate_poisson_schedule(
        self, arrival_rate: float, duration_s: float, warmup_count: int = 10
    ) -> List[ScheduledRequest]:
        """Generates arrival timestamps according to a Poisson process with parameter lambda (arrival_rate)."""
        schedule: List[ScheduledRequest] = []
        current_time = 0.0

        # Warmup requests: spaced evenly
        warmup_interval = 1.0 / max(arrival_rate, 0.1)
        for i in range(warmup_count):
            spec = self.trace_sampler.sample_request(f"warmup_{i:04d}", arrival_offset_s=current_time)
            schedule.append(ScheduledRequest(spec=spec, scheduled_timestamp_s=current_time))
            current_time += warmup_interval

        # Measurement window: exponential inter-arrival times (Poisson process)
        benchmark_start_offset = current_time
        while (current_time - benchmark_start_offset) < duration_s:
            # delta_t ~ Exponential(lambda) -> -ln(U) / lambda
            delta_t = self.rng.exponential(1.0 / arrival_rate)
            current_time += delta_t
            if (current_time - benchmark_start_offset) >= duration_s:
                break
            req_idx = len(schedule) - warmup_count
            spec = self.trace_sampler.sample_request(f"req_{req_idx:06d}", arrival_offset_s=current_time)
            schedule.append(ScheduledRequest(spec=spec, scheduled_timestamp_s=current_time))

        return schedule

    def generate_gamma_burst_schedule(
        self, shape_k: float, scale_theta: float, duration_s: float, warmup_count: int = 10
    ) -> List[ScheduledRequest]:
        """Generates arrival timestamps with a Gamma distribution for bursty, heavy-tailed traffic."""
        schedule: List[ScheduledRequest] = []
        current_time = 0.0

        for i in range(warmup_count):
            spec = self.trace_sampler.sample_request(f"warmup_{i:04d}", arrival_offset_s=current_time)
            schedule.append(ScheduledRequest(spec=spec, scheduled_timestamp_s=current_time))
            current_time += 0.5

        benchmark_start_offset = current_time
        while (current_time - benchmark_start_offset) < duration_s:
            delta_t = self.rng.gamma(shape=shape_k, scale=scale_theta)
            current_time += delta_t
            if (current_time - benchmark_start_offset) >= duration_s:
                break
            req_idx = len(schedule) - warmup_count
            spec = self.trace_sampler.sample_request(f"req_{req_idx:06d}", arrival_offset_s=current_time)
            schedule.append(ScheduledRequest(spec=spec, scheduled_timestamp_s=current_time))

        return schedule

    async def stream_requests(
        self, schedule: List[ScheduledRequest]
    ) -> AsyncGenerator[ScheduledRequest, None]:
        """Asynchronously dispatches requests at their scheduled wall-clock timestamps."""
        if not schedule:
            return

        start_wall_time = time.perf_counter()
        for item in schedule:
            target_time = start_wall_time + item.scheduled_timestamp_s
            now = time.perf_counter()
            sleep_duration = target_time - now
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

            yield item
