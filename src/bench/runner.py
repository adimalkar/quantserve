"""Benchmark runner for measured serving and explicit simulation sweeps."""
import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import yaml

from src.bench.workload_generator import WorkloadGenerator
from src.bench.metrics import MetricsCollector, BenchmarkResult, RequestRecord
from src.bench.engine_interface import MockServingEngine, ServingEngine
from src.bench.vllm_engine import VLLMServingEngine
from src.analysis.cost_model import HARDWARE_PRESETS


async def run_single_benchmark(
    engine: ServingEngine,
    workload_gen: WorkloadGenerator,
    arrival_rate: float,
    duration_s: float,
    model_name: str,
    precision: str,
    ttft_slo_ms: float = 150.0,
    tpot_slo_ms: float = 35.0,
    hourly_rate_usd: float = 0.45,
    warmup_requests: int = 5,
) -> BenchmarkResult:
    """Runs a single load test point at a given arrival rate."""
    collector = MetricsCollector(ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
    for i in range(warmup_requests):
        spec = workload_gen.trace_sampler.sample_request(f"warmup_{i:04d}")
        await engine.process_request(spec, 0.0)

    schedule = workload_gen.generate_poisson_schedule(
        arrival_rate=arrival_rate,
        duration_s=duration_s,
        warmup_count=0,
    )
    if not schedule:
        raise ValueError("No measured requests were scheduled; increase duration or arrival rate")

    start_bench_time = time.perf_counter()
    active_tasks: list[asyncio.Task] = []
    request_specs = []

    async for req in workload_gen.stream_requests(schedule):
        # Fire request asynchronously without blocking the schedule stream
        task = asyncio.create_task(
            engine.process_request(req.spec, req.scheduled_timestamp_s)
        )
        active_tasks.append(task)
        request_specs.append(req.spec)

    # Await all inflight requests to complete
    completed_records = await asyncio.gather(*active_tasks, return_exceptions=True)
    # Include the idle tail of the configured arrival window. A low-rate
    # Poisson schedule often ends before duration_s; shortening the denominator
    # would overstate throughput and SLO goodput.
    end_bench_time = max(time.perf_counter(), start_bench_time + duration_s)

    for spec, rec in zip(request_specs, completed_records):
        if isinstance(rec, Exception):
            rec = RequestRecord(
                request_id=spec.request_id,
                prompt_len=spec.prompt_len,
                output_len=0,
                scheduled_time=spec.arrival_offset_s,
                dispatched_time=end_bench_time,
                prefill_done_time=end_bench_time,
                completed_time=end_bench_time,
                error=f"{type(rec).__name__}: {rec}",
            )
        collector.record_request(rec)

    return collector.compute_summary(
        model_name=model_name,
        precision=precision,
        concurrency_or_rate=arrival_rate,
        start_time=start_bench_time,
        end_time=end_bench_time,
        hourly_rate_usd=hourly_rate_usd,
    )


async def run_benchmark_sweep(
    config_path: str,
    model_key: str = "Qwen/Qwen2.5-0.5B-Instruct",
    precisions: List[str] = None,
    backend: str = "vllm",
    hardware_key: str = "ada_4050",
    output_dir: str = "outputs",
    server_url: str = "http://127.0.0.1:8000",
) -> List[Dict[str, Any]]:
    """Runs a measured vLLM sweep or an explicitly simulated sweep."""
    if backend not in {"vllm", "mock"}:
        raise ValueError("backend must be 'vllm' or 'mock'")
    if backend == "vllm" and precisions:
        raise ValueError("A vLLM server has one configured precision; run each server configuration separately")
    with open(config_path, "rb") as f:
        config_bytes = f.read()
    cfg = yaml.safe_load(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if not isinstance(cfg, dict):
        raise ValueError("Benchmark config must be a YAML mapping")

    workload_cfg = cfg.get("workload", {})
    trace_cfg = cfg.get("trace_distribution", {})
    slo_cfg = cfg.get("slo_targets", {})

    arrival_rates = workload_cfg.get("arrival_rates_per_second", [1.0, 2.0, 4.0, 8.0, 16.0])
    duration_s = float(workload_cfg.get("duration_seconds", 5))  # Fast run default
    warmup_count = int(workload_cfg.get("warmup_requests", 3))
    workload_seed = int(workload_cfg.get("seed", 42))
    if not arrival_rates or any(float(rate) <= 0 for rate in arrival_rates):
        raise ValueError("arrival_rates_per_second must contain positive rates")
    if duration_s <= 0 or warmup_count < 0:
        raise ValueError("duration_seconds must be positive and warmup_requests nonnegative")

    ttft_slo = float(slo_cfg.get("ttft_p99_ms", 150.0))
    tpot_slo = float(slo_cfg.get("tpot_p99_ms", 35.0))

    if hardware_key not in HARDWARE_PRESETS:
        raise ValueError(f"Unknown cost preset {hardware_key!r}; choose from {sorted(HARDWARE_PRESETS)}")
    hw_profile = HARDWARE_PRESETS[hardware_key]
    hw_dict = {
        "memory_bandwidth_gbs": hw_profile.memory_bandwidth_gbs,
        "fp16_tflops": hw_profile.fp16_tflops,
        "fp8_tflops": hw_profile.fp8_tflops,
        "int8_tops": hw_profile.int8_tops,
    }

    test_precisions = (
        precisions or ["fp16", "int4_awq", "fp8_e4m3", "int8_smoothquant"]
    ) if backend == "mock" else ["server_configured"]
    all_results: List[Dict[str, Any]] = []

    print(f"\n=======================================================")
    print(f"Starting QuantServe-Bench {backend.upper()} Sweep (cost preset: {hw_profile.name})")
    print(f"Model: {model_key} | Precisions: {test_precisions}")
    print(f"Arrival Rates: {arrival_rates} req/s")
    print(f"SLO Targets: P99 TTFT <= {ttft_slo}ms, P99 TPOT <= {tpot_slo}ms")
    print(f"=======================================================\n")

    for prec in test_precisions:
        print(f"\n--- Benchmarking Precision: {prec.upper()} ---")
        if backend == "mock":
            engine = MockServingEngine(
                model_name=model_key,
                precision=prec,
                hardware_profile=hw_dict,
            )
        else:
            engine = VLLMServingEngine(model_name=model_key, base_url=server_url)

        try:
            await engine.initialize()
            workload_gen = WorkloadGenerator(workload_cfg, trace_cfg, seed=workload_seed)

            for rate in arrival_rates:
                res = await run_single_benchmark(
                    engine=engine,
                    workload_gen=workload_gen,
                    arrival_rate=rate,
                    duration_s=duration_s,
                    model_name=model_key,
                    precision=prec,
                    ttft_slo_ms=ttft_slo,
                    tpot_slo_ms=tpot_slo,
                    hourly_rate_usd=hw_profile.hourly_rate_usd or 0.45,
                    warmup_requests=warmup_count,
                )
                res_dict = res.to_dict()
                res_dict.update({
                    "schema_version": 1,
                    "measurement_source": "measured_e2e" if backend == "vllm" else "synthetic_simulation",
                    "measurement_status": "complete" if res.failed_requests == 0 else "partial_failed_requests",
                    "backend": backend,
                    "server_version": engine.server_version if backend == "vllm" else None,
                    "workload_config_sha256": config_sha256,
                    "requested_window_s": duration_s,
                    "warmup_requests": warmup_count,
                    "workload_seed": workload_seed,
                    "workload_source": "synthetic_request_lengths",
                    "hardware_profile_source": "preset_cost_estimate_only",
                    "tpot_method": "mean_from_stream_chunks_and_server_token_count" if backend == "vllm" else "simulated_token_timestamps",
                    "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                    "server_url": server_url if backend == "vllm" else None,
                    "request_errors": [r.error for r in res.records if r.error],
                })
                all_results.append(res_dict)

                print(
                    f"Rate {rate:4.1f} req/s | "
                    f"Throughput: {res_dict['throughput_tokens_per_s']:7.1f} tok/s | "
                    f"TTFT P99: {res_dict['ttft_p99_ms']:6.1f}ms | "
                    f"TPOT P99: {res_dict['tpot_p99_ms']:5.1f}ms | "
                    f"SLO Goodput: {res_dict['slo_goodput_requests_per_s']:4.1f} req/s ({res_dict['slo_compliance_rate_pct']:4.1f}%) | "
                    f"Failures: {res.failed_requests}"
                )
        finally:
            if isinstance(engine, VLLMServingEngine):
                await engine.close()

    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_key)
    out_file = os.path.join(output_dir, f"sweep_{safe_model_name}_{hardware_key}_{backend}.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, allow_nan=False)

    print(f"\nSweep complete! Saved results to {out_file}")
    if any(result["measurement_status"] != "complete" for result in all_results):
        raise RuntimeError(f"Benchmark had failed requests; inspect partial results in {out_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="QuantServe-Bench Serving Load Sweeper")
    parser.add_argument("--config", type=str, default="configs/workload_poisson.yaml")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--precisions", nargs="+", default=None, help="Simulation only")
    parser.add_argument("--hardware", type=str, default="ada_4050")
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark_sweep(
            config_path=args.config,
            model_key=args.model,
            precisions=args.precisions,
            backend=args.backend,
            hardware_key=args.hardware,
            output_dir=args.output_dir,
            server_url=args.server_url,
        )
    )


if __name__ == "__main__":
    main()
