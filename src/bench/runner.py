"""Benchmark runner executing load sweeps across model compression precisions."""
import argparse
import asyncio
import json
import os
import time
from typing import List, Dict, Any
import yaml

from src.bench.workload_generator import WorkloadGenerator
from src.bench.metrics import MetricsCollector, BenchmarkResult
from src.bench.engine_interface import MockServingEngine, HuggingFaceServingEngine, ServingEngine
from src.analysis.cost_model import HARDWARE_PRESETS, ServingCostModel


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
    schedule = workload_gen.generate_poisson_schedule(
        arrival_rate=arrival_rate,
        duration_s=duration_s,
        warmup_count=warmup_requests,
    )

    start_bench_time = time.perf_counter()
    active_tasks: list[asyncio.Task] = []

    async for req in workload_gen.stream_requests(schedule):
        # Fire request asynchronously without blocking the schedule stream
        task = asyncio.create_task(
            engine.process_request(req.spec, req.scheduled_timestamp_s)
        )
        active_tasks.append(task)

    # Await all inflight requests to complete
    completed_records = await asyncio.gather(*active_tasks, return_exceptions=True)
    end_bench_time = time.perf_counter()

    for rec in completed_records:
        if not isinstance(rec, Exception):
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
    model_key: str = "llama3_1b",
    precisions: List[str] = None,
    use_mock: bool = True,
    hardware_key: str = "ada_4050",
    output_dir: str = "outputs",
) -> List[Dict[str, Any]]:
    """Runs an open-loop Poisson arrival rate sweep across precisions."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    workload_cfg = cfg.get("workload", {})
    trace_cfg = cfg.get("trace_distribution", {})
    slo_cfg = cfg.get("slo_targets", {})

    arrival_rates = workload_cfg.get("arrival_rates_per_second", [1.0, 2.0, 4.0, 8.0, 16.0])
    duration_s = float(workload_cfg.get("duration_seconds", 5))  # Fast run default
    warmup_count = int(workload_cfg.get("warmup_requests", 3))

    ttft_slo = float(slo_cfg.get("ttft_p99_ms", 150.0))
    tpot_slo = float(slo_cfg.get("tpot_p99_ms", 35.0))

    hw_profile = HARDWARE_PRESETS.get(hardware_key, HARDWARE_PRESETS["ada_4050"])
    hw_dict = {
        "memory_bandwidth_gbs": hw_profile.memory_bandwidth_gbs,
        "fp16_tflops": hw_profile.fp16_tflops,
        "fp8_tflops": hw_profile.fp8_tflops,
        "int8_tops": hw_profile.int8_tops,
    }

    test_precisions = precisions or ["fp16", "int4_awq", "fp8_e4m3", "int8_smoothquant"]
    all_results: List[Dict[str, Any]] = []

    print(f"\n=======================================================")
    print(f"Starting QuantServe-Bench Sweep on {hw_profile.name}")
    print(f"Model: {model_key} | Precisions: {test_precisions}")
    print(f"Arrival Rates: {arrival_rates} req/s")
    print(f"SLO Targets: P99 TTFT <= {ttft_slo}ms, P99 TPOT <= {tpot_slo}ms")
    print(f"=======================================================\n")

    os.makedirs(output_dir, exist_ok=True)

    for prec in test_precisions:
        print(f"\n--- Benchmarking Precision: {prec.upper()} ---")
        if use_mock:
            engine = MockServingEngine(
                model_name=model_key,
                precision=prec,
                hardware_profile=hw_dict,
            )
        else:
            engine = HuggingFaceServingEngine(model_name=model_key, precision=prec)

        await engine.initialize()
        workload_gen = WorkloadGenerator(workload_cfg, trace_cfg, seed=42)

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
            all_results.append(res_dict)

            print(
                f"Rate {rate:4.1f} req/s | "
                f"Throughput: {res_dict['throughput_tokens_per_s']:7.1f} tok/s | "
                f"TTFT P99: {res_dict['ttft_p99_ms']:6.1f}ms | "
                f"TPOT P99: {res_dict['tpot_p99_ms']:5.1f}ms | "
                f"SLO Goodput: {res_dict['slo_goodput_requests_per_s']:4.1f} req/s ({res_dict['slo_compliance_rate_pct']:4.1f}%)"
            )

    out_file = os.path.join(output_dir, f"sweep_{model_key}_{hardware_key}.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSweep complete! Saved results to {out_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="QuantServe-Bench Serving Load Sweeper")
    parser.add_argument("--config", type=str, default="configs/workload_poisson.yaml")
    parser.add_argument("--model", type=str, default="llama3_1b")
    parser.add_argument("--precisions", nargs="+", default=["fp16", "int4_awq", "fp8_e4m3", "int8_smoothquant"])
    parser.add_argument("--hardware", type=str, default="ada_4050")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark_sweep(
            config_path=args.config,
            model_key=args.model,
            precisions=args.precisions,
            use_mock=args.mock,
            hardware_key=args.hardware,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
