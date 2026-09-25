"""Contract checks for the measured serving benchmark path."""

import asyncio
import json
import time

import pytest
import yaml
from aiohttp import web

from src.bench.metrics import RequestRecord
from src.bench.runner import run_benchmark_sweep, run_single_benchmark
from src.bench.trace_sampler import RequestSpec
from src.bench.vllm_engine import VLLMServingEngine
from src.bench.workload_generator import WorkloadGenerator


@pytest.fixture
async def completion_server():
    calls = []
    controls = {"include_usage": True}

    async def models(request):
        return web.json_response({"data": [{"id": "test-model"}]})

    async def version(request):
        return web.json_response({"version": "test-vllm-1"})

    async def completions(request):
        calls.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for text in ("one", " two", " three"):
            event = {"choices": [{"text": text}]}
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await asyncio.sleep(0.005)
        if controls["include_usage"]:
            await response.write(b'data: {"choices": [], "usage": {"prompt_tokens": 17, "completion_tokens": 3}}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_get("/version", version)
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", calls, controls
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_vllm_stream_uses_server_token_counts(completion_server):
    url, calls, _ = completion_server
    engine = VLLMServingEngine("test-model", url)
    await engine.initialize()
    try:
        record = await engine.process_request(RequestSpec("one", 5, 3), 0.0)
    finally:
        await engine.close()
    assert record.prompt_len == 17
    assert record.output_len == 3
    assert engine.server_version == "test-vllm-1"
    assert record.ttft_ms >= 0
    assert record.tpot_ms > 0
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert calls[0]["model"] == "test-model"


@pytest.mark.asyncio
async def test_same_workload_produces_same_prompts_across_runs(completion_server):
    url, calls, _ = completion_server
    for _ in range(2):
        engine = VLLMServingEngine("test-model", url)
        await engine.initialize()
        try:
            await engine.process_request(RequestSpec("same", 7, 3), 0.0)
        finally:
            await engine.close()
    assert calls[0]["prompt"] == calls[1]["prompt"]


@pytest.mark.asyncio
async def test_vllm_model_mismatch_fails_before_benchmark(completion_server):
    url, _, _ = completion_server
    engine = VLLMServingEngine("wrong-model", url)
    with pytest.raises(ValueError, match="not served"):
        await engine.initialize()
    assert engine.session is None


@pytest.mark.asyncio
async def test_vllm_missing_usage_fails_closed(completion_server):
    url, _, controls = completion_server
    controls["include_usage"] = False
    engine = VLLMServingEngine("test-model", url)
    await engine.initialize()
    try:
        with pytest.raises(ValueError, match="lacked completion, usage"):
            await engine.process_request(RequestSpec("one", 5, 3), 0.0)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_sweep_records_measured_provenance_and_fails_on_bad_stream(completion_server, tmp_path):
    url, _, controls = completion_server
    config = tmp_path / "workload.yaml"
    config.write_text(yaml.safe_dump({
        "workload": {"arrival_rates_per_second": [50.0], "duration_seconds": 0.2, "warmup_requests": 0, "seed": 7},
    }))
    output_dir = tmp_path / "results"
    results = await run_benchmark_sweep(
        str(config), "test-model", backend="vllm", server_url=url, output_dir=str(output_dir),
    )
    assert results[0]["measurement_source"] == "measured_e2e"
    assert results[0]["measurement_status"] == "complete"
    assert results[0]["server_version"] == "test-vllm-1"
    assert results[0]["requested_window_s"] == 0.2
    assert results[0]["workload_seed"] == 7
    assert len(results[0]["workload_config_sha256"]) == 64
    assert results[0]["total_prompt_tokens"] == 17 * results[0]["completed_requests"]
    output_file = output_dir / "sweep_test-model_ada_4050_vllm.json"
    assert json.loads(output_file.read_text()) == results

    controls["include_usage"] = False
    with pytest.raises(RuntimeError, match="failed requests"):
        await run_benchmark_sweep(
            str(config), "test-model", backend="vllm", server_url=url, output_dir=str(output_dir),
        )
    partial = json.loads(output_file.read_text())
    assert partial[0]["measurement_status"] == "partial_failed_requests"
    assert partial[0]["failed_requests"] == partial[0]["total_requests"]


@pytest.mark.asyncio
async def test_warmup_excluded_from_summary():
    class CountingEngine:
        def __init__(self):
            self.ids = []

        async def process_request(self, spec, scheduled_time):
            self.ids.append(spec.request_id)
            now = time.perf_counter()
            return RequestRecord(
                request_id=spec.request_id, prompt_len=spec.prompt_len,
                output_len=spec.output_len, scheduled_time=scheduled_time,
                dispatched_time=now, prefill_done_time=now + 0.001,
                token_timestamps=[now + 0.001, now + 0.002],
                completed_time=now + 0.002,
            )

    engine = CountingEngine()
    workload = WorkloadGenerator({}, {}, seed=1)
    result = await run_single_benchmark(engine, workload, 50.0, 0.2, "test-model", "fp16", warmup_requests=2)
    assert engine.ids[:2] == ["warmup_0000", "warmup_0001"]
    assert result.total_requests == len(engine.ids) - 2
    assert result.completed_requests == result.total_requests
    assert result.duration_s >= 0.2 - 1e-6
    assert result.throughput_tokens_per_s <= (result.total_prompt_tokens + result.total_output_tokens) / 0.2 + 1e-3
