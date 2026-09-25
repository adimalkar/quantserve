"""Measured request timing against a running vLLM OpenAI-compatible server."""

import json
import os
import time
from urllib.parse import urlparse

import aiohttp

from src.bench.engine_interface import ServingEngine
from src.bench.metrics import RequestRecord
from src.bench.trace_sampler import RequestSpec


class VLLMServingEngine(ServingEngine):
    """Benchmarks a server already configured with the desired model and precision.

    Prompt lengths are requested through synthetic text. The server's usage
    counters, rather than those requested lengths, supply measured token counts.
    """

    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:8000"):
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) server origin, such as http://127.0.0.1:8000")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.session = None
        self.server_version = None
        self.request_counter = 0

    async def initialize(self) -> None:
        headers = {}
        api_key = os.environ.get("QUANTSERVE_VLLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=120)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                response.raise_for_status()
                payload = await response.json()
            model_ids = {item["id"] for item in payload["data"]}
            if self.model_name not in model_ids:
                raise ValueError(
                    f"Model {self.model_name!r} is not served at {self.base_url}; "
                    f"available models: {sorted(model_ids)}"
                )
            async with self.session.get(f"{self.base_url}/version") as response:
                response.raise_for_status()
                version = (await response.json()).get("version")
            if not isinstance(version, str) or not version:
                raise ValueError("vLLM did not report a server version")
            self.server_version = version
        except aiohttp.ClientError as exc:
            await self.close()
            raise RuntimeError(
                f"Cannot query vLLM at {self.base_url}; start the server and check its URL and API key"
            ) from exc
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def process_request(self, spec: RequestSpec, scheduled_time: float) -> RequestRecord:
        if self.session is None:
            raise RuntimeError("vLLM engine is not initialized")
        if spec.prompt_len < 1 or spec.output_len < 1:
            raise ValueError("prompt_len and output_len must be positive")

        # The actual prompt token count comes from the server's usage object.
        self.request_counter += 1
        # Keep prompts identical across repeated configuration runs. The
        # counter also makes each request distinct within a sweep.
        prompt = f"QuantServe request {self.request_counter:08d}. " + ("data " * spec.prompt_len)
        request = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": spec.output_len,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }
        dispatched = time.perf_counter()
        chunk_times = []
        usage = None
        saw_done = False
        async with self.session.post(f"{self.base_url}/v1/completions", json=request) as response:
            response.raise_for_status()
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    raise ValueError("Unexpected non-SSE response from vLLM")
                data = line[6:]
                if data == "[DONE]":
                    saw_done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(f"vLLM stream error: {event['error']}")
                if event.get("usage") is not None:
                    usage = event["usage"]
                if any(choice.get("text") for choice in event.get("choices", [])):
                    chunk_times.append(time.perf_counter())

        completed = time.perf_counter()
        if not saw_done or usage is None or not chunk_times:
            raise ValueError("vLLM stream lacked completion, usage, or output chunks")
        prompt_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens < 1:
            raise ValueError("vLLM did not report a valid prompt token count")
        if not isinstance(output_tokens, int) or output_tokens < 1:
            raise ValueError("vLLM did not report a valid completion token count")
        if output_tokens > 1 and len(chunk_times) < 2:
            raise ValueError("vLLM streamed multiple tokens in one chunk; TPOT cannot be timed")
        tpot = ((chunk_times[-1] - chunk_times[0]) * 1000 / (output_tokens - 1)) if output_tokens > 1 else 0.0

        return RequestRecord(
            request_id=spec.request_id,
            prompt_len=prompt_tokens,
            output_len=output_tokens,
            scheduled_time=scheduled_time,
            dispatched_time=dispatched,
            prefill_done_time=chunk_times[0],
            token_timestamps=chunk_times,
            completed_time=completed,
            tpot_override_ms=tpot,
        )
