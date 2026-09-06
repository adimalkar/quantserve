"""Serving engine interfaces: Pluggable abstract engine, physics-calibrated simulator, and live PyTorch backend."""
import abc
import asyncio
import time
from typing import AsyncGenerator, Dict, Any, Optional
import torch

from src.bench.trace_sampler import RequestSpec
from src.bench.metrics import RequestRecord


class ServingEngine(abc.ABC):
    """Abstract interface for LLM serving backends."""

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initializes the model, allocates KV cache, and prepares workers."""
        pass

    @abc.abstractmethod
    async def process_request(self, spec: RequestSpec, scheduled_time: float) -> RequestRecord:
        """Executes a request, capturing exact prefill and per-token decode timestamps."""
        pass


class MockServingEngine(ServingEngine):
    """Physics-calibrated inference simulator modeling DRAM bandwidth and Tensor Core compute ceilings.
    Accurately reflects why INT4 wins at low batch and loses at high batch due to dequantization ALU overhead.
    """

    def __init__(
        self,
        model_name: str = "llama3_1b",
        precision: str = "fp16",
        kv_cache_precision: str = "fp16",
        hardware_profile: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.precision = precision.lower()
        self.kv_cache_precision = kv_cache_precision.lower()

        hw = hardware_profile or {}
        self.memory_bandwidth_gbs = hw.get("memory_bandwidth_gbs", 192.0)  # RTX 4050 default
        self.fp16_tflops = hw.get("fp16_tflops", 36.0)
        self.fp8_tflops = hw.get("fp8_tflops", 72.0)
        self.int8_tops = hw.get("int8_tops", 72.0)

        # Model params: Llama-3.2-1B = 1.23B params
        if "3b" in model_name:
            self.param_count = 3.21e9
            self.layers = 28
            self.hidden_dim = 3072
        else:
            self.param_count = 1.23e9
            self.layers = 16
            self.hidden_dim = 2048

        # Bytes per parameter by precision
        if self.precision in ("fp16", "bf16"):
            self.bytes_per_param = 2.0
            self.dequant_cost_factor = 1.0
            self.compute_cap = self.fp16_tflops * 1e12
        elif self.precision in ("int8", "int8_smoothquant", "w8a8"):
            self.bytes_per_param = 1.0
            self.dequant_cost_factor = 1.05  # Slight scaling overhead
            self.compute_cap = self.int8_tops * 1e12
        elif self.precision in ("fp8", "fp8_e4m3"):
            self.bytes_per_param = 1.0
            self.dequant_cost_factor = 1.02
            self.compute_cap = self.fp8_tflops * 1e12
        elif self.precision in ("int4", "int4_awq", "int4_gptq", "w4a16"):
            self.bytes_per_param = 0.5
            # Crucial real-world effect: W4A16 requires on-the-fly ALU dequantization to FP16 before GEMM
            self.dequant_cost_factor = 1.35
            self.compute_cap = self.fp16_tflops * 1e12
        elif self.precision in ("sparse_2_4", "sparse"):
            self.bytes_per_param = 1.0  # 50% dense weights packed
            self.dequant_cost_factor = 1.08
            self.compute_cap = (self.fp16_tflops * 2.0) * 1e12
        else:
            self.bytes_per_param = 2.0
            self.dequant_cost_factor = 1.0
            self.compute_cap = self.fp16_tflops * 1e12

        # Active requests in simulated continuous batching
        self._current_concurrency = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        await asyncio.sleep(0.01)

    def _calculate_prefill_time(self, prompt_tokens: int) -> float:
        """Prefill is compute-bound (GEMM): 2 * Params * PromptTokens FLOPs."""
        flops = 2.0 * self.param_count * prompt_tokens
        t_compute = flops / self.compute_cap
        # Small memory read for weights at start of prefill
        t_dram = (self.param_count * self.bytes_per_param) / (self.memory_bandwidth_gbs * 1e9)
        return max(t_compute, t_dram) + 0.002  # 2ms framework dispatch overhead

    def _calculate_token_decode_time(self, current_batch_size: int, context_len: int) -> float:
        """Decode step time under continuous batching:
        At batch 1: Memory-bandwidth bound (reading all weights to generate 1 token).
        At high batch: Compute bound (batch GEMV turns into compute-bound GEMM).
        """
        # DRAM memory transfer: read weights once per layer + read active KV cache
        weight_bytes = self.param_count * self.bytes_per_param
        kv_bytes_per_token = 2.0 if self.kv_cache_precision == "fp16" else 1.0
        kv_bytes = current_batch_size * context_len * self.layers * 2 * (self.hidden_dim // 4) * kv_bytes_per_token

        t_memory = (weight_bytes + kv_bytes) / (self.memory_bandwidth_gbs * 1e9)

        # Compute required for batch of tokens
        decode_flops = 2.0 * self.param_count * current_batch_size * self.dequant_cost_factor
        t_compute = decode_flops / self.compute_cap

        # Total decode iteration time divided by batch size gives time per output token
        step_time = max(t_memory, t_compute) + 0.0005
        return step_time

    async def process_request(self, spec: RequestSpec, scheduled_time: float) -> RequestRecord:
        dispatched_time = time.perf_counter()

        async with self._lock:
            self._current_concurrency += 1
            batch = self._current_concurrency

        try:
            # 1. Simulate Prefill
            t_prefill = self._calculate_prefill_time(spec.prompt_len)
            await asyncio.sleep(t_prefill)
            prefill_done_time = time.perf_counter()

            token_timestamps: list[float] = [prefill_done_time]

            # 2. Simulate Decode Tokens
            curr_context = spec.prompt_len
            for _ in range(spec.output_len - 1):
                t_decode = self._calculate_token_decode_time(batch, curr_context)
                await asyncio.sleep(t_decode)
                token_timestamps.append(time.perf_counter())
                curr_context += 1

            completed_time = time.perf_counter()

            return RequestRecord(
                request_id=spec.request_id,
                prompt_len=spec.prompt_len,
                output_len=spec.output_len,
                scheduled_time=scheduled_time,
                dispatched_time=dispatched_time,
                prefill_done_time=prefill_done_time,
                token_timestamps=token_timestamps,
                completed_time=completed_time,
                error=None,
            )

        finally:
            async with self._lock:
                self._current_concurrency = max(0, self._current_concurrency - 1)


class HuggingFaceServingEngine(ServingEngine):
    """Live PyTorch / Transformers engine supporting real CUDA execution and token timestamping."""

    def __init__(
        self,
        model_name: str,
        precision: str = "fp16",
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.precision = precision
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    async def initialize(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        # Load model onto target device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

    async def process_request(self, spec: RequestSpec, scheduled_time: float) -> RequestRecord:
        dispatched_time = time.perf_counter()
        loop = asyncio.get_running_loop()

        def _sync_generate():
            # Minimal dummy prompt of target length
            input_ids = torch.randint(100, 1000, (1, spec.prompt_len), device=self.device)
            t_prefill_start = time.perf_counter()

            with torch.inference_mode():
                # Prefill step
                out = self.model(input_ids, use_cache=True)
                past_key_values = out.past_key_values
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                t_prefill_done = time.perf_counter()

                timestamps = [t_prefill_done]

                # Decode loop
                for _ in range(spec.output_len - 1):
                    out = self.model(next_token, past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values
                    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    timestamps.append(time.perf_counter())

            t_complete = time.perf_counter()
            return t_prefill_done, timestamps, t_complete

        prefill_done, token_timestamps, completed_time = await loop.run_in_executor(None, _sync_generate)

        return RequestRecord(
            request_id=spec.request_id,
            prompt_len=spec.prompt_len,
            output_len=spec.output_len,
            scheduled_time=scheduled_time,
            dispatched_time=dispatched_time,
            prefill_done_time=prefill_done,
            token_timestamps=token_timestamps,
            completed_time=completed_time,
            error=None,
        )
