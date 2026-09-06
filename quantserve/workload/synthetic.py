"""Synthetic trace generator for Interactive Coding, RAG, and Batch workloads."""
from typing import List
import numpy as np
from quantserve.workload.trace_parser import TraceRecord


class SyntheticTraceGenerator:
    """Generates synthetic traces mimicking real-world production workload classes."""

    @staticmethod
    def generate_interactive_coding(num_requests: int = 100, seed: int = 42) -> List[TraceRecord]:
        """Interactive Coding Assistant: Short to medium prompts, fast generation, Poisson arrival."""
        rng = np.random.default_rng(seed)
        records = []
        t = 0.0
        for _ in range(num_requests):
            t += float(rng.exponential(0.2))  # ~5 req/s
            prompt = int(rng.normal(350, 100))
            prompt = max(32, min(prompt, 1024))
            gen = int(rng.normal(64, 20))
            gen = max(8, min(gen, 256))
            records.append(TraceRecord(timestamp_s=t, input_tokens=prompt, output_tokens=gen, concurrency=int(rng.integers(2, 12))))
        return records

    @staticmethod
    def generate_rag_assistant(num_requests: int = 100, seed: int = 42) -> List[TraceRecord]:
        """RAG Knowledge Assistant: Long retrieved contexts (2K-6K tokens), medium generation."""
        rng = np.random.default_rng(seed)
        records = []
        t = 0.0
        for _ in range(num_requests):
            t += float(rng.exponential(0.5))  # ~2 req/s
            prompt = int(rng.normal(2400, 600))
            prompt = max(512, min(prompt, 8192))
            gen = int(rng.normal(180, 40))
            gen = max(32, min(gen, 512))
            records.append(TraceRecord(timestamp_s=t, input_tokens=prompt, output_tokens=gen, concurrency=int(rng.integers(1, 8))))
        return records

    @staticmethod
    def generate_batch_processing(num_requests: int = 100, seed: int = 42) -> List[TraceRecord]:
        """Batch Document Processing: Heavy burst arrivals, long outputs, high volume."""
        rng = np.random.default_rng(seed)
        records = []
        t = 0.0
        for _ in range(num_requests):
            t += float(rng.gamma(shape=0.3, scale=0.1))  # Bursty arrivals
            prompt = int(rng.normal(800, 250))
            prompt = max(128, min(prompt, 2048))
            gen = int(rng.normal(400, 100))
            gen = max(64, min(gen, 1024))
            records.append(TraceRecord(timestamp_s=t, input_tokens=prompt, output_tokens=gen, concurrency=int(rng.integers(16, 64))))
        return records
