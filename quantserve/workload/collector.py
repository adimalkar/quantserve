"""Lightweight request telemetry collector for capturing anonymized production metadata."""
import time
from typing import List, Optional
from quantserve.workload.trace_parser import TraceRecord, TraceParser


class RequestTraceCollector:
    """Collects request timestamps, input/output token counts, and latencies without saving user prompt text."""

    def __init__(self):
        self.records: List[TraceRecord] = []

    def record_request(
        self,
        input_tokens: int,
        output_tokens: int,
        ttft_ms: Optional[float] = None,
        tpot_ms: Optional[float] = None,
        concurrency: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> TraceRecord:
        ts = timestamp_s if timestamp_s is not None else time.time()
        rec = TraceRecord(
            timestamp_s=ts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            concurrency=concurrency,
        )
        self.records.append(rec)
        return rec

    def save_to_jsonl(self, file_path: str) -> None:
        TraceParser.write_file(self.records, file_path)

    def clear(self) -> None:
        self.records.clear()
