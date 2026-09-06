"""Trace parser for reading and validating JSONL production workload logs."""
import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TraceRecord:
    timestamp_s: float
    input_tokens: int
    output_tokens: int
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    concurrency: Optional[int] = None


class TraceParser:
    """Parses anonymized JSONL request trace files."""

    @staticmethod
    def parse_file(file_path: str) -> List[TraceRecord]:
        records: List[TraceRecord] = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                records.append(
                    TraceRecord(
                        timestamp_s=float(d.get("timestamp_s", d.get("timestamp", 0.0))),
                        input_tokens=int(d.get("input_tokens", d.get("prompt_tokens", 128))),
                        output_tokens=int(d.get("output_tokens", d.get("decode_tokens", 32))),
                        ttft_ms=float(d["ttft_ms"]) if "ttft_ms" in d else None,
                        tpot_ms=float(d["tpot_ms"]) if "tpot_ms" in d else None,
                        concurrency=int(d["concurrency"]) if "concurrency" in d else None,
                    )
                )
        return records

    @staticmethod
    def write_file(records: List[TraceRecord], file_path: str) -> None:
        with open(file_path, "w") as f:
            for r in records:
                d = {
                    "timestamp_s": r.timestamp_s,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                }
                if r.ttft_ms is not None:
                    d["ttft_ms"] = round(r.ttft_ms, 2)
                if r.tpot_ms is not None:
                    d["tpot_ms"] = round(r.tpot_ms, 2)
                if r.concurrency is not None:
                    d["concurrency"] = r.concurrency
                f.write(json.dumps(d) + "\n")
