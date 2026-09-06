"""CLI command for generating hardware-aware deployment recommendations."""
import argparse
import json
import os
import yaml
from quantserve.advisor.hardware_probe import HardwareProber
from quantserve.advisor.optimizer import DeploymentOptimizer
from quantserve.workload.trace_parser import TraceParser
from quantserve.workload.fingerprint import WorkloadFingerprinter


def parse_latency_str(s: str) -> float:
    s = s.strip().lower()
    if s.endswith("ms"):
        return float(s[:-2])
    elif s.endswith("s"):
        return float(s[:-1]) * 1000.0
    return float(s)


def run_recommend_cmd(args):
    hw_preset = None if args.hardware == "auto" else args.hardware
    hardware = HardwareProber.probe(override_preset=hw_preset)

    if args.trace and os.path.exists(args.trace):
        records = TraceParser.parse_file(args.trace)
        fp = WorkloadFingerprinter.analyze(records)
        workload = {
            "avg_prompt_tokens": fp.avg_prompt_tokens,
            "avg_output_tokens": fp.avg_output_tokens,
            "concurrency": fp.peak_concurrency,
            "request_rate": fp.request_rate,
            "slo_ttft_ms": parse_latency_str(args.p95_ttft),
            "slo_tpot_ms": parse_latency_str(args.p95_tpot),
        }
    else:
        workload = {
            "avg_prompt_tokens": 350.0,
            "avg_output_tokens": 64.0,
            "concurrency": 8,
            "request_rate": 5.0,
            "slo_ttft_ms": parse_latency_str(args.p95_ttft),
            "slo_tpot_ms": parse_latency_str(args.p95_tpot),
        }

    model_name = args.model
    model_meta = {
        "params_billion": 0.5 if "0.5b" in model_name.lower() else (3.2 if "3b" in model_name.lower() else 1.23),
        "num_layers": 16 if "3b" not in model_name.lower() else 28,
        "hidden_size": 2048 if "3b" not in model_name.lower() else 3072,
        "num_kv_heads": 8,
        "num_heads": 32,
    }

    optimizer = DeploymentOptimizer()
    rec = optimizer.recommend(
        model_name=model_name,
        hardware=hardware,
        workload=workload,
        model_metadata=model_meta,
        p95_ttft_max_ms=parse_latency_str(args.p95_ttft),
        p95_tpot_max_ms=parse_latency_str(args.p95_tpot),
        quality_retention_min=args.quality_retention,
        objective=args.objective,
    )

    if args.json:
        print(json.dumps(rec.to_dict(), indent=2))
        return

    print(rec.format_cli_summary())

    if args.export:
        dirname = os.path.dirname(args.export)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.export, "w") as f:
            yaml.dump(rec.to_dict(), f, sort_keys=False)
        print(f"Exported recommendation to {args.export}\n")
