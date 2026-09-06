"""QuantServe top-level command-line interface entrypoint."""
import argparse
import sys

from quantserve.cli.hardware_cmd import run_hardware_cmd
from quantserve.cli.analyze_cmd import run_analyze_cmd
from quantserve.cli.recommend_cmd import run_recommend_cmd
from quantserve.cli.gate_cmd import run_gate_cmd
from quantserve.cli.export_cmd import run_export_cmd


def main():
    parser = argparse.ArgumentParser(
        prog="quantserve",
        description="QuantServe: Hardware-Aware LLM Deployment Optimizer",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Hardware probe
    hw_parser = subparsers.add_parser("hardware", help="Probe and inspect GPU hardware capabilities")
    hw_parser.add_argument("--preset", type=str, default=None, help="Override detected GPU with preset (ada_4050, a10g, l4, rtx_4090, h100)")
    hw_parser.add_argument("--json", action="store_true", help="Output hardware profile as JSON")

    # 2. Analyze trace
    an_parser = subparsers.add_parser("analyze", help="Analyze and fingerprint an anonymized JSONL workload trace")
    an_parser.add_argument("trace_file", type=str, help="Path to .jsonl workload trace file")
    an_parser.add_argument("--compare-baseline", type=str, default=None, help="Compare against baseline trace for drift detection")
    an_parser.add_argument("--json", action="store_true", help="Output fingerprint as JSON")

    # 3. Recommend deployment config
    rec_parser = subparsers.add_parser("recommend", help="Recommend optimal deployment configuration for model and workload")
    rec_parser.add_argument("--model", type=str, required=True, help="Hugging Face model identifier")
    rec_parser.add_argument("--hardware", type=str, default="auto", help="Hardware profile ('auto', 'ada_4050', 'a10g', 'l4', 'rtx_4090', 'h100')")
    rec_parser.add_argument("--trace", type=str, default=None, help="Path to request trace file for workload fingerprinting")
    rec_parser.add_argument("--p95-ttft", type=str, default="750ms", help="Maximum allowable P95 TTFT (e.g. '750ms', '1.0s')")
    rec_parser.add_argument("--p95-tpot", type=str, default="45ms", help="Maximum allowable P95 TPOT (e.g. '45ms')")
    rec_parser.add_argument("--quality-retention", type=float, default=0.98, help="Minimum allowable quality retention fraction (e.g. 0.98)")
    rec_parser.add_argument("--objective", type=str, default="minimize_cost", choices=["minimize_cost", "minimize_latency", "maximize_throughput"])
    rec_parser.add_argument("--export", type=str, default=None, help="Path to save recommendation YAML")
    rec_parser.add_argument("--json", action="store_true", help="Output recommendation as JSON")

    # 4. Gate CI/CD
    gate_parser = subparsers.add_parser("gate", help="Validate candidate deployment config against production baseline")
    gate_parser.add_argument("--baseline", type=str, required=True, help="Path to baseline YAML")
    gate_parser.add_argument("--candidate", type=str, required=True, help="Path to candidate YAML")
    gate_parser.add_argument("--max-ttft-increase", type=float, default=10.0, help="Max allowed TTFT increase %%")
    gate_parser.add_argument("--max-tpot-increase", type=float, default=10.0, help="Max allowed TPOT increase %%")
    gate_parser.add_argument("--max-quality-drop", type=float, default=2.0, help="Max allowed quality drop %%")

    # 5. Export deployment files
    exp_parser = subparsers.add_parser("export", help="Export deployment artifacts (vllm, docker, k8s)")
    exp_parser.add_argument("--recommendation", type=str, required=True, help="Path to recommendation YAML")
    exp_parser.add_argument("--format", type=str, default="vllm", choices=["vllm", "docker", "k8s"], help="Target format")
    exp_parser.add_argument("--output", type=str, default=None, help="Output destination filepath")

    # 6. Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run active benchmark sweep on model")
    bench_parser.add_argument("--config", type=str, default="configs/workload_fast.yaml")
    bench_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "hardware":
        run_hardware_cmd(args)
    elif args.command == "analyze":
        run_analyze_cmd(args)
    elif args.command == "recommend":
        run_recommend_cmd(args)
    elif args.command == "gate":
        run_gate_cmd(args)
    elif args.command == "export":
        run_export_cmd(args)
    elif args.command == "benchmark":
        from src.bench.runner import run_benchmark_sweep
        import asyncio
        asyncio.run(run_benchmark_sweep(args.config, args.model, use_mock=True))


if __name__ == "__main__":
    main()
