"""CLI command for CI/CD deployment regression gating."""
import sys
import yaml
from quantserve.quality.regression_gate import DeploymentGate


def run_gate_cmd(args):
    try:
        with open(args.baseline, "r") as f:
            base_data = yaml.safe_load(f)
        with open(args.candidate, "r") as f:
            cand_data = yaml.safe_load(f)
        res = DeploymentGate.evaluate(
            baseline=base_data,
            candidate=cand_data,
            max_ttft_increase_pct=args.max_ttft_increase,
            max_tpot_increase_pct=args.max_tpot_increase,
            max_throughput_drop_pct=args.max_throughput_drop,
            max_vram_increase_gb=args.max_vram_increase_gb,
            max_quality_drop_pct=args.max_quality_drop,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Deployment gate input error: {exc}", file=sys.stderr)
        sys.exit(2)

    print(res.format_cli_summary())

    if not res.passed:
        sys.exit(1)
