"""CLI command for CI/CD deployment regression gating."""
import argparse
import sys
import yaml
from quantserve.quality.regression_gate import DeploymentGate


def run_gate_cmd(args):
    with open(args.baseline, "r") as f:
        base_data = yaml.safe_load(f)

    with open(args.candidate, "r") as f:
        cand_data = yaml.safe_load(f)

    # If full recommendation objects are passed, flatten metrics
    base_m = base_data.get("predicted_metrics", base_data)
    cand_m = cand_data.get("predicted_metrics", cand_data)

    res = DeploymentGate.evaluate(
        baseline=base_m,
        candidate=cand_m,
        max_ttft_increase_pct=args.max_ttft_increase,
        max_tpot_increase_pct=args.max_tpot_increase,
        max_quality_drop_pct=args.max_quality_drop,
    )

    print(res.format_cli_summary())

    if not res.passed:
        sys.exit(1)
