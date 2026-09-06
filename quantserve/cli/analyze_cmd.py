"""CLI command for workload trace analysis and fingerprinting."""
import argparse
import json
import os
from quantserve.workload.trace_parser import TraceParser
from quantserve.workload.fingerprint import WorkloadFingerprinter
from quantserve.workload.drift import WorkloadDriftDetector


def run_analyze_cmd(args):
    if not os.path.exists(args.trace_file):
        print(f"Error: Trace file '{args.trace_file}' not found.")
        return

    records = TraceParser.parse_file(args.trace_file)
    fingerprint = WorkloadFingerprinter.analyze(records)

    if args.compare_baseline:
        if not os.path.exists(args.compare_baseline):
            print(f"Error: Baseline trace file '{args.compare_baseline}' not found.")
            return
        base_records = TraceParser.parse_file(args.compare_baseline)
        drift = WorkloadDriftDetector.compare(base_records, records)
        print(drift.format_cli_summary())
        return

    if args.json:
        print(json.dumps(fingerprint.to_dict(), indent=2))
        return

    print(fingerprint.format_cli_summary())
