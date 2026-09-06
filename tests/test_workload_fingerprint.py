"""Unit tests for workload trace parsing, fingerprinting, and drift detection."""
import pytest
from quantserve.workload.synthetic import SyntheticTraceGenerator
from quantserve.workload.fingerprint import WorkloadFingerprinter
from quantserve.workload.drift import WorkloadDriftDetector


def test_workload_fingerprinting_classification():
    # 1. Interactive Coding
    t_coding = SyntheticTraceGenerator.generate_interactive_coding(num_requests=50, seed=1)
    fp_coding = WorkloadFingerprinter.analyze(t_coding)
    assert "Interactive" in fp_coding.workload_class
    assert fp_coding.avg_prompt_tokens < 1000

    # 2. RAG Assistant
    t_rag = SyntheticTraceGenerator.generate_rag_assistant(num_requests=50, seed=2)
    fp_rag = WorkloadFingerprinter.analyze(t_rag)
    assert "RAG" in fp_rag.workload_class
    assert fp_rag.avg_prompt_tokens > 1500
    assert "Prefill" in fp_rag.likely_bottleneck

    # 3. Batch Processing
    t_batch = SyntheticTraceGenerator.generate_batch_processing(num_requests=50, seed=3)
    fp_batch = WorkloadFingerprinter.analyze(t_batch)
    assert "Batch" in fp_batch.workload_class


def test_workload_drift_detection():
    # Baseline: Interactive short prompts (mean ~350)
    baseline = SyntheticTraceGenerator.generate_interactive_coding(num_requests=100, seed=42)

    # Similar distribution (no drift)
    current_stable = SyntheticTraceGenerator.generate_interactive_coding(num_requests=100, seed=43)
    report_stable = WorkloadDriftDetector.compare(baseline, current_stable)
    assert report_stable.drift_detected is False
    assert report_stable.re_optimization_recommended is False

    # Shifted distribution: RAG long prompts (mean ~2400)
    current_drifted = SyntheticTraceGenerator.generate_rag_assistant(num_requests=100, seed=99)
    report_drifted = WorkloadDriftDetector.compare(baseline, current_drifted)
    assert report_drifted.drift_detected is True
    assert report_drifted.js_divergence_prompt > 0.25
    assert report_drifted.re_optimization_recommended is True
