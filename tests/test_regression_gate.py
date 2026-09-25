"""Measured deployment comparisons must fail closed on bad evidence."""
import copy
import subprocess
import sys

import pytest
import yaml

from quantserve.quality.regression_gate import DeploymentGate


@pytest.fixture
def baseline():
    return {
        "measurement_source": "measured_e2e",
        "measurement_status": "complete",
        "ttft_p95_ms": 500.0,
        "tpot_p95_ms": 40.0,
        "throughput_tokens_per_s": 100.0,
        "peak_vram_gb": 5.0,
        "quality_retention_pct": 99.0,
    }


def test_complete_measured_improvement_passes(baseline):
    candidate = copy.deepcopy(baseline)
    candidate.update(ttft_p95_ms=450.0, throughput_tokens_per_s=110.0)
    result = DeploymentGate.evaluate(baseline, candidate)
    assert result.passed
    assert "DEPLOYMENT GATE: PASS" in result.format_cli_summary()


@pytest.mark.parametrize(
    "change, error",
    [
        ({"measurement_source": "mock"}, "measurement_source"),
        ({"measurement_status": "partial"}, "measurement_status"),
        ({"ttft_p95_ms": None}, "ttft_p95_ms"),
        ({"ttft_p95_ms": float("nan")}, "finite"),
        ({"throughput_tokens_per_s": 0}, "positive"),
        ({"quality_retention_pct": 101}, "between 0 and 100"),
        ({"quality_pct": 97}, "conflicting values"),
    ],
)
def test_invalid_measurement_is_rejected(baseline, change, error):
    candidate = copy.deepcopy(baseline)
    candidate.update(change)
    with pytest.raises(ValueError, match=error):
        DeploymentGate.evaluate(baseline, candidate)


def test_missing_metric_is_rejected(baseline):
    candidate = copy.deepcopy(baseline)
    del candidate["peak_vram_gb"]
    with pytest.raises(ValueError, match="missing peak_vram_gb"):
        DeploymentGate.evaluate(baseline, candidate)


@pytest.mark.parametrize(
    "change, detail",
    [
        ({"ttft_p95_ms": 560.0}, "P95 TTFT increased"),
        ({"tpot_p95_ms": 45.0}, "P95 TPOT increased"),
        ({"throughput_tokens_per_s": 89.0}, "Throughput dropped"),
        ({"peak_vram_gb": 5.1}, "Peak VRAM increased"),
        ({"quality_retention_pct": 96.0}, "Quality dropped"),
    ],
)
def test_each_regression_fails(baseline, change, detail):
    candidate = copy.deepcopy(baseline)
    candidate.update(change)
    result = DeploymentGate.evaluate(baseline, candidate)
    assert not result.passed
    assert any(detail in item for item in result.details)


def test_custom_thresholds_apply(baseline):
    candidate = copy.deepcopy(baseline)
    candidate.update(throughput_tokens_per_s=89.0, peak_vram_gb=5.1)
    result = DeploymentGate.evaluate(
        baseline, candidate, max_throughput_drop_pct=12.0, max_vram_increase_gb=0.2,
    )
    assert result.passed


def test_gate_cli_exit_codes(tmp_path, baseline):
    baseline_path = tmp_path / "baseline.yaml"
    candidate_path = tmp_path / "candidate.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline))

    def run(candidate):
        candidate_path.write_text(yaml.safe_dump(candidate))
        return subprocess.run(
            [sys.executable, "-m", "quantserve", "gate", "--baseline", str(baseline_path),
             "--candidate", str(candidate_path)],
            capture_output=True, text=True,
        )

    assert run(baseline).returncode == 0
    regressed = {**baseline, "throughput_tokens_per_s": 80.0}
    assert run(regressed).returncode == 1
    predicted = {"predicted_metrics": baseline}
    error = run(predicted)
    assert error.returncode == 2
    assert "measurement_source" in error.stderr
