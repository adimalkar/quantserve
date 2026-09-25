"""Fail-closed regression gate for complete measured deployment metrics."""
from dataclasses import dataclass
from math import isfinite
from typing import Dict, Any, List


@dataclass
class GateResult:
    passed: bool
    ttft_delta_pct: float
    tpot_delta_pct: float
    throughput_delta_pct: float
    vram_delta_gb: float
    quality_delta_pct: float
    details: List[str]

    def format_cli_summary(self) -> str:
        status_line = "✓ DEPLOYMENT GATE: PASS" if self.passed else "✗ DEPLOYMENT GATE: FAIL"
        lines = [
            "\n" + "=" * 54,
            "           QuantServe Deployment Gate",
            "=" * 54,
            f"  P95 TTFT change:     {self.ttft_delta_pct:+.1f}%",
            f"  P95 TPOT change:     {self.tpot_delta_pct:+.1f}%",
            f"  Throughput change:   {self.throughput_delta_pct:+.1f}%",
            f"  Peak VRAM change:    {self.vram_delta_gb:+.2f} GB",
            f"  Quality change:      {self.quality_delta_pct:+.1f} pp",
            "-" * 54,
            *self.details,
            status_line,
            "=" * 54 + "\n",
        ]
        return "\n".join(lines)


class DeploymentGate:
    """Compares complete measured runs; missing or synthetic data cannot pass."""

    @staticmethod
    def _read_metrics(data: Dict[str, Any], label: str) -> Dict[str, float]:
        if not isinstance(data, dict):
            raise ValueError(f"{label} must be a YAML mapping")
        if data.get("measurement_source") != "measured_e2e":
            raise ValueError(f"{label} must have measurement_source: measured_e2e")
        if data.get("measurement_status") != "complete":
            raise ValueError(f"{label} must have measurement_status: complete")

        aliases = {
            "ttft_p95_ms": ("ttft_p95_ms",),
            "tpot_p95_ms": ("tpot_p95_ms",),
            "throughput_tokens_per_s": ("throughput_tokens_per_s", "throughput_tokens_s"),
            "peak_vram_gb": ("peak_vram_gb",),
            "quality_pct": ("quality_retention_pct", "quality_pct"),
        }
        values: Dict[str, float] = {}
        for key, names in aliases.items():
            present = [name for name in names if name in data]
            if not present:
                raise ValueError(f"{label} is missing {key}")
            if any(isinstance(data[name], bool) for name in present):
                raise ValueError(f"{label} {key} must be numeric")
            try:
                numbers = [float(data[name]) for name in present]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} {key} must be numeric") from exc
            if any(not isfinite(number) for number in numbers):
                raise ValueError(f"{label} {key} must be finite")
            if len(numbers) > 1 and numbers[0] != numbers[1]:
                raise ValueError(f"{label} has conflicting values for {key}")
            value = numbers[0]
            if key == "quality_pct":
                if not 0 <= value <= 100:
                    raise ValueError(f"{label} {key} must be between 0 and 100")
            elif key == "peak_vram_gb":
                if value < 0:
                    raise ValueError(f"{label} {key} must be nonnegative")
            elif value <= 0:
                raise ValueError(f"{label} {key} must be positive")
            values[key] = value
        return values

    @staticmethod
    def evaluate(
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
        max_ttft_increase_pct: float = 10.0,
        max_tpot_increase_pct: float = 10.0,
        max_throughput_drop_pct: float = 10.0,
        max_vram_increase_gb: float = 0.0,
        max_quality_drop_pct: float = 2.0,
    ) -> GateResult:
        thresholds = (
            max_ttft_increase_pct, max_tpot_increase_pct,
            max_throughput_drop_pct, max_vram_increase_gb, max_quality_drop_pct,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0 for value in thresholds):
            raise ValueError("Gate thresholds must be finite and nonnegative")
        base = DeploymentGate._read_metrics(baseline, "baseline")
        cand = DeploymentGate._read_metrics(candidate, "candidate")

        ttft_delta = ((cand["ttft_p95_ms"] - base["ttft_p95_ms"]) / base["ttft_p95_ms"]) * 100.0
        tpot_delta = ((cand["tpot_p95_ms"] - base["tpot_p95_ms"]) / base["tpot_p95_ms"]) * 100.0
        thru_delta = ((cand["throughput_tokens_per_s"] - base["throughput_tokens_per_s"]) / base["throughput_tokens_per_s"]) * 100.0
        vram_delta = cand["peak_vram_gb"] - base["peak_vram_gb"]
        qual_delta = cand["quality_pct"] - base["quality_pct"]

        details: List[str] = []
        if ttft_delta > max_ttft_increase_pct:
            details.append(f"P95 TTFT increased by {ttft_delta:.1f}% (max allowed: {max_ttft_increase_pct}%)")

        if tpot_delta > max_tpot_increase_pct:
            details.append(f"P95 TPOT increased by {tpot_delta:.1f}% (max allowed: {max_tpot_increase_pct}%)")

        if thru_delta < -max_throughput_drop_pct:
            details.append(f"Throughput dropped by {abs(thru_delta):.1f}% (max allowed: {max_throughput_drop_pct}%)")

        if vram_delta > max_vram_increase_gb:
            details.append(f"Peak VRAM increased by {vram_delta:.2f} GB (max allowed: {max_vram_increase_gb} GB)")

        if qual_delta < -max_quality_drop_pct:
            details.append(f"Quality dropped by {abs(qual_delta):.1f} pp (max allowed: {max_quality_drop_pct} pp)")

        passed = not details
        if not details:
            details.append("All performance, memory, and quality regression criteria met.")

        return GateResult(
            passed=passed,
            ttft_delta_pct=ttft_delta,
            tpot_delta_pct=tpot_delta,
            throughput_delta_pct=thru_delta,
            vram_delta_gb=vram_delta,
            quality_delta_pct=qual_delta,
            details=details,
        )
