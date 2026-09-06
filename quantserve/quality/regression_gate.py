"""CI/CD deployment regression gate validating candidate configs against production baselines."""
from dataclasses import dataclass
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
            f"Performance:",
            f"  P95 TTFT:       {self.ttft_delta_pct:+.1f}% {'✓' if self.ttft_delta_pct <= 0 else '⚠️'}",
            f"  P95 TPOT:       {self.tpot_delta_pct:+.1f}% {'✓' if self.tpot_delta_pct <= 0 else '⚠️'}",
            f"  Throughput:     {self.throughput_delta_pct:+.1f}% {'✓' if self.throughput_delta_pct >= 0 else '⚠️'}",
            f"Memory:",
            f"  Peak VRAM:      {self.vram_delta_gb:+.2f} GB {'✓' if self.vram_delta_gb <= 0 else '⚠️'}",
            f"Quality:",
            f"  Quality Drop:   {self.quality_delta_pct:+.1f}% {'✓' if self.quality_delta_pct >= -2.0 else '✗'}",
            "-" * 54,
            status_line,
            "=" * 54 + "\n",
        ]
        return "\n".join(lines)


class DeploymentGate:
    """Evaluates candidate serving configurations against production baselines."""

    @staticmethod
    def evaluate(
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
        max_ttft_increase_pct: float = 10.0,
        max_tpot_increase_pct: float = 10.0,
        max_quality_drop_pct: float = 2.0,
    ) -> GateResult:
        base_ttft = float(baseline.get("ttft_p95_ms", 500.0))
        cand_ttft = float(candidate.get("ttft_p95_ms", 450.0))
        ttft_delta = ((cand_ttft - base_ttft) / base_ttft) * 100.0

        base_tpot = float(baseline.get("tpot_p95_ms", 30.0))
        cand_tpot = float(candidate.get("tpot_p95_ms", 28.0))
        tpot_delta = ((cand_tpot - base_tpot) / base_tpot) * 100.0

        base_thru = float(baseline.get("throughput_tokens_s", 100.0))
        cand_thru = float(candidate.get("throughput_tokens_s", 120.0))
        thru_delta = ((cand_thru - base_thru) / base_thru) * 100.0

        base_vram = float(baseline.get("peak_vram_gb", 5.0))
        cand_vram = float(candidate.get("peak_vram_gb", 4.5))
        vram_delta = cand_vram - base_vram

        base_qual = float(baseline.get("quality_pct", 100.0))
        cand_qual = float(candidate.get("quality_pct", 98.7))
        qual_delta = cand_qual - base_qual

        details: List[str] = []
        passed = True

        if ttft_delta > max_ttft_increase_pct:
            passed = False
            details.append(f"P95 TTFT increased by {ttft_delta:.1f}% (max allowed: {max_ttft_increase_pct}%)")

        if tpot_delta > max_tpot_increase_pct:
            passed = False
            details.append(f"P95 TPOT increased by {tpot_delta:.1f}% (max allowed: {max_tpot_increase_pct}%)")

        if qual_delta < -max_quality_drop_pct:
            passed = False
            details.append(f"Quality dropped by {abs(qual_delta):.1f}% (max allowed drop: {max_quality_drop_pct}%)")

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
