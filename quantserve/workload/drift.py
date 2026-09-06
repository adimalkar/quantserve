"""Workload drift detection using Jensen-Shannon divergence and Population Stability Index (PSI)."""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from scipy.spatial.distance import jensenshannon
from quantserve.workload.trace_parser import TraceRecord


@dataclass
class DriftReport:
    drift_detected: bool
    js_divergence_prompt: float
    js_divergence_output: float
    psi_prompt: float
    severity: str  # "LOW", "MODERATE", "SEVERE"
    re_optimization_recommended: bool
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "js_divergence_prompt": round(self.js_divergence_prompt, 4),
            "js_divergence_output": round(self.js_divergence_output, 4),
            "psi_prompt": round(self.psi_prompt, 4),
            "severity": self.severity,
            "re_optimization_recommended": self.re_optimization_recommended,
            "details": self.details,
        }

    def format_cli_summary(self) -> str:
        status = "⚠️  DRIFT DETECTED" if self.drift_detected else "✓ NO DRIFT DETECTED"
        return (
            f"\n{status}\n"
            f"  • Severity:                     {self.severity}\n"
            f"  • Prompt Jensen-Shannon Div:    {self.js_divergence_prompt:.4f}\n"
            f"  • Output Jensen-Shannon Div:    {self.js_divergence_output:.4f}\n"
            f"  • Population Stability Index:   {self.psi_prompt:.4f}\n"
            f"  • Re-Optimization Recommended:  {self.re_optimization_recommended}\n"
            f"  • Analysis: {self.details}\n"
        )


class WorkloadDriftDetector:
    """Monitors live request traces against baseline optimization traces to alert on distribution shift."""

    @staticmethod
    def _compute_distribution(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(values, bins=bins)
        p = hist.astype(np.float64) + 1.0  # Laplace smoothing to avoid zero bins
        return p / np.sum(p)

    @classmethod
    def compare(
        cls,
        baseline_records: List[TraceRecord],
        current_records: List[TraceRecord],
        drift_threshold: float = 0.25,
    ) -> DriftReport:
        if not baseline_records or not current_records:
            return DriftReport(
                drift_detected=False,
                js_divergence_prompt=0.0,
                js_divergence_output=0.0,
                psi_prompt=0.0,
                severity="LOW",
                re_optimization_recommended=False,
                details="Insufficient data for drift comparison.",
            )

        base_p = np.array([r.input_tokens for r in baseline_records], dtype=np.float64)
        curr_p = np.array([r.input_tokens for r in current_records], dtype=np.float64)

        base_o = np.array([r.output_tokens for r in baseline_records], dtype=np.float64)
        curr_o = np.array([r.output_tokens for r in current_records], dtype=np.float64)

        # 5 Quantile bins calibrated on baseline
        quantiles = np.linspace(0, 100, 6)
        bins_p = np.percentile(base_p, quantiles)
        bins_p[0] = min(bins_p[0], float(np.min(curr_p))) - 1.0
        bins_p[-1] = max(bins_p[-1], float(np.max(curr_p))) + 1.0
        # Ensure strictly increasing bins
        bins_p = np.unique(bins_p)
        if len(bins_p) < 3:
            bins_p = np.linspace(0, max(float(np.max(base_p)), float(np.max(curr_p))), 6)

        p_base_dist = cls._compute_distribution(base_p, bins_p)
        p_curr_dist = cls._compute_distribution(curr_p, bins_p)

        # Jensen-Shannon divergence
        js_prompt = float(jensenshannon(p_base_dist, p_curr_dist))

        # Population Stability Index (PSI): sum (A - E) * ln(A / E)
        psi = float(np.sum((p_curr_dist - p_base_dist) * np.log(p_curr_dist / p_base_dist)))

        # Output length divergence
        bins_o = np.linspace(0, max(float(np.max(base_o)), float(np.max(curr_o))), 6)
        o_base_dist = cls._compute_distribution(base_o, bins_o)
        o_curr_dist = cls._compute_distribution(curr_o, bins_o)
        js_output = float(jensenshannon(o_base_dist, o_curr_dist))

        drift = (js_prompt > drift_threshold) or (psi > 0.25)
        if js_prompt > 0.40 or psi > 0.50:
            severity = "SEVERE"
        elif drift:
            severity = "MODERATE"
        else:
            severity = "LOW"

        mean_base = float(np.mean(base_p))
        mean_curr = float(np.mean(curr_p))

        if drift:
            details = (
                f"Workload shifted significantly: Mean prompt length moved from {mean_base:.0f} to {mean_curr:.0f} "
                f"tokens (JS={js_prompt:.3f}, PSI={psi:.3f}). Current serving batching or quantization settings may be suboptimal."
            )
        else:
            details = (
                f"Traffic characteristics remain stable: Prompt lengths (mean {mean_curr:.0f} vs {mean_base:.0f}) "
                f"are within normal statistical variance (JS={js_prompt:.3f}, PSI={psi:.3f})."
            )

        return DriftReport(
            drift_detected=drift,
            js_divergence_prompt=js_prompt,
            js_divergence_output=js_output,
            psi_prompt=psi,
            severity=severity,
            re_optimization_recommended=drift,
            details=details,
        )
