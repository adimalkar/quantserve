"""Automated Crossover Concurrency (C*) detector comparing INT4/FP8 against baseline FP16."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class CrossoverPoint:
    compressed_precision: str
    baseline_precision: str
    crossover_concurrency_or_rate: float
    crossover_metric: str
    peak_speedup_at_batch1: float
    high_load_penalty_pct: float
    crossover_found: bool


class CrossoverDetector:
    """Finds the crossover load where compression dequantization overhead flips net performance."""

    def __init__(self, baseline_precision: str = "fp16"):
        self.baseline_precision = baseline_precision.lower()

    def find_crossover(
        self,
        sweep_results: List[Dict[str, Any]],
        compressed_precision: str,
        metric: str = "throughput_tokens_per_s",
    ) -> CrossoverPoint:
        """Finds the concurrency/rate where compressed_precision throughput falls below baseline."""
        baseline_points = [
            r for r in sweep_results if r["precision"].lower() == self.baseline_precision
        ]
        compressed_points = [
            r for r in sweep_results if r["precision"].lower() == compressed_precision.lower()
        ]

        if not baseline_points or not compressed_points:
            return CrossoverPoint(
                compressed_precision=compressed_precision,
                baseline_precision=self.baseline_precision,
                crossover_concurrency_or_rate=0.0,
                crossover_metric=metric,
                peak_speedup_at_batch1=1.0,
                high_load_penalty_pct=0.0,
                crossover_found=False,
            )

        # Sort by concurrency_or_rate
        baseline_points = sorted(baseline_points, key=lambda x: x["concurrency_or_rate"])
        compressed_points = sorted(compressed_points, key=lambda x: x["concurrency_or_rate"])

        # Match points by rate
        rates: list[float] = []
        base_vals: list[float] = []
        comp_vals: list[float] = []

        base_map = {r["concurrency_or_rate"]: r[metric] for r in baseline_points}
        for c in compressed_points:
            rate = c["concurrency_or_rate"]
            if rate in base_map:
                rates.append(rate)
                comp_vals.append(c[metric])
                base_vals.append(base_map[rate])

        if len(rates) < 2:
            return CrossoverPoint(
                compressed_precision=compressed_precision,
                baseline_precision=self.baseline_precision,
                crossover_concurrency_or_rate=0.0,
                crossover_metric=metric,
                peak_speedup_at_batch1=1.0,
                high_load_penalty_pct=0.0,
                crossover_found=False,
            )

        # Calculate speedup at lowest rate (batch 1 regime)
        peak_speedup = comp_vals[0] / max(base_vals[0], 1e-5)

        # Look for sign change in (comp_vals - base_vals)
        crossover_rate = None
        for i in range(len(rates) - 1):
            diff_curr = comp_vals[i] - base_vals[i]
            diff_next = comp_vals[i + 1] - base_vals[i + 1]

            if diff_curr >= 0 and diff_next < 0:
                # Linear interpolation to estimate exact crossover point
                r1, r2 = rates[i], rates[i + 1]
                t = diff_curr / (diff_curr - diff_next)
                crossover_rate = r1 + t * (r2 - r1)
                break

        # Calculate high load penalty at highest rate
        high_penalty = 0.0
        if comp_vals[-1] < base_vals[-1]:
            high_penalty = ((base_vals[-1] - comp_vals[-1]) / base_vals[-1]) * 100.0

        return CrossoverPoint(
            compressed_precision=compressed_precision,
            baseline_precision=self.baseline_precision,
            crossover_concurrency_or_rate=float(crossover_rate) if crossover_rate else 0.0,
            crossover_metric=metric,
            peak_speedup_at_batch1=float(peak_speedup),
            high_load_penalty_pct=float(high_penalty),
            crossover_found=crossover_rate is not None,
        )
