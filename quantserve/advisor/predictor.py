"""Uncertainty estimation and 90% prediction intervals for performance forecasts."""
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from quantserve.advisor.surrogate import PerformanceSurrogate


@dataclass
class Interval:
    point: float
    lower_90: float
    upper_90: float

    def __str__(self) -> str:
        return f"{self.point:.1f} (90% PI: {self.lower_90:.1f}–{self.upper_90:.1f})"


@dataclass
class CalibratedPrediction:
    ttft: Interval
    tpot: Interval
    throughput: Interval
    peak_vram_gb: float
    slo_probability_pct: float


class PerformancePredictor:
    """Provides calibrated predictions with 90% prediction intervals across all target metrics."""

    def __init__(self, surrogate: PerformanceSurrogate):
        self.surrogate = surrogate

    def predict_with_uncertainty(
        self,
        config: Dict[str, Any],
        hardware: Dict[str, Any],
        workload: Dict[str, Any],
        model_meta: Dict[str, Any],
    ) -> CalibratedPrediction:
        """Predicts for a single configuration."""
        results = self.predict_batch_with_uncertainty([config], hardware, workload, model_meta)
        return results[0]

    def predict_batch_with_uncertainty(
        self,
        configs: List[Dict[str, Any]],
        hardware: Dict[str, Any],
        workload: Dict[str, Any],
        model_meta: Dict[str, Any],
    ) -> List[CalibratedPrediction]:
        """Vectorized batch prediction across all candidate configs in a single pass."""
        if not self.surrogate.is_fitted:
            self.surrogate.pretrain_on_synthetic_physics(samples=150)

        if not configs:
            return []

        # Vectorize feature extraction
        X_raw = np.array([
            self.surrogate._extract_feature_vector(c, hardware, workload, model_meta)
            for c in configs
        ], dtype=np.float32)

        X_scaled = self.surrogate.scaler.transform(X_raw)

        # Vectorized tree predictions across the entire batch (N_trees, N_candidates)
        # 50 calls instead of N_candidates * 50 calls!
        ttft_tree_preds = np.array([tree.predict(X_scaled) for tree in self.surrogate.model_ttft.estimators_])
        tpot_tree_preds = np.array([tree.predict(X_scaled) for tree in self.surrogate.model_tpot.estimators_])
        thru_tree_preds = np.array([tree.predict(X_scaled) for tree in self.surrogate.model_throughput.estimators_])
        vram_preds = self.surrogate.model_vram.predict(X_scaled)

        slo_ttft = float(workload.get("slo_ttft_ms", 500.0))
        slo_tpot = float(workload.get("slo_tpot_ms", 40.0))

        results: List[CalibratedPrediction] = []
        n_candidates = len(configs)

        for i in range(n_candidates):
            t_preds = ttft_tree_preds[:, i]
            p_preds = tpot_tree_preds[:, i]
            th_preds = thru_tree_preds[:, i]

            def _get_interval(preds: np.ndarray) -> Interval:
                point = float(np.mean(preds))
                low = float(np.percentile(preds, 5))
                high = float(np.percentile(preds, 95))
                low = max(low, point * 0.7)
                high = max(high, point * 1.3)
                return Interval(point=point, lower_90=low, upper_90=high)

            ttft_inv = _get_interval(t_preds)
            tpot_inv = _get_interval(p_preds)
            thru_inv = _get_interval(th_preds)

            prob_ttft = float(np.mean(t_preds <= slo_ttft))
            prob_tpot = float(np.mean(p_preds <= slo_tpot))
            slo_prob = (prob_ttft * prob_tpot) * 100.0

            results.append(
                CalibratedPrediction(
                    ttft=ttft_inv,
                    tpot=tpot_inv,
                    throughput=thru_inv,
                    peak_vram_gb=round(float(vram_preds[i]), 2),
                    slo_probability_pct=round(slo_prob, 1),
                )
            )

        return results
