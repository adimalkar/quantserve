"""Machine learning surrogate performance models for TTFT, TPOT, Throughput, and VRAM."""
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class SurrogatePrediction:
    ttft_ms: float
    tpot_ms: float
    throughput_tokens_s: float
    peak_vram_gb: float
    slo_compliance_prob: float


class PerformanceSurrogate:
    """Predicts deployment performance using learned regression ensembles."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model_ttft = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model_tpot = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model_throughput = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model_vram = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_fitted = False

    def _extract_feature_vector(
        self,
        config: Dict[str, Any],
        hardware: Dict[str, Any],
        workload: Dict[str, Any],
        model_meta: Dict[str, Any],
    ) -> np.ndarray:
        """Converts deployment, hardware, workload, and model metadata into a numerical vector."""
        # 1. Model specs
        param_b = float(model_meta.get("params_billion", 1.23))
        layers = float(model_meta.get("num_layers", 16))
        hidden_dim = float(model_meta.get("hidden_size", 2048))
        kv_heads = float(model_meta.get("num_kv_heads", 8))
        q_heads = float(model_meta.get("num_heads", 32))
        gqa_ratio = q_heads / max(kv_heads, 1.0)

        # 2. Compression specs
        q_str = str(config.get("quantization", "fp16")).lower()
        w_bits = 4.0 if "int4" in q_str else (8.0 if ("int8" in q_str or "fp8" in q_str) else 16.0)
        kv_str = str(config.get("kv_cache_precision", "fp16")).lower()
        kv_bits = 8.0 if ("int8" in kv_str or "fp8" in kv_str) else 16.0
        is_awq = 1.0 if "awq" in q_str else 0.0
        is_smoothquant = 1.0 if "smooth" in q_str else 0.0

        # 3. Hardware specs
        bw_gbs = float(hardware.get("memory_bandwidth_gbs", 192.0))
        tflops = float(hardware.get("fp16_tflops", 36.0))
        vram_gb = float(hardware.get("total_vram_gb", 6.0))

        # 4. Runtime options
        max_seqs = float(config.get("max_num_seqs", 16))
        max_tokens = float(config.get("max_num_batched_tokens", 2048))
        chunked_prefill = 1.0 if config.get("chunked_prefill", True) else 0.0

        # 5. Workload specs
        avg_prompt = float(workload.get("avg_prompt_tokens", 256))
        avg_gen = float(workload.get("avg_output_tokens", 64))
        concurrency = float(workload.get("concurrency", 8))
        rate = float(workload.get("request_rate", 4.0))

        return np.array([
            param_b, layers, hidden_dim, kv_heads, gqa_ratio,
            w_bits, kv_bits, is_awq, is_smoothquant,
            bw_gbs, tflops, vram_gb,
            max_seqs, max_tokens, chunked_prefill,
            avg_prompt, avg_gen, concurrency, rate,
        ], dtype=np.float32)

    def fit(self, X: np.ndarray, y_ttft: np.ndarray, y_tpot: np.ndarray, y_thru: np.ndarray, y_vram: np.ndarray):
        """Fits the regression models on training telemetry."""
        X_scaled = self.scaler.fit_transform(X)
        self.model_ttft.fit(X_scaled, y_ttft)
        self.model_tpot.fit(X_scaled, y_tpot)
        self.model_throughput.fit(X_scaled, y_thru)
        self.model_vram.fit(X_scaled, y_vram)
        self.is_fitted = True

    def pretrain_on_synthetic_physics(self, samples: int = 500):
        """Pre-trains on calibrated roofline and memory dynamics when no prior real telemetry exists."""
        rng = np.random.default_rng(42)
        X_list, y_ttft, y_tpot, y_thru, y_vram = [], [], [], [], []

        for _ in range(samples):
            param_b = rng.choice([0.5, 1.23, 1.5, 3.21])
            layers = 16 if param_b < 2.0 else 28
            hidden_dim = 2048 if param_b < 2.0 else 3072
            kv_heads = 8
            q_heads = 32
            gqa_ratio = 4.0

            w_bits = rng.choice([4.0, 8.0, 16.0])
            kv_bits = rng.choice([8.0, 16.0])
            is_awq = 1.0 if w_bits == 4.0 else 0.0
            is_smoothquant = 1.0 if w_bits == 8.0 else 0.0

            bw_gbs = rng.choice([192.0, 300.0, 600.0, 1008.0])
            tflops = (bw_gbs / 192.0) * 36.0
            vram_gb = 6.0 if bw_gbs <= 192.0 else 24.0

            max_seqs = rng.choice([4, 8, 16, 32, 64])
            max_tokens = rng.choice([512, 1024, 2048, 4096])
            chunked_prefill = rng.choice([0.0, 1.0])

            avg_prompt = rng.uniform(32, 2048)
            avg_gen = rng.uniform(16, 512)
            concurrency = min(max_seqs, rng.choice([1, 2, 4, 8, 16, 32, 64]))
            rate = concurrency * (1.0 / (avg_gen * 0.015))

            feat = np.array([
                param_b, layers, hidden_dim, kv_heads, gqa_ratio,
                w_bits, kv_bits, is_awq, is_smoothquant,
                bw_gbs, tflops, vram_gb,
                max_seqs, max_tokens, chunked_prefill,
                avg_prompt, avg_gen, concurrency, rate,
            ], dtype=np.float32)

            # Realistic physics targets
            b_per_p = w_bits / 8.0
            weight_bytes = param_b * 1e9 * b_per_p
            # TTFT (Prefill): Compute dominated
            flops_prefill = 2.0 * param_b * 1e9 * avg_prompt
            t_prefill = (flops_prefill / (tflops * 1e12)) * 1000.0 + (weight_bytes / (bw_gbs * 1e9)) * 1000.0 * 0.1
            t_prefill += rng.normal(5.0, 1.0)

            # TPOT (Decode): Memory-bound at low concurrency, compute-bound at high
            flops_dec = 2.0 * param_b * 1e9 * concurrency
            t_comp_dec = (flops_dec / (tflops * 1e12)) * 1000.0
            kv_bytes = concurrency * (avg_prompt + avg_gen) * layers * 2 * (hidden_dim // 4) * (kv_bits / 8.0)
            t_mem_dec = ((weight_bytes + kv_bytes) / (bw_gbs * 1e9)) * 1000.0
            tpot = max(t_mem_dec / concurrency, t_comp_dec)
            if w_bits == 4.0 and concurrency >= 32:
                tpot *= 1.25  # Dequant overhead at high batch

            tpot += rng.normal(1.0, 0.2)

            thru = (concurrency / max(tpot, 1.0)) * 1000.0
            vram = (weight_bytes / (1024**3)) + (kv_bytes / (1024**3)) + 0.5

            X_list.append(feat)
            y_ttft.append(max(t_prefill, 2.0))
            y_tpot.append(max(tpot, 1.0))
            y_thru.append(max(thru, 5.0))
            y_vram.append(vram)

        self.fit(np.array(X_list), np.array(y_ttft), np.array(y_tpot), np.array(y_thru), np.array(y_vram))

    def predict(
        self,
        config: Dict[str, Any],
        hardware: Dict[str, Any],
        workload: Dict[str, Any],
        model_meta: Dict[str, Any],
    ) -> SurrogatePrediction:
        """Predicts performance metrics for a candidate configuration."""
        if not self.is_fitted:
            self.pretrain_on_synthetic_physics()

        vec = self._extract_feature_vector(config, hardware, workload, model_meta).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)

        ttft = float(self.model_ttft.predict(vec_scaled)[0])
        tpot = float(self.model_tpot.predict(vec_scaled)[0])
        thru = float(self.model_throughput.predict(vec_scaled)[0])
        vram = float(self.model_vram.predict(vec_scaled)[0])

        # Heuristic SLO compliance probability based on latency margins
        slo_ttft = float(workload.get("slo_ttft_ms", 500.0))
        slo_tpot = float(workload.get("slo_tpot_ms", 40.0))
        prob_ttft = 1.0 if ttft <= slo_ttft else max(0.0, 1.0 - (ttft - slo_ttft) / slo_ttft)
        prob_tpot = 1.0 if tpot <= slo_tpot else max(0.0, 1.0 - (tpot - slo_tpot) / slo_tpot)
        slo_prob = prob_ttft * prob_tpot

        return SurrogatePrediction(
            ttft_ms=ttft,
            tpot_ms=tpot,
            throughput_tokens_s=thru,
            peak_vram_gb=vram,
            slo_compliance_prob=slo_prob,
        )
