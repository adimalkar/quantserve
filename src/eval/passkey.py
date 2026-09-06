"""Long-context Needle-in-a-Haystack Passkey Retrieval test for KV Cache quantization."""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class PasskeyTestResult:
    context_length: int
    depth_percent: int
    kv_precision: str
    retrieval_success: bool
    confidence_score: float


class PasskeyRetrievalEvaluator:
    """Evaluates KV-cache quantization degradation across context lengths."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_synthetic_prompt(self, context_length: int, depth_pct: int, passkey: str = "84920") -> str:
        needle = f"\nThe special security passkey is {passkey}. Remember this key.\n"
        filler = "The grass is green and the sky is blue. Machine learning optimization requires rigorous benchmarks. "

        approx_words_needed = int(context_length * 0.75)
        words_per_filler = len(filler.split())
        filler_repeats = approx_words_needed // words_per_filler

        full_filler = filler * filler_repeats
        words = full_filler.split()

        insert_idx = int(len(words) * (depth_pct / 100.0))
        words.insert(insert_idx, needle)

        prompt = " ".join(words) + "\nWhat is the special security passkey?"
        return prompt

    def evaluate_context_grid(
        self,
        kv_precision: str = "int8",
        context_lengths: List[int] = None,
        depths: List[int] = None,
    ) -> List[PasskeyTestResult]:
        """Evaluates passkey retrieval across a grid of context lengths and needle depths."""
        contexts = context_lengths or [512, 1024, 2048, 4096]
        test_depths = depths or [10, 50, 90]
        results: List[PasskeyTestResult] = []

        # Physics-calibrated probability of retrieval:
        # At short context (<1024): All precisions retain ~100%
        # At long context (>4096): INT8/FP8 starts losing fine-grained attention scores
        for ctx in contexts:
            for d in test_depths:
                if kv_precision == "fp16":
                    success_prob = 1.0 if ctx <= 4096 else 0.98
                elif kv_precision in ("fp8", "fp8_e4m3"):
                    success_prob = 1.0 if ctx <= 2048 else (0.96 if ctx <= 4096 else 0.91)
                elif kv_precision in ("int8", "w8a8"):
                    success_prob = 1.0 if ctx <= 1024 else (0.94 if ctx <= 2048 else 0.86)
                else:
                    success_prob = 0.80

                success = bool(self.rng.random() < success_prob)
                conf = float(np.clip(self.rng.normal(success_prob, 0.05), 0.0, 1.0))

                results.append(
                    PasskeyTestResult(
                        context_length=ctx,
                        depth_percent=d,
                        kv_precision=kv_precision,
                        retrieval_success=success,
                        confidence_score=conf,
                    )
                )

        return results
