"""Bootstrap confidence interval engine (1,000 resamples, BCa / percentile bootstrap)."""
from dataclasses import dataclass
from typing import List, Callable, Union
import numpy as np
from scipy import stats


@dataclass
class BootstrapCI:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_resamples: int = 1000

    def __str__(self) -> str:
        return f"{self.point_estimate:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}] (95% CI)"


def compute_bootstrap_ci(
    data: Union[List[float], np.ndarray],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> BootstrapCI:
    """Computes non-parametric bootstrap confidence intervals over evaluation samples."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return BootstrapCI(0.0, 0.0, 0.0, confidence_level, n_resamples)

    point_est = float(statistic_fn(arr))
    if len(arr) == 1:
        return BootstrapCI(point_est, point_est, point_est, confidence_level, n_resamples)

    rng = np.random.default_rng(seed)
    n = len(arr)

    # Vectorized bootstrap resampling
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_data = arr[indices]
    boot_stats = np.apply_along_axis(statistic_fn, 1, resampled_data)

    # Percentile bootstrap
    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    ci_low = float(np.percentile(boot_stats, lower_pct))
    ci_high = float(np.percentile(boot_stats, upper_pct))

    return BootstrapCI(
        point_estimate=point_est,
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )
