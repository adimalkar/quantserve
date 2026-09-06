"""Advisor module: Hardware probing, config space, surrogate modeling, and recommendation."""
from quantserve.advisor.hardware_probe import HardwareProfile, HardwareProber, HARDWARE_PRESETS
from quantserve.advisor.config_space import ConfigurationSpace, DeploymentConfig
from quantserve.advisor.surrogate import PerformanceSurrogate, SurrogatePrediction
from quantserve.advisor.predictor import PerformancePredictor, CalibratedPrediction, Interval
from quantserve.advisor.recommendation import RecommendationResult
from quantserve.advisor.optimizer import DeploymentOptimizer

__all__ = [
    "HardwareProfile",
    "HardwareProber",
    "HARDWARE_PRESETS",
    "ConfigurationSpace",
    "DeploymentConfig",
    "PerformanceSurrogate",
    "SurrogatePrediction",
    "PerformancePredictor",
    "CalibratedPrediction",
    "Interval",
    "RecommendationResult",
    "DeploymentOptimizer",
]
