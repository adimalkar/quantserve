"""Unit tests for GPU hardware probing and specification detection."""
import pytest
from quantserve.advisor.hardware_probe import HardwareProber, HARDWARE_PRESETS, HardwareProfile


def test_hardware_prober_presets():
    ada = HardwareProber.probe(override_preset="ada_4050")
    assert ada.total_vram_gb == 6.0
    assert ada.memory_bandwidth_gbs == 192.0
    assert ada.supports_fp8 is True
    assert ada.supports_sparse_tensor_cores is True

    a10 = HardwareProber.probe(override_preset="a10g")
    assert a10.total_vram_gb == 24.0
    assert a10.supports_fp8 is False  # Ampere has no native FP8
    assert a10.supports_sparse_tensor_cores is True

    h100 = HardwareProber.probe(override_preset="h100")
    assert h100.total_vram_gb == 80.0
    assert h100.memory_bandwidth_gbs == 3350.0
    assert h100.supports_fp8 is True


def test_hardware_prober_auto_detect():
    hw = HardwareProber.probe()
    assert isinstance(hw, HardwareProfile)
    assert hw.total_vram_gb > 0.0
    assert hw.memory_bandwidth_gbs > 0.0
    assert len(hw.supported_precisions) >= 2
