"""CLI command for GPU hardware inspection and probing."""
import argparse
import json
from quantserve.advisor.hardware_probe import HardwareProber


def run_hardware_cmd(args):
    profile = HardwareProber.probe(override_preset=args.preset)

    if args.json:
        print(json.dumps(profile.to_dict(), indent=2))
        return

    print("\n" + "=" * 54)
    print("           QuantServe: Hardware Inspection")
    print("=" * 54)
    print(f"Device Name:          {profile.name}")
    print(f"Architecture:         {profile.arch}")
    print(f"Compute Capability:   {profile.compute_capability}")
    print(f"Total VRAM:           {profile.total_vram_gb:.2f} GB")
    print(f"Memory Bandwidth:     {profile.memory_bandwidth_gbs:.1f} GB/s")
    print(f"Peak FP16 Compute:    {profile.fp16_tflops:.1f} TFLOPS")
    print(f"Peak FP8 Compute:     {profile.fp8_tflops:.1f} TFLOPS")
    print(f"Peak INT8 Compute:    {profile.int8_tops:.1f} TOPS")
    print(f"Sparse Tensor Cores:  {'Supported (2:4)' if profile.supports_sparse_tensor_cores else 'No'}")
    print(f"Native FP8 Support:   {'Yes (Ada / Hopper+)' if profile.supports_fp8 else 'No'}")
    print(f"Supported Precisions: {', '.join(p.upper() for p in profile.supported_precisions)}")
    print("=" * 54 + "\n")
