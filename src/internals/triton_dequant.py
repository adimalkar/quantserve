"""Triton Fused W4A16 Dequantization & Matrix-Vector Multiplication (GEMV) Kernel."""
from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return TRITON_AVAILABLE and torch.cuda.is_available()


if TRITON_AVAILABLE:

    @triton.jit
    def _w4a16_gemv_kernel(
        y_ptr,           # Output vector [M]
        x_ptr,           # Input activation vector [K]
        w_ptr,           # Packed weights uint8 [M, K // 2]
        scales_ptr,      # Scales float16 [M, num_groups]
        zeros_ptr,       # Zeros float16 [M, num_groups]
        M: tl.constexpr, # Rows
        K: tl.constexpr, # Columns
        GROUP_SIZE: tl.constexpr,
        BLOCK_K: tl.constexpr, # Elements of K per tile (e.g. 256)
    ):
        # Program ID corresponds to row index m
        m = tl.program_id(axis=0)
        if m >= M:
            return

        acc = 0.0

        # Loop over K in tiles of BLOCK_K
        for k_start in range(0, K, BLOCK_K):
            # Tile offsets
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K

            # Load x elements
            x_tile = tl.load(x_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)

            # Packed weights: BLOCK_K // 2 bytes
            packed_offsets = (m * (K // 2)) + (k_start // 2) + tl.arange(0, BLOCK_K // 2)
            packed_mask = (k_start // 2 + tl.arange(0, BLOCK_K // 2)) < (K // 2)
            packed_bytes = tl.load(w_ptr + packed_offsets, mask=packed_mask, other=0).to(tl.int32)

            # Unpack low and high nibbles
            low_nibble = (packed_bytes & 0x0F).to(tl.float32)
            high_nibble = ((packed_bytes >> 4) & 0x0F).to(tl.float32)

            # Group index for scales and zeros
            g_idx = k_offsets // GROUP_SIZE
            scale_offsets = m * (K // GROUP_SIZE) + (k_start // GROUP_SIZE)
            scale = tl.load(scales_ptr + scale_offsets).to(tl.float32)
            zero = tl.load(zeros_ptr + scale_offsets).to(tl.float32)

            # Dequantize: (nibble - zero) * scale
            w_dequant_low = (low_nibble - zero) * scale
            w_dequant_high = (high_nibble - zero) * scale

            # Slice corresponding x elements
            x_low = tl.load(x_ptr + k_start + tl.arange(0, BLOCK_K // 2) * 2, mask=packed_mask, other=0.0).to(tl.float32)
            x_high = tl.load(x_ptr + k_start + tl.arange(0, BLOCK_K // 2) * 2 + 1, mask=packed_mask, other=0.0).to(tl.float32)

            acc += tl.sum(w_dequant_low * x_low + w_dequant_high * x_high)

        # Store result in output vector
        tl.store(y_ptr + m, acc.to(tl.float16))


def triton_fused_w4a16_gemv(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    x: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dispatches the custom Triton fused W4A16 GEMV kernel.

    Args:
        qweight: Packed uint8 weights [M, K // 2]
        scales: Scale factors [M, num_groups]
        zeros: Zero points [M, num_groups]
        x: Input vector [K] or batch [1, K]
        group_size: Quantization group size (default 128)
    """
    if not is_triton_available():
        # Fallback to PyTorch unpack
        return _pytorch_unfused_w4a16_gemv(qweight, scales, zeros, x, group_size)

    x_flat = x.view(-1).contiguous()
    M, half_K = qweight.shape
    K = half_K * 2

    assert x_flat.shape[0] == K, f"Dimension mismatch: x is {x_flat.shape[0]}, expected {K}"
    assert x_flat.dtype == torch.float16, "x must be float16"

    y = torch.empty(M, device=x.device, dtype=torch.float16)

    BLOCK_K = 128
    grid = (M,)

    _w4a16_gemv_kernel[grid](
        y,
        x_flat,
        qweight,
        scales,
        zeros,
        M=M,
        K=K,
        GROUP_SIZE=group_size,
        BLOCK_K=BLOCK_K,
    )

    return y.view(*x.shape[:-1], M)


def _pytorch_unfused_w4a16_gemv(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    x: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reference unfused PyTorch implementation for CPU or validation."""
    M, half_K = qweight.shape
    q_low = qweight & 0x0F
    q_high = (qweight >> 4) & 0x0F

    unpacked = torch.empty((M, half_K * 2), dtype=torch.float16, device=x.device)
    unpacked[:, 0::2] = q_low.to(torch.float16)
    unpacked[:, 1::2] = q_high.to(torch.float16)

    num_groups = scales.shape[1]
    unpacked_r = unpacked.view(M, num_groups, group_size)
    dequant = ((unpacked_r - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)).view(M, -1)

    return torch.matmul(x.to(torch.float16), dequant.t())
