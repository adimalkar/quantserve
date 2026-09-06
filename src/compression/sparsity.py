"""NVIDIA Ampere/Ada 2:4 Semi-Structured Sparsity implementation and Sparse Tensor Core packing."""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredSparsity24:
    """Implements 2:4 semi-structured sparsity (2 non-zeros out of every 4 elements)."""

    @staticmethod
    def prune_to_2_4(tensor: torch.Tensor) -> torch.Tensor:
        """Enforces the 2:4 sparsity pattern along the last dimension:
        In every 4 contiguous elements, the 2 elements with smallest absolute magnitude are zeroed.
        """
        orig_shape = tensor.shape
        # Flatten except for blocks of 4
        assert orig_shape[-1] % 4 == 0, f"Last dimension {orig_shape[-1]} must be divisible by 4"

        reshaped = tensor.view(-1, 4)
        abs_vals = reshaped.abs()

        # Find top 2 indices in each block of 4
        _, top2_idx = torch.topk(abs_vals, k=2, dim=-1, largest=True, sorted=False)

        # Construct binary mask
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top2_idx, value=True)

        sparse_tensor = reshaped * mask
        return sparse_tensor.view(orig_shape)

    @staticmethod
    def verify_2_4_constraint(tensor: torch.Tensor) -> bool:
        """Verifies whether every 4-element block contains exactly 2 non-zero elements."""
        reshaped = tensor.view(-1, 4)
        non_zeros = (reshaped != 0).sum(dim=-1)
        # Note: If an original block had zeros, non_zeros could be <= 2, but never > 2
        return bool((non_zeros <= 2).all().item())

    @staticmethod
    def pack_dense_and_metadata(sparse_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packs a 2:4 sparse tensor into a dense 50% values matrix and a 2-bit metadata index tensor."""
        orig_shape = sparse_tensor.shape
        reshaped = sparse_tensor.view(-1, 4)
        abs_vals = reshaped.abs()

        # Sort top 2
        top2_vals, top2_idx = torch.topk(abs_vals, k=2, dim=-1, largest=True, sorted=True)

        # Get the actual signed values
        gathered_vals = torch.gather(reshaped, dim=-1, index=top2_idx)
        packed_vals = gathered_vals.view(orig_shape[0], orig_shape[1] // 2)

        # 2-bit indices packed into uint8
        # Each index is 0, 1, 2, or 3 (fits in 2 bits)
        idx_low = top2_idx[:, 0]
        idx_high = top2_idx[:, 1]
        metadata = (idx_low | (idx_high << 2)).to(torch.uint8).view(orig_shape[0], -1)

        return packed_vals, metadata


class Sparse24Linear(nn.Module):
    """Linear layer using 2:4 semi-structured sparsity."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert in_features % 4 == 0, "in_features must be divisible by 4"
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def prune_and_set(self, float_weight: torch.Tensor):
        pruned = StructuredSparsity24.prune_to_2_4(float_weight.to(torch.float16))
        self.weight.copy_(pruned)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
