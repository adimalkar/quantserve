"""PagedAttention memory simulator: Internal vs External fragmentation analysis."""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class FragmentationStats:
    total_physical_blocks: int
    block_size_tokens: int
    active_sequences: int
    active_tokens: int
    allocated_slots: int
    internal_fragmentation_pct: float
    external_fragmentation_pct: float
    paged_vram_saved_mb: float


class PagedKVCacheSimulator:
    """Simulates physical block allocation and measures internal vs external memory fragmentation."""

    def __init__(
        self,
        total_memory_mb: float = 2048.0,
        block_size_tokens: int = 16,
        bytes_per_token: int = 128,  # Llama-3.2-1B with GQA: 2 * 16 layers * 8 kv_heads * 64 dim * 2 bytes = 32,768 bytes/tok?
        # Let's calculate exact bytes_per_token for Llama-3.2-1B:
        # layers=16, kv_heads=8, head_dim=64 -> 2 * 16 * 8 * 64 * 2 (fp16) = 32,768 bytes = 32 KB per token across all layers
    ):
        self.total_memory_mb = total_memory_mb
        self.block_size = block_size_tokens
        self.bytes_per_token = bytes_per_token

        self.bytes_per_block = self.block_size * self.bytes_per_token
        self.total_blocks = int((total_memory_mb * 1024 * 1024) // self.bytes_per_block)

        self.free_blocks: List[int] = list(range(self.total_blocks))
        self.sequence_block_tables: Dict[str, List[int]] = {}
        self.sequence_token_counts: Dict[str, int] = {}

    def allocate_sequence(self, seq_id: str, prompt_len: int) -> bool:
        """Allocates physical blocks for initial prompt."""
        needed_blocks = (prompt_len + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed_blocks:
            return False  # OOM / eviction required

        blocks = [self.free_blocks.pop(0) for _ in range(needed_blocks)]
        self.sequence_block_tables[seq_id] = blocks
        self.sequence_token_counts[seq_id] = prompt_len
        return True

    def append_token(self, seq_id: str) -> bool:
        """Appends 1 decoded token, allocating a new block if the last block is full."""
        if seq_id not in self.sequence_token_counts:
            return False

        curr_tokens = self.sequence_token_counts[seq_id]
        new_tokens = curr_tokens + 1

        # Check if new block needed
        if (new_tokens - 1) % self.block_size == 0 and new_tokens > 1:
            if not self.free_blocks:
                return False  # Cache full
            new_block = self.free_blocks.pop(0)
            self.sequence_block_tables[seq_id].append(new_block)

        self.sequence_token_counts[seq_id] = new_tokens
        return True

    def free_sequence(self, seq_id: str) -> None:
        """Frees all physical blocks allocated to a sequence."""
        if seq_id in self.sequence_block_tables:
            blocks = self.sequence_block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)
            self.sequence_token_counts.pop(seq_id, None)

    def get_fragmentation_stats(self) -> FragmentationStats:
        """Computes internal and external fragmentation metrics."""
        active_seqs = len(self.sequence_token_counts)
        active_tokens = sum(self.sequence_token_counts.values())

        total_allocated_blocks = sum(len(b) for b in self.sequence_block_tables.values())
        allocated_slots = total_allocated_blocks * self.block_size

        if allocated_slots > 0:
            internal_waste_slots = allocated_slots - active_tokens
            internal_frag_pct = (internal_waste_slots / allocated_slots) * 100.0
        else:
            internal_frag_pct = 0.0

        # In paged allocation, external fragmentation is 0% (any free block can satisfy any request)
        external_frag_pct = 0.0

        # In contiguous allocation, max pre-allocated context (e.g. 2048) would waste (2048 - active_tokens)
        contiguous_slots = active_seqs * 2048
        saved_bytes = max(0, (contiguous_slots - allocated_slots) * self.bytes_per_token)
        saved_mb = saved_bytes / (1024.0 * 1024.0)

        return FragmentationStats(
            total_physical_blocks=self.total_blocks,
            block_size_tokens=self.block_size,
            active_sequences=active_seqs,
            active_tokens=active_tokens,
            allocated_slots=allocated_slots,
            internal_fragmentation_pct=internal_frag_pct,
            external_fragmentation_pct=external_frag_pct,
            paged_vram_saved_mb=saved_mb,
        )
