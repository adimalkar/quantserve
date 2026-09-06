"""Unit tests for Paged KV Cache simulator and memory fragmentation."""
import pytest
from src.internals.paged_cache_sim import PagedKVCacheSimulator


def test_paged_cache_allocation_and_fragmentation():
    # 256MB memory pool, block size = 16 tokens
    sim = PagedKVCacheSimulator(total_memory_mb=256.0, block_size_tokens=16, bytes_per_token=1024)

    # Allocate sequence with 20 tokens -> needs ceil(20 / 16) = 2 blocks = 32 slots
    ok = sim.allocate_sequence("seq_01", prompt_len=20)
    assert ok is True

    stats = sim.get_fragmentation_stats()
    assert stats.active_sequences == 1
    assert stats.active_tokens == 20
    assert stats.allocated_slots == 32
    # 12 wasted slots out of 32 = 37.5% internal fragmentation
    assert pytest.approx(stats.internal_fragmentation_pct, abs=0.1) == 37.5

    # Append tokens until 32
    for _ in range(12):
        sim.append_token("seq_01")

    stats = sim.get_fragmentation_stats()
    assert stats.active_tokens == 32
    assert stats.allocated_slots == 32
    assert stats.internal_fragmentation_pct == 0.0

    # Appending token 33 triggers allocation of 3rd block (16 more slots = 48 slots)
    sim.append_token("seq_01")
    stats = sim.get_fragmentation_stats()
    assert stats.active_tokens == 33
    assert stats.allocated_slots == 48

    # Free sequence returns all blocks
    sim.free_sequence("seq_01")
    stats = sim.get_fragmentation_stats()
    assert stats.active_sequences == 0
    assert stats.active_tokens == 0
    assert stats.allocated_slots == 0
