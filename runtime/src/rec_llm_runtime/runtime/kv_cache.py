from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PagedKVCache:
    """
    Minimal paged KV cache for decode-only MVP.

    Layout matches the CUDA kernel contract:
      key_cache/value_cache: [num_blocks, num_heads, block_size, head_dim]
      block_tables:          [B, max_blocks] (int32 physical block ids)
      context_lens:          [B] (int32)
    """

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    block_size: int
    _next_block: int

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        num_blocks: int,
        max_blocks: int,
        num_heads: int,
        block_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PagedKVCache":
        key_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)
        value_cache = torch.zeros_like(key_cache)
        block_tables = torch.full((batch_size, max_blocks), -1, device=device, dtype=torch.int32)
        context_lens = torch.zeros((batch_size,), device=device, dtype=torch.int32)
        return cls(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
            _next_block=0,
        )

    @property
    def batch_size(self) -> int:
        return int(self.block_tables.size(0))

    @property
    def num_heads(self) -> int:
        return int(self.key_cache.size(1))

    @property
    def head_dim(self) -> int:
        return int(self.key_cache.size(3))

    def _ensure_block(self, b: int, logical_block: int) -> int:
        cur = int(self.block_tables[b, logical_block].item())
        if cur >= 0:
            return cur
        if self._next_block >= int(self.key_cache.size(0)):
            raise RuntimeError("Out of KV cache blocks (increase num_blocks).")
        block_id = self._next_block
        self._next_block += 1
        self.block_tables[b, logical_block] = block_id
        return block_id

    def append_kv(
        self,
        key: torch.Tensor,   # [B, H, D]
        value: torch.Tensor, # [B, H, D]
        *,
        reshape_and_cache_fn,
    ) -> None:
        if key.shape != value.shape:
            raise ValueError("key/value shape mismatch.")
        if key.dim() != 3:
            raise ValueError("key/value must have shape [B, H, D].")
        if key.size(0) != self.batch_size:
            raise ValueError("batch size mismatch.")

        bsz, h, d = key.shape
        if h != self.num_heads or d != self.head_dim:
            raise ValueError("head dims mismatch.")

        # One new token per sequence (decode-only).
        slots = []
        for b in range(bsz):
            ctx = int(self.context_lens[b].item())
            logical_block = ctx // self.block_size
            offset = ctx % self.block_size
            block_id = self._ensure_block(b, logical_block)
            slots.append(block_id * self.block_size + offset)

        slot_mapping = torch.tensor(slots, device=key.device, dtype=torch.int32)
        reshape_and_cache_fn(key, value, self.key_cache, self.value_cache, slot_mapping)
        self.context_lens += 1

