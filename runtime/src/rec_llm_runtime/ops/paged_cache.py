from __future__ import annotations

import torch


def _ops():
    import rec_llm_kernels._C as _C  # type: ignore

    return _C.ops


def reshape_and_cache(
    key: torch.Tensor,        # [num_tokens, num_heads, head_dim]
    value: torch.Tensor,      # [num_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_dim]
    value_cache: torch.Tensor,# [num_blocks, num_heads, block_size, head_dim]
    slot_mapping: torch.Tensor # [num_tokens] global physical slot indices
) -> None:
    """
    Write per-token K/V into a paged KV cache.
    """
    return _ops().reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)


def paged_attention_decode(
    query: torch.Tensor,        # [B, H, D]
    key_cache: torch.Tensor,    # [NB, H, BS, D]
    value_cache: torch.Tensor,  # [NB, H, BS, D]
    block_tables: torch.Tensor, # [B, max_blocks] int32 physical block ids
    context_lens: torch.Tensor, # [B] int32
    scale: float,
) -> torch.Tensor:
    """
    Decode-only paged attention (MVP).
    """
    return _ops().paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, float(scale))

