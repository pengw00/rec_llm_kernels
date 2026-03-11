from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from rec_llm_runtime.ops.paged_cache import paged_attention_decode_v2
from rec_llm_runtime.ops.prefill_attention import flashinfer_prefill_attention
from rec_llm_runtime.runtime.kv_cache import PagedKVCache


def _build_packed_positions(cu_seqlens: torch.Tensor, base_context_lens: torch.Tensor) -> torch.Tensor:
    """
    Build packed RoPE positions for concatenated sequences.

    cu_seqlens: [B+1]
    base_context_lens: [B] int32/int64; offset added per sequence.
    Returns: [T] int64
    """
    if cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must have shape [B+1].")
    if base_context_lens.dim() != 1:
        raise ValueError("base_context_lens must have shape [B].")
    if cu_seqlens.numel() != base_context_lens.numel() + 1:
        raise ValueError("cu_seqlens must have length B+1.")

    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        cu_seqlens = cu_seqlens.to(dtype=torch.int32)

    B = int(base_context_lens.numel())
    seqlens = []
    cu_list = cu_seqlens.to(dtype=torch.int64).tolist()
    for b in range(B):
        seqlens.append(int(cu_list[b + 1] - cu_list[b]))
        if seqlens[-1] < 0:
            raise ValueError("cu_seqlens must be non-decreasing.")
    T = int(cu_list[-1])

    device = base_context_lens.device
    positions = torch.empty((T,), device=device, dtype=torch.int64)
    base = base_context_lens.to(dtype=torch.int64)
    for b in range(B):
        start = int(cu_list[b])
        end = int(cu_list[b + 1])
        if end == start:
            continue
        positions[start:end] = torch.arange(end - start, device=device, dtype=torch.int64) + base[b]
    return positions


def _get_reshape_and_cache_fn():
    try:
        import rec_llm_kernels._C as _C  # type: ignore

        return _C.ops.reshape_and_cache
    except Exception:
        from rec_llm_runtime.runtime.decode_mvp import _reshape_and_cache_torch

        return _reshape_and_cache_torch


@dataclass(frozen=True)
class LlamaAttentionConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_theta: float = 10000.0

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)


class LlamaAttention(nn.Module):
    """
    Minimal LLaMA attention module with:
      - Torch Linear for QKV / output projection
      - CUDA RoPE (`rec_llm_kernels.ops.apply_rope`) when available
      - FlashInfer prefill (packed tokens)
      - Paged KV cache + v2 paged decode (FlashInfer fast path with fallback)

    Note: This is no-GQA for now (num_kv_heads == num_heads).
    """

    def __init__(self, cfg: LlamaAttentionConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.hidden_size != cfg.num_heads * cfg.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim (no-GQA MVP).")

        self.wq = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from rec_llm_kernels.ops import apply_rope  # type: ignore

            return apply_rope(q, k, positions, base=self.cfg.rope_theta)
        except Exception:
            from rec_llm_runtime.ops.rope import apply_rope as torch_apply_rope

            return torch_apply_rope(q, k, positions, base=self.cfg.rope_theta)

    def prefill(self, x: torch.Tensor, cu_seqlens: torch.Tensor, cache: PagedKVCache) -> torch.Tensor:
        """
        Prefill attention over packed prompt tokens.

        x: [T, hidden]
        cu_seqlens: [B+1]
        cache: paged KV cache (will be updated in-place)
        Returns: [T, hidden] attention output projection (no residual add here)
        """
        if x.dim() != 2:
            raise ValueError("x must have shape [T, hidden].")
        T, hidden = x.shape
        if hidden != self.cfg.hidden_size:
            raise ValueError("hidden size mismatch.")

        q = self.wq(x).view(T, self.cfg.num_heads, self.cfg.head_dim)
        k = self.wk(x).view_as(q)
        v = self.wv(x).view_as(q)

        base_ctx = cache.context_lens
        if base_ctx.device != x.device:
            raise ValueError("cache must be on the same device as x.")
        positions = _build_packed_positions(cu_seqlens, base_ctx)
        q, k = self._apply_rope(q, k, positions)

        attn_out = flashinfer_prefill_attention(
            q, k, v, cu_seqlens, causal=True, kv_layout="NHD", sm_scale=self.cfg.scale, backend="auto"
        )

        reshape_and_cache_fn = _get_reshape_and_cache_fn()
        cache.prefill_kv(k, v, cu_seqlens, reshape_and_cache_fn=reshape_and_cache_fn)

        return self.wo(attn_out.reshape(T, -1))

    def decode(self, x: torch.Tensor, cache: PagedKVCache) -> torch.Tensor:
        """
        Decode attention for one token per sequence.

        x: [B, hidden]
        cache: paged KV cache (will be updated in-place)
        Returns: [B, hidden] attention output projection (no residual add here)
        """
        if x.dim() != 2:
            raise ValueError("x must have shape [B, hidden].")
        B, hidden = x.shape
        if hidden != self.cfg.hidden_size:
            raise ValueError("hidden size mismatch.")
        if B != cache.batch_size:
            raise ValueError("batch size mismatch with cache.")

        q = self.wq(x).view(B, self.cfg.num_heads, self.cfg.head_dim)
        k = self.wk(x).view_as(q)
        v = self.wv(x).view_as(q)

        # Position for the current token is the current context length.
        positions = cache.context_lens.to(dtype=torch.int64)
        q, k = self._apply_rope(q, k, positions)

        reshape_and_cache_fn = _get_reshape_and_cache_fn()
        cache.append_kv(k, v, reshape_and_cache_fn=reshape_and_cache_fn)

        attn_out = paged_attention_decode_v2(
            q, cache.key_cache, cache.value_cache, cache.block_tables, cache.context_lens, self.cfg.scale
        )
        return self.wo(attn_out.reshape(B, -1))

