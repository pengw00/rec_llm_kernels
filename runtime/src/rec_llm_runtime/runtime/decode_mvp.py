from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from rec_llm_runtime.ops.rope import apply_rope
from rec_llm_runtime.runtime.kv_cache import PagedKVCache


def _get_cuda_ops():
    try:
        import rec_llm_kernels._C as _C  # type: ignore

        return _C.ops
    except Exception:
        return None


def _torch_paged_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Vectorized reference implementation (padding+mask).
    bsz, nheads, dim = query.shape
    _, _, bs, _ = key_cache.shape

    max_ctx = int(context_lens.max().item()) if context_lens.numel() else 0
    if max_ctx == 0:
        return torch.zeros_like(query)

    pos = torch.arange(max_ctx, device=query.device, dtype=torch.int64).unsqueeze(0).expand(bsz, -1)
    mask = pos < context_lens.to(torch.int64).view(-1, 1)

    bt_idx = torch.div(pos, bs, rounding_mode="floor")
    off = pos - bt_idx * bs
    block_ids = block_tables.to(torch.int64).gather(1, bt_idx.to(torch.int64))

    h_ix = torch.arange(nheads, device=query.device, dtype=torch.int64).view(1, nheads, 1)
    block_ids_b = block_ids.view(bsz, 1, max_ctx).expand(-1, nheads, -1)
    off_b = off.view(bsz, 1, max_ctx).expand(-1, nheads, -1)

    K = key_cache[block_ids_b, h_ix, off_b]  # [B,H,T,D]
    V = value_cache[block_ids_b, h_ix, off_b]

    q = query.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)

    logits = (Kf * q.unsqueeze(2)).sum(dim=-1) * float(scale)  # [B,H,T]
    logits = logits.masked_fill(~mask.view(bsz, 1, max_ctx), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs.unsqueeze(-2), Vf).squeeze(-2)
    return out.to(dtype=query.dtype)


@dataclass(frozen=True)
class DecodeMVPConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_base: float = 10000.0

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)


class DecodeOnlyMVP(nn.Module):
    """
    Minimal decode-only block:
      RMSNorm -> QKV -> RoPE(q,k) -> append KV -> paged attention decode -> out proj -> residual

    Notes:
    - No MLP, no multi-layer stack.
    - Intended as a correctness/infra MVP.
    """

    def __init__(self, cfg: DecodeMVPConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.hidden_size, elementwise_affine=False)
        self.wq = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: PagedKVCache) -> torch.Tensor:
        bsz, hidden = x.shape
        if hidden != self.cfg.hidden_size:
            raise ValueError("hidden_size mismatch.")
        if positions.shape != (bsz,):
            raise ValueError("positions must be [B].")

        residual = x
        x = self.norm(x)

        q = self.wq(x).view(bsz, self.cfg.num_heads, self.cfg.head_dim)
        k = self.wk(x).view(bsz, self.cfg.num_heads, self.cfg.head_dim)
        v = self.wv(x).view(bsz, self.cfg.num_heads, self.cfg.head_dim)

        q, k = apply_rope(q, k, positions, base=self.cfg.rope_base)

        cuda_ops = _get_cuda_ops()
        if cuda_ops is not None and x.is_cuda:
            cache.append_kv(k, v, reshape_and_cache_fn=cuda_ops.reshape_and_cache)
            attn_out = cuda_ops.paged_attention_decode(
                q, cache.key_cache, cache.value_cache, cache.block_tables, cache.context_lens, self.cfg.scale
            )
        else:
            cache.append_kv(k, v, reshape_and_cache_fn=lambda kk, vv, kc, vc, sm: _reshape_and_cache_torch(kk, vv, kc, vc, sm))
            attn_out = _torch_paged_attention_decode(
                q, cache.key_cache, cache.value_cache, cache.block_tables, cache.context_lens, self.cfg.scale
            )

        y = self.wo(attn_out.reshape(bsz, -1))
        return residual + y


def _reshape_and_cache_torch(key, value, key_cache, value_cache, slot_mapping):
    # torch reference, slot semantics match CUDA kernel: slot = block_id*BS + off
    nb, h, bs, d = key_cache.shape
    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    if valid.sum().item() == 0:
        return
    slots = slots[valid]
    k_flat = key_cache.permute(0, 2, 1, 3).reshape(nb * bs, h, d)
    v_flat = value_cache.permute(0, 2, 1, 3).reshape(nb * bs, h, d)
    idx = slots.view(-1, 1, 1).expand(-1, h, d)
    k_flat = k_flat.scatter(0, idx, key[valid])
    v_flat = v_flat.scatter(0, idx, value[valid])
    key_cache.copy_(k_flat.reshape(nb, bs, h, d).permute(0, 2, 1, 3))
    value_cache.copy_(v_flat.reshape(nb, bs, h, d).permute(0, 2, 1, 3))
