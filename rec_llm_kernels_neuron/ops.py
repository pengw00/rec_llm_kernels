from __future__ import annotations

import math
from typing import Optional

import torch


def _sync_if_xla() -> None:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        xm.mark_step()
    except Exception:
        return


def rms_norm(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    x_f = x.to(torch.float32)
    w_f = weight.to(torch.float32)
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps) * w_f
    out.copy_(y.to(dtype=out.dtype))
    _sync_if_xla()


def reshape_and_cache(
    key: torch.Tensor,  # [T, H, D]
    value: torch.Tensor,  # [T, H, D]
    key_cache: torch.Tensor,  # [NB, H, BS, D]
    value_cache: torch.Tensor,  # [NB, H, BS, D]
    slot_mapping: torch.Tensor,  # [T] int32
) -> None:
    nb, h, bs, d = key_cache.shape
    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    if valid.sum().item() == 0:
        return

    slots = slots[valid]
    k_flat = key_cache.reshape(nb * bs, h, d)
    v_flat = value_cache.reshape(nb * bs, h, d)

    k_flat.index_copy_(0, slots, key[valid])
    v_flat.index_copy_(0, slots, value[valid])
    _sync_if_xla()


def paged_attention_decode(
    query: torch.Tensor,  # [B, H, D]
    key_cache: torch.Tensor,  # [NB, H, BS, D]
    value_cache: torch.Tensor,  # [NB, H, BS, D]
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    context_lens: torch.Tensor,  # [B] int32
    scale: Optional[float] = None,
) -> torch.Tensor:
    bsz, nheads, dim = query.shape
    _, _, bs, _ = key_cache.shape

    s = float(scale) if scale is not None else 1.0 / math.sqrt(dim)

    max_ctx = int(context_lens.max().item()) if context_lens.numel() else 0
    if max_ctx == 0:
        return torch.zeros_like(query)

    pos = torch.arange(max_ctx, device=query.device, dtype=torch.int32).unsqueeze(0).expand(bsz, -1)
    mask = pos < context_lens.view(-1, 1)

    bt_idx = torch.div(pos, bs, rounding_mode="floor")  # [B, max_ctx]
    off = pos - bt_idx * bs  # [B, max_ctx]

    block_ids = block_tables.gather(1, bt_idx.to(block_tables.dtype))  # [B, max_ctx]

    b_ix = torch.arange(bsz, device=query.device).view(bsz, 1, 1)
    h_ix = torch.arange(nheads, device=query.device).view(1, nheads, 1)

    block_ids_b = block_ids.view(bsz, 1, max_ctx).expand(-1, nheads, -1)
    off_b = off.view(bsz, 1, max_ctx).expand(-1, nheads, -1)

    K = key_cache[block_ids_b, h_ix, off_b]  # [B, H, max_ctx, D]
    V = value_cache[block_ids_b, h_ix, off_b]

    q = query.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)

    logits = (Kf * q.unsqueeze(2)).sum(dim=-1) * s
    logits = logits.masked_fill(~mask.view(bsz, 1, max_ctx), float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, Vf).to(dtype=query.dtype)
    _sync_if_xla()
    return out

