from __future__ import annotations

import math
from typing import Optional

import torch


def _sync_if_xla() -> None:
    try:
        import torch_xla  # type: ignore

        # Newer torch-xla recommends `torch_xla.sync()` over `xm.mark_step()`.
        if hasattr(torch_xla, "sync"):
            torch_xla.sync()
        else:
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
    # Layout note:
    # key_cache/value_cache are [NB, H, BS, D]. The logical "slot index" is
    # (block_id * BS + offset). To make dim0 align with that slot index, we need
    # to move BS next to NB before flattening.
    k_flat = key_cache.permute(0, 2, 1, 3).reshape(nb * bs, h, d)
    v_flat = value_cache.permute(0, 2, 1, 3).reshape(nb * bs, h, d)

    # XLA/Neuron is often more reliable with `scatter` than in-place `index_copy_`.
    idx = slots.view(-1, 1, 1).expand(-1, h, d)
    k_flat = k_flat.scatter(0, idx, key[valid])
    v_flat = v_flat.scatter(0, idx, value[valid])

    key_cache.copy_(k_flat.reshape(nb, bs, h, d).permute(0, 2, 1, 3))
    value_cache.copy_(v_flat.reshape(nb, bs, h, d).permute(0, 2, 1, 3))
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

    # XLA advanced indexing requires all index tensors to have the same integer dtype.
    h_ix = torch.arange(nheads, device=query.device, dtype=torch.int64).view(1, nheads, 1)
    block_ids_b = block_ids.to(torch.int64).view(bsz, 1, max_ctx).expand(-1, nheads, -1)
    off_b = off.to(torch.int64).view(bsz, 1, max_ctx).expand(-1, nheads, -1)

    K = key_cache[block_ids_b, h_ix, off_b]  # [B, H, max_ctx, D]
    V = value_cache[block_ids_b, h_ix, off_b]

    q = query.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)

    logits = (Kf * q.unsqueeze(2)).sum(dim=-1) * s
    logits = logits.masked_fill(~mask.view(bsz, 1, max_ctx), float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    # Weighted sum over the context length dimension.
    # probs: [B, H, T], V: [B, H, T, D] -> out: [B, H, D]
    out = torch.matmul(probs.unsqueeze(-2), Vf).squeeze(-2).to(dtype=query.dtype)
    _sync_if_xla()
    return out


def flash_att_forward(
    q: torch.Tensor,  # [B, H, Tq, D] or [B, H, D] (treated as Tq=1)
    k: torch.Tensor,  # [B, H, Tk, D] or [B, H, D] (treated as Tk=1)
    v: torch.Tensor,  # [B, H, Tk, D] or [B, H, D] (treated as Tk=1)
    out: torch.Tensor,  # same shape as q
    *,
    scale: Optional[float] = None,
    causal: bool = False,
) -> None:
    """
    Reference attention implementation intended for Neuron/XLA compilation.

    This is not a hand-written NKI kernel. It is expressed as PyTorch ops so it
    can be lowered by torch-xla and compiled by Neuron.
    """
    if q.dim() == 3:
        q_ = q.unsqueeze(2)
        k_ = k.unsqueeze(2)
        v_ = v.unsqueeze(2)
        squeeze_t = True
    else:
        q_, k_, v_ = q, k, v
        squeeze_t = False

    bsz, nheads, tq, dim = q_.shape
    tk = k_.shape[2]

    s = float(scale) if scale is not None else 1.0 / math.sqrt(dim)

    qf = q_.to(torch.float32)
    kf = k_.to(torch.float32)
    vf = v_.to(torch.float32)

    # [B, H, Tq, Tk]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * s
    if causal:
        # Allow attending only to positions <= current query index.
        i = torch.arange(tq, device=q.device).view(1, 1, tq, 1)
        j = torch.arange(tk, device=q.device).view(1, 1, 1, tk)
        scores = scores.masked_fill(j > i, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    y = torch.matmul(probs, vf).to(dtype=out.dtype)  # [B, H, Tq, D]

    if squeeze_t:
        out.copy_(y.squeeze(2))
    else:
        out.copy_(y)

    _sync_if_xla()
