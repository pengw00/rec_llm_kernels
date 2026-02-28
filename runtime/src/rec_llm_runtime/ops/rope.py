from __future__ import annotations

import torch


def build_rope_cache(
    positions: torch.Tensor,
    head_dim: int,
    *,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build RoPE cos/sin caches for given positions.

    positions: [B] int32/int64
    Returns: (cos, sin) each [B, head_dim/2] float32
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")

    positions_f = positions.to(dtype=torch.float32)
    half = head_dim // 2
    inv_freq = (base ** (-torch.arange(0, half, device=positions.device, dtype=torch.float32) / half))
    freqs = positions_f[:, None] * inv_freq[None, :]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Llama-style rotary embedding to q/k.

    q, k: [B, H, D]
    positions: [B]
    Returns rotated (q, k) with same dtype/device.
    """
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape.")
    if q.dim() != 3:
        raise ValueError("q/k must have shape [B, H, D].")

    bsz, nheads, head_dim = q.shape
    if positions.shape != (bsz,):
        raise ValueError("positions must have shape [B].")

    cos, sin = build_rope_cache(positions, head_dim, base=base)
    cos = cos[:, None, :].to(dtype=torch.float32)
    sin = sin[:, None, :].to(dtype=torch.float32)

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)

    q1, q2 = qf[..., : head_dim // 2], qf[..., head_dim // 2 :]
    k1, k2 = kf[..., : head_dim // 2], kf[..., head_dim // 2 :]

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rot.to(dtype=q.dtype), k_rot.to(dtype=k.dtype)

