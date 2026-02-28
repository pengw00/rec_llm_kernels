from __future__ import annotations

import torch


def _ops():
    """
    Thin wrapper around the compiled extension.

    Public API surface:
      - reshape_and_cache
      - rms_norm
      - flash_att_forward (placeholder kernel)
      - paged_attention_decode
    """
    from . import _C  # local import to keep import errors localized

    return _C.ops


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return _ops().reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)


def rms_norm(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    return _ops().rms_norm(out, x, weight, float(eps))


def flash_att_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> None:
    return _ops().flash_att_forward(q, k, v, out)


def paged_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return _ops().paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, float(scale))
