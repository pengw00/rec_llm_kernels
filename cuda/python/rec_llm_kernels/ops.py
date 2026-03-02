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


def _flashinfer_page_table_from_block_tables(
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    context_lens: torch.Tensor,  # [B] int32
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert our vLLM-style block tables + context lens into FlashInfer's CSR page table:
      - indptr:        [B+1] int32
      - indices:       [indptr[-1]] int32
      - last_page_len: [B] int32, 1 <= last_page_len <= page_size
    """
    if block_tables.dim() != 2:
        raise ValueError("block_tables must have shape [B, max_blocks].")
    if context_lens.dim() != 1:
        raise ValueError("context_lens must have shape [B].")
    if block_tables.size(0) != context_lens.size(0):
        raise ValueError("block_tables/context_lens batch size mismatch.")
    if block_tables.dtype != torch.int32:
        block_tables = block_tables.to(torch.int32)
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)

    B, max_blocks = block_tables.shape
    if B == 0:
        device = block_tables.device
        indptr = torch.zeros((1,), device=device, dtype=torch.int32)
        indices = torch.empty((0,), device=device, dtype=torch.int32)
        last_page_len = torch.empty((0,), device=device, dtype=torch.int32)
        return indptr, indices, last_page_len

    ctx = context_lens.to(torch.int64)
    page_size_i64 = int(page_size)
    if page_size_i64 <= 0:
        raise ValueError("page_size must be > 0.")

    num_pages = torch.div(ctx + (page_size_i64 - 1), page_size_i64, rounding_mode="floor")  # [B]
    max_pages = int(num_pages.max().item()) if num_pages.numel() else 0
    max_pages = min(max_pages, max_blocks)

    indptr = torch.zeros((B + 1,), device=block_tables.device, dtype=torch.int32)
    indptr[1:] = torch.cumsum(num_pages.to(torch.int32), dim=0)

    # last_page_len must be in [1, page_size] for FlashInfer.
    last = (ctx % page_size_i64).to(torch.int32)
    last = torch.where((ctx > 0) & (last == 0), torch.tensor(page_size_i64, device=last.device, dtype=last.dtype), last)
    last = torch.where(ctx == 0, torch.ones_like(last), last)

    if max_pages == 0:
        indices = torch.empty((0,), device=block_tables.device, dtype=torch.int32)
        return indptr, indices, last

    pages = block_tables[:, :max_pages].contiguous().to(torch.int32)  # [B, max_pages]
    page_ix = torch.arange(max_pages, device=block_tables.device, dtype=torch.int64).view(1, -1).expand(B, -1)
    mask = page_ix < num_pages.view(-1, 1)
    indices = pages[mask].contiguous()
    return indptr, indices, last


def paged_attention_decode_v2_flashinfer(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    *,
    backend: str = "auto",
    workspace_bytes: int = 128 * 1024 * 1024,
) -> torch.Tensor:
    """
    Decode-only paged attention via FlashInfer (preferred fast path).

    Expects cache layout: [num_pages, num_kv_heads, page_size, head_dim] (HND).
    """
    try:
        import flashinfer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "flashinfer is required for paged_attention_decode_v2_flashinfer. "
            "Install flashinfer or call paged_attention_decode() / paged_attention_decode_v2()."
        ) from e

    if query.dim() != 3:
        raise ValueError("query must have shape [B, H, D].")
    if key_cache.dim() != 4:
        raise ValueError("key_cache must have shape [NB, H, BS, D].")
    if value_cache.sizes() != key_cache.sizes():
        raise ValueError("value_cache must match key_cache shape.")

    B, H, D = query.shape
    NB, Hk, BS, Dk = key_cache.shape
    if Hk != H or Dk != D:
        raise ValueError("key_cache shape mismatch with query.")
    if NB <= 0:
        raise ValueError("key_cache must have at least one page.")

    device = query.device
    if device.type != "cuda":
        raise ValueError("FlashInfer decode requires CUDA tensors.")
    if key_cache.device != device or value_cache.device != device:
        raise ValueError("key_cache/value_cache must be on the same device as query.")

    indptr, indices, last_page_len = _flashinfer_page_table_from_block_tables(
        block_tables.to(device=device),
        context_lens.to(device=device),
        page_size=BS,
    )

    workspace = torch.zeros(int(workspace_bytes), dtype=torch.uint8, device=device)
    decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="HND", backend=backend)
    decode_wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads=H,
        num_kv_heads=Hk,
        head_dim=D,
        page_size=BS,
        pos_encoding_mode="NONE",
        q_data_type=query.dtype,
        kv_data_type=key_cache.dtype,
        o_data_type=query.dtype,
        sm_scale=float(scale),
    )
    return decode_wrapper.run(query, (key_cache, value_cache))


def paged_attention_decode_v2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    v2 decode path: use FlashInfer if available, otherwise fall back to the reference CUDA kernel.
    """
    try:
        return paged_attention_decode_v2_flashinfer(query, key_cache, value_cache, block_tables, context_lens, scale)
    except Exception:
        return paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale)
