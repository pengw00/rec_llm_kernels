from __future__ import annotations

import torch


def flashinfer_prefill_attention(
    q: torch.Tensor,  # [T, Hq, D]
    k: torch.Tensor,  # [T, Hk, D]
    v: torch.Tensor,  # [T, Hk, D]
    cu_seqlens: torch.Tensor,  # [B+1] int32/int64
    *,
    causal: bool = True,
    kv_layout: str = "NHD",
    sm_scale: float | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """
    Prefill attention via FlashInfer (per-sequence loop).

    This is a correctness-first MVP used to bring up LLaMA prefill with minimal
    infra changes. It intentionally uses the single-request FlashInfer API in a
    Python loop; once the end-to-end pipeline is stable, this can be upgraded to
    a batch wrapper for performance.

    Shapes:
      q/k/v:     [T, H, D] (packed tokens)
      cu_seqlens: [B+1], cu_seqlens[0]=0, cu_seqlens[-1]=T
    """
    try:
        import flashinfer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("flashinfer is required for flashinfer_prefill_attention") from e

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v shape mismatch.")
    if q.dim() != 3:
        raise ValueError("q/k/v must have shape [T, H, D].")
    if cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must have shape [B+1].")

    T = int(q.size(0))
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError("cu_seqlens must start with 0.")
    if int(cu_seqlens[-1].item()) != T:
        raise ValueError("cu_seqlens[-1] must equal T.")

    # FlashInfer expects CUDA tensors for kernels.
    if q.device.type != "cuda":
        raise ValueError("q must be a CUDA tensor.")
    if k.device != q.device or v.device != q.device or cu_seqlens.device != q.device:
        raise ValueError("q/k/v/cu_seqlens must be on the same device.")

    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        cu_seqlens = cu_seqlens.to(dtype=torch.int32)

    if sm_scale is None:
        sm_scale = 1.0 / (float(q.size(-1)) ** 0.5)

    B = int(cu_seqlens.numel() - 1)
    outs: list[torch.Tensor] = []
    for b in range(B):
        start = int(cu_seqlens[b].item())
        end = int(cu_seqlens[b + 1].item())
        if end < start:
            raise ValueError("cu_seqlens must be non-decreasing.")
        if end == start:
            continue
        qb = q[start:end]
        kb = k[start:end]
        vb = v[start:end]
        try:
            ob = flashinfer.prefill.single_prefill_with_kv_cache(
                qb,
                kb,
                vb,
                causal=bool(causal),
                kv_layout=str(kv_layout),
                pos_encoding_mode="NONE",
                sm_scale=float(sm_scale),
                backend=str(backend),
            )
        except RuntimeError as e:
            # Some FlashInfer builds have limited support for small head_dim or
            # specific (H, D, L) combinations. Provide a clear error message.
            msg = str(e)
            if "Invalid configuration" in msg:
                raise RuntimeError(
                    "FlashInfer prefill kernel reports an invalid configuration for the given shapes. "
                    "Try a different head_dim (e.g. 64/128), upgrade flashinfer, or switch the backend."
                ) from e
            raise
        outs.append(ob)
    if not outs:
        return torch.empty_like(q)
    return torch.cat(outs, dim=0)
