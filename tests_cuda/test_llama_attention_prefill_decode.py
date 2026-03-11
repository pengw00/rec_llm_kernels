import math

import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for LlamaAttention prefill/decode test.")


def _require_flashinfer() -> None:
    try:
        import flashinfer  # noqa: F401
    except Exception:
        pytest.skip("flashinfer is required for LlamaAttention prefill/decode test.")


def _torch_apply_rope(q: "torch.Tensor", k: "torch.Tensor", positions: "torch.Tensor", base: float) -> tuple["torch.Tensor", "torch.Tensor"]:
    # q/k: [N, H, D]
    N, _, D = q.shape
    if D % 2 != 0:
        raise ValueError("head_dim must be even")
    half = D // 2
    pos_f = positions.to(dtype=torch.float32)
    inv_freq = base ** (-torch.arange(0, half, device=q.device, dtype=torch.float32) / half)
    freqs = pos_f[:, None] * inv_freq[None, :]
    cos = torch.cos(freqs)[:, None, :]
    sin = torch.sin(freqs)[:, None, :]

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    q1, q2 = qf[..., :half], qf[..., half:]
    k1, k2 = kf[..., :half], kf[..., half:]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot.to(dtype=q.dtype), k_rot.to(dtype=k.dtype)


def _ref_causal_attention(q: "torch.Tensor", k: "torch.Tensor", v: "torch.Tensor", sm_scale: float) -> "torch.Tensor":
    # q/k/v: [L, H, D]
    L, H, D = q.shape
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    out = torch.empty((L, H, D), device=q.device, dtype=torch.float32)
    for t in range(L):
        qt = qf[t]  # [H, D]
        kt = kf[: t + 1]  # [t+1, H, D]
        vt = vf[: t + 1]
        logits = (kt * qt.unsqueeze(0)).sum(dim=-1) * float(sm_scale)  # [t+1, H]
        probs = torch.softmax(logits.transpose(0, 1), dim=-1)  # [H, t+1]
        out[t] = torch.einsum("ht,thd->hd", probs, vt)
    return out.to(dtype=q.dtype)


def test_llama_attention_prefill_then_decode_matches_reference():
    _require_cuda()
    _require_flashinfer()

    from rec_llm_runtime.model_executor.llama_attention import LlamaAttention, LlamaAttentionConfig
    from rec_llm_runtime.runtime.kv_cache import PagedKVCache
    from rec_llm_runtime.runtime.decode_mvp import _torch_paged_attention_decode, _reshape_and_cache_torch

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # Model dims (no-GQA MVP)
    B = 2
    H = 4
    D = 64
    hidden = H * D
    rope_theta = 10000.0
    sm_scale = 1.0 / math.sqrt(D)

    # Prompt lengths => packed T
    seqlens = [8, 5]
    cu = [0]
    for l in seqlens:
        cu.append(cu[-1] + l)
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
    T = cu[-1]

    cache = PagedKVCache.allocate(
        batch_size=B,
        num_blocks=64,
        max_blocks=16,
        num_heads=H,
        block_size=8,
        head_dim=D,
        device=torch.device(device),
        dtype=dtype,
    )

    attn = LlamaAttention(LlamaAttentionConfig(hidden_size=hidden, num_heads=H, head_dim=D, rope_theta=rope_theta))
    attn = attn.to(device=device, dtype=dtype).eval()

    x_prompt = torch.randn(T, hidden, device=device, dtype=dtype)
    y_prompt = attn.prefill(x_prompt, cu_seqlens, cache)
    torch.cuda.synchronize()

    # Reference prefill
    q = attn.wq(x_prompt).view(T, H, D)
    k = attn.wk(x_prompt).view(T, H, D)
    v = attn.wv(x_prompt).view(T, H, D)
    positions = torch.empty((T,), device=device, dtype=torch.int64)
    for b in range(B):
        start, end = cu[b], cu[b + 1]
        positions[start:end] = torch.arange(end - start, device=device, dtype=torch.int64)
    q, k = _torch_apply_rope(q, k, positions, base=rope_theta)

    ref_outs = []
    for b in range(B):
        s, e = cu[b], cu[b + 1]
        ref_outs.append(_ref_causal_attention(q[s:e], k[s:e], v[s:e], sm_scale))
    ref_prompt = torch.cat(ref_outs, dim=0).reshape(T, -1)
    ref_prompt = attn.wo(ref_prompt)

    torch.testing.assert_close(y_prompt, ref_prompt, rtol=1e-2, atol=1e-2)

    # Decode one step and compare against torch reference built via paged-cache read.
    x_step = torch.randn(B, hidden, device=device, dtype=dtype)

    # Snapshot cache state for reference path.
    cache_ref = PagedKVCache(
        key_cache=cache.key_cache.clone(),
        value_cache=cache.value_cache.clone(),
        block_tables=cache.block_tables.clone(),
        context_lens=cache.context_lens.clone(),
        block_size=cache.block_size,
        _next_block=cache._next_block,
    )

    y_step = attn.decode(x_step, cache)
    torch.cuda.synchronize()

    # Reference decode: apply rope, write KV via torch reshape, then read via torch paged attention.
    qd = attn.wq(x_step).view(B, H, D)
    kd = attn.wk(x_step).view(B, H, D)
    vd = attn.wv(x_step).view(B, H, D)
    pos_d = cache_ref.context_lens.to(dtype=torch.int64)
    qd, kd = _torch_apply_rope(qd, kd, pos_d, base=rope_theta)
    cache_ref.append_kv(kd, vd, reshape_and_cache_fn=_reshape_and_cache_torch)
    attn_ref = _torch_paged_attention_decode(
        qd, cache_ref.key_cache, cache_ref.value_cache, cache_ref.block_tables, cache_ref.context_lens, sm_scale
    )
    y_step_ref = attn.wo(attn_ref.reshape(B, -1))

    torch.testing.assert_close(y_step, y_step_ref, rtol=1e-2, atol=1e-2)

