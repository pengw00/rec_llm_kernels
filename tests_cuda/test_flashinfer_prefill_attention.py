import math

import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashInfer prefill attention test.")


def _require_flashinfer() -> None:
    try:
        import flashinfer  # noqa: F401
    except Exception:
        pytest.skip("flashinfer is required for FlashInfer prefill attention test.")


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
        out[t] = probs @ vt.transpose(0, 1)  # [H, D]
    return out.to(dtype=q.dtype)


def test_flashinfer_prefill_attention_matches_reference():
    _require_cuda()
    _require_flashinfer()

    from rec_llm_runtime.ops.prefill_attention import flashinfer_prefill_attention

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # FlashInfer has limited support for very small head_dim on some GPUs/builds.
    # Use a more realistic head_dim (e.g. 64) to exercise the kernel path.
    seqlens = [8, 5]  # packed T=13
    cu = [0]
    for l in seqlens:
        cu.append(cu[-1] + l)
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)

    T = cu[-1]
    H = 4
    D = 64
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(T, H, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = flashinfer_prefill_attention(q, k, v, cu_seqlens, causal=True, sm_scale=sm_scale, backend="auto")
    torch.cuda.synchronize()

    # Reference: per-sequence causal attention
    refs = []
    for b in range(len(seqlens)):
        s = cu[b]
        e = cu[b + 1]
        refs.append(_ref_causal_attention(q[s:e], k[s:e], v[s:e], sm_scale))
    ref = torch.cat(refs, dim=0)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
