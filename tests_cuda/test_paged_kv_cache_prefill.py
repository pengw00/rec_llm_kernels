import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for paged KV cache prefill test.")


def test_prefill_kv_writes_cache_and_decode_reads_correctly():
    _require_cuda()

    from rec_llm_runtime.runtime.kv_cache import PagedKVCache
    from rec_llm_runtime.runtime.decode_mvp import _torch_paged_attention_decode

    _C = pytest.importorskip("rec_llm_kernels._C")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    B = 2
    H = 4
    D = 64
    BS = 8
    NB = 16
    max_blocks = 8

    # Prefill lengths: seq0=10, seq1=5 => packed T=15
    seqlens = [10, 5]
    cu = [0]
    for l in seqlens:
        cu.append(cu[-1] + l)
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
    T = cu[-1]

    cache = PagedKVCache.allocate(
        batch_size=B,
        num_blocks=NB,
        max_blocks=max_blocks,
        num_heads=H,
        block_size=BS,
        head_dim=D,
        device=torch.device(device),
        dtype=dtype,
    )

    # Random packed KV to write.
    k = torch.randn(T, H, D, device=device, dtype=dtype)
    v = torch.randn_like(k)

    cache.prefill_kv(k, v, cu_seqlens, reshape_and_cache_fn=_C.ops.reshape_and_cache)
    torch.cuda.synchronize()

    # Decode query for each sequence (one token per seq), and compare CUDA decode against torch reference.
    query = torch.randn(B, H, D, device=device, dtype=dtype)
    scale = 1.0 / (D**0.5)

    out = _C.ops.paged_attention_decode(query, cache.key_cache, cache.value_cache, cache.block_tables, cache.context_lens, scale)
    torch.cuda.synchronize()

    ref = _torch_paged_attention_decode(query, cache.key_cache, cache.value_cache, cache.block_tables, cache.context_lens, scale)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

