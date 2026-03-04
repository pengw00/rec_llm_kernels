import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RoPE kernel test.")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [16, 64, 128])
def test_apply_rope_matches_torch_reference(dtype, head_dim):
    _require_cuda()

    from rec_llm_kernels.ops import apply_rope as cuda_apply_rope
    from rec_llm_runtime.ops.rope import apply_rope as torch_apply_rope

    torch.manual_seed(0)
    device = "cuda"

    B = 4
    H = 8
    base = 10000.0

    q = torch.randn(B, H, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, H, head_dim, device=device, dtype=dtype)
    positions = torch.tensor([0, 1, 17, 1024], device=device, dtype=torch.int64)

    q2, k2 = cuda_apply_rope(q, k, positions, base=base)
    torch.cuda.synchronize()

    ref_q, ref_k = torch_apply_rope(q, k, positions, base=base)

    torch.testing.assert_close(q2, ref_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k2, ref_k, rtol=1e-2, atol=1e-2)

