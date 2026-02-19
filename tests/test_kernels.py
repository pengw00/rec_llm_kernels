import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for these kernel tests.")


def test_flash_att_forward_smoke_adds_q_and_k():
    _require_cuda()
    _C = pytest.importorskip("rec_llm_kernels._C")

    q = torch.randn(128, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)

    _C.ops.flash_att_forward(q, k, v, out)
    torch.cuda.synchronize()

    # Current kernel is a placeholder: out = q + k
    torch.testing.assert_close(out, q + k, rtol=0, atol=0)

