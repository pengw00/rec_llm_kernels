import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RMSNorm kernel test.")


def _ref_rms_norm(x: "torch.Tensor", weight: "torch.Tensor", eps: float) -> "torch.Tensor":
    # x: [rows, hidden]
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    y = x.to(torch.float32) * inv_rms * weight.to(torch.float32)
    return y.to(dtype=x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rms_norm_matches_reference(dtype):
    _require_cuda()
    _C = pytest.importorskip("rec_llm_kernels._C")

    rows = 128
    hidden = 256
    eps = 1e-6

    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    out = torch.empty_like(x)

    _C.ops.rms_norm(out, x, weight, eps)
    torch.cuda.synchronize()

    ref = _ref_rms_norm(x, weight, eps)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

