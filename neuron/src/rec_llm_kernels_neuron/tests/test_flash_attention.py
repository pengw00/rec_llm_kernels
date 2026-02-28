import math

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_xla")
import torch_xla.core.xla_model as xm  # type: ignore

from rec_llm_kernels_neuron import ops


@pytest.fixture(scope="session")
def device():
    return xm.xla_device()


def _ref_attention(q, k, v, scale: float, causal: bool):
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        b, h, tq, tk = scores.shape
        i = torch.arange(tq, device=q.device).view(1, 1, tq, 1)
        j = torch.arange(tk, device=q.device).view(1, 1, 1, tk)
        scores = scores.masked_fill(j > i, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).to(dtype=q.dtype)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_att_forward_matches_reference(device, causal):
    torch.manual_seed(0)
    b, h, tq, tk, d = 2, 4, 3, 5, 16
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(b, h, tq, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, tk, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, tk, d, device=device, dtype=torch.float32)
    out = torch.empty_like(q)

    ops.flash_att_forward(q, k, v, out, scale=scale, causal=causal)
    ref = _ref_attention(q, k, v, scale=scale, causal=causal)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

