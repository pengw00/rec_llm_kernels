import math

import pytest


torch = pytest.importorskip("torch")

torch_xla = pytest.importorskip("torch_xla")
import torch_xla.core.xla_model as xm  # type: ignore

from rec_llm_kernels_neuron import ops


@pytest.fixture(scope="session")
def device():
    return xm.xla_device()

def _ref_rms_norm(x: "torch.Tensor", weight: "torch.Tensor", eps: float) -> "torch.Tensor":
    x_f = x.to(torch.float32)
    w_f = weight.to(torch.float32)
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps) * w_f
    return y.to(dtype=x.dtype)


def test_rms_norm_runs_on_inf2(device):
    rows, hidden = 32, 64
    x = torch.randn(rows, hidden, device=device, dtype=torch.float32)
    w = torch.randn(hidden, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, w, 1e-6)
    assert out.shape == x.shape
    ref = _ref_rms_norm(x, w, 1e-6)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_reshape_and_cache_runs_on_inf2(device):
    t, h, d = 4, 4, 16
    bs = 8
    nb = 4

    key = torch.randn(t, h, d, device=device, dtype=torch.float32)
    val = torch.randn(t, h, d, device=device, dtype=torch.float32)
    k_cache = torch.zeros(nb, h, bs, d, device=device, dtype=torch.float32)
    v_cache = torch.zeros(nb, h, bs, d, device=device, dtype=torch.float32)

    base_slot = 2 * bs
    slot_mapping = torch.tensor([base_slot + i for i in range(t)], device=device, dtype=torch.int32)

    ops.reshape_and_cache(key, val, k_cache, v_cache, slot_mapping)

    out_k = k_cache[2, :, 0:t, :]
    ref_k = key.transpose(0, 1)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-3, atol=1e-3)

def test_reshape_and_cache_skips_negative_slots(device):
    t, h, d = 4, 2, 8
    bs = 8
    nb = 3

    key = torch.randn(t, h, d, device=device, dtype=torch.float32)
    val = torch.randn(t, h, d, device=device, dtype=torch.float32)
    k_cache = torch.zeros(nb, h, bs, d, device=device, dtype=torch.float32)
    v_cache = torch.zeros(nb, h, bs, d, device=device, dtype=torch.float32)

    base_slot = 1 * bs
    slot_mapping = torch.tensor([base_slot + 0, -1, base_slot + 2, -1], device=device, dtype=torch.int32)

    ops.reshape_and_cache(key, val, k_cache, v_cache, slot_mapping)

    # Only positions 0 and 2 should be written.
    torch.testing.assert_close(k_cache[1, :, 0, :], key[0].transpose(0, 1), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_cache[1, :, 2, :], key[2].transpose(0, 1), rtol=1e-3, atol=1e-3)
    assert torch.all(k_cache[1, :, 1, :] == 0)
    assert torch.all(k_cache[1, :, 3, :] == 0)


def test_paged_attention_decode_matches_reference(device):
    torch.manual_seed(0)

    B, H, D = 2, 4, 16
    BS, NB, max_blocks = 8, 6, 4
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, H, D, device=device, dtype=torch.float32)
    key_cache = torch.randn(NB, H, BS, D, device=device, dtype=torch.float32)
    value_cache = torch.randn(NB, H, BS, D, device=device, dtype=torch.float32)

    context_lens = torch.tensor([12, 5], device=device, dtype=torch.int32)
    block_tables = torch.full((B, max_blocks), 0, device=device, dtype=torch.int32)
    block_tables[0, 0] = 1
    block_tables[0, 1] = 3
    block_tables[1, 0] = 2

    out = ops.paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale)

    # Reference (same math, explicit loop)
    ref = torch.zeros((B, H, D), device=device, dtype=torch.float32)
    for b in range(B):
        ctx = int(context_lens[b].item())
        for h in range(H):
            q = query[b, h]
            keys = []
            vals = []
            for pos in range(ctx):
                bt = pos // BS
                block_id = int(block_tables[b, bt].item())
                off = pos - bt * BS
                keys.append(key_cache[block_id, h, off])
                vals.append(value_cache[block_id, h, off])
            K = torch.stack(keys, dim=0)
            V = torch.stack(vals, dim=0)
            logits = (K @ q) * scale
            probs = torch.softmax(logits, dim=0)
            ref[b, h] = probs @ V

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
