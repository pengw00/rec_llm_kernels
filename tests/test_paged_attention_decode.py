import math

import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for paged attention decode test.")


def _ref_paged_attention_decode(
    query: "torch.Tensor",  # [B, H, D]
    key_cache: "torch.Tensor",  # [NB, H, BS, D]
    value_cache: "torch.Tensor",  # [NB, H, BS, D]
    block_tables: "torch.Tensor",  # [B, max_blocks]
    context_lens: "torch.Tensor",  # [B]
    scale: float,
) -> "torch.Tensor":
    B, H, D = query.shape
    _, _, BS, _ = key_cache.shape
    max_blocks = block_tables.shape[1]

    out = torch.zeros((B, H, D), device=query.device, dtype=torch.float32)
    for b in range(B):
        ctx = int(context_lens[b].item())
        for h in range(H):
            q = query[b, h].to(torch.float32)  # [D]
            keys = []
            vals = []
            for pos in range(ctx):
                bt_idx = pos // BS
                assert bt_idx < max_blocks
                block_id = int(block_tables[b, bt_idx].item())
                off = pos - bt_idx * BS
                keys.append(key_cache[block_id, h, off].to(torch.float32))
                vals.append(value_cache[block_id, h, off].to(torch.float32))
            K = torch.stack(keys, dim=0)  # [ctx, D]
            V = torch.stack(vals, dim=0)  # [ctx, D]
            logits = (K @ q) * scale  # [ctx]
            probs = torch.softmax(logits, dim=0)
            out[b, h] = probs @ V
    return out.to(dtype=query.dtype)


def test_paged_attention_decode_matches_reference():
    _require_cuda()
    _C = pytest.importorskip("rec_llm_kernels._C")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    B = 2
    H = 4
    D = 16
    BS = 8
    NB = 6
    max_blocks = 4

    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, H, D, device=device, dtype=dtype)
    key_cache = torch.randn(NB, H, BS, D, device=device, dtype=dtype)
    value_cache = torch.randn(NB, H, BS, D, device=device, dtype=dtype)

    # Context lengths for each sequence (decode-only)
    context_lens = torch.tensor([12, 5], device=device, dtype=torch.int32)

    # Each row maps logical blocks to physical block IDs.
    # ctx=12 with BS=8 => needs 2 blocks; ctx=5 => needs 1 block.
    block_tables = torch.full((B, max_blocks), 0, device=device, dtype=torch.int32)
    block_tables[0, 0] = 1
    block_tables[0, 1] = 3
    block_tables[1, 0] = 2

    out = _C.ops.paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale)
    torch.cuda.synchronize()

    ref = _ref_paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

