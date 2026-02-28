import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for llama layer wiring test.")


def _ref_rms_norm(x: "torch.Tensor", weight: "torch.Tensor", eps: float) -> "torch.Tensor":
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    y = x.to(torch.float32) * inv_rms * weight.to(torch.float32)
    return y.to(dtype=x.dtype)


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


def test_llama_layer_wiring_decode_only_matches_reference():
    _require_cuda()

    from rec_llm_runtime.model_executor.models.llama import LlamaLayer, LlamaLayerConfig

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    B = 2
    H = 4
    D = 16
    hidden = H * D

    BS = 8
    NB = 6
    max_blocks = 4

    eps = 1e-6
    scale = 1.0 / (D**0.5)

    layer = LlamaLayer(LlamaLayerConfig(hidden_size=hidden, num_heads=H, head_dim=D, scale=scale)).to(device).to(dtype)
    layer.input_layernorm.weight.data.fill_(1.0)
    layer.input_layernorm.variance_epsilon = eps

    hidden_states = torch.randn(B, hidden, device=device, dtype=dtype)
    key_cache = torch.randn(NB, H, BS, D, device=device, dtype=dtype)
    value_cache = torch.randn(NB, H, BS, D, device=device, dtype=dtype)

    context_lens = torch.tensor([12, 5], device=device, dtype=torch.int32)
    block_tables = torch.full((B, max_blocks), 0, device=device, dtype=torch.int32)
    block_tables[0, 0] = 1
    block_tables[0, 1] = 3
    block_tables[1, 0] = 2

    out = layer(hidden_states, key_cache, value_cache, block_tables, context_lens)
    torch.cuda.synchronize()

    # Reference: residual + (rmsnorm -> paged attention decode)
    normed = _ref_rms_norm(hidden_states, torch.ones(hidden, device=device, dtype=dtype), eps)
    query = normed.view(B, H, D)
    ref_attn = _ref_paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale)
    ref = hidden_states + ref_attn.view(B, hidden)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
