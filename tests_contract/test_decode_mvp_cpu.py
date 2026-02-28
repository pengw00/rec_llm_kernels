import pytest


torch = pytest.importorskip("torch")

from rec_llm_runtime.runtime.decode_mvp import DecodeMVPConfig, DecodeOnlyMVP
from rec_llm_runtime.runtime.kv_cache import PagedKVCache


def test_decode_only_mvp_cpu_runs_and_shapes():
    # CPU-only correctness smoke test (no CUDA extension required).
    torch.manual_seed(0)
    bsz = 2
    num_heads = 4
    head_dim = 8
    hidden = num_heads * head_dim

    model = DecodeOnlyMVP(DecodeMVPConfig(hidden_size=hidden, num_heads=num_heads, head_dim=head_dim)).eval()

    cache = PagedKVCache.allocate(
        batch_size=bsz,
        num_blocks=8,
        max_blocks=4,
        num_heads=num_heads,
        block_size=4,
        head_dim=head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    x = torch.randn(bsz, hidden, dtype=torch.float32)
    positions = torch.tensor([0, 5], dtype=torch.int64)

    with torch.inference_mode():
        y = model(x, positions, cache)

    assert y.shape == x.shape
    assert cache.context_lens.tolist() == [1, 1]
