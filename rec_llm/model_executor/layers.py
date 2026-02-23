import torch
from torch import nn

try:
    import rec_llm_kernels._C as _C  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled extension `rec_llm_kernels._C`. "
        "Build/install the project first (e.g. `pip install -e .`)."
    ) from e


def _require_cuda(t: torch.Tensor) -> None:
    if not t.is_cuda:
        raise ValueError("This module expects CUDA tensors.")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 权重还是用 PyTorch 原生的，方便管理
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor):
        """
        x: [num_tokens, hidden_size]
        """
        # 1. 确保输入是连续的，CUDA Kernel 极其依赖连续内存
        if not x.is_contiguous():
            x = x.contiguous()

        weight = self.weight
        if weight.device != x.device:
            weight = weight.to(device=x.device)
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)

        out = torch.empty_like(x)
        _C.ops.rms_norm(out, x, weight, float(self.variance_epsilon))
        return out


class PagedAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, scale: float | None = None):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scale = float(scale) if scale is not None else 1.0 / (float(self.head_dim) ** 0.5)

    def forward(
        self,
        query: torch.Tensor,         # [B, H, D]
        key_cache: torch.Tensor,     # [NB, H, BS, D]
        value_cache: torch.Tensor,   # [NB, H, BS, D]
        block_tables: torch.Tensor,  # [B, max_blocks]
        context_lens: torch.Tensor,  # [B]
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Decode-only paged attention wrapper (MVP).
        """
        _require_cuda(query)
        _require_cuda(key_cache)
        _require_cuda(value_cache)
        _require_cuda(block_tables)
        _require_cuda(context_lens)

        if query.dim() != 3:
            raise ValueError("query must have shape [B, H, D].")
        if query.size(1) != self.num_heads or query.size(2) != self.head_dim:
            raise ValueError("query shape mismatch with num_heads/head_dim.")

        s = self.scale if scale is None else float(scale)
        return _C.ops.paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, s)

# Back-compat alias (older code used Attention)
Attention = PagedAttention
