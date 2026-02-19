import torch
from torch import nn

try:
    import rec_llm_kernels._C as _C  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled extension `rec_llm_kernels._C`. "
        "Build/install the project first (e.g. `pip install -e .`)."
    ) from e

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
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Minimal wrapper for the current placeholder FlashAttention kernel.
        Ignores paged-KV-cache arguments for now.
        """
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("PagedAttention expects CUDA tensors.")
        if q.dtype != torch.float32:
            raise ValueError("Current flash_att_forward placeholder expects float32 tensors.")

        out = torch.empty_like(q)
        _C.ops.flash_att_forward(q, k, v, out)
        return out


# Back-compat alias (older code used Attention)
Attention = PagedAttention
