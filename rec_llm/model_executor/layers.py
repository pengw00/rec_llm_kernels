import torch
from torch import nn
from rec_llm import _C  # 导入你的 C 扩展

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
            
        # 2. 调用你的 C++ 算子
        # 传入 x (输入/输出), self.weight (权重), self.variance_epsilon (超参)
        _C.ops.rms_norm(x, self.weight, self.variance_epsilon)
        
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 其它初始化...

    def forward(self, q, k, v, kv_cache, slot_mapping):
        """
        模拟 vLLM 的 PagedAttention 调用
        """
        # 调用 C++ 算子
        _C.ops.paged_attention(
            q, k, v, 
            kv_cache, 
            slot_mapping,
            self.config.num_heads,
            self.config.head_size,
            self.config.scale
        )
        return q
