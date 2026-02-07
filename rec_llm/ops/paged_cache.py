import torch
from typing import Optional

# 尝试导入编译好的 C++ 扩展 (你的 rec_llm_kernels_lib)
try:
    import rec_llm_kernels_lib as _C
except ImportError:
    raise ImportError("请先运行 setup.py install 编译 rec_llm_kernels_lib")

class PagedCacheOps:
    """
    Paged KV Cache 的算子封装类。
    负责将新 Token 的 KV 写入 Cache，并调用 FlashInfer 进行 Paged Attention 计算。
    """
    
    @staticmethod
    def reshape_and_cache(
        key: torch.Tensor,        # [num_tokens, num_heads, head_dim]
        value: torch.Tensor,      # [num_tokens, num_heads, head_dim]
        key_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_dim]
        value_cache: torch.Tensor,# [num_blocks, num_heads, block_size, head_dim]
        slot_mapping: torch.Tensor # [num_tokens] 存储每个 token 对应的全局物理位置索引
    ) -> None:
        """
        将新生成的 key 和 value 写入到预分配的物理显存块中。
        """
        # 调用你在 paged_cache.cu 中写的 C++ 函数
        _C.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping
        )

    @staticmethod
    def paged_attention(
        query: torch.Tensor,            # [num_queries, num_heads, head_dim]
        key_cache: torch.Tensor,        # [num_blocks, num_heads, block_size, head_dim]
        value_cache: torch.Tensor,      # [num_blocks, num_heads, block_size, head_dim]
        block_tables: torch.Tensor,     # [num_queries, max_num_blocks_per_seq] 映射表
        context_lens: torch.Tensor,     # [num_queries] 每个序列目前的有效长度
        scale: float,                   # 注意力缩放因子 (通常是 1/sqrt(head_dim))
        max_context_len: int,           # 当前 batch 中最长的序列长度
        alibi_slopes: Optional[torch.Tensor] = None, # ALiBi 偏置（如果是 Llama 则为 None）
    ) -> torch.Tensor:
        """
        调用 FlashInfer 后端执行 Paged Attention 计算。
        """
        # 这里的 _C.paged_attention 内部实际上是调用了 FlashInfer 的 
        # BatchDecodeWithPagedKVCache 或 BatchPrefillWithPagedKVCache
        output = _C.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            scale,
            max_context_len,
            alibi_slopes
        )
        return output

# 为了方便调用，暴露简单的函数接口
reshape_and_cache = PagedCacheOps.reshape_and_cache
paged_attention = PagedCacheOps.paged_attention
