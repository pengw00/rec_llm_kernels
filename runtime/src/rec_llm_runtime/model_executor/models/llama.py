from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rec_llm_runtime.model_executor.layers import PagedAttention, RMSNorm


@dataclass(frozen=True)
class LlamaLayerConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    scale: float | None = None


class LlamaLayer(nn.Module):
    """
    Minimal decode-only Llama block skeleton.

    This is intentionally incomplete (no QKV projection / MLP / RoPE). It exists to
    validate that the wiring between RMSNorm and paged attention decode is consistent.
    """

    def __init__(self, config: LlamaLayerConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.attention = PagedAttention(config.num_heads, config.head_dim, scale=config.scale)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, hidden]
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        bsz = hidden_states.size(0)
        query = hidden_states.view(bsz, self.config.num_heads, self.config.head_dim)
        attn_out = self.attention(
            query, key_cache, value_cache, block_tables, context_lens, scale=self.attention.scale
        )
        attn_out = attn_out.view_as(hidden_states)
        return residual + attn_out
