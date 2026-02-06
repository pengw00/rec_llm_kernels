import os
import torch

def get_attention_backend():
    # 模拟 vLLM 的 Selector 逻辑
    if os.getenv("VLLM_USE_FLASHINFER") == "1":
        from . import flashinfer_wrapper
        return flashinfer_wrapper
    else:
        from . import my_simple_ops
        return my_simple_ops

# 外部调用时完全不关心底层是谁
backend = get_attention_backend()
backend.forward(q, k, v)