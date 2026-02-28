import torch
import pytest
import rec_llm_kernels._C as _C  # 调用你编译的 C++ 扩展

def test_reshape_and_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for reshape_and_cache test.")

    # --- 1. 配置参数 (以 Llama 架构为例) ---
    num_tokens = 4
    num_heads = 32
    head_dim = 128
    block_size = 16  # 每个物理块存 16 个 token
    num_blocks = 10  # 预分配 10 个物理块
    dtype = torch.bfloat16 # A100 标配
    device = "cuda"

    # --- 2. 构造模拟输入 (新生成的 K/V) ---
    # 形状: [num_tokens, num_heads, head_dim]
    k = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=dtype)

    # --- 3. 构造显存池 (KV Cache) ---
    # 形状: [num_blocks, num_heads, block_size, head_dim]
    k_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)

    # --- 4. 构造映射表 (Slot Mapping) ---
    # 假设我们要把这 4 个 token 存到第 5 个物理块 (block_id=5) 的前 4 个位置
    # slot_idx = block_id * block_size + offset
    base_slot = 5 * block_size
    slot_mapping = torch.tensor([base_slot + i for i in range(num_tokens)], device=device, dtype=torch.int32)

    # --- 5. 执行你的 CUDA Kernel ---
    print(f"\n正在启动 Kernel 测试: 写入 {num_tokens} 个 tokens 到 Block 5...")
    _C.ops.reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)
    torch.cuda.synchronize() # 确保 GPU 计算完成

    # --- 6. 验证结果 ---
    # 检查 Block 5 的前 4 个位置是否等于我们的输入 k 和 v
    out_k = k_cache[5, :, 0:num_tokens, :] # 形状: [num_heads, 4, head_dim]
    out_v = v_cache[5, :, 0:num_tokens, :]
    
    # 调整输入形状以进行对比 [4, heads, dim] -> [heads, 4, dim]
    ref_k = k.transpose(0, 1)
    ref_v = v.transpose(0, 1)

    # 断言验证精度
    torch.testing.assert_close(out_k, ref_k, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_v, ref_v, rtol=1e-3, atol=1e-3)
    
    print("✅ Unit Test Passed: Reshape and Cache 寻址正确！")

if __name__ == "__main__":
    test_reshape_and_cache()
