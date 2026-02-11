import torch
from rec_llm.model_executor.layers import RMSNorm, PagedAttention

@torch.inference_mode()
def test_layer_accuracy():
    device = "cuda"
    dtype = torch.float16
    
    # --- Input Data ---
    x = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=dtype)
    
    # --- 1. Test RMSNorm ---
    my_norm = RMSNorm(HIDDEN_SIZE).to(device).to(dtype)
    # Load identity weights for simplicity
    my_norm.weight.data.fill_(1.0)
    
    custom_output = my_norm(x.clone())
    golden_output = ref_rms_norm(x, my_norm.weight, my_norm.variance_epsilon)
    
    # Compare
    torch.testing.assert_close(custom_output, golden_output, atol=1e-3, rtol=1e-3)
    print("✅ RMSNorm Accuracy Check Passed!")

    # --- 2. Test Attention & Cache ---
    # Prepare Q, K, V
    q = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_SIZE, device=device, dtype=dtype)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_SIZE, device=device, dtype=dtype)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_SIZE, device=device, dtype=dtype)
    
    paged_attn = PagedAttention().to(device)
    
    # Execute your custom kernel flow
    # This calls reshape_and_cache + flash_att_forward inside
    custom_attn_out = paged_attn(q, k, v, kv_cache, slot_mapping)
    
    # Execute Reference
    golden_attn_out = ref_attention(q, k, v)
    
    # Compare
    torch.testing.assert_close(custom_attn_out, golden_attn_out, atol=1e-2, rtol=1e-2)
    print("✅ PagedAttention Accuracy Check Passed!")

if __name__ == "__main__":
    test_layer_accuracy()
