#include <torch/extension.h>

// 1. 声明你的 CUDA Kernel 启动函数
void launch_flash_att(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out);
void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping);

// 2. 声明我们要新加的 RMSNorm (刚才写的那个)
void launch_rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, float epsilon);

// 3. Decode-only paged attention
torch::Tensor paged_attention_decode(torch::Tensor query,
                                     torch::Tensor key_cache,
                                     torch::Tensor value_cache,
                                     torch::Tensor block_tables,
                                     torch::Tensor context_lens,
                                     double scale);

// 绑定模块
// NOTE: The filename is `_C.so` and Python imports `rec_llm_kernels._C`,
// so the init symbol must be `PyInit__C`. Use a fixed module name to avoid
// mismatches when building via plain CMake.
PYBIND11_MODULE(_C, m) {
    // 创建 ops 子模块，模仿 vllm._C.ops
    auto ops = m.def_submodule("ops", "rec_llm kernels");

    // 将已有算子挂载到 ops 下
    ops.def("reshape_and_cache", &reshape_and_cache, "将 KV 写入 Paged Cache (A100 优化)");
    ops.def("flash_att_forward", &launch_flash_att, "FlashAttention Forward (A100 Optimized)");
    
    // 挂载新写的 RMSNorm
    ops.def("rms_norm", &launch_rms_norm, "RMSNorm CUDA kernel");

    // Paged attention (decode-only MVP)
    ops.def("paged_attention_decode", &paged_attention_decode, "Paged Attention Decode (naive MVP)");
}
