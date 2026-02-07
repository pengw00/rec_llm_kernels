#include <torch/extension.h>

// 声明你在 .cu 里写的函数
void launch_flash_att(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out);

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping);

// 绑定模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reshape_and_cache", &reshape_and_cache, "将 KV 写入 Paged Cache (A100 优化)");
    m.def("flash_att_forward", &launch_flash_att, "FlashAttention Forward (A100 Optimized)");
}
