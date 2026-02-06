#include <torch/extension.h>

// 声明你在 .cu 里写的函数
void launch_flash_att(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out);

// 绑定模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_att_forward", &launch_flash_att, "FlashAttention Forward (A100 Optimized)");
}
