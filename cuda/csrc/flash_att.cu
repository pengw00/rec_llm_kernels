#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void simple_att_kernel(const float* q, const float* k, const float* v, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 先写个简单的赋值，证明链路通了
        out[idx] = q[idx] + k[idx]; 
    }
}

void launch_flash_att(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out) {
    int size = q.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    simple_att_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), size);
}
