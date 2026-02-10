#include <cuda_runtime.h>
#include <torch/extension.h>

// 使用 Warp Shuffle 实现高效的求和规约
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 核心 Kernel
template<typename T>
__global__ void rms_norm_kernel(
    T* __restrict__ out,           // [num_tokens, hidden_size]
    const T* __restrict__ input,    // [num_tokens, hidden_size]
    const T* __restrict__ weight,   // [hidden_size]
    const float epsilon,
    const int hidden_size) {
    
    int row = blockIdx.x;       // 每个 Block 处理一个 Token (行)
    int tid = threadIdx.x;      // 线程 ID

    float sum = 0.0f;
    // 1. 计算当前行所有元素的平方和
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = (float)input[row * hidden_size + i];
        sum += val * val;
    }

    // 2. Warp 内规约
    sum = warpReduceSum(sum);

    // 3. 使用共享内存进行 Block 内规约 (处理多于 32 线程的情况)
    static __shared__ float s_sum;
    if (tid % 32 == 0) atomicAdd(&s_sum, sum);
    __syncthreads();

    // 4. 计算 RMS 倒数
    float inv_rms = rsqrtf(s_sum / hidden_size + epsilon);

    // 5. 归一化并写回结果
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = (float)input[row * hidden_size + i];
        out[row * hidden_size + i] = (T)(val * inv_rms * (float)weight[i]);
    }
    
    if (tid == 0) s_sum = 0.0f; // 重置共享内存供下次使用
}

// C++ 包装层：用于对齐 PyTorch Tensor 并启动 Kernel
void launch_rms_norm(
    torch::Tensor& out,    // 输出
    torch::Tensor& input,  // 输入
    torch::Tensor& weight, // 权重
    float epsilon) {
    
    int num_tokens = input.size(0);
    int hidden_size = input.size(1);

    // 配置线程：通常取 256 或 512
    int threads = 256;
    int blocks = num_tokens;

    // 根据数据类型分发模板（支持 FP16/FP32）
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_kernel", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            epsilon,
            hidden_size);
    }));
}
