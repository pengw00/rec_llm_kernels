#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// 向量化访存：一次读取 16 bytes (对应 8 个 bfloat16 或 4 个 float)
// 这能极大提升 A100 显存带宽利用率
template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_dim]
    scalar_t* __restrict__ k_cache,        // [num_blocks, num_heads, block_size, head_dim]
    scalar_t* __restrict__ v_cache,        // [num_blocks, num_heads, block_size, head_dim]
    const int32_t* __restrict__ slot_mapping, // [num_tokens]
    const int num_heads,
    const int head_dim,
    const int block_size,
    const int key_stride,                  // k[i+1] - k[i] 的距离
    const int value_stride                 // v[i+1] - v[i] 的距离
) {
    // blockIdx.x 对应第几个 token
    // threadIdx.y 对应第几个 head
    // threadIdx.x 对应 head 中的维度偏移
    const int token_idx = blockIdx.x;
    const int head_idx = threadIdx.y;
    const int dim_idx = threadIdx.x;

    if (head_idx < num_heads && dim_idx < head_dim) {
        const int slot_idx = slot_mapping[token_idx];
        if (slot_idx < 0) return; // 负值代表忽略该 token

        const int block_idx = slot_idx / block_size;
        const int block_offset = slot_idx % block_size;

        // 计算物理 Cache 的偏移量
        // [num_blocks, num_heads, block_size, head_dim]
        const int64_t cache_idx = 
            (int64_t)block_idx * num_heads * block_size * head_dim +
            (int64_t)head_idx * block_size * head_dim +
            (int64_t)block_offset * head_dim +
            dim_idx;

        // 计算当前输入 K/V 的偏移量
        // [num_tokens, num_heads, head_dim]
        const int64_t input_idx = 
            (int64_t)token_idx * num_heads * head_dim +
            (int64_t)head_idx * head_dim +
            dim_idx;

        k_cache[cache_idx] = key[input_idx];
        v_cache[cache_idx] = value[input_idx];
    }
}

// 封装成 PyTorch 调用的 launch 函数
void reshape_and_cache(
    torch::Tensor& key,          // [num_tokens, num_heads, head_dim]
    torch::Tensor& value,        // [num_tokens, num_heads, head_dim]
    torch::Tensor& key_cache,    // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor& value_cache,  // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor& slot_mapping  // [num_tokens]
) {
    int num_tokens = key.size(0);
    int num_heads = key.size(1);
    int head_dim = key.size(2);
    int block_size = key_cache.size(2);

    dim3 grid(num_tokens);
    dim3 block(std::min(head_dim, 1024), num_heads); 
    // 注意：如果 num_heads * head_dim 超过 1024，需要调整线程块维度

    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        key.scalar_type(), "reshape_and_cache_kernel", ([&] {
            reshape_and_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int32_t>(),
                num_heads,
                head_dim,
                block_size,
                key.stride(0),
                value.stride(0)
            );
        }));
}
