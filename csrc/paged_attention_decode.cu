#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

template <typename scalar_t>
__global__ void paged_attention_decode_kernel(
    const scalar_t* __restrict__ query,          // [B, H, D]
    const scalar_t* __restrict__ key_cache,      // [NB, H, BS, D]
    const scalar_t* __restrict__ value_cache,    // [NB, H, BS, D]
    const int32_t* __restrict__ block_tables,    // [B, max_blocks]
    const int32_t* __restrict__ context_lens,    // [B]
    scalar_t* __restrict__ out,                  // [B, H, D]
    const int batch_size,
    const int num_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks,
    const float scale) {
  const int bh = (int)blockIdx.x;
  const int b = bh / num_heads;
  const int h = bh - b * num_heads;
  const int tid = (int)threadIdx.x;

  if (b >= batch_size) {
    return;
  }

  const int ctx = (int)context_lens[b];
  if (ctx <= 0) {
    for (int d = tid; d < head_dim; d += blockDim.x) {
      out[((b * num_heads + h) * head_dim) + d] = (scalar_t)0;
    }
    return;
  }

  extern __shared__ float smem[];
  float* reduce_buf = smem;                 // [blockDim.x]
  float* shared_scalar = smem + blockDim.x; // [3]
  float& s_dot = shared_scalar[0];
  float& s_weight = shared_scalar[1];
  float& s_denom = shared_scalar[2];

  const scalar_t* q_ptr = query + ((b * num_heads + h) * head_dim);

  float max_logit = -INFINITY;

  // Pass 1: compute max logit for stable softmax.
  for (int pos = 0; pos < ctx; ++pos) {
    const int bt_idx = pos / block_size;
    const int block_id = (int)block_tables[b * max_blocks + bt_idx];
    const int block_off = pos - bt_idx * block_size;

    const scalar_t* k_ptr =
        key_cache + (((block_id * num_heads + h) * block_size + block_off) * head_dim);

    float partial = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
      partial += (float)q_ptr[d] * (float)k_ptr[d];
    }

    reduce_buf[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_buf[tid] += reduce_buf[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      s_dot = reduce_buf[0] * scale;
      if (s_dot > max_logit) {
        max_logit = s_dot;
      }
    }
    __syncthreads();
  }

  // Pass 2: compute denom and output numerator.
  float denom = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out[((b * num_heads + h) * head_dim) + d] = (scalar_t)0;
  }
  __syncthreads();

  for (int pos = 0; pos < ctx; ++pos) {
    const int bt_idx = pos / block_size;
    const int block_id = (int)block_tables[b * max_blocks + bt_idx];
    const int block_off = pos - bt_idx * block_size;

    const scalar_t* k_ptr =
        key_cache + (((block_id * num_heads + h) * block_size + block_off) * head_dim);
    const scalar_t* v_ptr =
        value_cache + (((block_id * num_heads + h) * block_size + block_off) * head_dim);

    float partial = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
      partial += (float)q_ptr[d] * (float)k_ptr[d];
    }

    reduce_buf[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_buf[tid] += reduce_buf[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      s_dot = reduce_buf[0] * scale;
      s_weight = expf(s_dot - max_logit);
      denom += s_weight;
    }
    __syncthreads();

    const float w = s_weight;
    for (int d = tid; d < head_dim; d += blockDim.x) {
      const int out_idx = ((b * num_heads + h) * head_dim) + d;
      const float prev = (float)out[out_idx];
      const float next = prev + w * (float)v_ptr[d];
      out[out_idx] = (scalar_t)next;
    }
    __syncthreads();
  }

  if (tid == 0) {
    s_denom = denom;
  }
  __syncthreads();

  const float inv = 1.0f / s_denom;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const int out_idx = ((b * num_heads + h) * head_dim) + d;
    out[out_idx] = (scalar_t)((float)out[out_idx] * inv);
  }
}

torch::Tensor paged_attention_decode(torch::Tensor query,
                                    torch::Tensor key_cache,
                                    torch::Tensor value_cache,
                                    torch::Tensor block_tables,
                                    torch::Tensor context_lens,
                                    double scale) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be a CUDA tensor");
  TORCH_CHECK(context_lens.is_cuda(), "context_lens must be a CUDA tensor");

  TORCH_CHECK(query.dim() == 3, "query must have shape [B, H, D]");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must have shape [NB, H, BS, D]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache must match key_cache shape");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must have shape [B, max_blocks]");
  TORCH_CHECK(context_lens.dim() == 1, "context_lens must have shape [B]");

  TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
  TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");

  query = query.contiguous();
  key_cache = key_cache.contiguous();
  value_cache = value_cache.contiguous();
  block_tables = block_tables.contiguous();
  context_lens = context_lens.contiguous();

  const int batch_size = (int)query.size(0);
  const int num_heads = (int)query.size(1);
  const int head_dim = (int)query.size(2);

  TORCH_CHECK(key_cache.size(1) == num_heads, "key_cache num_heads mismatch");
  TORCH_CHECK(key_cache.size(3) == head_dim, "key_cache head_dim mismatch");

  const int block_size = (int)key_cache.size(2);
  const int max_blocks = (int)block_tables.size(1);

  auto out = torch::empty_like(query);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int threads = 256;
  const int blocks = batch_size * num_heads;
  const size_t shmem_bytes = (threads * sizeof(float)) + (3 * sizeof(float));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "paged_attention_decode_kernel", [&] {
    paged_attention_decode_kernel<scalar_t><<<blocks, threads, shmem_bytes, stream>>>(
        query.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        block_tables.data_ptr<int32_t>(),
        context_lens.data_ptr<int32_t>(),
        out.data_ptr<scalar_t>(),
        batch_size,
        num_heads,
        head_dim,
        block_size,
        max_blocks,
        (float)scale);
  });

  return out;
}

