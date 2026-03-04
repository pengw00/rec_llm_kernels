#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#
#include <cmath>
#
// Llama-style RoPE:
// For each (pos, d) where d in [0, rotary_dim/2):
//   angle = pos * base ** (-d/(rotary_dim/2))
//   [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
//
// Supports q/k shapes:
//   - [B, H, D] with positions [B]
//   - [T, H, D] with positions [T]
//
// Rotary is applied to the first rotary_dim dims (defaults to D).

namespace {

__device__ __forceinline__ float rope_inv_freq(int d, int half_rotary_dim, float log_base) {
  // base ** (-d/half) = exp(-log(base) * d / half)
  return expf(-log_base * ((float)d / (float)half_rotary_dim));
}

template <typename scalar_t>
__global__ void apply_rope_kernel(
    const scalar_t* __restrict__ q_in,  // [T, H, D] or [B, H, D] flattened
    const scalar_t* __restrict__ k_in,  // same shape
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    const int64_t* __restrict__ positions,  // [T] or [B]
    int tokens,
    int num_heads,
    int head_dim,
    int rotary_dim,
    float base) {
  const int token = (int)blockIdx.x;
  const int head = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;

  if (token >= tokens || head >= num_heads) {
    return;
  }

  const int half = rotary_dim / 2;
  const float log_base = logf(base);
  const float pos = (float)positions[token];

  const int64_t base_idx = ((int64_t)token * num_heads + head) * head_dim;

  // Rotate first rotary_dim dims (2*half). Copy remaining dims unchanged.
  for (int d = tid; d < half; d += blockDim.x) {
    const float inv = rope_inv_freq(d, half, log_base);
    const float angle = pos * inv;
    float c, s;
    // CUDA provides sincosf for efficient paired computation
    sincosf(angle, &s, &c);

    const int idx1 = (int)base_idx + d;
    const int idx2 = (int)base_idx + d + half;

    const float q1 = (float)q_in[idx1];
    const float q2 = (float)q_in[idx2];
    const float k1 = (float)k_in[idx1];
    const float k2 = (float)k_in[idx2];

    q_out[idx1] = (scalar_t)(q1 * c - q2 * s);
    q_out[idx2] = (scalar_t)(q1 * s + q2 * c);
    k_out[idx1] = (scalar_t)(k1 * c - k2 * s);
    k_out[idx2] = (scalar_t)(k1 * s + k2 * c);
  }

  // Copy tail dims (if any)
  for (int d = tid + rotary_dim; d < head_dim; d += blockDim.x) {
    const int idx = (int)base_idx + d;
    q_out[idx] = q_in[idx];
    k_out[idx] = k_in[idx];
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> apply_rope(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor positions,
    double base,
    int64_t rotary_dim) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
  TORCH_CHECK(q.device() == k.device(), "q/k device mismatch");
  TORCH_CHECK(q.device() == positions.device(), "positions must be on the same device as q/k");

  TORCH_CHECK(q.dim() == 3, "q must have shape [T, H, D] or [B, H, D]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must have the same shape as q");
  TORCH_CHECK(positions.dim() == 1, "positions must have shape [T] or [B]");
  TORCH_CHECK(positions.size(0) == q.size(0), "positions length must match q.size(0)");

  const int tokens = (int)q.size(0);
  const int num_heads = (int)q.size(1);
  const int head_dim = (int)q.size(2);

  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for RoPE");

  int rdim = (int)(rotary_dim <= 0 ? head_dim : rotary_dim);
  TORCH_CHECK(rdim % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(rdim <= head_dim, "rotary_dim must be <= head_dim");

  // Ensure contiguous for simple indexing math.
  q = q.contiguous();
  k = k.contiguous();
  if (positions.scalar_type() != at::kLong) {
    positions = positions.to(at::kLong);
  }
  positions = positions.contiguous();

  auto q_out = torch::empty_like(q);
  auto k_out = torch::empty_like(k);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(tokens, num_heads);
  int threads = 256;
  threads = std::min(threads, rdim / 2);
  if (threads < 32) {
    threads = 32;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
                                  "apply_rope_kernel", [&] {
    apply_rope_kernel<scalar_t><<<grid, threads, 0, stream>>>(
        (const scalar_t*)q.data_ptr<scalar_t>(),
        (const scalar_t*)k.data_ptr<scalar_t>(),
        (scalar_t*)q_out.data_ptr<scalar_t>(),
        (scalar_t*)k_out.data_ptr<scalar_t>(),
        (const int64_t*)positions.data_ptr<int64_t>(),
        tokens,
        num_heads,
        head_dim,
        rdim,
        (float)base);
  });

  return {q_out, k_out};
}

