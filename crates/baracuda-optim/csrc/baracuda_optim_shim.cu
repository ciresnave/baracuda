// baracuda-optim Phase 49 — C-ABI shim for the vendored Apex optimizer
// functors. Bridges the safe Rust wrapper to the device-side functors
// in `../vendor/apex/multi_tensor_{adam,lamb,sgd}.cuh`.
//
// Status code convention (matches baracuda-kernels-sys/src/lib.rs):
//   0 = ok
//   1 = misaligned operand
//   2 = invalid problem (n_tensors <= 0, depth mismatch, etc.)
//   3 = unsupported (unknown dtype combo)
//   4 = workspace too small
//   5 = internal CUDA launch failure

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../vendor/apex/multi_tensor_apply.cuh"
#include "../vendor/apex/multi_tensor_adam.cuh"
#include "../vendor/apex/multi_tensor_lamb.cuh"
#include "../vendor/apex/multi_tensor_sgd.cuh"

namespace ba = baracuda_apex;

namespace {

constexpr int STATUS_OK            = 0;
constexpr int STATUS_INVALID_ARG   = 2;
constexpr int STATUS_UNSUPPORTED   = 3;
constexpr int STATUS_LAUNCH_FAIL   = 5;

// Apex chunk size — Apex uses 2048 for all current optimizer functors
// (single-precision SIMD-friendly chunk size, ~8 KiB per chunk).
constexpr int CHUNK_SIZE = 2048;

// Element-count -> number of chunks for that tensor.
inline int chunks_for(int n) {
  return (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
}

// Pack a single launch's metadata (caller-supplied per-tensor pointer
// arrays + element counts) into the on-stack TensorListMetadata. Returns
// the number of (tensor, chunk) blocks the launch will dispatch.
//
// `start_tensor` is the index of the first tensor packed in this launch;
// caller advances it across multi-launch batches.
//
// `end_tensor` is set to the (exclusive) one-past-last index of the
// tensor consumed by this launch — the caller uses it as the start
// of the next launch. The split point is wherever we hit
// MAX_TENSORS_PER_LAUNCH OR MAX_BLOCKS_PER_LAUNCH.
template<int Depth>
int pack_metadata(
    ba::TensorListMetadata<Depth>& tl,
    int n_tensors,
    int start_tensor,
    int* end_tensor_out,
    const int*    sizes,
    void* const*  addrs_per_slot[Depth])
{
  int n_blocks_total = 0;
  int t = start_tensor;
  int packed_tensors = 0;

  tl.start_tensor_this_launch = start_tensor;

  while (t < n_tensors && packed_tensors < ba::MAX_TENSORS_PER_LAUNCH) {
    const int n_chunks = chunks_for(sizes[t]);
    if (n_blocks_total + n_chunks > ba::MAX_BLOCKS_PER_LAUNCH) {
      // Splitting at this tensor would overflow the per-launch block
      // budget. If we've already packed at least one tensor, we stop;
      // if not (huge single tensor), we still pack it and let it
      // consume more than MAX_BLOCKS_PER_LAUNCH worth of chunks.
      // (Apex handles the >MAX_BLOCKS case the same way.)
      if (packed_tensors > 0) break;
    }

    tl.sizes[packed_tensors] = sizes[t];
    #pragma unroll
    for (int d = 0; d < Depth; ++d) {
      tl.addresses[d][packed_tensors] = addrs_per_slot[d][t];
    }

    for (int c = 0; c < n_chunks; ++c) {
      if (n_blocks_total >= ba::MAX_BLOCKS_PER_LAUNCH) break;
      tl.block_to_tensor[n_blocks_total] = static_cast<unsigned char>(packed_tensors);
      tl.block_to_chunk[n_blocks_total]  = c;
      n_blocks_total++;
    }

    packed_tensors++;
    t++;
  }

  *end_tensor_out = t;
  return n_blocks_total;
}

// Adam launch wrapper template — instantiated per dtype combo.
template<typename GRAD_T, typename PARAM_T, typename STATE_T>
__global__ void adam_kernel_launch(
    int chunk_size,
    ba::TensorListMetadata<4> tl,
    float beta1, float beta2,
    float beta1_corr, float beta2_corr,
    float eps, float lr,
    ba::adamMode_t mode, float decay)
{
  ba::AdamFunctor<GRAD_T, PARAM_T, STATE_T> f;
  f(chunk_size, tl, beta1, beta2, beta1_corr, beta2_corr, eps, lr, mode, decay);
}

// SGD launch wrapper.
template<typename GRAD_T, typename PARAM_T, typename MOMENT_T>
__global__ void sgd_kernel_launch(
    int chunk_size,
    ba::TensorListMetadata<3> tl,
    float weight_decay, float momentum, float dampening, float lr,
    bool nesterov, bool first_run, bool wd_after_momentum,
    float grad_scale)
{
  ba::SgdFunctor<GRAD_T, PARAM_T, MOMENT_T> f;
  f(chunk_size, tl, weight_decay, momentum, dampening, lr,
    nesterov, first_run, wd_after_momentum, grad_scale);
}

// LAMB stage 1 kernel.
template<typename T>
__global__ void lamb_stage1_kernel_launch(
    int chunk_size,
    ba::TensorListMetadata<4> tl,
    void** u_scratch_per_tensor,
    float* w_norm_partial,
    float* u_norm_partial,
    float beta1, float beta2,
    float beta1_corr, float beta2_corr,
    float eps, float decay,
    ba::adamMode_t mode,
    float global_grad_norm,
    float max_global_grad_norm)
{
  ba::LambStage1Functor<T> f;
  f(chunk_size, tl, u_scratch_per_tensor, w_norm_partial, u_norm_partial,
    beta1, beta2, beta1_corr, beta2_corr, eps, decay, mode,
    global_grad_norm, max_global_grad_norm);
}

// Per-tensor sqrt — runs on `num_tensors` elements only, so we use
// a tiny single-block launch.
__global__ void elementwise_sqrt_inplace(float* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = sqrtf(x[i]);
}

// LAMB stage 2 kernel.
template<typename T>
__global__ void lamb_stage2_kernel_launch(
    int chunk_size,
    ba::TensorListMetadata<2> tl,
    const float* w_norm,
    const float* u_norm,
    float lr, float lr_lo, float lr_hi)
{
  ba::LambStage2Functor<T> f;
  f(chunk_size, tl, w_norm, u_norm, lr, lr_lo, lr_hi);
}

} // namespace

extern "C" {

// ============================================================================
// Adam — depth-4 multi-tensor apply
// ============================================================================

// Run the multi-tensor Adam update.
//
// Dtype convention: this entry is f32-throughout (param, grad, state
// all f32). Half/bf16 variants are name-suffixed (`_f16` / `_bf16` for
// param+grad with f32 state — standard mixed-precision wiring).
//
// Pointer-array contract: `param_ptrs[t]` / `grad_ptrs[t]` /
// `exp_avg_ptrs[t]` / `exp_avg_sq_ptrs[t]` all point at the start of
// the same logical tensor `t` (no offsets). `sizes[t]` = element count.
//
// Bias-correction: when `bias_correction == 0`, the caller is
// responsible for pre-scaling lr (Apex convention).
int baracuda_optim_adam_f32_run(
    int32_t n_tensors,
    const int32_t* sizes,
    void* const*   param_ptrs,
    void* const*   grad_ptrs,
    void* const*   exp_avg_ptrs,
    void* const*   exp_avg_sq_ptrs,
    int32_t        step,            // 1-indexed; used for bias correction
    float          lr,
    float          beta1,
    float          beta2,
    float          epsilon,
    float          weight_decay,
    int32_t        bias_correction, // 1 = on, 0 = off (caller pre-scales lr)
    int32_t        adamw_mode,      // 1 = AdamW (decoupled decay), 0 = classic
    void*          stream)
{
  if (n_tensors <= 0) return STATUS_INVALID_ARG;
  if (!sizes || !param_ptrs || !grad_ptrs || !exp_avg_ptrs || !exp_avg_sq_ptrs)
    return STATUS_INVALID_ARG;

  const float beta1_corr = bias_correction ? (1.0f - powf(beta1, static_cast<float>(step))) : 1.0f;
  const float beta2_corr = bias_correction ? (1.0f - powf(beta2, static_cast<float>(step))) : 1.0f;
  const ba::adamMode_t mode = adamw_mode ? ba::adamMode_t::ADAMW : ba::adamMode_t::ADAM;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

  // Multi-launch loop — Apex's pattern.
  void* const* addrs[4] = { param_ptrs, grad_ptrs, exp_avg_ptrs, exp_avg_sq_ptrs };
  int start = 0;
  while (start < n_tensors) {
    ba::TensorListMetadata<4> tl;
    std::memset(&tl, 0, sizeof(tl));
    int end = start;
    int n_blocks = pack_metadata<4>(tl, n_tensors, start, &end, sizes, addrs);
    if (n_blocks == 0) break;
    adam_kernel_launch<float, float, float><<<n_blocks, ba::BLOCK_SIZE, 0, s>>>(
        CHUNK_SIZE, tl, beta1, beta2, beta1_corr, beta2_corr,
        epsilon, lr, mode, weight_decay);
    start = end;
  }

  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? STATUS_OK : STATUS_LAUNCH_FAIL;
}

// Mixed precision: f16 param+grad with f32 state.
int baracuda_optim_adam_f16_run(
    int32_t n_tensors,
    const int32_t* sizes,
    void* const*   param_ptrs,       // __half*
    void* const*   grad_ptrs,        // __half*
    void* const*   exp_avg_ptrs,     // float*
    void* const*   exp_avg_sq_ptrs,  // float*
    int32_t step, float lr,
    float beta1, float beta2, float epsilon, float weight_decay,
    int32_t bias_correction, int32_t adamw_mode, void* stream)
{
  if (n_tensors <= 0) return STATUS_INVALID_ARG;
  if (!sizes || !param_ptrs || !grad_ptrs || !exp_avg_ptrs || !exp_avg_sq_ptrs)
    return STATUS_INVALID_ARG;

  const float beta1_corr = bias_correction ? (1.0f - powf(beta1, static_cast<float>(step))) : 1.0f;
  const float beta2_corr = bias_correction ? (1.0f - powf(beta2, static_cast<float>(step))) : 1.0f;
  const ba::adamMode_t mode = adamw_mode ? ba::adamMode_t::ADAMW : ba::adamMode_t::ADAM;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  void* const* addrs[4] = { param_ptrs, grad_ptrs, exp_avg_ptrs, exp_avg_sq_ptrs };
  int start = 0;
  while (start < n_tensors) {
    ba::TensorListMetadata<4> tl;
    std::memset(&tl, 0, sizeof(tl));
    int end = start;
    int n_blocks = pack_metadata<4>(tl, n_tensors, start, &end, sizes, addrs);
    if (n_blocks == 0) break;
    adam_kernel_launch<__half, __half, float><<<n_blocks, ba::BLOCK_SIZE, 0, s>>>(
        CHUNK_SIZE, tl, beta1, beta2, beta1_corr, beta2_corr,
        epsilon, lr, mode, weight_decay);
    start = end;
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? STATUS_OK : STATUS_LAUNCH_FAIL;
}

// Mixed precision: bf16 param+grad with f32 state.
int baracuda_optim_adam_bf16_run(
    int32_t n_tensors,
    const int32_t* sizes,
    void* const*   param_ptrs,
    void* const*   grad_ptrs,
    void* const*   exp_avg_ptrs,
    void* const*   exp_avg_sq_ptrs,
    int32_t step, float lr,
    float beta1, float beta2, float epsilon, float weight_decay,
    int32_t bias_correction, int32_t adamw_mode, void* stream)
{
  if (n_tensors <= 0) return STATUS_INVALID_ARG;
  if (!sizes || !param_ptrs || !grad_ptrs || !exp_avg_ptrs || !exp_avg_sq_ptrs)
    return STATUS_INVALID_ARG;

  const float beta1_corr = bias_correction ? (1.0f - powf(beta1, static_cast<float>(step))) : 1.0f;
  const float beta2_corr = bias_correction ? (1.0f - powf(beta2, static_cast<float>(step))) : 1.0f;
  const ba::adamMode_t mode = adamw_mode ? ba::adamMode_t::ADAMW : ba::adamMode_t::ADAM;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  void* const* addrs[4] = { param_ptrs, grad_ptrs, exp_avg_ptrs, exp_avg_sq_ptrs };
  int start = 0;
  while (start < n_tensors) {
    ba::TensorListMetadata<4> tl;
    std::memset(&tl, 0, sizeof(tl));
    int end = start;
    int n_blocks = pack_metadata<4>(tl, n_tensors, start, &end, sizes, addrs);
    if (n_blocks == 0) break;
    adam_kernel_launch<__nv_bfloat16, __nv_bfloat16, float><<<n_blocks, ba::BLOCK_SIZE, 0, s>>>(
        CHUNK_SIZE, tl, beta1, beta2, beta1_corr, beta2_corr,
        epsilon, lr, mode, weight_decay);
    start = end;
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? STATUS_OK : STATUS_LAUNCH_FAIL;
}

// ============================================================================
// SGD — depth-3 multi-tensor apply
// ============================================================================

int baracuda_optim_sgd_f32_run(
    int32_t n_tensors,
    const int32_t* sizes,
    void* const*   param_ptrs,
    void* const*   grad_ptrs,
    void* const*   momentum_ptrs,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    int32_t nesterov,
    int32_t first_run,
    int32_t wd_after_momentum,
    float grad_scale,
    void* stream)
{
  if (n_tensors <= 0) return STATUS_INVALID_ARG;
  if (!sizes || !param_ptrs || !grad_ptrs || !momentum_ptrs) return STATUS_INVALID_ARG;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  void* const* addrs[3] = { param_ptrs, grad_ptrs, momentum_ptrs };
  int start = 0;
  while (start < n_tensors) {
    ba::TensorListMetadata<3> tl;
    std::memset(&tl, 0, sizeof(tl));
    int end = start;
    int n_blocks = pack_metadata<3>(tl, n_tensors, start, &end, sizes, addrs);
    if (n_blocks == 0) break;
    sgd_kernel_launch<float, float, float><<<n_blocks, ba::BLOCK_SIZE, 0, s>>>(
        CHUNK_SIZE, tl, weight_decay, momentum, dampening, lr,
        nesterov != 0, first_run != 0, wd_after_momentum != 0, grad_scale);
    start = end;
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? STATUS_OK : STATUS_LAUNCH_FAIL;
}

int baracuda_optim_sgd_f16_run(
    int32_t n_tensors,
    const int32_t* sizes,
    void* const*   param_ptrs,
    void* const*   grad_ptrs,
    void* const*   momentum_ptrs,    // f32 momentum (mixed-precision)
    float lr, float momentum, float dampening, float weight_decay,
    int32_t nesterov, int32_t first_run, int32_t wd_after_momentum,
    float grad_scale, void* stream)
{
  if (n_tensors <= 0) return STATUS_INVALID_ARG;
  if (!sizes || !param_ptrs || !grad_ptrs || !momentum_ptrs) return STATUS_INVALID_ARG;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  void* const* addrs[3] = { param_ptrs, grad_ptrs, momentum_ptrs };
  int start = 0;
  while (start < n_tensors) {
    ba::TensorListMetadata<3> tl;
    std::memset(&tl, 0, sizeof(tl));
    int end = start;
    int n_blocks = pack_metadata<3>(tl, n_tensors, start, &end, sizes, addrs);
    if (n_blocks == 0) break;
    sgd_kernel_launch<__half, __half, float><<<n_blocks, ba::BLOCK_SIZE, 0, s>>>(
        CHUNK_SIZE, tl, weight_decay, momentum, dampening, lr,
        nesterov != 0, first_run != 0, wd_after_momentum != 0, grad_scale);
    start = end;
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? STATUS_OK : STATUS_LAUNCH_FAIL;
}

int baracuda_optim_sgd_bf16_run(
    int32_t n_tensors,
    const int32_t* sizes,
    void* const*   param_ptrs,
    void* const*   grad_ptrs,
    void* const*   momentum_ptrs,
    float lr, float momentum, float dampening, float weight_decay,
    int32_t nesterov, int32_t first_run, int32_t wd_after_momentum,
    float grad_scale, void* stream)
{
  if (n_tensors <= 0) return STATUS_INVALID_ARG;
  if (!sizes || !param_ptrs || !grad_ptrs || !momentum_ptrs) return STATUS_INVALID_ARG;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  void* const* addrs[3] = { param_ptrs, grad_ptrs, momentum_ptrs };
  int start = 0;
  while (start < n_tensors) {
    ba::TensorListMetadata<3> tl;
    std::memset(&tl, 0, sizeof(tl));
    int end = start;
    int n_blocks = pack_metadata<3>(tl, n_tensors, start, &end, sizes, addrs);
    if (n_blocks == 0) break;
    sgd_kernel_launch<__nv_bfloat16, __nv_bfloat16, float><<<n_blocks, ba::BLOCK_SIZE, 0, s>>>(
        CHUNK_SIZE, tl, weight_decay, momentum, dampening, lr,
        nesterov != 0, first_run != 0, wd_after_momentum != 0, grad_scale);
    start = end;
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? STATUS_OK : STATUS_LAUNCH_FAIL;
}

// ============================================================================
// LAMB — two-stage multi-tensor apply
// ============================================================================
//
// Two-stage LAMB:
//   Stage 1 — Adam update + per-tensor norm accumulation (atomicAdd).
//   Sqrt   — single small kernel converting w_norm² and u_norm² to norms.
//   Stage 2 — trust-ratio + parameter update.
//
// Caller must provide:
//   - u_scratch device array, per-tensor [num_tensors] pointer-array,
//     each entry pointing at a tensor-sized scratch buffer.
//   - w_norm_partial: f32[num_tensors], device, initialized to 0.
//   - u_norm_partial: f32[num_tensors], device, initialized to 0.
//
// The shim zeros the two norm buffers (cudaMemsetAsync) before launch
// so the caller doesn't have to.

int baracuda_optim_lamb_f32_run(
    int32_t n_tensors,
    const int32_t* sizes,
    void* const*   param_ptrs,
    void* const*   grad_ptrs,
    void* const*   exp_avg_ptrs,
    void* const*   exp_avg_sq_ptrs,
    void* const*   u_scratch_ptrs,     // [num_tensors] -> per-tensor scratch
    void*          u_scratch_host_to_dev, // void**[num_tensors] device array;
                                          // caller responsible for staging
                                          // (we don't allocate inside FFI)
    void*          w_norm_dev,          // float[num_tensors] device
    void*          u_norm_dev,          // float[num_tensors] device
    int32_t        step,
    float          lr,
    float          beta1, float beta2,
    float          epsilon,
    float          weight_decay,
    int32_t        bias_correction,
    int32_t        adamw_mode,
    float          global_grad_norm,
    float          max_global_grad_norm,
    float          lr_lower_bound,
    float          lr_upper_bound,
    void*          stream)
{
  (void)param_ptrs; (void)grad_ptrs; (void)exp_avg_ptrs;
  (void)exp_avg_sq_ptrs; (void)u_scratch_ptrs;
  if (n_tensors <= 0) return STATUS_INVALID_ARG;
  if (!sizes || !w_norm_dev || !u_norm_dev || !u_scratch_host_to_dev)
    return STATUS_INVALID_ARG;

  const float beta1_corr = bias_correction ? (1.0f - powf(beta1, static_cast<float>(step))) : 1.0f;
  const float beta2_corr = bias_correction ? (1.0f - powf(beta2, static_cast<float>(step))) : 1.0f;
  const ba::adamMode_t mode = adamw_mode ? ba::adamMode_t::ADAMW : ba::adamMode_t::ADAM;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

  // Zero the norm scratch on the stream.
  cudaMemsetAsync(w_norm_dev, 0, sizeof(float) * n_tensors, s);
  cudaMemsetAsync(u_norm_dev, 0, sizeof(float) * n_tensors, s);

  // Stage 1.
  {
    void* const* addrs[4] = { param_ptrs, grad_ptrs, exp_avg_ptrs, exp_avg_sq_ptrs };
    int start = 0;
    while (start < n_tensors) {
      ba::TensorListMetadata<4> tl;
      std::memset(&tl, 0, sizeof(tl));
      int end = start;
      int n_blocks = pack_metadata<4>(tl, n_tensors, start, &end, sizes, addrs);
      if (n_blocks == 0) break;
      lamb_stage1_kernel_launch<float><<<n_blocks, ba::BLOCK_SIZE, 0, s>>>(
          CHUNK_SIZE, tl,
          static_cast<void**>(u_scratch_host_to_dev),
          static_cast<float*>(w_norm_dev),
          static_cast<float*>(u_norm_dev),
          beta1, beta2, beta1_corr, beta2_corr,
          epsilon, weight_decay, mode,
          global_grad_norm, max_global_grad_norm);
      start = end;
    }
  }

  // sqrt the norms in-place. One small launch — num_tensors typically
  // <= 2000 for an LLM, so a single block suffices.
  {
    const int threads = 128;
    const int blocks  = (n_tensors + threads - 1) / threads;
    elementwise_sqrt_inplace<<<blocks, threads, 0, s>>>(static_cast<float*>(w_norm_dev), n_tensors);
    elementwise_sqrt_inplace<<<blocks, threads, 0, s>>>(static_cast<float*>(u_norm_dev), n_tensors);
  }

  // Stage 2.
  {
    void* const* addrs2[2] = { param_ptrs, u_scratch_ptrs };
    int start = 0;
    while (start < n_tensors) {
      ba::TensorListMetadata<2> tl2;
      std::memset(&tl2, 0, sizeof(tl2));
      int end = start;
      int n_blocks = pack_metadata<2>(tl2, n_tensors, start, &end, sizes, addrs2);
      if (n_blocks == 0) break;
      lamb_stage2_kernel_launch<float><<<n_blocks, ba::BLOCK_SIZE, 0, s>>>(
          CHUNK_SIZE, tl2,
          static_cast<const float*>(w_norm_dev),
          static_cast<const float*>(u_norm_dev),
          lr, lr_lower_bound, lr_upper_bound);
      start = end;
    }
  }

  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? STATUS_OK : STATUS_LAUNCH_FAIL;
}

// LAMB host-side helper — for the safe-Rust wrapper to compute the
// chunk count for a given problem (used to size the launch loop and
// for the multi-tensor-dispatch perf smoke test).
int32_t baracuda_optim_chunk_count(int32_t total_elements) {
  if (total_elements <= 0) return 0;
  return (total_elements + CHUNK_SIZE - 1) / CHUNK_SIZE;
}

int32_t baracuda_optim_chunk_size() {
  return CHUNK_SIZE;
}

int32_t baracuda_optim_max_tensors_per_launch() {
  return ba::MAX_TENSORS_PER_LAUNCH;
}

int32_t baracuda_optim_max_blocks_per_launch() {
  return ba::MAX_BLOCKS_PER_LAUNCH;
}

} // extern "C"
