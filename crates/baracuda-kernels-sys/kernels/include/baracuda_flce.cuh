// baracuda_flce.cuh
//
// Phase 47 — Fused Linear Cross-Entropy kernels.
//
// Math/algorithm credit: LinkedIn Liger-Kernel
//   `src/liger_kernel/ops/fused_linear_cross_entropy.py` (BSD-2-Clause).
//   https://github.com/linkedin/Liger-Kernel
// Original Triton kernel by Pin-Lun Hsu et al. (LinkedIn, 2024).
// This is a clean-room CUDA reimplementation following the same
// chunked-tile algorithm; no Liger source is vendored.
//
// What this header covers (the per-chunk fused step):
//
//   Input per chunk of size `n_rows × V`:
//     `logits`   — pre-computed by the Rust-side `GemmPlan`
//                  (`logits = input_chunk @ weight.T`), `[n_rows, V]`,
//                  row-major.
//     `target`   — `i64[n_rows]` class indices.
//
//   The kernel does, per row, in **one** pass over `V`:
//     1. find max (numerical stability)
//     2. sum(exp(logit - max))                  — partial sums into smem
//     3. write per-cell gradient
//          `grad_logits[i, j] = (softmax(logits)[i, j] - 1{j == target[i]})
//                                · scale`
//        IN PLACE (overwriting `logits`).
//     4. write per-row loss term `-(logit[target] - max - log(sum_exp))`
//        into `loss_1d[i]`.
//
//   The caller's outer loop then runs (still per-chunk):
//     * `grad_input_chunk = grad_logits_chunk @ weight`         (cuBLAS)
//     * `grad_weight += grad_logits_chunk^T @ input_chunk`      (cuBLAS,
//                                                                 beta=1)
//
//   Final reduction: a single-block tree reduction over `loss_1d` to
//   produce the scalar loss (Mean / Sum). Reused
//   `loss_reduce_finalize_kernel` from `baracuda_loss.cuh`.
//
// Key value-prop: never materializes the full `[BT, V]` logits tensor.
// At chunk_size=2048, V=128K, BT=16K, the saving is `(BT - chunk) * V
// * sizeof(T)` = ~3.5 GiB in bf16 — exactly the memory cliff LLM
// trainers see at vocab >= 128K.
//
// Reduction modes:
//   0 = None  (per-row loss; no scalar reduction)
//   1 = Mean  (scalar loss = sum / N_non_ignore)
//   2 = Sum   (scalar loss = sum)
//
// `scale_per_row` is the factor folded into each row's gradient at
// per-chunk fused step. For Mean it's `1 / N_non_ignore`; for Sum and
// None it's `1`. (Liger applies the divide once up front so the
// gradient is "loss-reduction-aware".)
//
// `ignore_index`: rows whose `target[i] == ignore_index` contribute
// zero loss AND zero gradient (`grad_logits[i, :] = 0`).
//
// Status codes follow the rest of the kernels-sys ABI:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

#ifndef BARACUDA_FLCE_CUH
#define BARACUDA_FLCE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_loss.cuh" // for load_as_acc / store_from_acc / loss_reduce_finalize_kernel

namespace baracuda { namespace flce {

// Bring in the f32-detour load/store helpers from `baracuda::loss`. They
// have the same dtype-detour contract we need here (f16/bf16 detour
// through f32, native otherwise) — no point duplicating.
using baracuda::loss::load_as_acc;
using baracuda::loss::store_from_acc;

// One block per row; threads cooperatively scan the V dimension.
// Block size is fixed at 256 to keep occupancy reasonable across all
// V sizes (the inner loop is gmem-bound — bigger blocks don't help).
constexpr int kFlceBlock = 256;

// Per-block 3-pass kernel: max, sum-exp, gradient-write.
// Reads & writes `logits` once each for steps 1+2, then once more for
// step 3 — total 3 passes over the row. Acceptable: V is the dominant
// term and modern GDDR6X / HBM is ~7 GB/s/SM; the per-row work is
// O(V) regardless of how we split it.
template <typename T>
__global__ void flce_per_row_kernel(
    T* __restrict__ logits,             // [n_rows, V] row-major; mutated to grad_logits
    const int64_t* __restrict__ target, // [n_rows]
    float* __restrict__ loss_1d,        // [n_rows] fp32 accumulator
    int32_t v,                          // vocab extent
    int64_t row_stride,                 // logits row stride (elements)
    int64_t target_ignore,              // ignore_index (default -100)
    float scale_per_row)                // gradient pre-scale (1/N for Mean)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float smem[kFlceBlock];

    const int64_t t = target[row];
    T* row_ptr = logits + (int64_t)row * row_stride;

    // Ignore-index path: zero the gradient row and the loss; nothing else.
    if (t == target_ignore) {
        if (tid == 0) loss_1d[row] = 0.0f;
        for (int j = tid; j < v; j += kFlceBlock) {
            row_ptr[j] = store_from_acc<T>(0.0f);
        }
        return;
    }

    // --- Pass 1: find max -----------------------------------------------
    float local_max = -INFINITY;
    for (int j = tid; j < v; j += kFlceBlock) {
        float val = load_as_acc<T>(row_ptr[j]);
        if (val > local_max) local_max = val;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = kFlceBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = smem[tid + s];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }
    float row_max = smem[0];

    // --- Pass 2: sum(exp(logit - max)) ----------------------------------
    float local_sum = 0.0f;
    for (int j = tid; j < v; j += kFlceBlock) {
        local_sum += expf(load_as_acc<T>(row_ptr[j]) - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = kFlceBlock / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];
    float inv_row_sum = 1.0f / row_sum;
    float log_row_sum = logf(row_sum);

    // --- Loss term (per-row scalar) -------------------------------------
    // Bounds-check target — same contract as `cross_entropy_per_row_kernel`.
    if (tid == 0) {
        if (t < 0 || t >= (int64_t)v) {
            loss_1d[row] = 0.0f;
        } else {
            float xt = load_as_acc<T>(row_ptr[t]);
            // -log_softmax[t] = -(xt - row_max - log(row_sum))
            loss_1d[row] = -xt + row_max + log_row_sum;
        }
    }

    // --- Pass 3: in-place gradient write --------------------------------
    // grad_logits[i, j] = (softmax[i, j] - 1{j == t}) · scale_per_row
    const int t_int = (t >= 0 && t < (int64_t)v) ? (int)t : -1;
    for (int j = tid; j < v; j += kFlceBlock) {
        float p = expf(load_as_acc<T>(row_ptr[j]) - row_max) * inv_row_sum;
        float one_hot = (j == t_int) ? 1.0f : 0.0f;
        row_ptr[j] = store_from_acc<T>((p - one_hot) * scale_per_row);
    }
}

// f64 specialization — keeps accumulators in double; no `expf`/`logf` for fp64.
template <>
__global__ void flce_per_row_kernel<double>(
    double* __restrict__ logits,
    const int64_t* __restrict__ target,
    float* __restrict__ loss_1d,
    int32_t v,
    int64_t row_stride,
    int64_t target_ignore,
    float scale_per_row)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ double smem[kFlceBlock];

    const int64_t t = target[row];
    double* row_ptr = logits + (int64_t)row * row_stride;

    if (t == target_ignore) {
        if (tid == 0) loss_1d[row] = 0.0f;
        for (int j = tid; j < v; j += kFlceBlock) {
            row_ptr[j] = 0.0;
        }
        return;
    }

    double local_max = -INFINITY;
    for (int j = tid; j < v; j += kFlceBlock) {
        double val = row_ptr[j];
        if (val > local_max) local_max = val;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = kFlceBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double other = smem[tid + s];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }
    double row_max = smem[0];

    double local_sum = 0.0;
    for (int j = tid; j < v; j += kFlceBlock) {
        local_sum += exp(row_ptr[j] - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = kFlceBlock / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    double row_sum = smem[0];
    double inv_row_sum = 1.0 / row_sum;
    double log_row_sum = log(row_sum);

    if (tid == 0) {
        if (t < 0 || t >= (int64_t)v) {
            loss_1d[row] = 0.0f;
        } else {
            double xt = row_ptr[t];
            // Cast scalar loss to f32 for the shared finalize kernel.
            // The accumulator buffer is f32 because the finalizer is
            // single-template across all dtypes for code-reuse — f64
            // precision in the per-row loss isn't a meaningful win
            // (we'd lose it again in the finalize step), and the
            // cross-entropy term is bounded log(V) anyway.
            loss_1d[row] = (float)(-xt + row_max + log_row_sum);
        }
    }

    const int t_int = (t >= 0 && t < (int64_t)v) ? (int)t : -1;
    double scale_d = (double)scale_per_row;
    for (int j = tid; j < v; j += kFlceBlock) {
        double p = exp(row_ptr[j] - row_max) * inv_row_sum;
        double one_hot = (j == t_int) ? 1.0 : 0.0;
        row_ptr[j] = (p - one_hot) * scale_d;
    }
}

// =============================================================================
// Loss-1d → scalar finalize. We reuse `loss_reduce_finalize_kernel` from
// `baracuda_loss.cuh` against the **f32** accumulator buffer, then cast
// the final scalar to T. For Mean: the per-row scale is already folded
// into the gradient (so the gradient is "post-mean"), but the LOSS still
// needs `sum / N_non_ignore` at the end. For Sum: no divide.
//
// `denom_inv = 1 / N_non_ignore` for Mean, 1.0 for Sum.
// =============================================================================

template <typename T>
__global__ void flce_scalar_finalize_kernel(
    const float* __restrict__ loss_1d,
    T* __restrict__ scalar_out,
    int64_t n_rows,
    float denom_inv)
{
    __shared__ float smem[256];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int64_t i = tid; i < n_rows; i += 256) {
        acc += loss_1d[i];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        scalar_out[0] = store_from_acc<T>(smem[0] * denom_inv);
    }
}

// f64 specialization — accumulate in double for the finalize step.
template <>
__global__ void flce_scalar_finalize_kernel<double>(
    const float* __restrict__ loss_1d,
    double* __restrict__ scalar_out,
    int64_t n_rows,
    float denom_inv)
{
    __shared__ double smem[256];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int64_t i = tid; i < n_rows; i += 256) {
        acc += (double)loss_1d[i];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        scalar_out[0] = smem[0] * (double)denom_inv;
    }
}

// =============================================================================
// Per-row loss copy: cast f32 loss_1d → T per-row out (reduction_mode=0).
// =============================================================================

template <typename T>
__global__ void flce_per_row_cast_kernel(
    const float* __restrict__ loss_1d,
    T* __restrict__ out,
    int64_t n_rows)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        out[i] = store_from_acc<T>(loss_1d[i]);
    }
}

template <>
__global__ void flce_per_row_cast_kernel<double>(
    const float* __restrict__ loss_1d,
    double* __restrict__ out,
    int64_t n_rows)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        out[i] = (double)loss_1d[i];
    }
}

// =============================================================================
// Multiply-in-place: dout *= scalar.  Used in BW to apply the upstream
// scalar `grad_output` to `grad_input` / `grad_weight` saved during FW.
//
// `scalar_dy_host` is a HOST f32 (caller pre-multiplies the f32 cast of
// the device scalar) — keeps the BW interface dead-simple and avoids a
// device-pointer-scalar dance.
// =============================================================================

template <typename T>
__global__ void flce_inplace_scale_kernel(
    T* __restrict__ buf,
    int64_t numel,
    float scalar)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (int64_t k = i; k < numel; k += step) {
        float v = load_as_acc<T>(buf[k]) * scalar;
        buf[k] = store_from_acc<T>(v);
    }
}

template <>
__global__ void flce_inplace_scale_kernel<double>(
    double* __restrict__ buf,
    int64_t numel,
    float scalar_f)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;
    double scalar = (double)scalar_f;
    for (int64_t k = i; k < numel; k += step) {
        buf[k] = buf[k] * scalar;
    }
}

// =============================================================================
// count_non_ignore: writes the i64 count of `target[i] != ignore_index`
// into `count_out[0]`. Single-block tree reduction; caller pre-zeros
// count_out.
// =============================================================================

// Plain `__global__` (not `__global__ inline` — that combo is rejected
// by some nvcc versions). Defined `static` to avoid multiple-definition
// link errors when this header is included by more than one TU.
static __global__ void flce_count_non_ignore_kernel(
    const int64_t* __restrict__ target,
    int64_t* __restrict__ count_out,  // [1]; pre-zeroed
    int32_t bt,
    int64_t ignore_index)
{
    __shared__ int64_t smem[256];
    int tid = threadIdx.x;
    int64_t local = 0;
    for (int i = tid; i < bt; i += 256) {
        if (target[i] != ignore_index) local += 1;
    }
    smem[tid] = local;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) count_out[0] = smem[0];
}

} } // namespace baracuda::flce

// =============================================================================
// Host launcher macros.
// =============================================================================

// Per-chunk fused step. Mutates `logits` -> `grad_logits` in place.
// Returns the per-row f32 loss in `loss_1d` (caller-owned).
#define BARACUDA_KERNELS_FLCE_PER_ROW_INSTANTIATE(NAME, T)                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t n_rows,                                                                             \
        int32_t v,                                                                                  \
        int64_t row_stride,                                                                         \
        int64_t target_ignore,                                                                      \
        float scale_per_row,                                                                        \
        void* logits, const void* target, void* loss_1d,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0 || v < 1) return 2;                                                         \
        if (n_rows == 0) return 0;                                                                  \
        if (logits == nullptr || target == nullptr || loss_1d == nullptr) return 2;                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        dim3 grid(n_rows);                                                                          \
        dim3 block(baracuda::flce::kFlceBlock);                                                     \
        baracuda::flce::flce_per_row_kernel<T><<<grid, block, 0, stream>>>(                         \
            static_cast<T*>(logits),                                                                \
            static_cast<const int64_t*>(target),                                                    \
            static_cast<float*>(loss_1d),                                                           \
            v, row_stride, target_ignore, scale_per_row);                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// Per-row loss copy (reduction_mode = None). Casts f32 loss_1d → T per-cell.
#define BARACUDA_KERNELS_FLCE_PER_ROW_CAST_INSTANTIATE(NAME, T)                                     \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows,                                                                             \
        const void* loss_1d, void* out,                                                             \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0) return 2;                                                                   \
        if (n_rows == 0) return 0;                                                                  \
        if (loss_1d == nullptr || out == nullptr) return 2;                                         \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int blocks = (int)((n_rows + kBlock - 1) / kBlock);                                         \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::flce::flce_per_row_cast_kernel<T><<<blocks, kBlock, 0, stream>>>(                 \
            static_cast<const float*>(loss_1d),                                                     \
            static_cast<T*>(out), n_rows);                                                          \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// Scalar finalize (reduction_mode = Mean / Sum). f32 loss_1d → scalar T.
#define BARACUDA_KERNELS_FLCE_SCALAR_FINALIZE_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows,                                                                             \
        float denom_inv,                                                                            \
        const void* loss_1d, void* out,                                                             \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0) return 2;                                                                   \
        if (loss_1d == nullptr || out == nullptr) return 2;                                         \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        baracuda::flce::flce_scalar_finalize_kernel<T><<<1, 256, 0, stream>>>(                      \
            static_cast<const float*>(loss_1d),                                                     \
            static_cast<T*>(out), n_rows, denom_inv);                                               \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// In-place scale: `buf *= scalar` over `numel` elements. Used in BW to
// apply the upstream grad_output to saved grad_input / grad_weight.
#define BARACUDA_KERNELS_FLCE_INPLACE_SCALE_INSTANTIATE(NAME, T)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        float scalar,                                                                               \
        void* buf,                                                                                  \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                    \
        if (numel == 0) return 0;                                                                   \
        if (buf == nullptr) return 2;                                                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::flce::flce_inplace_scale_kernel<T><<<blocks, kBlock, 0, stream>>>(                \
            static_cast<T*>(buf), numel, scalar);                                                   \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// Count non-ignore launcher. No template params (single int64_t signature).
#define BARACUDA_KERNELS_FLCE_COUNT_NON_IGNORE_INSTANTIATE_BODY()                                  \
    extern "C" int32_t baracuda_kernels_loss_flce_count_non_ignore_run(                            \
        int32_t bt,                                                                                 \
        int64_t ignore_index,                                                                       \
        const void* target,                                                                         \
        void* count_out, /* [1] int64_t; caller pre-zeroes */                                       \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (bt < 0) return 2;                                                                       \
        if (target == nullptr || count_out == nullptr) return 2;                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        if (bt == 0) {                                                                              \
            cudaMemsetAsync(count_out, 0, sizeof(int64_t), stream);                                 \
            return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                     \
        }                                                                                           \
        baracuda::flce::flce_count_non_ignore_kernel<<<1, 256, 0, stream>>>(                        \
            static_cast<const int64_t*>(target),                                                    \
            static_cast<int64_t*>(count_out), bt, ignore_index);                                    \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

#endif // BARACUDA_FLCE_CUH
