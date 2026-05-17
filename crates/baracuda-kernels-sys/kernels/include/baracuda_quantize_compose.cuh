// baracuda_quantize_compose.cuh
//
// Phase 8 Milestone 8.3 — composing quantization ops on top of the
// per-tensor / per-channel / per-token / per-group primitives from
// Milestones 8.1 and 8.2.
//
// Two op families live here:
//
//   1. `dynamic_range_quantize` — compute scale (and optionally
//      zero_point) from the runtime dynamic range of the input, then
//      quantize. Symmetric per-token trailblazer math:
//
//          max_abs[n] = max_d |x[n, d]|
//          scale[n]   = max_abs[n] / qmax       (qmax is e.g. 127 for s8)
//          zero_point[n] = 0                    (symmetric)
//          q[n, d]    = clamp(round(x[n, d] / scale[n]), qmin, qmax)
//
//      The kernel writes both the quantized tensor AND the computed
//      `scale[n]` (FP vector, length N) — callers need the scale for the
//      matching `dequantize_per_token` step downstream.
//
//   2. `quantized_linear` — fused W8A8 quantized matmul. Pipeline:
//
//          (a) compute per-row scale_a[m] from the FP activation row's
//              max-abs (the same DRQ recipe in (1)),
//          (b) quantize activation per-row to int8,
//          (c) accumulate int32 acc[m, n] = Σ_k a_q[m, k] * w_q[n, k],
//          (d) FP store out[m, n] = scale_a[m] * scale_w[n] * (float)acc.
//
//      Weight is `[C_out, K]` int8 row-major (per the LLM convention,
//      one row per output channel); `weight_scale` is `[C_out]` FP.
//      Activation is `[M, K]` FP. Output is `[M, C_out]` FP.
//
//      The trailblazer kernel is **deliberately a naive SIMT
//      implementation**: one thread per output cell, K-loop accumulates
//      in int32, dequant + store in FP. Performance optimization (tiled
//      smem, mma.sync m16n8k32, double-buffered) is a follow-up
//      milestone; correctness + scaffolded plan composition is the goal
//      here.
//
// Dtype coverage (Phase 8.3 trailblazer):
//   - dynamic_range_quantize: TIn ∈ {f32, f64} × TOut ∈ {s8}.
//   - quantized_linear      : TIn ∈ {f32, f64} (activation + scale),
//                             weight = s8, output = TIn.
//
// f16 / bf16 + u8 + asymmetric variants follow in fanout milestones.
//
// Status codes: 0 success / 1 misalign (reserved) / 2 invalid /
// 3 unsupported (reserved) / 4 workspace too small (reserved) /
// 5 kernel launch error.

#ifndef BARACUDA_QUANTIZE_COMPOSE_CUH
#define BARACUDA_QUANTIZE_COMPOSE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace quantize_compose {

// =============================================================================
// dtype helpers — same load-as-f32 / store-from-f32 pattern as the 8.2
// header (`baracuda_quantize_per_token_group.cuh`). Kept local so this
// translation unit can be built standalone — see the duplicate-symbol
// note in `baracuda_quantize_per_token_group.cuh`: each header isolates
// its inline helpers in its own anonymous-style namespace path.
// =============================================================================

template <typename T>
__host__ __device__ __forceinline__ float qc_load_as_f32(T x) { return (float)x; }
template <>
__host__ __device__ __forceinline__ float qc_load_as_f32<double>(double x) { return (float)x; }
template <>
__host__ __device__ __forceinline__ float qc_load_as_f32<__half>(__half x) {
#if defined(__CUDA_ARCH__)
    return __half2float(x);
#else
    return (float)x;
#endif
}
template <>
__host__ __device__ __forceinline__ float qc_load_as_f32<__nv_bfloat16>(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__)
    return __bfloat162float(x);
#else
    return (float)x;
#endif
}

template <typename T>
__host__ __device__ __forceinline__ T qc_store_from_f32(float v) { return (T)v; }
template <>
__host__ __device__ __forceinline__ double qc_store_from_f32<double>(float v) { return (double)v; }
template <>
__host__ __device__ __forceinline__ __half qc_store_from_f32<__half>(float v) {
#if defined(__CUDA_ARCH__)
    return __float2half(v);
#else
    return __half(v);
#endif
}
template <>
__host__ __device__ __forceinline__ __nv_bfloat16 qc_store_from_f32<__nv_bfloat16>(float v) {
#if defined(__CUDA_ARCH__)
    return __float2bfloat16(v);
#else
    return __nv_bfloat16(v);
#endif
}

// Saturating round-to-nearest-int (round-half-to-even) into [qmin, qmax].
// Mirrors `qptg_round_sat` from the 8.2 header.
__host__ __device__ __forceinline__ int32_t qc_round_sat(
    float x, int32_t qmin, int32_t qmax)
{
#if defined(__CUDA_ARCH__)
    int32_t r = __float2int_rn(x);
#else
    int32_t r = (int32_t)nearbyintf(x);
#endif
    if (r < qmin) r = qmin;
    if (r > qmax) r = qmax;
    return r;
}

// =============================================================================
// DYNAMIC RANGE QUANTIZE — symmetric per-token.
// =============================================================================
//
// One block per row. Each block:
//   1. Each thread accumulates a local `max_abs` over its strided slice
//      of `x[row, :]`.
//   2. Block-wide reduction in shared memory gives `row_max_abs`.
//   3. Thread 0 computes `scale[row] = max_abs / qmax`, stores it (with
//      a small epsilon to avoid divide-by-zero for an all-zero row).
//   4. All threads quantize their slice: `q = clamp(round(x/scale),
//      qmin, qmax)`. Zero point is 0 (symmetric).
//
// Block size is fixed at 256. D > 256 is handled via the stride loop in
// both the reduce + quantize passes.

template <typename TIn, typename TOut, int BLOCK = 256>
__global__ void dynamic_range_quantize_per_token_symmetric_kernel(
    const TIn*  __restrict__ input,    // [N, D]
    TIn*        __restrict__ scale,    // [N]
    TOut*       __restrict__ output,   // [N, D]
    int32_t     N,
    int32_t     D,
    int32_t     qmin,
    int32_t     qmax)
{
    int32_t n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;
    int64_t row_off = (int64_t)n * (int64_t)D;

    // --- pass 1: max-abs reduce ----------------------------------------------
    float local_max = 0.0f;
    for (int d = tid; d < D; d += BLOCK) {
        float v = qc_load_as_f32<TIn>(input[row_off + (int64_t)d]);
        float a = fabsf(v);
        if (a > local_max) local_max = a;
    }

    __shared__ float smem[BLOCK];
    smem[tid] = local_max;
    __syncthreads();

    // tree reduce — power-of-two stride.
    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float other = smem[tid + s];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }

    // --- pass 2: compute scale, broadcast, quantize --------------------------
    __shared__ float s_scale_inv;
    if (tid == 0) {
        float row_max_abs = smem[0];
        // Use qmax as the symmetric denominator. Guard divide-by-zero
        // for an all-zero row by emitting scale = 1 (any non-zero value
        // is fine; the row's quantized output is all-zero anyway).
        float qmaxf = (float)qmax;
        float s_val = (row_max_abs > 0.0f) ? (row_max_abs / qmaxf) : 1.0f;
        scale[n] = qc_store_from_f32<TIn>(s_val);
        s_scale_inv = (s_val != 0.0f) ? (1.0f / s_val) : 0.0f;
    }
    __syncthreads();

    float inv_s = s_scale_inv;
    for (int d = tid; d < D; d += BLOCK) {
        float v = qc_load_as_f32<TIn>(input[row_off + (int64_t)d]);
        int32_t q = qc_round_sat(v * inv_s, qmin, qmax);
        output[row_off + (int64_t)d] = static_cast<TOut>(q);
    }
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_dynamic_range_quantize_per_token_symmetric(
    const TIn* input, TIn* scale, TOut* output,
    int32_t N, int32_t D, int32_t qmin, int32_t qmax,
    cudaStream_t stream)
{
    if (N < 0 || D < 0) return 2;
    if (qmax <= 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (input == nullptr || scale == nullptr || output == nullptr) return 2;
    constexpr int kBlock = 256;
    if (N > 65535) return 2;  // legacy grid limit; future fanout: tile rows.
    dynamic_range_quantize_per_token_symmetric_kernel<TIn, TOut, kBlock>
        <<<N, kBlock, 0, stream>>>(input, scale, output, N, D, qmin, qmax);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// QUANTIZED LINEAR — W8A8 fused naive kernel.
// =============================================================================
//
// Computes:  out[m, n] = scale_a[m] * scale_w[n] *
//                          Σ_k a_q[m, k] * w_q[n, k]
//
//   activation_q : `[M, K]`   int8   (already-quantized activation)
//   weight_q     : `[C_out, K]` int8 (already-quantized weight, row-major)
//   scale_a      : `[M]`    FP
//   scale_w      : `[C_out]` FP
//   output       : `[M, C_out]` FP
//
// Layout convention chosen for cache locality of the K reduction:
// weight is `[C_out, K]` row-major (NOT `[K, C_out]`), so a single
// output cell `(m, n)` reads `a_q[m, *]` (row-stride K) and
// `w_q[n, *]` (row-stride K) — both contiguous K spans.
//
// Trailblazer is one-thread-per-output-cell. Performance optimization
// (smem tiling, mma.sync m16n8k32, vectorized loads) is a follow-up
// milestone — this kernel is for correctness scaffolding of the
// composing op, not throughput.

template <typename TIn, typename TWQ>
__global__ void quantized_linear_w8a8_kernel(
    const TWQ*  __restrict__ weight_q,    // [C_out, K] int8
    const int8_t* __restrict__ act_q,     // [M, K] int8 (per-token quantized)
    const TIn*  __restrict__ scale_a,     // [M]
    const TIn*  __restrict__ scale_w,     // [C_out]
    TIn*        __restrict__ output,      // [M, C_out]
    int32_t     M,
    int32_t     C_out,
    int32_t     K)
{
    int64_t total = (int64_t)M * (int64_t)C_out;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t cell = tid; cell < total; cell += step) {
        int32_t m = (int32_t)(cell / (int64_t)C_out);
        int32_t n = (int32_t)(cell - (int64_t)m * (int64_t)C_out);
        int64_t a_off = (int64_t)m * (int64_t)K;
        int64_t w_off = (int64_t)n * (int64_t)K;

        int32_t acc = 0;
        for (int32_t k = 0; k < K; ++k) {
            int32_t av = (int32_t)act_q[a_off + (int64_t)k];
            int32_t wv = (int32_t)(int8_t)weight_q[w_off + (int64_t)k];
            acc += av * wv;
        }
        float sa = qc_load_as_f32<TIn>(scale_a[m]);
        float sw = qc_load_as_f32<TIn>(scale_w[n]);
        float y = sa * sw * (float)acc;
        output[cell] = qc_store_from_f32<TIn>(y);
    }
}

template <typename TIn>
__host__ inline int32_t launch_quantized_linear_w8a8(
    const int8_t* weight_q, const int8_t* act_q,
    const TIn* scale_a, const TIn* scale_w,
    TIn* output,
    int32_t M, int32_t C_out, int32_t K,
    cudaStream_t stream)
{
    if (M < 0 || C_out < 0 || K < 0) return 2;
    int64_t total = (int64_t)M * (int64_t)C_out;
    if (total == 0) return 0;
    if (weight_q == nullptr || act_q == nullptr ||
        scale_a == nullptr || scale_w == nullptr ||
        output == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantized_linear_w8a8_kernel<TIn, int8_t><<<blocks, kBlock, 0, stream>>>(
        weight_q, act_q, scale_a, scale_w, output, M, C_out, K);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::quantize_compose

// =============================================================================
// INSTANTIATE macros
// =============================================================================

#define BARACUDA_KERNELS_DYNAMIC_RANGE_QUANTIZE_PER_TOKEN_SYM_INSTANTIATE(           \
        NAME, TIN, TOUT)                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                  \
        int32_t N, int32_t D, int32_t qmin, int32_t qmax,                              \
        const void* input,                                                              \
        void* scale,                                                                    \
        void* output,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                \
        void* stream_ptr)                                                               \
    {                                                                                  \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                    \
        return baracuda::quantize_compose                                               \
            ::launch_dynamic_range_quantize_per_token_symmetric<TIN, TOUT>(             \
                static_cast<const TIN*>(input),                                         \
                static_cast<TIN*>(scale),                                               \
                static_cast<TOUT*>(output),                                             \
                N, D, qmin, qmax, stream);                                              \
    }

#define BARACUDA_KERNELS_QUANTIZED_LINEAR_W8A8_INSTANTIATE(NAME, TIN)                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                  \
        int32_t M, int32_t C_out, int32_t K,                                            \
        const void* weight_q,                                                           \
        const void* act_q,                                                              \
        const void* scale_a,                                                            \
        const void* scale_w,                                                            \
        void* output,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                \
        void* stream_ptr)                                                               \
    {                                                                                  \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                    \
        return baracuda::quantize_compose::launch_quantized_linear_w8a8<TIN>(           \
            static_cast<const int8_t*>(weight_q),                                       \
            static_cast<const int8_t*>(act_q),                                          \
            static_cast<const TIN*>(scale_a),                                           \
            static_cast<const TIN*>(scale_w),                                           \
            static_cast<TIN*>(output),                                                  \
            M, C_out, K, stream);                                                       \
    }

#endif // BARACUDA_QUANTIZE_COMPOSE_CUH
