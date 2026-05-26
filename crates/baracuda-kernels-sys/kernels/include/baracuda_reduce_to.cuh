// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_reduce_to.cuh
//
// Phase 31 — broadcast-REVERSE reductions: `ReduceSumTo` and
// `ReduceMaxTo`. Used by autograd's `Op::ReduceSumTo` / `Op::ReduceMaxTo`
// to undo a forward `Op::BroadcastTo`.
//
// Semantics:
//   * `output_shape` is left-padded with 1s to match `input_rank`
//     (caller's responsibility).
//   * For each output cell at multi-coord `c_out`, the set of input
//     cells that broadcast TO it consists of every multi-coord `c_in`
//     where `c_in[d] == c_out[d]` if `output_shape[d] > 1`, and
//     `c_in[d]` ranges over `[0, input_shape[d])` if
//     `output_shape[d] == 1`.
//   * Sum: `dst[c_out] = Σ src[c_in]` over that set.
//   * Max: `dst[c_out] = max src[c_in]` over that set; result is
//     `-inf` (or the most-negative finite value for non-FP) if the
//     set is empty (any `input_shape[d] == 0`).
//
// One thread per output cell. Each thread:
//   1. Decomposes its linear `out_id` into `c_out` against
//      `output_shape`.
//   2. Walks the broadcast dims' input ranges in a nested-loop pattern
//      encoded via a stack of running coords + an unrolled outer
//      iteration count.
//   3. Computes the dot of (c_in, input_stride) for each combination
//      and accumulates `src[off]` into the running acc.
//   4. Writes `acc` to `dst[out_id]`.
//
// Layout: the output is contiguous over `output_shape` (the caller
// allocates a strictly contiguous dst with the same rank as the
// input, broadcast dims being size-1). The input may have arbitrary
// (non-contig) strides — common in autograd traces where the input
// is itself a view.
//
// Dtype: f32 / f64 / f16 / bf16. Half-precision dtypes accumulate in
// f32 / f64 for sum (matches PyTorch convention) and use the natural
// half-precision comparison for max (`fmaxf` after `__half2float`).

#ifndef BARACUDA_REDUCE_TO_CUH
#define BARACUDA_REDUCE_TO_CUH

#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace reduce_to {

// MAX_RANK matches the rest of the elementwise family — keeps the
// stack-allocated coord arrays cache-friendly.
constexpr int MAX_RANK = 8;

// ----------------------------------------------------------------------------
// Per-dtype accumulator policies — match PyTorch's autograd convention
// of accumulating half/bf16 sums in f32 to avoid overflow and ULP loss
// on long broadcast axes.
// ----------------------------------------------------------------------------

// Common API across both policies (Sum + Max):
//   * `AccT`        — accumulator type
//   * `identity()`  — sum's 0, max's most-negative value
//   * `load(p)`     — read one source element into AccT (half/bf16 widen)
//   * `store(p, v)` — write AccT back to dst (narrows for half/bf16)
//   * `combine(a, b)` — per-element reduce step

template <typename T> struct AccSum {
    using AccT = T;
    static __device__ __forceinline__ AccT identity() { return AccT(0); }
    static __device__ __forceinline__ AccT load(const T* p) { return *p; }
    static __device__ __forceinline__ void store(T* p, AccT v) { *p = v; }
    static __device__ __forceinline__ AccT combine(AccT a, AccT b) { return a + b; }
};

template <> struct AccSum<__half> {
    using AccT = float;
    static __device__ __forceinline__ AccT identity() { return 0.0f; }
    static __device__ __forceinline__ AccT load(const __half* p) { return __half2float(*p); }
    static __device__ __forceinline__ void store(__half* p, AccT v) { *p = __float2half(v); }
    static __device__ __forceinline__ AccT combine(AccT a, AccT b) { return a + b; }
};

template <> struct AccSum<__nv_bfloat16> {
    using AccT = float;
    static __device__ __forceinline__ AccT identity() { return 0.0f; }
    static __device__ __forceinline__ AccT load(const __nv_bfloat16* p) { return __bfloat162float(*p); }
    static __device__ __forceinline__ void store(__nv_bfloat16* p, AccT v) { *p = __float2bfloat16(v); }
    static __device__ __forceinline__ AccT combine(AccT a, AccT b) { return a + b; }
};

template <typename T> struct AccMax {
    using AccT = T;
    static __device__ __forceinline__ AccT identity();
    static __device__ __forceinline__ AccT load(const T* p) { return *p; }
    static __device__ __forceinline__ void store(T* p, AccT v) { *p = v; }
    static __device__ __forceinline__ AccT combine(AccT a, AccT b) { return (a > b) ? a : b; }
};

template <> __device__ __forceinline__ float AccMax<float>::identity() { return -FLT_MAX; }
template <> __device__ __forceinline__ double AccMax<double>::identity() { return -DBL_MAX; }

template <> struct AccMax<__half> {
    using AccT = float;
    static __device__ __forceinline__ AccT identity() { return -FLT_MAX; }
    static __device__ __forceinline__ AccT load(const __half* p) { return __half2float(*p); }
    static __device__ __forceinline__ void store(__half* p, AccT v) { *p = __float2half(v); }
    static __device__ __forceinline__ AccT combine(AccT a, AccT b) { return (a > b) ? a : b; }
};

template <> struct AccMax<__nv_bfloat16> {
    using AccT = float;
    static __device__ __forceinline__ AccT identity() { return -FLT_MAX; }
    static __device__ __forceinline__ AccT load(const __nv_bfloat16* p) { return __bfloat162float(*p); }
    static __device__ __forceinline__ void store(__nv_bfloat16* p, AccT v) { *p = __float2bfloat16(v); }
    static __device__ __forceinline__ AccT combine(AccT a, AccT b) { return (a > b) ? a : b; }
};

struct DimsI32 { int32_t v[MAX_RANK]; };
struct DimsI64 { int64_t v[MAX_RANK]; };

// ----------------------------------------------------------------------------
// Per-output-cell reduction kernel. POLICY is `AccSum<T>` or `AccMax<T>`;
// both expose the same `identity / load / store / combine` API so the
// kernel body needs no compile-time switch between the two ops.
// ----------------------------------------------------------------------------

template <typename T, typename POLICY>
__global__ void reduce_to_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    DimsI32 input_shape,
    DimsI64 input_stride,
    int32_t rank,
    DimsI32 output_shape,
    int64_t output_numel)
{
    using AccT = typename POLICY::AccT;

    int64_t out_id = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (out_id >= output_numel) return;

    // Decompose out_id into multi-coord against output_shape (contig).
    int32_t out_coord[MAX_RANK] = {0};
    {
        int64_t lin = out_id;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = output_shape.v[d];
            if (s <= 0) { out_coord[d] = 0; continue; }
            out_coord[d] = (int32_t)(lin % (int64_t)s);
            lin /= (int64_t)s;
        }
    }

    // Identify broadcast dims (output_shape[d] == 1, input_shape[d] != 1).
    // For non-broadcast dims, c_in[d] == out_coord[d]. For broadcast
    // dims, c_in[d] ranges over [0, input_shape[d]).
    //
    // We compute the total iteration count = Π input_shape[d] over
    // broadcast dims, and iterate by decomposing the linear iter index
    // into per-broadcast-dim coords.
    int32_t bcast_lens[MAX_RANK];
    int32_t n_bcast = 0;
    int64_t bcast_total = 1;
    for (int d = 0; d < rank; ++d) {
        if (output_shape.v[d] == 1 && input_shape.v[d] != 1) {
            bcast_lens[n_bcast] = input_shape.v[d];
            bcast_total *= (int64_t)input_shape.v[d];
            n_bcast++;
        }
    }

    AccT acc = POLICY::identity();

    // For non-broadcast dims, validate the input has matching extent.
    // If a non-broadcast input dim is 0 → empty set → identity.
    bool empty = false;
    for (int d = 0; d < rank; ++d) {
        if (output_shape.v[d] != 1 && input_shape.v[d] == 0) {
            empty = true;
            break;
        }
    }
    if (empty || bcast_total == 0) {
        POLICY::store(dst + out_id, acc);
        return;
    }

    // Walk the broadcast iteration space.
    for (int64_t bcast_iter = 0; bcast_iter < bcast_total; ++bcast_iter) {
        // Decompose bcast_iter into per-broadcast-dim coords (row-major).
        int32_t bcast_coord[MAX_RANK] = {0};
        int64_t lin = bcast_iter;
        for (int i = n_bcast - 1; i >= 0; --i) {
            int32_t s = bcast_lens[i];
            bcast_coord[i] = (int32_t)(lin % (int64_t)s);
            lin /= (int64_t)s;
        }

        // Build the full input coord.
        int64_t in_off = 0;
        int bi = 0;
        for (int d = 0; d < rank; ++d) {
            int32_t c;
            if (output_shape.v[d] == 1 && input_shape.v[d] != 1) {
                c = bcast_coord[bi++];
            } else {
                // Non-broadcast: c_in[d] == out_coord[d]. If input
                // happens to have size-1 here (no real broadcast on
                // this dim), clamp to 0.
                c = (input_shape.v[d] == 1) ? 0 : out_coord[d];
            }
            in_off += (int64_t)c * input_stride.v[d];
        }

        AccT v = POLICY::load(src + in_off);
        acc = POLICY::combine(acc, v);
    }

    POLICY::store(dst + out_id, acc);
}

// ----------------------------------------------------------------------------
// Host launchers.
// ----------------------------------------------------------------------------

template <typename T, typename POLICY>
__host__ inline int32_t launch_reduce_to(
    const T* src, T* dst,
    const int32_t* input_shape_host,
    const int64_t* input_stride_host,
    int32_t rank,
    const int32_t* output_shape_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;

    DimsI32 in_shape = {};
    DimsI64 in_stride = {};
    DimsI32 out_shape = {};
    int64_t output_numel = 1;
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i] = input_shape_host[i];
        in_stride.v[i] = input_stride_host[i];
        out_shape.v[i] = output_shape_host[i];
        if (out_shape.v[i] < 0) return 2;
        output_numel *= (int64_t)out_shape.v[i];
    }

    if (output_numel == 0) return 0;

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = (1LL << 30);
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;

    reduce_to_kernel<T, POLICY><<<blocks, kBlock, 0, stream>>>(
        src, dst, in_shape, in_stride, rank, out_shape, output_numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::reduce_to

// ----------------------------------------------------------------------------
// INSTANTIATE macros — emit one `extern "C"` launcher pair per (op, dtype).
// ----------------------------------------------------------------------------

#define BARACUDA_KERNELS_REDUCE_SUM_TO_INSTANTIATE(SUFFIX, T)                                         \
    extern "C" int32_t baracuda_kernels_reduce_sum_to_##SUFFIX##_run(                                \
        const void* src, void* dst,                                                                   \
        const int32_t* input_shape, const int64_t* input_stride,                                      \
        int32_t rank_in,                                                                              \
        const int32_t* output_shape,                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                             \
    {                                                                                                 \
        if (rank_in < 0) return 2;                                                                   \
        if (src == nullptr || dst == nullptr) return 2;                                              \
        if (input_shape == nullptr || input_stride == nullptr || output_shape == nullptr) return 2;   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::reduce_to::launch_reduce_to<T, baracuda::reduce_to::AccSum<T>>(              \
            static_cast<const T*>(src), static_cast<T*>(dst),                                         \
            input_shape, input_stride, rank_in, output_shape, stream);                                \
    }                                                                                                 \
    extern "C" int32_t baracuda_kernels_reduce_sum_to_##SUFFIX##_can_implement(                      \
        const void* /*src*/, const void* /*dst*/,                                                     \
        const int32_t* /*input_shape*/, const int64_t* /*input_stride*/,                              \
        int32_t rank_in,                                                                              \
        const int32_t* /*output_shape*/)                                                              \
    {                                                                                                 \
        if (rank_in < 0 || rank_in > baracuda::reduce_to::MAX_RANK) return 2;                         \
        return 0;                                                                                     \
    }

#define BARACUDA_KERNELS_REDUCE_MAX_TO_INSTANTIATE(SUFFIX, T)                                         \
    extern "C" int32_t baracuda_kernels_reduce_max_to_##SUFFIX##_run(                                \
        const void* src, void* dst,                                                                   \
        const int32_t* input_shape, const int64_t* input_stride,                                      \
        int32_t rank_in,                                                                              \
        const int32_t* output_shape,                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                             \
    {                                                                                                 \
        if (rank_in < 0) return 2;                                                                   \
        if (src == nullptr || dst == nullptr) return 2;                                              \
        if (input_shape == nullptr || input_stride == nullptr || output_shape == nullptr) return 2;   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::reduce_to::launch_reduce_to<T, baracuda::reduce_to::AccMax<T>>(              \
            static_cast<const T*>(src), static_cast<T*>(dst),                                         \
            input_shape, input_stride, rank_in, output_shape, stream);                                \
    }                                                                                                 \
    extern "C" int32_t baracuda_kernels_reduce_max_to_##SUFFIX##_can_implement(                      \
        const void* /*src*/, const void* /*dst*/,                                                     \
        const int32_t* /*input_shape*/, const int64_t* /*input_stride*/,                              \
        int32_t rank_in,                                                                              \
        const int32_t* /*output_shape*/)                                                              \
    {                                                                                                 \
        if (rank_in < 0 || rank_in > baracuda::reduce_to::MAX_RANK) return 2;                         \
        return 0;                                                                                     \
    }

#endif // BARACUDA_REDUCE_TO_CUH
