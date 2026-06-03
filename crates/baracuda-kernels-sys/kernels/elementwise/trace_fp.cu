// baracuda-kernels Phase 4 reduction: matrix trace for FP types.
//
// `y = trace(M) = sum(diag(M)) = Σ M[i, i] for i in 0..N` — for a 2-D
// square matrix `M` of shape `[N, N]` with the diagonal selected by
// stride_row + stride_col. Output is a rank-0 scalar (caller passes a
// 1-element output buffer).
//
// This doesn't fit `ReducePlan<T, N>` cleanly — trace reduces *both*
// axes via the i==i constraint rather than a single reduce_axis. We
// ship a dedicated `TracePlan<T>` (no rank generic; always 2D in,
// scalar out) and a dedicated kernel here.
//
// Trailblazer kernel: single block, single thread sequentially walks
// the diagonal accumulating in the compute dtype (f32 for half-precision,
// otherwise native). For large N a parallel-reduce version will land in
// fanout — the typical model use of trace (e.g. attention sanity
// checks, regularizers) keeps N at a few hundred elements at most so
// the naive walk is fine for the trailblazer.

#include "../include/baracuda_elementwise.cuh"
#include <math.h>

namespace baracuda { namespace elementwise {

// Pick the compute dtype: f32 for half-precision inputs (f32-detour),
// otherwise native.
template <typename T> struct TraceCompute       { using Type = T; };
template <>            struct TraceCompute<__half>         { using Type = float; };
template <>            struct TraceCompute<__nv_bfloat16>  { using Type = float; };

template <typename T> __device__ __forceinline__ typename TraceCompute<T>::Type trace_load(T v) {
    return static_cast<typename TraceCompute<T>::Type>(v);
}
template <> __device__ __forceinline__ float trace_load<__half>(__half v) {
    return __half2float(v);
}
template <> __device__ __forceinline__ float trace_load<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T trace_store(typename TraceCompute<T>::Type v) {
    return static_cast<T>(v);
}
template <> __device__ __forceinline__ __half trace_store<__half>(float v) {
    return __float2half(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 trace_store<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__global__ void trace_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int32_t n,
    int64_t stride_row,
    int64_t stride_col)
{
    // Single thread walks the diagonal. Grid is launched as <<<1, 1>>>.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    using C = typename TraceCompute<T>::Type;
    C acc = C(0);
    int64_t step = stride_row + stride_col;
    int64_t off = 0;
    for (int32_t i = 0; i < n; ++i) {
        acc = acc + trace_load<T>(x[off]);
        off += step;
    }
    y[0] = trace_store<T>(acc);
}

template <typename T>
__host__ inline int32_t launch_trace(
    const T* x, T* y,
    int32_t n,
    int64_t stride_row,
    int64_t stride_col,
    cudaStream_t stream)
{
    if (n < 0) return 2;
    if (n == 0) {
        // Empty trace = 0. Emit a zero in the output cell using a
        // single-thread kernel so the caller can still synchronize.
        // We piggy-back on the trace_kernel with n=0 — the loop body
        // doesn't execute, the store writes 0.
    }
    trace_kernel<T><<<1, 1, 0, stream>>>(x, y, n, stride_row, stride_col);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================
//
// FFI: (rows, stride_row, stride_col, x, y, ws, ws_bytes, stream).
// Square matrix only (rows == cols); the Rust plan enforces that.

#define BARACUDA_KERNELS_TRACE_INSTANTIATE(NAME, T)                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                            \
        int32_t rows,                                                                             \
        int64_t stride_row,                                                                       \
        int64_t stride_col,                                                                       \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                             \
        if (rows < 0) return 2;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                             \
        return baracuda::elementwise::launch_trace<T>(                                           \
            static_cast<const T*>(x), static_cast<T*>(y),                                        \
            rows, stride_row, stride_col, stream);                                               \
    }                                                                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                  \
        int32_t rows,                                                                             \
        int64_t /*stride_row*/,                                                                   \
        int64_t /*stride_col*/,                                                                   \
        const void* /*x*/, const void* /*y*/)                                                     \
    {                                                                                             \
        if (rows < 0) return 2;                                                                  \
        return 0;                                                                                 \
    }

BARACUDA_KERNELS_TRACE_INSTANTIATE(trace_f32, float)
BARACUDA_KERNELS_TRACE_INSTANTIATE(trace_f16, __half)
BARACUDA_KERNELS_TRACE_INSTANTIATE(trace_bf16, __nv_bfloat16)
BARACUDA_KERNELS_TRACE_INSTANTIATE(trace_f64, double)
