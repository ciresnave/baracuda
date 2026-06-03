// baracuda-kernels Phase 4 reduction: axis-LogSumExp for FP types.
//
// `y = log(sum(exp(x - max), axis=k)) + max` — the numerically stable
// formulation: subtract the per-output-cell max before exp, add it
// back after log. Critical primitive for Phase 5's log_softmax /
// cross-entropy loss (which is just LSE expressed in log space).
//
// This op cannot reuse `BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE`
// because it is genuinely **two-pass over the reduce axis**:
//   pass 1: m = max(x along axis)
//   pass 2: s = sum(exp(x - m) along axis)
//   store : y = log(s) + m
// The single-functor `init() → op(acc, x) → finalize(acc, extent)`
// shape used by Sum / Mean / Max / Min / Prod / Norm2 only sees one
// element at a time and can't pre-pass for `m`. We ship a dedicated
// kernel here, with the same parameter ABI as the simple-reduce family
// so the Rust dispatcher can wire it through the same FFI shape.
//
// All four FP dtypes are wired. f32 / f64 use libdevice
// `expf`/`exp` and `logf`/`log`; f16 / bf16 detour through f32 for
// every load, op, and store (consistent with the existing f16/bf16
// reduction specializations elsewhere in this header).

#include "../include/baracuda_elementwise.cuh"
#include <math.h>

namespace baracuda { namespace elementwise {

// Trait selecting libdevice flavour per element dtype. The compute
// path always runs in f32 (or f64 for double); load / store routes
// through the dtype's converters for the half-precision variants.
template <typename T>
struct LseDtype;

template <>
struct LseDtype<float> {
    using Compute = float;
    static __device__ __forceinline__ float load(float v) { return v; }
    static __device__ __forceinline__ float store_from(float v) { return v; }
    static __device__ __forceinline__ float neg_infinity() { return -INFINITY; }
    static __device__ __forceinline__ float gmax(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float gexp(float v) { return __expf(v); }
    static __device__ __forceinline__ float glog(float v) { return logf(v); }
};

template <>
struct LseDtype<double> {
    using Compute = double;
    static __device__ __forceinline__ double load(double v) { return v; }
    static __device__ __forceinline__ double store_from(double v) { return v; }
    static __device__ __forceinline__ double neg_infinity() { return -INFINITY; }
    static __device__ __forceinline__ double gmax(double a, double b) { return fmax(a, b); }
    static __device__ __forceinline__ double gexp(double v) { return exp(v); }
    static __device__ __forceinline__ double glog(double v) { return log(v); }
};

template <>
struct LseDtype<__half> {
    using Compute = float;
    static __device__ __forceinline__ float load(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half store_from(float v) { return __float2half(v); }
    static __device__ __forceinline__ float neg_infinity() { return -INFINITY; }
    static __device__ __forceinline__ float gmax(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float gexp(float v) { return __expf(v); }
    static __device__ __forceinline__ float glog(float v) { return logf(v); }
};

template <>
struct LseDtype<__nv_bfloat16> {
    using Compute = float;
    static __device__ __forceinline__ float load(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    static __device__ __forceinline__ __nv_bfloat16 store_from(float v) {
        return __float2bfloat16(v);
    }
    static __device__ __forceinline__ float neg_infinity() { return -INFINITY; }
    static __device__ __forceinline__ float gmax(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float gexp(float v) { return __expf(v); }
    static __device__ __forceinline__ float glog(float v) { return logf(v); }
};

// Two-pass LSE kernel. One thread per output cell:
//   pass 1: walk reduce axis tracking max `m`.
//   pass 2: walk reduce axis accumulating `s = sum(exp(x - m))`.
//   write : y = log(s) + m.
//
// Empty-axis (`reduce_extent == 0`) is rejected by the launcher (see
// argmax convention) — LSE over zero elements is `-inf` and PyTorch
// raises rather than silently returning that, so we keep the same
// guard for consistency.
template <typename T>
__global__ void reduce_logsumexp_axis_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 output_shape,
    DimsI64 stride_x,
    DimsI64 stride_y,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x)
{
    using DT = LseDtype<T>;
    using C  = typename DT::Compute;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = output_shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != reduce_axis) {
                off_x_base += coord * stride_x.v[d];
            }
        }
        // Pass 1: max.
        C m = DT::neg_infinity();
        for (int32_t k = 0; k < reduce_extent; ++k) {
            C v = DT::load(x[off_x_base + (int64_t)k * reduce_stride_x]);
            m = DT::gmax(m, v);
        }
        // Pass 2: sum(exp(x - m)).
        C s = C(0);
        for (int32_t k = 0; k < reduce_extent; ++k) {
            C v = DT::load(x[off_x_base + (int64_t)k * reduce_stride_x]);
            s = s + DT::gexp(v - m);
        }
        C out = DT::glog(s) + m;
        y[off_y] = DT::store_from(out);
    }
}

template <typename T>
__host__ inline int32_t launch_reduce_logsumexp_axis(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* output_shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (reduce_axis < 0 || reduce_axis >= rank) return 2;
    if (reduce_extent <= 0) return 2;
    DimsI32 out_shape = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        out_shape.v[i] = output_shape_host[i];
        sx.v[i]        = stride_x_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_logsumexp_axis_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, out_shape, sx, sy,
        reduce_axis, reduce_extent, reduce_stride_x);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================
//
// ABI mirrors `BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE` exactly — the
// Rust dispatcher's `dispatch!` macro reaches both families through
// the same parameter shape.

#define BARACUDA_KERNELS_REDUCE_LOGSUMEXP_INSTANTIATE(NAME, T)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* output_shape,                                                              \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        int32_t reduce_axis,                                                                      \
        int32_t reduce_extent,                                                                    \
        int64_t reduce_stride_x,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (output_shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;      \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_logsumexp_axis<T>(                            \
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                     \
            output_shape, stride_x, stride_y,                                                     \
            reduce_axis, reduce_extent, reduce_stride_x,                                          \
            stream);                                                                              \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int64_t output_numel, int32_t rank,                                                       \
        const int32_t* output_shape,                                                              \
        const int64_t* stride_x, const int64_t* stride_y,                                         \
        int32_t reduce_axis, int32_t reduce_extent, int64_t reduce_stride_x,                     \
        const void* /*x*/, const void* /*y*/)                                                     \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (rank < 0) return 2;                                                                   \
        if (reduce_extent <= 0) return 2;                                                         \
        if (reduce_axis < 0 || reduce_axis >= rank) return 2;                                     \
        if (output_numel > 0 && (output_shape == nullptr || stride_x == nullptr ||                \
                                  stride_y == nullptr)) return 2;                                 \
        (void)reduce_stride_x;                                                                    \
        return 0;                                                                                  \
    }

BARACUDA_KERNELS_REDUCE_LOGSUMEXP_INSTANTIATE(reduce_logsumexp_f32, float)
BARACUDA_KERNELS_REDUCE_LOGSUMEXP_INSTANTIATE(reduce_logsumexp_f16, __half)
BARACUDA_KERNELS_REDUCE_LOGSUMEXP_INSTANTIATE(reduce_logsumexp_bf16, __nv_bfloat16)
BARACUDA_KERNELS_REDUCE_LOGSUMEXP_INSTANTIATE(reduce_logsumexp_f64, double)
