// baracuda_random.cuh
//
// Templated kernels and INSTANTIATE macros for the random / sampling op
// family (Phase 4.5 Category Q of the comprehensive plan).
//
// Scope today:
//   * Bernoulli      — `y = (rand < p) ? 1 : 0`, Bool output. Reads a
//                       caller-supplied uniform-rand `float` buffer.
//   * Dropout (FW)   — `y = mask * x * scale`, `mask = (rand < 1 - p)`
//                       Bool. `scale = 1 / (1 - p)`.
//   * Dropout (BW)   — `dx = mask * dy * scale`.
//
// `Uniform` and `Normal` route directly through the cuRAND host API at
// the safe-plan layer — no bespoke kernel needed.
//
// Status codes mirror the elementwise family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

#ifndef BARACUDA_RANDOM_CUH
#define BARACUDA_RANDOM_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace baracuda { namespace random {

// Bernoulli — one Bool (uint8_t 0/1) per output cell.
__global__ inline void bernoulli_kernel(
    const float* __restrict__ rand,
    uint8_t* __restrict__ y,
    int64_t numel,
    float p)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = (rand[i] < p) ? (uint8_t)1 : (uint8_t)0;
    }
}

__host__ inline int32_t launch_bernoulli(
    const float* rand,
    uint8_t* y,
    int64_t numel,
    float p,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    bernoulli_kernel<<<blocks, kBlock, 0, stream>>>(rand, y, numel, p);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Dropout FW — writes `y` and `mask` simultaneously. `scale` is passed
// by value (caller computes `1 / (1 - p)` to avoid a divide-by-zero check
// inside the kernel hot path; the safe-plan layer handles `p == 1` by
// short-circuiting before invoking this kernel).
//
// `T` is the element type for `x` and `y`. The uniform-rand buffer is
// always `float` regardless of `T` (cuRAND's f32 uniform generator is
// fast and the `<` comparison loses no useful entropy for the dropout
// use case).
template <typename T, typename Scale>
__global__ void dropout_fw_kernel(
    const T* __restrict__ x,
    const float* __restrict__ rand,
    T* __restrict__ y,
    uint8_t* __restrict__ mask,
    int64_t numel,
    float keep_prob,
    Scale scale)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        uint8_t m = (rand[i] < keep_prob) ? (uint8_t)1 : (uint8_t)0;
        mask[i] = m;
        // mask * scale is 0 or scale; multiplying x by that yields the
        // dropout output without a branch on the data path.
        y[i] = static_cast<T>(static_cast<Scale>(x[i]) * (m ? scale : (Scale)0));
    }
}

template <typename T, typename Scale>
__host__ inline int32_t launch_dropout_fw(
    const T* x,
    const float* rand,
    T* y,
    uint8_t* mask,
    int64_t numel,
    float keep_prob,
    Scale scale,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dropout_fw_kernel<T, Scale><<<blocks, kBlock, 0, stream>>>(
        x, rand, y, mask, numel, keep_prob, scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Dropout BW — `dx = dy * mask * scale`.
template <typename T, typename Scale>
__global__ void dropout_bw_kernel(
    const T* __restrict__ dy,
    const uint8_t* __restrict__ mask,
    T* __restrict__ dx,
    int64_t numel,
    Scale scale)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        uint8_t m = mask[i];
        dx[i] = static_cast<T>(static_cast<Scale>(dy[i]) * (m ? scale : (Scale)0));
    }
}

template <typename T, typename Scale>
__host__ inline int32_t launch_dropout_bw(
    const T* dy,
    const uint8_t* mask,
    T* dx,
    int64_t numel,
    Scale scale,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dropout_bw_kernel<T, Scale><<<blocks, kBlock, 0, stream>>>(
        dy, mask, dx, numel, scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// In-place affine `y = scale * y + offset`. Used to remap a cuRAND
// uniform-(0, 1] buffer into Uniform(low, high].
template <typename T>
__global__ void affine_inplace_kernel(
    T* __restrict__ y,
    int64_t numel,
    T scale,
    T offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = static_cast<T>(static_cast<T>(y[i]) * scale + offset);
    }
}

template <typename T>
__host__ inline int32_t launch_affine_inplace(
    T* y,
    int64_t numel,
    T scale,
    T offset,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_inplace_kernel<T><<<blocks, kBlock, 0, stream>>>(y, numel, scale, offset);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Half-precision in-place affine — compute at f32, store at __half.
// Mirrors the forward `affine_contig_kernel_f16` upcast/downcast pattern;
// `__half * __half + __half` has no native operator path (would require
// `__hmul`/`__hadd` intrinsics).
__global__ inline void affine_inplace_kernel_f16(
    __half* __restrict__ y,
    int64_t numel,
    float scale,
    float offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float yi = __half2float(y[i]);
        y[i] = __float2half(scale * yi + offset);
    }
}

// bf16 in-place affine — compute at f32, store at __nv_bfloat16.
__global__ inline void affine_inplace_kernel_bf16(
    __nv_bfloat16* __restrict__ y,
    int64_t numel,
    float scale,
    float offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float yi = __bfloat162float(y[i]);
        y[i] = __float2bfloat16(scale * yi + offset);
    }
}

__host__ inline int32_t launch_affine_inplace_f16(
    __half* y,
    int64_t numel,
    float scale,
    float offset,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_inplace_kernel_f16<<<blocks, kBlock, 0, stream>>>(y, numel, scale, offset);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_affine_inplace_bf16(
    __nv_bfloat16* y,
    int64_t numel,
    float scale,
    float offset,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_inplace_kernel_bf16<<<blocks, kBlock, 0, stream>>>(y, numel, scale, offset);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Phase 62 — strided in-place affine `y[off(i)] = scale * y[off(i)] + offset`.
//
// Stride-aware variant of `affine_inplace_kernel<T>`. Each thread decomposes
// its linear index `i` into a multi-coord, computes a single stride-aware
// offset `off(i)` into the target buffer, reads `y[off(i)]`, applies the
// affine, writes back. Single-pointer ABI: caller passes `y` + `stride_y`
// only; in-place is implicit in the kernel structure.
//
// Per-thread access pattern is identical to contig in-place: one read
// then one write at the same address, so the same aliasing-safety
// reasoning holds. The pre-launch contract for callers that want this
// to *replace* a forward strided affine call with `x_ptr == y_ptr` is:
// `stride_x == stride_y` AND `stride_y` is a valid permutation (no
// zero strides on `y`, no two `i` mapping to the same `off`). Both are
// caller obligations — the kernel does no validation.
// =============================================================================

inline constexpr int MAX_RANK = 8;
struct DimsI32 { int32_t v[MAX_RANK]; };
struct DimsI64 { int64_t v[MAX_RANK]; };

// Templated strided in-place affine — compute and storage at T.
template <typename T>
__global__ void affine_inplace_strided_kernel(
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_y,
    T scale,
    T offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += c * stride_y.v[d];
        }
        y[off_y] = static_cast<T>(static_cast<T>(y[off_y]) * scale + offset);
    }
}

// f16-storage / f32-compute strided in-place — matches forward
// `affine_strided_kernel_f16` upcast/downcast pattern.
__global__ inline void affine_inplace_strided_kernel_f16(
    __half* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_y,
    float scale,
    float offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += c * stride_y.v[d];
        }
        float yi = __half2float(y[off_y]);
        y[off_y] = __float2half(scale * yi + offset);
    }
}

// bf16-storage / f32-compute strided in-place.
__global__ inline void affine_inplace_strided_kernel_bf16(
    __nv_bfloat16* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_y,
    float scale,
    float offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += c * stride_y.v[d];
        }
        float yi = __bfloat162float(y[off_y]);
        y[off_y] = __float2bfloat16(scale * yi + offset);
    }
}

template <typename T>
__host__ inline int32_t launch_affine_inplace_strided(
    T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_y_host,
    T scale,
    T offset,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape{};
    DimsI64 sy{};
    for (int d = 0; d < rank; ++d) {
        shape.v[d] = shape_host[d];
        sy.v[d]    = stride_y_host[d];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_inplace_strided_kernel<T><<<blocks, kBlock, 0, stream>>>(
        y, numel, rank, shape, sy, scale, offset);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_affine_inplace_strided_f16(
    __half* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_y_host,
    float scale,
    float offset,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape{};
    DimsI64 sy{};
    for (int d = 0; d < rank; ++d) {
        shape.v[d] = shape_host[d];
        sy.v[d]    = stride_y_host[d];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_inplace_strided_kernel_f16<<<blocks, kBlock, 0, stream>>>(
        y, numel, rank, shape, sy, scale, offset);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_affine_inplace_strided_bf16(
    __nv_bfloat16* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_y_host,
    float scale,
    float offset,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape{};
    DimsI64 sy{};
    for (int d = 0; d < rank; ++d) {
        shape.v[d] = shape_host[d];
        sy.v[d]    = stride_y_host[d];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_inplace_strided_kernel_bf16<<<blocks, kBlock, 0, stream>>>(
        y, numel, rank, shape, sy, scale, offset);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::random

// =============================================================================
// Instantiation macros — emit `extern "C"` launcher symbols.
// =============================================================================

// ABI: `(numel, p, rand, y, ws, ws_bytes, stream) -> i32`.
//
// `y` is a packed-Bool buffer (`uint8_t`). `rand` is a `float` buffer
// generated by cuRAND.
#define BARACUDA_KERNELS_BERNOULLI_INSTANTIATE()                                                   \
    extern "C" int32_t baracuda_kernels_bernoulli_run(                                             \
        int64_t numel,                                                                              \
        float p,                                                                                    \
        const void* rand,                                                                           \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (rand == nullptr || y == nullptr) return 2;                                             \
        if (!(p >= 0.0f && p <= 1.0f)) return 2;                                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::random::launch_bernoulli(                                                  \
            static_cast<const float*>(rand),                                                        \
            static_cast<uint8_t*>(y),                                                               \
            numel, p, stream);                                                                      \
    }

// ABI: `(numel, p, scale, x, rand, y, mask, ws, ws_bytes, stream) -> i32`.
//
// `x` / `y` are `T`-typed; `rand` is `float`; `mask` is packed-Bool
// (`uint8_t`). The safe-plan layer enforces `0 <= p < 1` before
// dispatching (so `scale = 1/(1-p)` is finite).
#define BARACUDA_KERNELS_DROPOUT_INSTANTIATE(NAME, T, SCALE_T)                                     \
    extern "C" int32_t baracuda_kernels_dropout_##NAME##_run(                                      \
        int64_t numel,                                                                              \
        float p,                                                                                    \
        SCALE_T scale,                                                                              \
        const void* x,                                                                              \
        const void* rand,                                                                           \
        void* y,                                                                                    \
        void* mask,                                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || rand == nullptr || y == nullptr || mask == nullptr) return 2;          \
        if (!(p >= 0.0f && p < 1.0f)) return 2;                                                    \
        float keep_prob = 1.0f - p;                                                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::random::launch_dropout_fw<T, SCALE_T>(                                     \
            static_cast<const T*>(x),                                                               \
            static_cast<const float*>(rand),                                                        \
            static_cast<T*>(y),                                                                     \
            static_cast<uint8_t*>(mask),                                                            \
            numel, keep_prob, scale, stream);                                                       \
    }

// ABI: `(numel, scale, dy, mask, dx, ws, ws_bytes, stream) -> i32`.
#define BARACUDA_KERNELS_DROPOUT_BACKWARD_INSTANTIATE(NAME, T, SCALE_T)                            \
    extern "C" int32_t baracuda_kernels_dropout_backward_##NAME##_run(                             \
        int64_t numel,                                                                              \
        SCALE_T scale,                                                                              \
        const void* dy,                                                                             \
        const void* mask,                                                                           \
        void* dx,                                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (dy == nullptr || mask == nullptr || dx == nullptr) return 2;                           \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::random::launch_dropout_bw<T, SCALE_T>(                                     \
            static_cast<const T*>(dy),                                                              \
            static_cast<const uint8_t*>(mask),                                                      \
            static_cast<T*>(dx),                                                                    \
            numel, scale, stream);                                                                  \
    }

// ABI: `(numel, scale, offset, y, ws, ws_bytes, stream) -> i32`.
//
// In-place affine map `y[i] = scale * y[i] + offset`. Used to remap a
// cuRAND uniform-(0, 1] buffer into Uniform(low, high] without a second
// kernel set, and (Phase 61) for in-place weight-decay / Op::AddScalar /
// Op::MulScalar on contiguous tensors in Fuel's executor.
#define BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_affine_inplace_##NAME##_run(                               \
        int64_t numel,                                                                              \
        T scale,                                                                                    \
        T offset,                                                                                   \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (y == nullptr) return 2;                                                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::random::launch_affine_inplace<T>(                                          \
            static_cast<T*>(y), numel, scale, offset, stream);                                      \
    }

// f32-scalar variant for half-precision storage types (Phase 61).
//
// Same ABI shape, but `scale` and `offset` are always `float` regardless
// of the storage dtype. This matches the forward `affine_{f16,bf16}_run`
// convention (avoids passing `__half` / `__nv_bfloat16` by value through
// C ABI, which is compiler-dependent) and matches how Fuel's CPU
// `affine_inplace_{f16,bf16}` kernels carry their scalars. Half-precision
// kernels use the same upcast-to-f32, compute, downcast pattern as the
// forward `affine_contig_kernel_{f16,bf16}`.
//
// NAME : `f16` or `bf16` — must match the suffix of the
//        `launch_affine_inplace_<NAME>` host wrapper.
// T    : `__half` or `__nv_bfloat16` — storage type, for the y pointer
//        cast.
#define BARACUDA_KERNELS_AFFINE_INPLACE_F32SCALAR_INSTANTIATE(NAME, T)                             \
    extern "C" int32_t baracuda_kernels_affine_inplace_##NAME##_run(                               \
        int64_t numel,                                                                              \
        float scale,                                                                                \
        float offset,                                                                               \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (y == nullptr) return 2;                                                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::random::launch_affine_inplace_##NAME(                                      \
            static_cast<T*>(y), numel, scale, offset, stream);                                      \
    }

// Phase 62 — strided in-place affine, ABI:
//   (numel, rank, shape, stride_y, scale, offset, y, ws, ws_bytes, stream) -> i32
//
// shape and stride_y are host-side arrays of length `rank` (int32_t for
// shape, int64_t for stride). MAX_RANK = 8; rank > 8 returns
// STATUS_INVALID_ARG.
#define BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_affine_inplace_##NAME##_strided_run(                       \
        int64_t numel,                                                                              \
        int32_t rank,                                                                               \
        const int32_t* shape,                                                                       \
        const int64_t* stride_y,                                                                    \
        T scale,                                                                                    \
        T offset,                                                                                   \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (y == nullptr) return 2;                                                                 \
        if (rank > 0 && (shape == nullptr || stride_y == nullptr)) return 2;                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::random::launch_affine_inplace_strided<T>(                                  \
            static_cast<T*>(y), numel, rank, shape, stride_y, scale, offset, stream);              \
    }

// Phase 62 — f32-scalar strided in-place variant for half-precision.
// Matches the contig f32-scalar pattern (`scale` and `offset` are
// always `float`).
#define BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_F32SCALAR_INSTANTIATE(NAME, T)                     \
    extern "C" int32_t baracuda_kernels_affine_inplace_##NAME##_strided_run(                       \
        int64_t numel,                                                                              \
        int32_t rank,                                                                               \
        const int32_t* shape,                                                                       \
        const int64_t* stride_y,                                                                    \
        float scale,                                                                                \
        float offset,                                                                               \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (y == nullptr) return 2;                                                                 \
        if (rank > 0 && (shape == nullptr || stride_y == nullptr)) return 2;                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::random::launch_affine_inplace_strided_##NAME(                              \
            static_cast<T*>(y), numel, rank, shape, stride_y, scale, offset, stream);              \
    }

#endif // BARACUDA_RANDOM_CUH
