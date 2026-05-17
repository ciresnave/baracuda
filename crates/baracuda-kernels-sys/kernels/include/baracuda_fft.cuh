// baracuda_fft.cuh
//
// Bespoke companion kernels for the cuFFT wrap (Milestone 6.4):
//
//   * fftshift / ifftshift — index permutation along the last axis of
//     a `[batch, n]` tensor. cuFFT has no native shift. Element-width
//     specialized (4 / 8 / 16 bytes per cell) so the same kernel covers
//     `f32`, `f64` / `Complex32`, and `Complex64` without templating
//     on the complex struct types.
//
//   * scale_inplace_{c32,c64,real_f32,real_f64} — multiply an in-place
//     buffer by a real scalar. Used to apply `1/N` normalization after
//     cuFFT's inverse exec (cuFFT returns N · IFFT(x); PyTorch's
//     `norm="backward"` convention wants IFFT(x)).
//
// Status codes mirror the elementwise family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

#ifndef BARACUDA_FFT_CUH
#define BARACUDA_FFT_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace fft {

// fftshift along the last axis of a [batch, n] tensor:
//   y[b, i] = x[b, (i + n/2) % n]
//
// Templated on a byte-storage struct `Cell` so the kernel is element-
// agnostic — the safe-plan layer picks the cell width matching the
// tensor element type. We use uint32_t (4 bytes), uint2 (8 bytes), and
// uint4 (16 bytes) as the storage types, which lets nvcc emit aligned
// vector loads/stores for the larger cells.
template <typename Cell>
__global__ void fftshift_kernel(
    const Cell* __restrict__ x,
    Cell* __restrict__ y,
    int64_t batch,
    int32_t n)
{
    int64_t total = batch * (int64_t)n;
    int64_t tid   = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step  = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int32_t half  = n / 2;
    for (int64_t i = tid; i < total; i += step) {
        int64_t b   = i / (int64_t)n;
        int32_t idx = (int32_t)(i - b * (int64_t)n);
        int32_t src = idx - half;
        if (src < 0) src += n;       // (idx + n/2) % n with branch-light maths
        y[i] = x[b * (int64_t)n + (int64_t)src];
    }
}

// ifftshift: y[b, i] = x[b, (i + n/2) % n]
//
// Matches NumPy / PyTorch convention: `ifftshift(x) = roll(x, -(n//2))`
// which rearranges into `y[i] = x[(i + n//2) % n]`. For even n this
// equals fftshift (the n/2 offset is its own inverse mod n). For odd
// n the two differ by one cell — fftshift uses offset `(n+1)/2`
// (equivalently `n - n//2`) and ifftshift uses offset `n/2`. This is
// the genuine inverse pair: `ifftshift(fftshift(x)) == x` for any n.
template <typename Cell>
__global__ void ifftshift_kernel(
    const Cell* __restrict__ x,
    Cell* __restrict__ y,
    int64_t batch,
    int32_t n)
{
    int64_t total = batch * (int64_t)n;
    int64_t tid   = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step  = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int32_t off   = n / 2;
    for (int64_t i = tid; i < total; i += step) {
        int64_t b   = i / (int64_t)n;
        int32_t idx = (int32_t)(i - b * (int64_t)n);
        int32_t src = idx + off;
        if (src >= n) src -= n;
        y[i] = x[b * (int64_t)n + (int64_t)src];
    }
}

template <typename Cell>
__host__ inline int32_t launch_fftshift(
    const Cell* x,
    Cell* y,
    int64_t batch,
    int32_t n,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t total = batch * (int64_t)n;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fftshift_kernel<Cell><<<blocks, kBlock, 0, stream>>>(x, y, batch, n);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename Cell>
__host__ inline int32_t launch_ifftshift(
    const Cell* x,
    Cell* y,
    int64_t batch,
    int32_t n,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t total = batch * (int64_t)n;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    ifftshift_kernel<Cell><<<blocks, kBlock, 0, stream>>>(x, y, batch, n);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// In-place scale of a single-precision complex buffer: y[i] *= scale
// (componentwise). Used to apply 1/N normalization after `cufftExecC2C`
// in the inverse direction.
struct Complex32 { float x; float y; };
struct Complex64 { double x; double y; };

__global__ inline void scale_inplace_c32_kernel(
    Complex32* __restrict__ y,
    int64_t numel,
    float scale)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        Complex32 v = y[i];
        v.x *= scale;
        v.y *= scale;
        y[i] = v;
    }
}

__global__ inline void scale_inplace_c64_kernel(
    Complex64* __restrict__ y,
    int64_t numel,
    double scale)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        Complex64 v = y[i];
        v.x *= scale;
        v.y *= scale;
        y[i] = v;
    }
}

template <typename T>
__global__ void scale_inplace_real_kernel(
    T* __restrict__ y,
    int64_t numel,
    T scale)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = y[i] * scale;
    }
}

__host__ inline int32_t launch_scale_inplace_c32(
    Complex32* y,
    int64_t numel,
    float scale,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    scale_inplace_c32_kernel<<<blocks, kBlock, 0, stream>>>(y, numel, scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_scale_inplace_c64(
    Complex64* y,
    int64_t numel,
    double scale,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    scale_inplace_c64_kernel<<<blocks, kBlock, 0, stream>>>(y, numel, scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_scale_inplace_real(
    T* y,
    int64_t numel,
    T scale,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    scale_inplace_real_kernel<T><<<blocks, kBlock, 0, stream>>>(y, numel, scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// N-D fftshift / ifftshift — single-pass general-permutation kernel.
// =============================================================================
//
// One thread per output element; each thread decomposes its flat output
// index into per-axis coords using the (dense, contiguous) output
// strides, then for every axis in the "shifted axes" set rotates that
// coord by the per-axis shift amount (`(out_coord + shift) % n`).
// Source flat index is recomposed from the rotated coords against the
// same strides — both `x` and `y` are dense in the same logical layout
// so the strides are shared.
//
// Rank cap is `MAX_RANK_ND = 8`, matching the `RollPlan` cap. The
// per-axis `shift_amt` array is the same length as `shape` — slots for
// non-shifted "batch" axes carry `0` (a no-op rotation that reduces to
// the identity for any coord). This collapses the kernel to a single
// inner loop and removes the need to carry a separate "which axes are
// shifted" bit-mask.
//
// Bandwidth-bound — one read + one write per element. Beats chained 1-D
// shifts (N · 2 = 2N traversals) at any rank > 1.

constexpr int kFftShiftNdMaxRank = 8;

// Pass-by-value descriptor arrays — the per-axis shape / shift / stride
// data lives in the kernel parameter block, no gmem indirection.
struct FftShiftNdDimsI32 { int32_t v[kFftShiftNdMaxRank]; };
struct FftShiftNdDimsI64 { int64_t v[kFftShiftNdMaxRank]; };

template <typename Cell>
__global__ void fftshift_nd_kernel(
    const Cell* __restrict__ x,
    Cell* __restrict__ y,
    int64_t total,
    int32_t rank,
    FftShiftNdDimsI32 shape,
    FftShiftNdDimsI32 shift_amt,
    FftShiftNdDimsI64 stride)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;

    for (int64_t i = tid; i < total; i += step) {
        // Decompose flat output index `i` into per-axis coords using
        // strides. Both tensors are dense / contiguous in the same
        // layout so `stride[d] = product(shape[d+1..rank])` and
        // `stride[0]` is the largest stride.
        int64_t rem = i;
        int64_t src = 0;
        #pragma unroll
        for (int d = 0; d < kFftShiftNdMaxRank; ++d) {
            if (d >= rank) break;
            int64_t s = stride.v[d];
            int32_t n = shape.v[d];
            int32_t out_coord = (int32_t)(rem / s);
            rem -= (int64_t)out_coord * s;
            int32_t shift = shift_amt.v[d];
            int32_t src_coord = out_coord + shift;
            if (src_coord >= n) src_coord -= n;  // shift in [0, n) keeps src_coord in [0, 2n)
            src += (int64_t)src_coord * s;
        }
        y[i] = x[src];
    }
}

template <typename Cell>
__host__ inline int32_t launch_fftshift_nd(
    const Cell* x,
    Cell* y,
    int64_t total,
    int32_t rank,
    const int32_t* shape_host,
    const int32_t* shift_amt_host,
    const int64_t* stride_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > kFftShiftNdMaxRank) return 2;
    FftShiftNdDimsI32 shape = {}, shift_amt = {};
    FftShiftNdDimsI64 stride = {};
    for (int d = 0; d < rank; ++d) {
        shape.v[d]     = shape_host[d];
        shift_amt.v[d] = shift_amt_host[d];
        stride.v[d]    = stride_host[d];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fftshift_nd_kernel<Cell><<<blocks, kBlock, 0, stream>>>(
        x, y, total, rank, shape, shift_amt, stride);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::fft

// =============================================================================
// Instantiation macros — emit `extern "C"` launcher symbols.
// =============================================================================

// ABI: `(batch, n, x, y, ws, ws_bytes, stream) -> i32`. The element type
// is selected by the instantiation macro's `BYTES` width (4 / 8 / 16)
// and routed through `uint32_t` / `uint2` / `uint4` storage so nvcc can
// emit aligned vector loads.

#define BARACUDA_KERNELS_FFTSHIFT_INSTANTIATE(BYTES, CELL_T)                                        \
    extern "C" int32_t baracuda_kernels_fftshift_##BYTES##_run(                                     \
        int64_t batch,                                                                              \
        int32_t n,                                                                                  \
        const void* x,                                                                              \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (batch < 0 || n < 0) return 2;                                                           \
        if (batch == 0 || n == 0) return 0;                                                         \
        if (x == nullptr || y == nullptr) return 2;                                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::fft::launch_fftshift<CELL_T>(                                              \
            static_cast<const CELL_T*>(x),                                                          \
            static_cast<CELL_T*>(y),                                                                \
            batch, n, stream);                                                                      \
    }

#define BARACUDA_KERNELS_IFFTSHIFT_INSTANTIATE(BYTES, CELL_T)                                       \
    extern "C" int32_t baracuda_kernels_ifftshift_##BYTES##_run(                                    \
        int64_t batch,                                                                              \
        int32_t n,                                                                                  \
        const void* x,                                                                              \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (batch < 0 || n < 0) return 2;                                                           \
        if (batch == 0 || n == 0) return 0;                                                         \
        if (x == nullptr || y == nullptr) return 2;                                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::fft::launch_ifftshift<CELL_T>(                                             \
            static_cast<const CELL_T*>(x),                                                          \
            static_cast<CELL_T*>(y),                                                                \
            batch, n, stream);                                                                      \
    }

// ABI: `(numel, scale, y, ws, ws_bytes, stream) -> i32`. In-place scale
// of a complex buffer by a real scalar. Two flavors (c32 / c64) because
// the scale type matches the underlying float width.

#define BARACUDA_KERNELS_SCALE_INPLACE_C32_INSTANTIATE()                                            \
    extern "C" int32_t baracuda_kernels_scale_inplace_c32_run(                                      \
        int64_t numel,                                                                              \
        float scale,                                                                                \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                    \
        if (numel == 0) return 0;                                                                   \
        if (y == nullptr) return 2;                                                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::fft::launch_scale_inplace_c32(                                             \
            static_cast<baracuda::fft::Complex32*>(y), numel, scale, stream);                       \
    }

#define BARACUDA_KERNELS_SCALE_INPLACE_C64_INSTANTIATE()                                            \
    extern "C" int32_t baracuda_kernels_scale_inplace_c64_run(                                      \
        int64_t numel,                                                                              \
        double scale,                                                                               \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                    \
        if (numel == 0) return 0;                                                                   \
        if (y == nullptr) return 2;                                                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::fft::launch_scale_inplace_c64(                                             \
            static_cast<baracuda::fft::Complex64*>(y), numel, scale, stream);                       \
    }

#define BARACUDA_KERNELS_SCALE_INPLACE_REAL_INSTANTIATE(NAME, T)                                    \
    extern "C" int32_t baracuda_kernels_scale_inplace_real_##NAME##_run(                            \
        int64_t numel,                                                                              \
        T scale,                                                                                    \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                    \
        if (numel == 0) return 0;                                                                   \
        if (y == nullptr) return 2;                                                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::fft::launch_scale_inplace_real<T>(                                         \
            static_cast<T*>(y), numel, scale, stream);                                              \
    }

// ABI: `(total, rank, shape, shift_amt, stride, x, y, ws, ws_bytes, stream) -> i32`.
// N-D fftshift / ifftshift, byte-width specialized (matches the 1-D
// variant's element-agnostic design). Caller must marshal `shape`,
// `shift_amt`, `stride` to host arrays; the launcher copies them to
// device-side cmdline-arg memory implicitly via the kernel param block.
// `shape`, `shift_amt`, `stride` therefore point to host memory and are
// passed by value through the kernel launch's argument-marshaling path.
//
// Per-axis `shift_amt[d]` semantics:
//   - For shifted axes: `n/2` for fftshift, `n - n/2` for ifftshift.
//   - For pass-through (batch) axes: `0` (collapses to identity).

#define BARACUDA_KERNELS_FFTSHIFT_ND_INSTANTIATE(BYTES, CELL_T)                                     \
    extern "C" int32_t baracuda_kernels_fftshift_nd_##BYTES##_run(                                  \
        int64_t total,                                                                              \
        int32_t rank,                                                                               \
        const int32_t* shape,                                                                       \
        const int32_t* shift_amt,                                                                   \
        const int64_t* stride,                                                                      \
        const void* x,                                                                              \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (total < 0 || rank < 0 || rank > baracuda::fft::kFftShiftNdMaxRank) return 2;            \
        if (total == 0) return 0;                                                                   \
        if (x == nullptr || y == nullptr) return 2;                                                 \
        if (shape == nullptr || shift_amt == nullptr || stride == nullptr) return 2;                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::fft::launch_fftshift_nd<CELL_T>(                                           \
            static_cast<const CELL_T*>(x),                                                          \
            static_cast<CELL_T*>(y),                                                                \
            total, rank, shape, shift_amt, stride, stream);                                         \
    }

#endif // BARACUDA_FFT_CUH
