// baracuda_elementwise.cuh
//
// Templated kernels and INSTANTIATE macros for the elementwise op family
// (Phase 3 of the baracuda-kernels comprehensive plan).
//
// Each elementwise op family is one templated __global__ in this header,
// parameterized on the element type `T` and a binary / unary / ternary
// functor type `F`. Per-op .cu files supply the functor type and
// invoke the matching INSTANTIATE macro to emit `extern "C"` launcher
// symbols.
//
// Status codes returned by the launchers mirror the GEMM family:
//   0 success
//   1 misaligned operand
//   2 invalid problem (e.g. negative numel)
//   3 unsupported
//   4 workspace too small
//   5 internal kernel error (typically a launch failure)
//
// All launchers take `(workspace, workspace_bytes)` for ABI parity with
// the GEMM family even though the elementwise kernels here don't need
// scratch — pass `(nullptr, 0)` from Rust.

#ifndef BARACUDA_ELEMENTWISE_CUH
#define BARACUDA_ELEMENTWISE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace elementwise {

// Contiguous binary-elementwise kernel. Both inputs and the output are
// fully contiguous and laid out identically in gmem; the kernel is a
// pure linear sweep. Each thread handles a stride of grid_size elements
// so a grid capped at 65535 blocks still covers an arbitrarily large
// numel via a per-thread loop.
//
// T is the scalar element type (e.g. `float` for f32). F is the binary
// functor type, expected to define
//   `__device__ T operator()(T, T) const`
template <typename T, typename F>
__global__ void binary_pointwise_contig_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ y,
    int64_t numel,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = op(a[i], b[i]);
    }
}

// Internal launch helper — wraps the kernel launch + last-error check.
// Centralized so future tile / vectorization changes only touch this
// function, not every INSTANTIATE site.
template <typename T, typename F>
__host__ inline int32_t launch_binary_pointwise_contig(
    const T* a, const T* b, T* y,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    binary_pointwise_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, y, numel, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Strided / broadcast binary-elementwise kernel.
// =============================================================================
//
// Handles every non-contiguous case the contig fast path can't: broadcast
// (stride 0 on an axis), transposed views, gather-like strided views.
// The same kernel covers all of them — we do per-axis stride math
// independently for each operand, with no special "broadcast" branch.
//
// Inputs:
//   `rank`    : number of valid axes, in [0, MAX_RANK]. The kernel
//               iterates the rightmost (innermost) axis first.
//   `shape`   : the OUTPUT shape (`y.shape[d]`). Always populated from
//               y — operands `a` and `b` are read via their own strides
//               at the same coords, so an operand with shape[d] == 1
//               and stride[d] == 0 is broadcast along axis d.
//   `stride_*`: per-operand element strides. A stride of 0 along axis
//               d marks a broadcast operand. Negative strides are not
//               currently supported (no flip-like views).
//
// Performance note: each output element costs `rank` integer
// divmods to map the linear index back to a coord — measurable
// overhead on bandwidth-bound elementwise kernels but acceptable
// for v1. Future tuning can specialize on rank (template
// instantiation per rank ∈ {1, 2, 3, 4}) or use magic-number
// division.

inline constexpr int MAX_RANK = 8;

// Plain-data structs used to pass the fixed-rank-8 shape / stride
// arrays through the kernel parameter block by value. CUDA kernel args
// are passed by value, so these need to be small (≤4 KB total) and
// trivially copyable — both criteria are met (32 B for DimsI32,
// 64 B for DimsI64).
struct DimsI32 { int32_t v[MAX_RANK]; };
struct DimsI64 { int64_t v[MAX_RANK]; };

template <typename T, typename F>
__global__ void binary_pointwise_strided_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_a,
    DimsI64 stride_b,
    DimsI64 stride_y,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_a = 0, off_b = 0, off_y = 0;
        // Unravel from the rightmost (innermost / fastest-varying) dim.
        // Loop bound is `rank`, NOT MAX_RANK — unused trailing slots
        // contribute zero stride and would self-cancel, but we skip
        // them to avoid `linear % 0` undefined behavior.
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_a += coord * stride_a.v[d];
            off_b += coord * stride_b.v[d];
            off_y += coord * stride_y.v[d];
        }
        y[off_y] = op(a[off_a], b[off_b]);
    }
}

// Internal launch helper — copies host-side shape/stride arrays into
// kernel param structs, then launches. Same grid-cap loop convention
// as the contig launcher.
template <typename T, typename F>
__host__ inline int32_t launch_binary_pointwise_strided(
    const T* a, const T* b, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_a_host,
    const int64_t* stride_b_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape  = {};
    DimsI64 sa     = {};
    DimsI64 sb     = {};
    DimsI64 sy     = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sa.v[i]    = stride_a_host[i];
        sb.v[i]    = stride_b_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    binary_pointwise_strided_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, y, numel, rank, shape, sa, sb, sy, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Binary backward kernels — Phase 3 backward family
// =============================================================================
//
// Two kernel templates by saved-tensor requirement:
//   * `binary_backward_nosave_kernel` — for ops whose gradient depends
//     only on `dy` (Add: (dy, dy); Sub: (dy, -dy)). Functor signature:
//       `__device__ void operator()(T dy, T& da, T& db) const`.
//   * `binary_backward_saves_kernel` — for ops whose gradient references
//     the saved forward inputs `a`, `b` (Mul: (dy*b, dy*a); Div:
//     (dy/b, -dy*a/b²)). Functor signature:
//       `__device__ void operator()(T dy, T a, T b, T& da, T& db) const`.
//
// Same launch convention as the forward elementwise family — pure SIMT,
// linear sweep, grid-cap loop for unbounded numel.

template <typename T, typename F>
__global__ void binary_backward_nosave_kernel(
    const T* __restrict__ dy,
    T* __restrict__ da,
    T* __restrict__ db,
    int64_t numel,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        T da_out, db_out;
        op(dy[i], da_out, db_out);
        da[i] = da_out;
        db[i] = db_out;
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_binary_backward_nosave(
    const T* dy, T* da, T* db,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    binary_backward_nosave_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        dy, da, db, numel, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void binary_backward_saves_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ da,
    T* __restrict__ db,
    int64_t numel,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        T da_out, db_out;
        op(dy[i], a[i], b[i], da_out, db_out);
        da[i] = da_out;
        db[i] = db_out;
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_binary_backward_saves(
    const T* dy, const T* a, const T* b, T* da, T* db,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    binary_backward_saves_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        dy, a, b, da, db, numel, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Parameterized unary / binary kernels — Phase 3 deferred (Threshold, Lerp)
// =============================================================================
//
// Sibling shapes of the plain unary / binary pointwise kernels above,
// extended with one or two `float` scalar parameters threaded through to
// the device functor. The functor's `operator()` takes the param(s) by
// value as additional arguments — kernels are still SIMT linear sweeps,
// contig-only (no strided variant for the trailblazer).
//
// Functor signatures:
//   Unary param (2 scalars):     `T operator()(T x, float p0, float p1) const`
//   Unary param BW (2 scalars):  `T operator()(T dy, T x, float p0, float p1) const`
//   Binary param (1 scalar):     `T operator()(T a, T b, float p) const`
//   Binary param BW (1 scalar):  `void operator()(T dy, float p, T& da, T& db) const`
//
// Today's wired ops:
//   Threshold FW / BW — 2 params (t, v).
//   Lerp     FW / BW — 1 param  (weight).
//
// Future param-bearing ops (LeakyRelu, ELU, Hardshrink, Softshrink) can
// re-emit through these macros to expose their hardcoded coefficients
// as runtime args.

template <typename T, typename F>
__global__ void unary_param_pointwise_contig_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    float p0,
    float p1,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = op(x[i], p0, p1);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_unary_param_pointwise_contig(
    const T* x, T* y,
    int64_t numel,
    float p0, float p1,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unary_param_pointwise_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, p0, p1, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void unary_param_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    T* __restrict__ dx,
    int64_t numel,
    float p0,
    float p1,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        dx[i] = op(dy[i], x[i], p0, p1);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_unary_param_backward(
    const T* dy, const T* x, T* dx,
    int64_t numel,
    float p0, float p1,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unary_param_backward_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        dy, x, dx, numel, p0, p1, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Parameterized unary strided kernels — Phase 14.2 (PowI)
// =============================================================================
//
// Sibling of `unary_param_pointwise_contig_kernel` that walks non-contig
// input / output views. One thread per output cell — for each linear
// output index we decompose into a multi-coord, then dot with `stride_x`
// and `stride_y` (signed i64) to land at the operand offsets. Same
// 1-thread-per-element pattern as `unary_pointwise_strided_kernel`,
// extended with `p0`/`p1` passthrough to the functor.
//
// Backward sibling: same shape, but with three stride arrays
// (`stride_x`, `stride_dy`, `stride_dx`) so the BW launcher can route
// each of the three operands through its own view.

template <typename T, typename F>
__global__ void unary_param_pointwise_strided_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_x,
    DimsI64 stride_y,
    float p0,
    float p1,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_x = 0, off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_x += coord * stride_x.v[d];
            off_y += coord * stride_y.v[d];
        }
        y[off_y] = op(x[off_x], p0, p1);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_unary_param_pointwise_strided(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    float p0, float p1,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 sx    = {};
    DimsI64 sy    = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unary_param_pointwise_strided_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, rank, shape, sx, sy, p0, p1, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void unary_param_backward_strided_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_x,
    DimsI64 stride_dy,
    DimsI64 stride_dx,
    float p0,
    float p1,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_x = 0, off_dy = 0, off_dx = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_x  += coord * stride_x.v[d];
            off_dy += coord * stride_dy.v[d];
            off_dx += coord * stride_dx.v[d];
        }
        dx[off_dx] = op(dy[off_dy], x[off_x], p0, p1);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_unary_param_backward_strided(
    const T* dy, const T* x, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_dx_host,
    float p0, float p1,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 sx    = {};
    DimsI64 sdy   = {};
    DimsI64 sdx   = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sdy.v[i]   = stride_dy_host[i];
        sdx.v[i]   = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unary_param_backward_strided_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        dy, x, dx, numel, rank, shape, sx, sdy, sdx, p0, p1, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void binary_param_pointwise_contig_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ y,
    int64_t numel,
    float p,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = op(a[i], b[i], p);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_binary_param_pointwise_contig(
    const T* a, const T* b, T* y,
    int64_t numel,
    float p,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    binary_param_pointwise_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, y, numel, p, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void binary_param_backward_nosave_kernel(
    const T* __restrict__ dy,
    T* __restrict__ da,
    T* __restrict__ db,
    int64_t numel,
    float p,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        T da_out, db_out;
        op(dy[i], p, da_out, db_out);
        da[i] = da_out;
        db[i] = db_out;
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_binary_param_backward_nosave(
    const T* dy, T* da, T* db,
    int64_t numel,
    float p,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    binary_param_backward_nosave_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        dy, da, db, numel, p, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Unary backward kernel — Phase 3 unary-backward trailblazer
// =============================================================================
//
// `dx = f'(saved) * dy` for any unary op where the gradient is a
// pointwise function of `dy` and one saved tensor. Two callers exist
// in practice:
//   * Saved-x ops (Sin, Cos, Log, ...): the gradient references the
//     forward input `x`, so the caller passes `x` as `saved`.
//   * Saved-y ops (Exp, Sigmoid, Tanh, Sqrt, ...): the gradient
//     references the forward output `y`, so the caller passes `y`
//     as `saved`.
//
// The kernel itself is save-shape-agnostic — it just sees `(dy, saved)`
// and applies the functor. Which save to pass is a Rust-side concern.

template <typename T, typename F>
__global__ void unary_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ saved,
    T* __restrict__ dx,
    int64_t numel,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        dx[i] = op(dy[i], saved[i]);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_unary_backward(
    const T* dy, const T* saved, T* dx,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unary_backward_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        dy, saved, dx, numel, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Reduce-sum backward / broadcast-copy kernel — Phase 4 BW trailblazer
// =============================================================================
//
// `dx[c] = dy[c_strided]` for every coord `c` in dx's shape. The caller
// arranges `stride_dy[reduced_axis] = 0` so reading varies-coord-on-
// reduced-axis collapses to the singleton dy slot, effectively
// broadcasting dy across the reduced axis. Sum BW is exactly this — no
// op, just a strided copy.
//
// Pattern matches `binary_pointwise_strided_kernel` but with one input
// instead of two. Future Mean BW will reuse this kernel with a
// `dy[i] / k` scaling functor (or a separate launcher).

template <typename T>
__global__ void reduce_sum_backward_kernel(
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_dx)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_dx = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dy += coord * stride_dy.v[d];
            off_dx += coord * stride_dx.v[d];
        }
        dx[off_dx] = dy[off_dy];
    }
}

template <typename T>
__host__ inline int32_t launch_reduce_sum_backward(
    const T* dy, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_dx_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape  = {};
    DimsI64 s_dy   = {};
    DimsI64 s_dx   = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_sum_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, dx, numel, rank, shape, s_dy, s_dx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Reduce-mean backward kernel — same shape as Sum BW with an additional
// `inv_extent` scale. `dx[c] = dy[c_strided] * inv_extent` where
// `inv_extent = 1 / reduced_extent`. The Rust dispatcher computes
// `inv_extent` once on the host (in f64 for maximum precision) and the
// kernel casts to T at use.

template <typename T>
__global__ void reduce_mean_backward_kernel(
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_dx,
    double inv_extent_d)
{
    T inv_extent = T(inv_extent_d);
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_dx = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dy += coord * stride_dy.v[d];
            off_dx += coord * stride_dx.v[d];
        }
        dx[off_dx] = dy[off_dy] * inv_extent;
    }
}

template <typename T>
__host__ inline int32_t launch_reduce_mean_backward(
    const T* dy, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_dx_host,
    double inv_extent,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape  = {};
    DimsI64 s_dy   = {};
    DimsI64 s_dx   = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_mean_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, dx, numel, rank, shape, s_dy, s_dx, inv_extent);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Reduce-max / reduce-min backward kernel. Used by BOTH Max BW and Min
// BW — the routing logic is identical: `dx[c] = dy[c_reduced]` for
// every input position where `x[c] == y[c_reduced]`, else 0. y is the
// forward output (max or min) — whichever forward kind ran, the BW
// kernel just compares.
//
// Tie semantic: every tied position gets the FULL gradient (split-
// across-ties / "share" convention, matching JAX). PyTorch routes
// gradient to the first tied index only — that requires a saved
// argmax/argmin tensor, deferred.
//
// dy and y are read with `stride[reduce_axis] = 0` (broadcast); x and
// dx walk the full input shape.

template <typename T>
__global__ void reduce_max_min_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_x = 0, off_y = 0, off_dx = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dy += coord * stride_dy.v[d];
            off_x  += coord * stride_x.v[d];
            off_y  += coord * stride_y.v[d];
            off_dx += coord * stride_dx.v[d];
        }
        T x_val = x[off_x];
        T y_val = y[off_y];
        // T(0) is the additive identity; works for all four FP dtypes.
        dx[off_dx] = (x_val == y_val) ? dy[off_dy] : T(0);
    }
}

template <typename T>
__host__ inline int32_t launch_reduce_max_min_backward(
    const T* dy, const T* x, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 s_dy  = {};
    DimsI64 s_x   = {};
    DimsI64 s_y   = {};
    DimsI64 s_dx  = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_x.v[i]   = stride_x_host[i];
        s_y.v[i]   = stride_y_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_max_min_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, x, y, dx, numel, rank, shape, s_dy, s_x, s_y, s_dx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Reduce-prod backward kernel — `dx[c] = dy[c_reduced] * y[c_reduced]
// / x[c]`. Saved-x AND saved-y, same ABI shape as Max/Min BW.
// Caller must ensure no `x[c] == 0` (PyTorch's BW also produces NaN
// in that case).

template <typename T>
__global__ void reduce_prod_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_x = 0, off_y = 0, off_dx = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dy += coord * stride_dy.v[d];
            off_x  += coord * stride_x.v[d];
            off_y  += coord * stride_y.v[d];
            off_dx += coord * stride_dx.v[d];
        }
        dx[off_dx] = dy[off_dy] * y[off_y] / x[off_x];
    }
}

template <typename T>
__host__ inline int32_t launch_reduce_prod_backward(
    const T* dy, const T* x, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 s_dy  = {};
    DimsI64 s_x   = {};
    DimsI64 s_y   = {};
    DimsI64 s_dx  = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_x.v[i]   = stride_x_host[i];
        s_y.v[i]   = stride_y_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_prod_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, x, y, dx, numel, rank, shape, s_dy, s_x, s_y, s_dx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Reduce-norm2 backward kernel — `dx[c] = dy[c_reduced] * x[c] /
// y[c_reduced]` where y = sqrt(sum(x²)). Saved-x AND saved-y, same
// ABI shape. Caller must ensure no `y[c_reduced] == 0` (only happens
// when all x in the reduced group are zero).

template <typename T>
__global__ void reduce_norm2_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_x = 0, off_y = 0, off_dx = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dy += coord * stride_dy.v[d];
            off_x  += coord * stride_x.v[d];
            off_y  += coord * stride_y.v[d];
            off_dx += coord * stride_dx.v[d];
        }
        dx[off_dx] = dy[off_dy] * x[off_x] / y[off_y];
    }
}

template <typename T>
__host__ inline int32_t launch_reduce_norm2_backward(
    const T* dy, const T* x, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 s_dy  = {};
    DimsI64 s_x   = {};
    DimsI64 s_y   = {};
    DimsI64 s_dx  = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_x.v[i]   = stride_x_host[i];
        s_y.v[i]   = stride_y_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_norm2_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, x, y, dx, numel, rank, shape, s_dy, s_x, s_y, s_dx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Reduce-LogSumExp backward kernel — `dx[c] = dy[c_reduced] *
// exp(x[c] - y[c_reduced])` where `y = lse(x) = log(sum(exp(x))) ≥
// max(x) ≥ x[c]`. Numerically safe: `x - y ∈ (-∞, 0]` so the exp
// result is in `(0, 1]` and never overflows. f16/bf16 detour through
// f32 for the exp (same pattern as the FW two-pass kernel). Saved-x
// AND saved-y, same ABI shape as Prod/Norm2 BW.

template <typename T>
struct LseBwDtype;

template <>
struct LseBwDtype<float> {
    static __device__ __forceinline__ float compute(float dy, float xv, float yv) {
        return dy * __expf(xv - yv);
    }
};

template <>
struct LseBwDtype<double> {
    static __device__ __forceinline__ double compute(double dy, double xv, double yv) {
        return dy * exp(xv - yv);
    }
};

template <>
struct LseBwDtype<__half> {
    static __device__ __forceinline__ __half compute(__half dy, __half xv, __half yv) {
        float xf = __half2float(xv);
        float yf = __half2float(yv);
        float dyf = __half2float(dy);
        float out = dyf * __expf(xf - yf);
        return __float2half(out);
    }
};

template <>
struct LseBwDtype<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 compute(
        __nv_bfloat16 dy, __nv_bfloat16 xv, __nv_bfloat16 yv)
    {
        float xf  = __bfloat162float(xv);
        float yf  = __bfloat162float(yv);
        float dyf = __bfloat162float(dy);
        float out = dyf * __expf(xf - yf);
        return __float2bfloat16(out);
    }
};

template <typename T>
__global__ void reduce_logsumexp_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_x = 0, off_y = 0, off_dx = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dy += coord * stride_dy.v[d];
            off_x  += coord * stride_x.v[d];
            off_y  += coord * stride_y.v[d];
            off_dx += coord * stride_dx.v[d];
        }
        dx[off_dx] = LseBwDtype<T>::compute(dy[off_dy], x[off_x], y[off_y]);
    }
}

template <typename T>
__host__ inline int32_t launch_reduce_logsumexp_backward(
    const T* dy, const T* x, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 s_dy  = {};
    DimsI64 s_x   = {};
    DimsI64 s_y   = {};
    DimsI64 s_dx  = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_x.v[i]   = stride_x_host[i];
        s_y.v[i]   = stride_y_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_logsumexp_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, x, y, dx, numel, rank, shape, s_dy, s_x, s_y, s_dx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Reduce-Welford (Var / Std) backward kernel — one thread per dx cell.
//
// Forward:
//   mean = sum(x) / n
//   var  = sum((x - mean)^2) / m,  where m = max(n - correction, 1)
//   std  = sqrt(var)
//
// Backward (DoSqrt = false → Var):
//   dx[c] = dy[c_reduced] * 2 * (x[c] - mean[c_reduced]) / m
// Backward (DoSqrt = true  → Std):
//   dx[c] = dy[c_reduced] * (x[c] - mean[c_reduced]) / (m * y[c_reduced])
//
// `mean[c_reduced]` is recomputed inline (one pass over the reduce
// axis on `x`) — keeps the dual-save ABI (no third saved tensor). Cost
// is `n` extra reads per output cell; acceptable for n in 16–1024.
//
// Templated on T (dtype). Internal accumulation runs in `WelfordAcc<T>`
// — `float` for f32/f16/bf16, `double` for f64. Welford's update is
// numerically delicate; staying at the wider precision is critical for
// f16/bf16 (whose final cast back to T introduces 1 ULP at store time).

// Accumulator dtype mapping for Welford — `float` for f32/half/bfloat,
// `double` for f64. Compile-time selection; no runtime branch.
template <typename T> struct WelfordAcc { using type = float; };
template <>          struct WelfordAcc<double> { using type = double; };

// Load `x[i]` (typed) as the Welford accumulator dtype.
template <typename T>
__device__ __forceinline__ typename WelfordAcc<T>::type
welford_load_as_acc(const T& v) {
    return static_cast<typename WelfordAcc<T>::type>(v);
}
template <>
__device__ __forceinline__ float welford_load_as_acc<__half>(const __half& v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float welford_load_as_acc<__nv_bfloat16>(const __nv_bfloat16& v) {
    return __bfloat162float(v);
}

// Store accumulator back to T.
template <typename T>
__device__ __forceinline__ T
welford_store_from_acc(typename WelfordAcc<T>::type acc) {
    return static_cast<T>(acc);
}
template <>
__device__ __forceinline__ __half welford_store_from_acc<__half>(float acc) {
    return __float2half(acc);
}
template <>
__device__ __forceinline__ __nv_bfloat16
welford_store_from_acc<__nv_bfloat16>(float acc) {
    return __float2bfloat16(acc);
}

// Acc-typed `sqrt` (use `sqrt` for double, `sqrtf` for float).
template <typename Acc>
__device__ __forceinline__ Acc welford_sqrt(Acc v);
template <>
__device__ __forceinline__ float welford_sqrt<float>(float v) { return sqrtf(v); }
template <>
__device__ __forceinline__ double welford_sqrt<double>(double v) { return sqrt(v); }

template <typename T, bool DoSqrt>
__global__ void reduce_welford_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x,
    int32_t correction)
{
    using Acc = typename WelfordAcc<T>::type;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    Acc denom = static_cast<Acc>(reduce_extent - correction);
    if (!(denom > (Acc)0)) denom = (Acc)1;
    Acc inv_n = (reduce_extent > 0) ? ((Acc)1 / static_cast<Acc>(reduce_extent)) : (Acc)0;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_x = 0, off_y = 0, off_dx = 0;
        // Coord of this dx cell, plus the base offset into x for the
        // reduce-axis loop (x with reduce-axis coord = 0).
        int64_t off_x_base = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dy += coord * stride_dy.v[d];
            off_x  += coord * stride_x.v[d];
            off_y  += coord * stride_y.v[d];
            off_dx += coord * stride_dx.v[d];
            if (d != reduce_axis) {
                off_x_base += coord * stride_x.v[d];
            }
        }
        // Recompute mean for this reduce group: single-pass sum / n in Acc.
        Acc sum = (Acc)0;
        for (int32_t k = 0; k < reduce_extent; ++k) {
            sum += welford_load_as_acc<T>(x[off_x_base + (int64_t)k * reduce_stride_x]);
        }
        Acc mean = sum * inv_n;
        Acc xc = welford_load_as_acc<T>(x[off_x]);
        Acc dyc = welford_load_as_acc<T>(dy[off_dy]);
        Acc diff = xc - mean;
        Acc out;
        if (DoSqrt) {
            // Std BW: dy * (x - mean) / (m * y). Caller must ensure y != 0.
            Acc yc = welford_load_as_acc<T>(y[off_y]);
            out = dyc * diff / (denom * yc);
        } else {
            // Var BW: dy * 2 * (x - mean) / m.
            out = dyc * (Acc)2 * diff / denom;
        }
        dx[off_dx] = welford_store_from_acc<T>(out);
    }
}

template <typename T, bool DoSqrt>
__host__ inline int32_t launch_reduce_welford_backward(
    const T* dy, const T* x, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x,
    int32_t correction,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (reduce_axis < 0 || reduce_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 s_dy  = {};
    DimsI64 s_x   = {};
    DimsI64 s_y   = {};
    DimsI64 s_dx  = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_x.v[i]   = stride_x_host[i];
        s_y.v[i]   = stride_y_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_welford_backward_kernel<T, DoSqrt><<<blocks, kBlock, 0, stream>>>(
        dy, x, y, dx, numel, rank, shape, s_dy, s_x, s_y, s_dx,
        reduce_axis, reduce_extent, reduce_stride_x, correction);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Repeat (tile) kernel — Category N (per-axis replication; output >
// input)
// =============================================================================
//
// `y = repeat(x, repeats)` where `output[d] = input.shape[d] *
// repeats[d]`. For each output coord c, input coord
// `c'[d] = c[d] % input.shape[d]` (modular wrap). Same kernel pattern
// as Flip/Roll but with output shape larger than input.

template <typename T>
__global__ void repeat_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 input_shape,
    DimsI32 output_shape,
    DimsI64 stride_x,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s_out = output_shape.v[d];
            int64_t c = (s_out == 0) ? 0 : (linear % (int64_t)s_out);
            if (s_out != 0) linear /= (int64_t)s_out;
            off_y += c * stride_y.v[d];
            int32_t s_in = input_shape.v[d];
            int64_t in_c = (s_in == 0) ? 0 : (c % (int64_t)s_in);
            off_x += in_c * stride_x.v[d];
        }
        y[off_y] = x[off_x];
    }
}

template <typename T>
__host__ inline int32_t launch_repeat(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* input_shape_host,
    const int32_t* output_shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 in_shape = {}, out_shape = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i]  = input_shape_host[i];
        out_shape.v[i] = output_shape_host[i];
        sx.v[i]        = stride_x_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    repeat_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, in_shape, out_shape, sx, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Welford reduction kernels — Phase 4 (Var / Std via one-pass online
// algorithm)
// =============================================================================
//
// Welford's algorithm for numerically-stable variance in one pass:
//   M2 = 0; mean = 0;
//   for k in 0..n:
//       delta  = x[k] - mean;
//       mean  += delta / (k + 1);
//       delta2 = x[k] - mean;
//       M2    += delta * delta2;
//   variance = M2 / (n - correction);   // correction=1 for sample, 0 for population
//
// `T` is the value dtype. Internal Welford state runs in `WelfordAcc<T>`
// — `float` for f32/f16/bf16, `double` for f64. The result is cast
// back to T at store time. `DoSqrt` template parameter chooses var
// (false) vs std (true). The `WelfordAcc<T>` / `welford_load_as_acc<T>`
// / `welford_store_from_acc<T>` helpers are defined near the Welford
// BW kernel above.

template <typename T, bool DoSqrt>
__global__ void reduce_welford_axis_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 output_shape,
    DimsI64 stride_x,
    DimsI64 stride_y,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x,
    int32_t correction)
{
    using Acc = typename WelfordAcc<T>::type;
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
        Acc mean = (Acc)0;
        Acc m2 = (Acc)0;
        for (int32_t k = 0; k < reduce_extent; ++k) {
            Acc v = welford_load_as_acc<T>(x[off_x_base + (int64_t)k * reduce_stride_x]);
            Acc delta = v - mean;
            mean += delta / static_cast<Acc>(k + 1);
            Acc delta2 = v - mean;
            m2 += delta * delta2;
        }
        Acc denom = static_cast<Acc>(reduce_extent - correction);
        Acc variance = (denom > (Acc)0) ? (m2 / denom) : (Acc)0;
        Acc result = DoSqrt ? welford_sqrt<Acc>(variance) : variance;
        y[off_y] = welford_store_from_acc<T>(result);
    }
}

template <typename T, bool DoSqrt>
__host__ inline int32_t launch_reduce_welford_axis(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* output_shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x,
    int32_t correction,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (reduce_axis < 0 || reduce_axis >= rank) return 2;
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
    reduce_welford_axis_kernel<T, DoSqrt><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, out_shape, sx, sy,
        reduce_axis, reduce_extent, reduce_stride_x, correction);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Flip kernel — Category N (reverse along selected axes)
// =============================================================================
//
// `y = flip(x, dims)`. Output shape == input shape. For each output
// coord c, the input coord is `c'[d] = (shape[d] - 1 - c[d])` if
// `flip_axes[d] != 0`, else `c[d]`. `flip_axes` is a per-axis mask
// (0/1) stored as DimsI32.

template <typename T>
__global__ void flip_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI32 flip_axes,
    DimsI64 stride_x,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_x = 0;
        int64_t off_y = 0;
        // Unravel output coord and accumulate both offsets.
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += c * stride_y.v[d];
            int64_t in_c = (flip_axes.v[d] != 0) ? ((int64_t)s - 1 - c) : c;
            off_x += in_c * stride_x.v[d];
        }
        y[off_y] = x[off_x];
    }
}

template <typename T>
__host__ inline int32_t launch_flip(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int32_t* flip_axes_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {}, flip_axes = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i]     = shape_host[i];
        flip_axes.v[i] = flip_axes_host[i];
        sx.v[i]        = stride_x_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    flip_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, rank, shape, flip_axes, sx, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Roll kernel — Category N (cyclic shift along axes)
// =============================================================================
//
// `y = roll(x, shifts)`. Output shape == input shape. For each output
// coord c, the input coord is `c'[d] = ((c[d] - shifts[d]) mod
// shape[d])`. Negative or large shifts are normalized via `((c - s)
// % len + len) % len` arithmetic (defensive against C's signed-mod
// behavior on negatives).

template <typename T>
__global__ void roll_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI32 shifts,
    DimsI64 stride_x,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_x = 0;
        int64_t off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += c * stride_y.v[d];
            int64_t in_c = c;
            if (s != 0) {
                int64_t sh = (int64_t)shifts.v[d];
                // Normalize: ((c - sh) % s + s) % s for any sign of sh.
                int64_t mod_s = (int64_t)s;
                int64_t raw = c - sh;
                int64_t m = raw % mod_s;
                if (m < 0) m += mod_s;
                in_c = m;
            }
            off_x += in_c * stride_x.v[d];
        }
        y[off_y] = x[off_x];
    }
}

template <typename T>
__host__ inline int32_t launch_roll(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int32_t* shifts_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {}, shifts = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i]  = shape_host[i];
        shifts.v[i] = shifts_host[i];
        sx.v[i]     = stride_x_host[i];
        sy.v[i]     = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    roll_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, rank, shape, shifts, sx, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Permute kernel — Category N (axis permutation, materialized)
// =============================================================================
//
// `y = x.permute(dims)` where `dims[d]` is the input axis index that
// becomes output axis d. Output shape: `output[d] = input[dims[d]]`.
//
// Iterates INPUT cells (one thread per input cell). For each input
// linear index, unravels to input coord c, then writes to the output
// using `output_coord[d] = c[dims[d]]`. This avoids needing the
// inverse permutation.
//
// Used when the caller needs a materialized contiguous output (the
// strided-view-only path is the same data but with reshuffled strides;
// callers that can consume the strided view don't need this kernel).

template <typename T>
__global__ void permute_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t input_numel,
    int32_t rank,
    DimsI32 input_shape,
    DimsI32 dims,
    DimsI64 stride_x,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < input_numel; i += step) {
        int64_t linear = i;
        int64_t coord[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int64_t off_x = 0;
        // Unravel input coord (innermost dim first).
        for (int k = rank - 1; k >= 0; --k) {
            int32_t s = input_shape.v[k];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[k] = c;
            off_x += c * stride_x.v[k];
        }
        // Compute output offset: for each output axis d, the source
        // coord is `coord[dims[d]]` (which input axis became this
        // output axis).
        int64_t off_y = 0;
        for (int d = 0; d < rank; ++d) {
            int32_t in_axis = dims.v[d];
            off_y += coord[in_axis] * stride_y.v[d];
        }
        y[off_y] = x[off_x];
    }
}

template <typename T>
__host__ inline int32_t launch_permute(
    const T* x, T* y,
    int64_t input_numel,
    int32_t rank,
    const int32_t* input_shape_host,
    const int32_t* dims_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 in_shape = {}, dims = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i] = input_shape_host[i];
        dims.v[i]     = dims_host[i];
        sx.v[i]       = stride_x_host[i];
        sy.v[i]       = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (input_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    permute_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, input_numel, rank, in_shape, dims, sx, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Concat (2-input) kernel — Category N (output shape > input shape,
// variable axis)
// =============================================================================
//
// `y = cat(a, b, dim=k)` — output shape per-axis: matches a/b except
// `output[k] = a.shape[k] + b.shape[k]`. For each output cell, branch
// on the concat-axis coord:
//   if coord[k] < a.shape[k] → copy from a at the same coord
//   else                     → copy from b at coord with k adjusted
//                              (`coord[k] -= a.shape[k]`)
//
// Trailblazer is 2-input only — N-input variable-arity follows in a
// future session (would need device-side packing of N pointers + N
// stride arrays through kernel-param-by-value, separate plan shape).

template <typename T>
__global__ void concat2_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 output_shape,
    int32_t concat_dim,
    int32_t split_offset,  // a.shape[concat_dim]
    DimsI64 stride_a,
    DimsI64 stride_b,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t coords[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int64_t off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = output_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coords[d] = c;
            off_y += c * stride_y.v[d];
        }
        bool from_a = coords[concat_dim] < (int64_t)split_offset;
        int64_t adj_coord = from_a ? coords[concat_dim]
                                   : (coords[concat_dim] - (int64_t)split_offset);
        int64_t off_in = 0;
        if (from_a) {
            for (int d = 0; d < rank; ++d) {
                int64_t cc = (d == concat_dim) ? adj_coord : coords[d];
                off_in += cc * stride_a.v[d];
            }
            y[off_y] = a[off_in];
        } else {
            for (int d = 0; d < rank; ++d) {
                int64_t cc = (d == concat_dim) ? adj_coord : coords[d];
                off_in += cc * stride_b.v[d];
            }
            y[off_y] = b[off_in];
        }
    }
}

template <typename T>
__host__ inline int32_t launch_concat2(
    const T* a, const T* b, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* output_shape_host,
    int32_t concat_dim,
    int32_t split_offset,
    const int64_t* stride_a_host,
    const int64_t* stride_b_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (concat_dim < 0 || concat_dim >= rank) return 2;
    DimsI32 out_shape = {};
    DimsI64 sa = {}, sb = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        out_shape.v[i] = output_shape_host[i];
        sa.v[i]        = stride_a_host[i];
        sb.v[i]        = stride_b_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    concat2_kernel<T><<<blocks, kBlock, 0, stream>>>(
        a, b, y, output_numel, rank, out_shape,
        concat_dim, split_offset, sa, sb, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Pad kernel — Category N (output shape > input shape, constant mode)
// =============================================================================
//
// `y[c] = x[c - pad_low]` if `c - pad_low ∈ [0, input_shape)` per-axis,
// else `value`. Output shape is derived per-axis as
// `output[d] = input[d] + pad_low[d] + pad_high[d]`. Only constant
// mode is wired today (other modes change the pad-region branch).
//
// The kernel iterates output cells linearly. For each output cell:
//   1. Unravel linear index `i` into output coord c[d].
//   2. Compute input coord c_in[d] = c[d] - pad_low[d].
//   3. If c_in[d] ∈ [0, input_shape[d]) for all d, copy x[c_in]; else
//      write `value`.
// Output is conventionally contiguous; input can be strided.

template <typename T>
__global__ void pad_constant_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 input_shape,
    DimsI32 output_shape,
    DimsI32 pad_low,
    DimsI64 stride_x,
    DimsI64 stride_y,
    T value)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x = 0;
        bool in_input = true;
        // Unravel from innermost (rightmost) dim.
        for (int d = rank - 1; d >= 0; --d) {
            int32_t out_s = output_shape.v[d];
            int64_t coord = (out_s == 0) ? 0 : (linear % (int64_t)out_s);
            if (out_s != 0) linear /= (int64_t)out_s;
            off_y += coord * stride_y.v[d];
            int64_t in_coord = coord - (int64_t)pad_low.v[d];
            if (in_coord < 0 || in_coord >= (int64_t)input_shape.v[d]) {
                in_input = false;
            }
            off_x += in_coord * stride_x.v[d];
        }
        y[off_y] = in_input ? x[off_x] : value;
    }
}

template <typename T>
__host__ inline int32_t launch_pad_constant(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* input_shape_host,
    const int32_t* output_shape_host,
    const int32_t* pad_low_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    T value,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 in_shape = {}, out_shape = {}, plow = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i]  = input_shape_host[i];
        out_shape.v[i] = output_shape_host[i];
        plow.v[i]      = pad_low_host[i];
        sx.v[i]        = stride_x_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    pad_constant_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, in_shape, out_shape, plow, sx, sy, value);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// -----------------------------------------------------------------------------
// Pad — Reflect / Replicate / Circular modes
// -----------------------------------------------------------------------------
//
// Sibling kernels of `pad_constant_kernel` that share the iteration
// shape (one thread per output cell, unravel + read from input) but
// differ in how a pad-region coord (`in_coord` outside `[0, extent)`)
// is mapped back into the valid input range. None of these modes have
// a constant-value parameter — every output cell reads from the
// input.
//
//   Reflect:   mirror the input across each boundary (no edge dup).
//                c_in < 0          → -c_in
//                c_in >= extent    → 2*extent - 2 - c_in
//                extent == 1       → 0 (degenerate)
//   Replicate: clamp to the edge.
//                c_in = max(0, min(extent - 1, c_in))
//   Circular:  cyclic wrap (defensive double-mod for negatives).
//                c_in = ((c_in % extent) + extent) % extent
//
// All three accept arbitrary strided inputs / outputs.

template <typename T>
__global__ void pad_reflect_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 input_shape,
    DimsI32 output_shape,
    DimsI32 pad_low,
    DimsI64 stride_x,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t out_s = output_shape.v[d];
            int64_t coord = (out_s == 0) ? 0 : (linear % (int64_t)out_s);
            if (out_s != 0) linear /= (int64_t)out_s;
            off_y += coord * stride_y.v[d];
            int64_t in_coord = coord - (int64_t)pad_low.v[d];
            int64_t extent = (int64_t)input_shape.v[d];
            if (extent <= 1) {
                in_coord = 0;
            } else if (in_coord < 0) {
                in_coord = -in_coord;
            } else if (in_coord >= extent) {
                in_coord = 2 * extent - 2 - in_coord;
            }
            // Defensive: clamp into [0, extent) in case caller padded
            // beyond the first reflection.
            if (in_coord < 0) in_coord = 0;
            if (in_coord >= extent) in_coord = extent - 1;
            off_x += in_coord * stride_x.v[d];
        }
        y[off_y] = x[off_x];
    }
}

template <typename T>
__global__ void pad_replicate_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 input_shape,
    DimsI32 output_shape,
    DimsI32 pad_low,
    DimsI64 stride_x,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t out_s = output_shape.v[d];
            int64_t coord = (out_s == 0) ? 0 : (linear % (int64_t)out_s);
            if (out_s != 0) linear /= (int64_t)out_s;
            off_y += coord * stride_y.v[d];
            int64_t in_coord = coord - (int64_t)pad_low.v[d];
            int64_t extent = (int64_t)input_shape.v[d];
            if (extent <= 0) {
                in_coord = 0;
            } else {
                if (in_coord < 0) in_coord = 0;
                if (in_coord > extent - 1) in_coord = extent - 1;
            }
            off_x += in_coord * stride_x.v[d];
        }
        y[off_y] = x[off_x];
    }
}

template <typename T>
__global__ void pad_circular_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 input_shape,
    DimsI32 output_shape,
    DimsI32 pad_low,
    DimsI64 stride_x,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t out_s = output_shape.v[d];
            int64_t coord = (out_s == 0) ? 0 : (linear % (int64_t)out_s);
            if (out_s != 0) linear /= (int64_t)out_s;
            off_y += coord * stride_y.v[d];
            int64_t in_coord = coord - (int64_t)pad_low.v[d];
            int64_t extent = (int64_t)input_shape.v[d];
            if (extent <= 0) {
                in_coord = 0;
            } else {
                // Defensive double-mod: C's `%` is sign-of-dividend.
                int64_t r = in_coord % extent;
                if (r < 0) r += extent;
                in_coord = r;
            }
            off_x += in_coord * stride_x.v[d];
        }
        y[off_y] = x[off_x];
    }
}

template <typename T>
__host__ inline int32_t launch_pad_reflect(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* input_shape_host,
    const int32_t* output_shape_host,
    const int32_t* pad_low_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 in_shape = {}, out_shape = {}, plow = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i]  = input_shape_host[i];
        out_shape.v[i] = output_shape_host[i];
        plow.v[i]      = pad_low_host[i];
        sx.v[i]        = stride_x_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    pad_reflect_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, in_shape, out_shape, plow, sx, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_pad_replicate(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* input_shape_host,
    const int32_t* output_shape_host,
    const int32_t* pad_low_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 in_shape = {}, out_shape = {}, plow = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i]  = input_shape_host[i];
        out_shape.v[i] = output_shape_host[i];
        plow.v[i]      = pad_low_host[i];
        sx.v[i]        = stride_x_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    pad_replicate_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, in_shape, out_shape, plow, sx, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_pad_circular(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* input_shape_host,
    const int32_t* output_shape_host,
    const int32_t* pad_low_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 in_shape = {}, out_shape = {}, plow = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i]  = input_shape_host[i];
        out_shape.v[i] = output_shape_host[i];
        plow.v[i]      = pad_low_host[i];
        sx.v[i]        = stride_x_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    pad_circular_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, in_shape, out_shape, plow, sx, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Pad constant — backward (slice)
// =============================================================================
//
// `dx = dy[pad_low : pad_low + input_shape]` per axis. The forward
// constant-pad is `y = pad(x, pad_low, pad_high, value)`; its backward
// discards the gradient at pad-region cells and copies the gradient
// from the central (in-bounds) region back to `dx`. Since the
// pad-region cells in `y` were a constant (the `value` argument),
// their gradient w.r.t. the input `x` is identically zero — we never
// touch those cells of `dy`.
//
// One thread per `dx` cell (input_numel total). For each output coord
// `c`, the source `dy` coord is `c + pad_low` per axis — always
// in-bounds because `0 <= pad_low[d]` and
// `input_shape[d] + pad_low[d] <= output_shape[d]`.

template <typename T>
__global__ void pad_constant_backward_kernel(
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int64_t input_numel,
    int32_t rank,
    DimsI32 input_shape,
    DimsI32 pad_low,
    DimsI64 stride_dy,
    DimsI64 stride_dx)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < input_numel; i += step) {
        int64_t linear = i;
        int64_t off_dx = 0;
        int64_t off_dy = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t in_s = input_shape.v[d];
            int64_t coord = (in_s == 0) ? 0 : (linear % (int64_t)in_s);
            if (in_s != 0) linear /= (int64_t)in_s;
            off_dx += coord * stride_dx.v[d];
            off_dy += (coord + (int64_t)pad_low.v[d]) * stride_dy.v[d];
        }
        dx[off_dx] = dy[off_dy];
    }
}

template <typename T>
__host__ inline int32_t launch_pad_constant_backward(
    const T* dy, T* dx,
    int64_t input_numel,
    int32_t rank,
    const int32_t* input_shape_host,
    const int32_t* pad_low_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_dx_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 in_shape = {}, plow = {};
    DimsI64 sdy = {}, sdx = {};
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i] = input_shape_host[i];
        plow.v[i]     = pad_low_host[i];
        sdy.v[i]      = stride_dy_host[i];
        sdx.v[i]      = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (input_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    pad_constant_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, dx, input_numel, rank, in_shape, plow, sdy, sdx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Repeat backward kernel — Category N (Phase 3 BW for `torch.repeat`)
// =============================================================================
//
// Forward: `y[c_out] = x[c_out[d] % input_shape[d]]` per axis. BW is the
// gather-adjoint: `dx[c_in] = sum_{k} dy[c_in + k * input_shape]` where
// each `k_d` ranges in `[0, repeats[d])`. One thread per dx cell; each
// thread loops over the per-axis repeats grid (product of `repeats[d]`)
// and accumulates `dy` values into a single store. f16 / bf16 accumulate
// in `float` for stability (mirrors the f32-detour pattern used by the
// reduce-sum functor); f32 / f64 accumulate in their own dtype.

// Accumulator dtype mapping — `float` for half / bfloat16, `T` for the
// native FP types. Only template-detected at compile time; no runtime
// branch.
template <typename T> struct RepeatBwAcc        { using type = T; };
template <>          struct RepeatBwAcc<__half> { using type = float; };
template <>          struct RepeatBwAcc<__nv_bfloat16> { using type = float; };

template <typename T>
__device__ __forceinline__ typename RepeatBwAcc<T>::type
repeat_bw_load_as_acc(const T& v) { return static_cast<typename RepeatBwAcc<T>::type>(v); }
template <>
__device__ __forceinline__ float repeat_bw_load_as_acc<__half>(const __half& v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float repeat_bw_load_as_acc<__nv_bfloat16>(const __nv_bfloat16& v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T
repeat_bw_store_from_acc(typename RepeatBwAcc<T>::type acc) { return static_cast<T>(acc); }
template <>
__device__ __forceinline__ __half repeat_bw_store_from_acc<__half>(float acc) {
    return __float2half(acc);
}
template <>
__device__ __forceinline__ __nv_bfloat16
repeat_bw_store_from_acc<__nv_bfloat16>(float acc) {
    return __float2bfloat16(acc);
}

template <typename T>
__global__ void repeat_backward_kernel(
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int64_t input_numel,
    int32_t rank,
    DimsI32 input_shape,
    DimsI32 repeats,
    DimsI64 stride_dy,
    DimsI64 stride_dx)
{
    using AccT = typename RepeatBwAcc<T>::type;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < input_numel; i += step) {
        // Unravel dx-coord (in_coord) and compute the base dy offset
        // corresponding to `k = 0` along every axis. Also compute the
        // total number of dy cells contributing to this dx cell as the
        // product of `repeats[d]`, plus the per-axis "block stride" of
        // `dy` for stepping by one tile along axis d
        // (= input_shape[d] * stride_dy[d]).
        int64_t linear = i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0;
        int32_t in_coord[MAX_RANK];
        int64_t tile_stride_dy[MAX_RANK];
        int32_t rep[MAX_RANK];
        for (int d = rank - 1; d >= 0; --d) {
            int32_t in_s = input_shape.v[d];
            int64_t coord = (in_s == 0) ? 0 : (linear % (int64_t)in_s);
            if (in_s != 0) linear /= (int64_t)in_s;
            in_coord[d] = (int32_t)coord;
            off_dx += coord * stride_dx.v[d];
            off_dy_base += coord * stride_dy.v[d];
            tile_stride_dy[d] = (int64_t)in_s * stride_dy.v[d];
            rep[d] = repeats.v[d];
        }
        // Walk the repeats grid (product over axes of `repeats[d]`)
        // using a multi-index k[0..rank). Linearize total = prod(rep[d]).
        int64_t total = 1;
        for (int d = 0; d < rank; ++d) total *= (int64_t)rep[d];

        AccT acc = AccT(0);
        for (int64_t t = 0; t < total; ++t) {
            int64_t rem = t;
            int64_t off_dy = off_dy_base;
            // Unravel `t` in row-major order across the rep grid and add
            // each axis' offset contribution.
            for (int d = rank - 1; d >= 0; --d) {
                int32_t r = rep[d];
                int64_t k = (r == 0) ? 0 : (rem % (int64_t)r);
                if (r != 0) rem /= (int64_t)r;
                off_dy += k * tile_stride_dy[d];
            }
            acc += repeat_bw_load_as_acc<T>(dy[off_dy]);
        }
        dx[off_dx] = repeat_bw_store_from_acc<T>(acc);
    }
}

template <typename T>
__host__ inline int32_t launch_repeat_backward(
    const T* dy, T* dx,
    int64_t input_numel,
    int32_t rank,
    const int32_t* input_shape_host,
    const int32_t* repeats_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_dx_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 in_shape = {}, reps = {};
    DimsI64 sdy = {}, sdx = {};
    for (int i = 0; i < rank; ++i) {
        in_shape.v[i] = input_shape_host[i];
        reps.v[i]     = repeats_host[i];
        sdy.v[i]      = stride_dy_host[i];
        sdx.v[i]      = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (input_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    repeat_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, dx, input_numel, rank, in_shape, reps, sdy, sdx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Concat2 backward kernel — Category N (Phase 3 BW for 2-input concat)
// =============================================================================
//
// Forward: `y = cat(a, b, dim=k)` — output shape per-axis matches a/b
// except `output[k] = a.shape[k] + b.shape[k]`. Backward is the pure
// inverse routing: every `dy` cell maps to exactly one of `da` or `db`.
//
//   if coord[k] < split_offset → da at coord
//   else                       → db at coord with k adjusted (`-= split_offset`)
//
// One thread per output cell (output_numel = dy.numel() =
// da.numel() + db.numel()). Bit-exact across every wired dtype — pure
// element copy, no arithmetic.
//
// Mirror of `concat2_kernel` with the pointer flow reversed (one input
// becomes two outputs).
template <typename T>
__global__ void concat2_backward_kernel(
    const T* __restrict__ dy,
    T* __restrict__ da,
    T* __restrict__ db,
    int64_t output_numel,
    int32_t rank,
    DimsI32 output_shape,
    int32_t concat_dim,
    int32_t split_offset,
    DimsI64 stride_dy,
    DimsI64 stride_da,
    DimsI64 stride_db)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t coords[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int64_t off_dy = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = output_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coords[d] = c;
            off_dy += c * stride_dy.v[d];
        }
        bool to_a = coords[concat_dim] < (int64_t)split_offset;
        int64_t adj_coord = to_a ? coords[concat_dim]
                                 : (coords[concat_dim] - (int64_t)split_offset);
        int64_t off_out = 0;
        if (to_a) {
            for (int d = 0; d < rank; ++d) {
                int64_t cc = (d == concat_dim) ? adj_coord : coords[d];
                off_out += cc * stride_da.v[d];
            }
            da[off_out] = dy[off_dy];
        } else {
            for (int d = 0; d < rank; ++d) {
                int64_t cc = (d == concat_dim) ? adj_coord : coords[d];
                off_out += cc * stride_db.v[d];
            }
            db[off_out] = dy[off_dy];
        }
    }
}

template <typename T>
__host__ inline int32_t launch_concat2_backward(
    const T* dy, T* da, T* db,
    int64_t output_numel,
    int32_t rank,
    const int32_t* output_shape_host,
    int32_t concat_dim,
    int32_t split_offset,
    const int64_t* stride_dy_host,
    const int64_t* stride_da_host,
    const int64_t* stride_db_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (concat_dim < 0 || concat_dim >= rank) return 2;
    DimsI32 out_shape = {};
    DimsI64 sdy = {}, sda = {}, sdb = {};
    for (int i = 0; i < rank; ++i) {
        out_shape.v[i] = output_shape_host[i];
        sdy.v[i]       = stride_dy_host[i];
        sda.v[i]       = stride_da_host[i];
        sdb.v[i]       = stride_db_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    concat2_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, da, db, output_numel, rank, out_shape,
        concat_dim, split_offset, sdy, sda, sdb);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Argmax / Argmin kernels — Phase 4 (axis reduction, i64 index output)
// =============================================================================
//
// `y = argmax(x, axis=k)` — index of the max along axis k. Output dtype
// is `int64_t` (PyTorch convention for indices). Output shape == input
// shape with `[reduce_axis]` collapsed to size 1 (keepdim convention).
//
// The kernel iterates output cells; for each, walks the reduce axis
// tracking the best (value, index) pair. Ties are broken by FIRST
// occurrence (smaller index wins on equality) — matches PyTorch.
//
// `F` is an arg-reduce policy with a static `bool prefer(new_v,
// new_i, best_v, best_i)` predicate that returns true when the new
// candidate should replace the current best.

// Phase 12.2: generalized in the output dtype `OutI` (u32 / i32 / i64).
// Internal best-index tracking stays `int64_t` (max range is the reduce
// axis extent — i64 is safe); only the final store narrows to `OutI`.
template <typename T, typename F, typename OutI>
__global__ void arg_reduce_axis_kernel(
    const T* __restrict__ x,
    OutI* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 output_shape,
    DimsI64 stride_x,
    DimsI64 stride_y,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x)
{
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
        // Initialize with k=0.
        T best_v = x[off_x_base];
        int64_t best_i = 0;
        F policy{};
        for (int32_t k = 1; k < reduce_extent; ++k) {
            T v = x[off_x_base + (int64_t)k * reduce_stride_x];
            if (policy.prefer(v, (int64_t)k, best_v, best_i)) {
                best_v = v;
                best_i = (int64_t)k;
            }
        }
        y[off_y] = static_cast<OutI>(best_i);
    }
}

template <typename T, typename F, typename OutI>
__host__ inline int32_t launch_arg_reduce_axis(
    const T* x, OutI* y,
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
    if (reduce_extent <= 0) return 2; // can't argmax over empty axis
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
    arg_reduce_axis_kernel<T, F, OutI><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, out_shape, sx, sy,
        reduce_axis, reduce_extent, reduce_stride_x);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Reduction kernels — Phase 4 (axis reduction; output shape = input
// shape with reduced axis collapsed to 1)
// =============================================================================
//
// Single-axis reduction trailblazer: each output cell sums (or otherwise
// reduces) a slice of `reduce_extent` input values along the reduced
// axis. One thread per output cell — naive but correct; tile/warp
// optimization is Phase 4 follow-up.
//
// `F` is a binary reduce functor `T operator()(T acc, T x) const` plus
// a static `T init()` and a static `T finalize(T acc, int32_t extent)`
// step (Sum / Max / Min / Prod are pass-through; Mean uses it to divide
// by the reduce extent). Examples:
//   SumReduce:  { init = 0;   op(a, x) = a + x;       finalize = acc }
//   MeanReduce: { init = 0;   op(a, x) = a + x;       finalize = acc/extent }
//   ProdReduce: { init = 1;   op(a, x) = a * x;       finalize = acc }
//   MaxReduce:  { init = -∞;  op(a, x) = max(a, x);   finalize = acc }

template <typename T, typename F>
__global__ void reduce_axis_kernel(
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
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        // Unravel output coord. The reduced axis is size 1 in output,
        // so coord on that axis is always 0 — we walk it ourselves
        // along `reduce_axis` using `reduce_stride_x`.
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = output_shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            // For the reduced axis, output extent is 1 and coord is 0;
            // off_x_base contribution is 0 anyway.
            if (d != reduce_axis) {
                off_x_base += coord * stride_x.v[d];
            }
        }
        T acc = F::init();
        F op{};
        for (int32_t k = 0; k < reduce_extent; ++k) {
            int64_t off_x = off_x_base + (int64_t)k * reduce_stride_x;
            acc = op(acc, x[off_x]);
        }
        y[off_y] = F::finalize(acc, reduce_extent);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_reduce_axis(
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
    reduce_axis_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, out_shape, sx, sy,
        reduce_axis, reduce_extent, reduce_stride_x);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Heterogeneous-output-dtype reduction kernel — Phase 4 deferral 4.4
// =============================================================================
//
// Same loop shape as `reduce_axis_kernel`, but the output element type
// is independent of the input element type. Used by `Any` / `All`
// (output: `uint8_t` Bool) and `count_nonzero` (output: `int64_t`).
//
// Functor `F` shape:
//   static __device__ T_out init();
//   __device__ T_out operator()(T_out acc, T_in x) const;
//
// No `finalize()` step (Any / All / CountNonzero are pass-through).

template <typename T_in, typename T_out, typename F>
__global__ void reduce_axis_hetero_kernel(
    const T_in* __restrict__ x,
    T_out* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 output_shape,
    DimsI64 stride_x,
    DimsI64 stride_y,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x)
{
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
        T_out acc = F::init();
        F op{};
        for (int32_t k = 0; k < reduce_extent; ++k) {
            int64_t off_x = off_x_base + (int64_t)k * reduce_stride_x;
            acc = op(acc, x[off_x]);
        }
        y[off_y] = acc;
    }
}

template <typename T_in, typename T_out, typename F>
__host__ inline int32_t launch_reduce_axis_hetero(
    const T_in* x, T_out* y,
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
    reduce_axis_hetero_kernel<T_in, T_out, F><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, out_shape, sx, sy,
        reduce_axis, reduce_extent, reduce_stride_x);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Scan kernel — Phase 4 (Category F: associative prefix scans)
// =============================================================================
//
// `y = scan(x, dim=k, reverse)` — output shape == input shape (no axis
// collapse). For each output cell at coord c with `c[scan_axis] = k`,
// accumulate via the functor `F` over `x[..., 0..=k, ...]` (forward) or
// `x[..., k..extent, ...]` (reverse).
//
// Trailblazer is naive — one thread per output cell, O(extent) work
// per cell. Total work O(numel · extent), suboptimal vs a parallel-scan
// (Blelloch / Hillis-Steele) but trivially correct and fits the same
// dispatch shape as the reduce family. Replacement with parallel-scan
// for large extents is a future optimization.
//
// Functor `F` shape mirrors the reduce functor: provide
// `static T init()`, `T operator()(T acc, T x)`, and
// `static T finalize(T acc, int32_t extent)` (pass-through for
// Cumsum / Cumprod / Cummax / Cummin; not used yet but kept for ABI
// parity with the reduce family).

template <typename T, typename F>
__global__ void scan_axis_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_x,
    DimsI64 stride_y,
    int32_t scan_axis,
    int32_t scan_extent,
    int64_t scan_stride_x,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != scan_axis) {
                off_x_base += coord * stride_x.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        T acc = F::init();
        F op{};
        if (reverse != 0) {
            for (int32_t j = scan_extent - 1; j >= k; --j) {
                int64_t off_x = off_x_base + (int64_t)j * scan_stride_x;
                acc = op(acc, x[off_x]);
            }
        } else {
            for (int32_t j = 0; j <= k; ++j) {
                int64_t off_x = off_x_base + (int64_t)j * scan_stride_x;
                acc = op(acc, x[off_x]);
            }
        }
        y[off_y] = F::finalize(acc, scan_extent);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_scan_axis(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t scan_axis,
    int32_t scan_extent,
    int64_t scan_stride_x,
    int32_t reverse,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (scan_axis < 0 || scan_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    scan_axis_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, rank, shape, sx, sy,
        scan_axis, scan_extent, scan_stride_x, reverse);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Scan BW kernels — Phase 4 (Category F)
// =============================================================================
//
// Cumprod / Cummax / Cummin BWs cannot reuse the FW scan kernel: the
// gradient formulas reference the FW input `x` (Cumprod, Cummax, Cummin)
// and the FW output `y` (Cumprod only). One thread per dx cell at coord
// `c` with `c[scan_axis] = j`. Each thread does an O(extent) walk along
// the scan axis.
//
// Cumprod BW: `dx[j] = Σ_{i in suffix} dy[i] * y[i] / x[j]` where the
// "suffix" is `{i ≥ j}` for forward FW and `{i ≤ j}` for reverse FW.
// Caller must ensure `x[j] != 0`.
//
// Cummax / Cummin BW: walk the full forward scan (from 0 to extent-1
// for forward, extent-1 to 0 for reverse) maintaining a running
// max/min value and the *first-occurrence* argmax/argmin. Each step
// `i`, dy[i] flows to dx[running_argmax_or_argmin] — so a thread at
// position `j` accumulates dy[i] only when the running winner equals
// `j`. Ties broken by first occurrence (PyTorch convention).

template <typename T, typename Acc>
__global__ void scan_cumprod_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0, off_y_base = 0;
        int32_t j = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                j = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        int64_t stride_y_axis  = stride_y.v[scan_axis];
        // Read x[j] for the per-thread divisor.
        T x_j = x[off_x_base + (int64_t)j * stride_x_axis];
        Acc inv_x_j = Acc(1) / static_cast<Acc>(x_j);
        Acc acc = Acc(0);
        // Suffix over i: forward FW => i in [j, extent); reverse FW => i in [0, j].
        int32_t i_lo, i_hi;
        if (reverse != 0) {
            i_lo = 0;
            i_hi = j;
        } else {
            i_lo = j;
            i_hi = scan_extent - 1;
        }
        for (int32_t ii = i_lo; ii <= i_hi; ++ii) {
            T dy_i = dy[off_dy_base + (int64_t)ii * stride_dy_axis];
            T y_i  = y [off_y_base  + (int64_t)ii * stride_y_axis];
            acc += static_cast<Acc>(dy_i) * static_cast<Acc>(y_i) * inv_x_j;
        }
        dx[off_dx] = static_cast<T>(acc);
    }
}

// f16 specialization — accumulator is float, but the multiply-and-store
// goes through __half2float / __float2half.
template <>
__global__ void scan_cumprod_backward_kernel<__half, float>(
    const __half* __restrict__ dy,
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    __half* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0, off_y_base = 0;
        int32_t j = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                j = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        int64_t stride_y_axis  = stride_y.v[scan_axis];
        float x_j = __half2float(x[off_x_base + (int64_t)j * stride_x_axis]);
        float inv_x_j = 1.0f / x_j;
        float acc = 0.0f;
        int32_t i_lo, i_hi;
        if (reverse != 0) { i_lo = 0; i_hi = j; }
        else              { i_lo = j; i_hi = scan_extent - 1; }
        for (int32_t ii = i_lo; ii <= i_hi; ++ii) {
            float dy_i = __half2float(dy[off_dy_base + (int64_t)ii * stride_dy_axis]);
            float y_i  = __half2float(y [off_y_base  + (int64_t)ii * stride_y_axis]);
            acc += dy_i * y_i * inv_x_j;
        }
        dx[off_dx] = __float2half(acc);
    }
}

// bf16 specialization — same f32-detour pattern.
template <>
__global__ void scan_cumprod_backward_kernel<__nv_bfloat16, float>(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0, off_y_base = 0;
        int32_t j = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                j = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        int64_t stride_y_axis  = stride_y.v[scan_axis];
        float x_j = __bfloat162float(x[off_x_base + (int64_t)j * stride_x_axis]);
        float inv_x_j = 1.0f / x_j;
        float acc = 0.0f;
        int32_t i_lo, i_hi;
        if (reverse != 0) { i_lo = 0; i_hi = j; }
        else              { i_lo = j; i_hi = scan_extent - 1; }
        for (int32_t ii = i_lo; ii <= i_hi; ++ii) {
            float dy_i = __bfloat162float(dy[off_dy_base + (int64_t)ii * stride_dy_axis]);
            float y_i  = __bfloat162float(y [off_y_base  + (int64_t)ii * stride_y_axis]);
            acc += dy_i * y_i * inv_x_j;
        }
        dx[off_dx] = __float2bfloat16(acc);
    }
}

template <typename T, typename Acc>
__host__ inline int32_t launch_scan_cumprod_backward(
    const T* dy, const T* x, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (scan_axis < 0 || scan_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 s_dy = {}, s_x = {}, s_y = {}, s_dx = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_x.v[i]   = stride_x_host[i];
        s_y.v[i]   = stride_y_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    scan_cumprod_backward_kernel<T, Acc><<<blocks, kBlock, 0, stream>>>(
        dy, x, y, dx, numel, rank, shape, s_dy, s_x, s_y, s_dx,
        scan_axis, scan_extent, reverse);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Cummax / Cummin BW kernel — single template, bool template param
// IsMax selects max-vs-min semantics. Walks the forward sequence (or
// reverse, when `reverse != 0`) tracking running winner value AND its
// first-occurrence index. Each output step i, dy[i] is added to the
// thread's dx accumulator iff that thread's coord-along-axis equals
// the current running winner.

template <typename T, typename Acc, bool IsMax>
__global__ void scan_cummax_min_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin_i = tid; lin_i < numel; lin_i += step) {
        int64_t linear = lin_i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0;
        int32_t j = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
            } else {
                j = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        Acc acc = Acc(0);
        // Walk the FW sequence in scan order; track first-occurrence
        // running winner. running_arg < 0 means "uninitialized".
        Acc running_winner = Acc(0);
        int32_t running_arg = -1;
        int32_t i_start, i_end, i_step;
        if (reverse != 0) {
            i_start = scan_extent - 1;
            i_end   = -1;
            i_step  = -1;
        } else {
            i_start = 0;
            i_end   = scan_extent;
            i_step  = 1;
        }
        for (int32_t ii = i_start; ii != i_end; ii += i_step) {
            T x_ii_raw = x[off_x_base + (int64_t)ii * stride_x_axis];
            Acc x_ii = static_cast<Acc>(x_ii_raw);
            bool is_better;
            if (running_arg < 0) {
                is_better = true;
            } else if (IsMax) {
                is_better = (x_ii > running_winner);
            } else {
                is_better = (x_ii < running_winner);
            }
            if (is_better) {
                running_winner = x_ii;
                running_arg = ii;
            }
            if (running_arg == j) {
                Acc dy_ii = static_cast<Acc>(dy[off_dy_base + (int64_t)ii * stride_dy_axis]);
                acc += dy_ii;
            }
        }
        dx[off_dx] = static_cast<T>(acc);
    }
}

// f16 specialization — accumulator stays f32 throughout.
template <>
__global__ void scan_cummax_min_backward_kernel<__half, float, true>(
    const __half* __restrict__ dy,
    const __half* __restrict__ x,
    __half* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin_i = tid; lin_i < numel; lin_i += step) {
        int64_t linear = lin_i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0;
        int32_t j = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
            } else {
                j = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        float acc = 0.0f;
        float running_winner = 0.0f;
        int32_t running_arg = -1;
        int32_t i_start, i_end, i_step;
        if (reverse != 0) { i_start = scan_extent - 1; i_end = -1;          i_step = -1; }
        else              { i_start = 0;               i_end = scan_extent; i_step = 1;  }
        for (int32_t ii = i_start; ii != i_end; ii += i_step) {
            float x_ii = __half2float(x[off_x_base + (int64_t)ii * stride_x_axis]);
            bool is_better = (running_arg < 0) || (x_ii > running_winner);
            if (is_better) { running_winner = x_ii; running_arg = ii; }
            if (running_arg == j) {
                acc += __half2float(dy[off_dy_base + (int64_t)ii * stride_dy_axis]);
            }
        }
        dx[off_dx] = __float2half(acc);
    }
}

template <>
__global__ void scan_cummax_min_backward_kernel<__half, float, false>(
    const __half* __restrict__ dy,
    const __half* __restrict__ x,
    __half* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin_i = tid; lin_i < numel; lin_i += step) {
        int64_t linear = lin_i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0;
        int32_t j = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
            } else {
                j = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        float acc = 0.0f;
        float running_winner = 0.0f;
        int32_t running_arg = -1;
        int32_t i_start, i_end, i_step;
        if (reverse != 0) { i_start = scan_extent - 1; i_end = -1;          i_step = -1; }
        else              { i_start = 0;               i_end = scan_extent; i_step = 1;  }
        for (int32_t ii = i_start; ii != i_end; ii += i_step) {
            float x_ii = __half2float(x[off_x_base + (int64_t)ii * stride_x_axis]);
            bool is_better = (running_arg < 0) || (x_ii < running_winner);
            if (is_better) { running_winner = x_ii; running_arg = ii; }
            if (running_arg == j) {
                acc += __half2float(dy[off_dy_base + (int64_t)ii * stride_dy_axis]);
            }
        }
        dx[off_dx] = __float2half(acc);
    }
}

template <>
__global__ void scan_cummax_min_backward_kernel<__nv_bfloat16, float, true>(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin_i = tid; lin_i < numel; lin_i += step) {
        int64_t linear = lin_i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0;
        int32_t j = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
            } else {
                j = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        float acc = 0.0f;
        float running_winner = 0.0f;
        int32_t running_arg = -1;
        int32_t i_start, i_end, i_step;
        if (reverse != 0) { i_start = scan_extent - 1; i_end = -1;          i_step = -1; }
        else              { i_start = 0;               i_end = scan_extent; i_step = 1;  }
        for (int32_t ii = i_start; ii != i_end; ii += i_step) {
            float x_ii = __bfloat162float(x[off_x_base + (int64_t)ii * stride_x_axis]);
            bool is_better = (running_arg < 0) || (x_ii > running_winner);
            if (is_better) { running_winner = x_ii; running_arg = ii; }
            if (running_arg == j) {
                acc += __bfloat162float(dy[off_dy_base + (int64_t)ii * stride_dy_axis]);
            }
        }
        dx[off_dx] = __float2bfloat16(acc);
    }
}

template <>
__global__ void scan_cummax_min_backward_kernel<__nv_bfloat16, float, false>(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin_i = tid; lin_i < numel; lin_i += step) {
        int64_t linear = lin_i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0;
        int32_t j = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
            } else {
                j = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        float acc = 0.0f;
        float running_winner = 0.0f;
        int32_t running_arg = -1;
        int32_t i_start, i_end, i_step;
        if (reverse != 0) { i_start = scan_extent - 1; i_end = -1;          i_step = -1; }
        else              { i_start = 0;               i_end = scan_extent; i_step = 1;  }
        for (int32_t ii = i_start; ii != i_end; ii += i_step) {
            float x_ii = __bfloat162float(x[off_x_base + (int64_t)ii * stride_x_axis]);
            bool is_better = (running_arg < 0) || (x_ii < running_winner);
            if (is_better) { running_winner = x_ii; running_arg = ii; }
            if (running_arg == j) {
                acc += __bfloat162float(dy[off_dy_base + (int64_t)ii * stride_dy_axis]);
            }
        }
        dx[off_dx] = __float2bfloat16(acc);
    }
}

template <typename T, typename Acc, bool IsMax>
__host__ inline int32_t launch_scan_cummax_min_backward(
    const T* dy, const T* x, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_x_host,
    const int64_t* stride_dx_host,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (scan_axis < 0 || scan_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 s_dy = {}, s_x = {}, s_dx = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_x.v[i]   = stride_x_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    scan_cummax_min_backward_kernel<T, Acc, IsMax><<<blocks, kBlock, 0, stream>>>(
        dy, x, dx, numel, rank, shape, s_dy, s_x, s_dx,
        scan_axis, scan_extent, reverse);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// LogCumsumExp FW + BW — Phase 4 scan fanout (closes the delicate-
// numerics scan gap).
// =============================================================================
//
// FW: `y[k] = log(Σ_{j ≤ k} exp(x[j]))` (inclusive prefix LSE) — and the
// reverse-direction analog `y[k] = log(Σ_{j ≥ k} exp(x[j]))` when
// `reverse != 0`. Implemented via the standard online-LSE
// running-max-rescaling algorithm to avoid overflow inside `exp`:
//
//   running_max = -INF, running_sum = 0
//   for each new x:
//       if x > running_max:
//           running_sum *= exp(running_max - x)
//           running_max = x
//       running_sum += exp(x - running_max)
//       y_out = log(running_sum) + running_max
//
// The "rescale on new max" line is the load-bearing step. Without it,
// values like x[0]=100, x[1]=1 would mix raw `exp(100)` with `exp(1)`
// and overflow inside f32 (and even f64). After it, every term inside
// the running_sum is `exp(x_j - m)` with `m ≥ x_j`, so every term is
// in `[0, 1]` and the sum stays bounded.
//
// Same per-thread shape as the simple scan kernel: one thread per
// output cell, walks the prefix (or suffix) accumulating O(extent).
// f16 / bf16 use a float accumulator throughout (every load goes
// through __half2float / __bfloat162float; the final store goes back
// through __float2half / __float2bfloat16). Single-rounding-at-store
// semantics, like the rest of the half-precision family.
//
// BW: gradient of `log(Σ_{j ≤ i} exp(x[j]))` w.r.t. `x[k]` is
// `exp(x[k] - y[i])` when `k ≤ i`, else 0. Aggregated over all
// dy positions:
//
//   forward FW: dx[k] = Σ_{i ≥ k} dy[i] * exp(x[k] - y[i])
//   reverse FW: dx[k] = Σ_{i ≤ k} dy[i] * exp(x[k] - y[i])
//
// Per-thread: load x[k] once, walk the i-range on saved y and dy.
// `x[k] - y[i]` is always ≤ 0 (since `y[i] = LSE(prefix) ≥ x[k]`
// whenever k is inside the prefix), so `exp(.)` stays in `[0, 1]`.
// Stable by construction; no extra max-tracking required in BW.

// LSE-style dtype trait — mirror of `LseDtype` from reduce_logsumexp_fp.cu
// but inlined here so the scan kernel doesn't drag a foreign .cu's
// header in. f64 stays in double everywhere; f16 / bf16 detour through
// float.
template <typename T>
struct LogCumsumExpDtype;

template <>
struct LogCumsumExpDtype<float> {
    using Compute = float;
    static __device__ __forceinline__ float load(float v) { return v; }
    static __device__ __forceinline__ float store_from(float v) { return v; }
    static __device__ __forceinline__ float neg_infinity() { return -INFINITY; }
    static __device__ __forceinline__ float gexp(float v) { return __expf(v); }
    static __device__ __forceinline__ float glog(float v) { return logf(v); }
};

template <>
struct LogCumsumExpDtype<double> {
    using Compute = double;
    static __device__ __forceinline__ double load(double v) { return v; }
    static __device__ __forceinline__ double store_from(double v) { return v; }
    static __device__ __forceinline__ double neg_infinity() { return -INFINITY; }
    static __device__ __forceinline__ double gexp(double v) { return exp(v); }
    static __device__ __forceinline__ double glog(double v) { return log(v); }
};

template <>
struct LogCumsumExpDtype<__half> {
    using Compute = float;
    static __device__ __forceinline__ float load(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half store_from(float v) { return __float2half(v); }
    static __device__ __forceinline__ float neg_infinity() { return -INFINITY; }
    static __device__ __forceinline__ float gexp(float v) { return __expf(v); }
    static __device__ __forceinline__ float glog(float v) { return logf(v); }
};

template <>
struct LogCumsumExpDtype<__nv_bfloat16> {
    using Compute = float;
    static __device__ __forceinline__ float load(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    static __device__ __forceinline__ __nv_bfloat16 store_from(float v) {
        return __float2bfloat16(v);
    }
    static __device__ __forceinline__ float neg_infinity() { return -INFINITY; }
    static __device__ __forceinline__ float gexp(float v) { return __expf(v); }
    static __device__ __forceinline__ float glog(float v) { return logf(v); }
};

template <typename T>
__global__ void log_cumsum_exp_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_x,
    DimsI64 stride_y,
    int32_t scan_axis,
    int32_t scan_extent,
    int64_t scan_stride_x,
    int32_t reverse)
{
    using DT = LogCumsumExpDtype<T>;
    using C  = typename DT::Compute;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != scan_axis) {
                off_x_base += coord * stride_x.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        // Online-LSE running-max algorithm.
        C running_max = DT::neg_infinity();
        C running_sum = C(0);
        int32_t j_start, j_end, j_step;
        if (reverse != 0) {
            // Suffix `j ∈ [k, extent)` walked high-to-low; this matches
            // the simple-scan kernel's "reverse FW" walk order and gives
            // y[k] = LSE(x[k..extent]).
            j_start = scan_extent - 1;
            j_end   = k - 1;
            j_step  = -1;
        } else {
            j_start = 0;
            j_end   = k + 1;
            j_step  = 1;
        }
        for (int32_t j = j_start; j != j_end; j += j_step) {
            C x_j = DT::load(x[off_x_base + (int64_t)j * scan_stride_x]);
            if (x_j > running_max) {
                // First iteration: running_max == -inf so this branch
                // also handles the init. `exp(-inf - x_j)` would be 0
                // but we skip the multiply by guarding on running_sum
                // being zero anyway; explicit check on isinf keeps
                // running_sum from going NaN on the first step.
                if (running_max == DT::neg_infinity()) {
                    running_sum = C(0);
                } else {
                    running_sum *= DT::gexp(running_max - x_j);
                }
                running_max = x_j;
            }
            running_sum += DT::gexp(x_j - running_max);
        }
        // y[k] = log(running_sum) + running_max. running_sum ≥ 1
        // (the term at j == k is `exp(x_k - running_max) = exp(0) = 1`
        // since x_k becomes running_max no later than its own step),
        // so the log is well-defined.
        C out = DT::glog(running_sum) + running_max;
        y[off_y] = DT::store_from(out);
    }
}

template <typename T>
__host__ inline int32_t launch_log_cumsum_exp(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t scan_axis,
    int32_t scan_extent,
    int64_t scan_stride_x,
    int32_t reverse,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (scan_axis < 0 || scan_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    log_cumsum_exp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, rank, shape, sx, sy,
        scan_axis, scan_extent, scan_stride_x, reverse);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// LogCumsumExp BW kernel. Reuses LogCumsumExpDtype above. The walk
// range and direction are the *complement* of the FW's: forward FW
// (reverse == 0) → BW thread at coord-along-axis `k` reads dy/y at
// indices `i ∈ [k, extent)`; reverse FW → BW reads `i ∈ [0, k]`. Each
// step accumulates `dy[i] * exp(x[k] - y[i])` into the f32/f64
// accumulator, then stores back through the dtype's converter.
template <typename T>
__global__ void log_cumsum_exp_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_dy,
    DimsI64 stride_x,
    DimsI64 stride_y,
    DimsI64 stride_dx,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse)
{
    using DT = LogCumsumExpDtype<T>;
    using C  = typename DT::Compute;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin_i = tid; lin_i < numel; lin_i += step) {
        int64_t linear = lin_i;
        int64_t off_dx = 0;
        int64_t off_dy_base = 0, off_x_base = 0, off_y_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != scan_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_x_base  += coord * stride_x.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        int64_t stride_dy_axis = stride_dy.v[scan_axis];
        int64_t stride_x_axis  = stride_x.v[scan_axis];
        int64_t stride_y_axis  = stride_y.v[scan_axis];
        C x_k = DT::load(x[off_x_base + (int64_t)k * stride_x_axis]);
        C acc = C(0);
        int32_t i_lo, i_hi;
        if (reverse != 0) {
            // reverse FW: y[i] = LSE(x[i..extent]); k contributes to
            // y[i] iff i ≤ k. Walk i ∈ [0, k].
            i_lo = 0;
            i_hi = k;
        } else {
            // forward FW: y[i] = LSE(x[0..=i]); k contributes to y[i]
            // iff i ≥ k. Walk i ∈ [k, extent).
            i_lo = k;
            i_hi = scan_extent - 1;
        }
        for (int32_t ii = i_lo; ii <= i_hi; ++ii) {
            C dy_i = DT::load(dy[off_dy_base + (int64_t)ii * stride_dy_axis]);
            C y_i  = DT::load(y [off_y_base  + (int64_t)ii * stride_y_axis]);
            acc += dy_i * DT::gexp(x_k - y_i);
        }
        dx[off_dx] = DT::store_from(acc);
    }
}

template <typename T>
__host__ inline int32_t launch_log_cumsum_exp_backward(
    const T* dy, const T* x, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    int32_t scan_axis,
    int32_t scan_extent,
    int32_t reverse,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (scan_axis < 0 || scan_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 s_dy = {}, s_x = {}, s_y = {}, s_dx = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        s_dy.v[i]  = stride_dy_host[i];
        s_x.v[i]   = stride_x_host[i];
        s_y.v[i]   = stride_y_host[i];
        s_dx.v[i]  = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    log_cumsum_exp_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, x, y, dx, numel, rank, shape, s_dy, s_x, s_y, s_dx,
        scan_axis, scan_extent, reverse);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Scaled ternary pointwise kernels — contig and strided
// =============================================================================
//
// Variant of the standard ternary kernel family with an extra `float
// scale` parameter threaded through to the functor. Used by ops that
// take a scalar multiplier (PyTorch's `addcmul(input, t1, t2, value)`
// and `addcdiv(input, t1, t2, value)`). The functor is 4-arg:
// `__device__ T operator()(T a, T b, T c, float scale) const`.
//
// Scale is f32 regardless of T — for f64 ops it widens to double inside
// the functor; for f16 / bf16 it stays f32 (the f32-detour pattern uses
// f32 internally anyway). This matches the convention for alpha / beta
// in the float-GEMM plans.

template <typename T, typename F>
__global__ void ternary_scaled_pointwise_contig_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ c,
    T* __restrict__ y,
    int64_t numel,
    float scale,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = op(a[i], b[i], c[i], scale);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_ternary_scaled_pointwise_contig(
    const T* a, const T* b, const T* c, T* y,
    int64_t numel,
    float scale,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    ternary_scaled_pointwise_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, c, y, numel, scale, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void ternary_scaled_pointwise_strided_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ c,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_a,
    DimsI64 stride_b,
    DimsI64 stride_c,
    DimsI64 stride_y,
    float scale,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_a = 0, off_b = 0, off_c = 0, off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_a += coord * stride_a.v[d];
            off_b += coord * stride_b.v[d];
            off_c += coord * stride_c.v[d];
            off_y += coord * stride_y.v[d];
        }
        y[off_y] = op(a[off_a], b[off_b], c[off_c], scale);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_ternary_scaled_pointwise_strided(
    const T* a, const T* b, const T* c, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_a_host,
    const int64_t* stride_b_host,
    const int64_t* stride_c_host,
    const int64_t* stride_y_host,
    float scale,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 sa = {}, sb = {}, sc = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sa.v[i]    = stride_a_host[i];
        sb.v[i]    = stride_b_host[i];
        sc.v[i]    = stride_c_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    ternary_scaled_pointwise_strided_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, c, y, numel, rank, shape, sa, sb, sc, sy, scale, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Ternary backward kernels — Phase 3 backward family (Milestone F)
// =============================================================================
//
// Sibling of `binary_backward_saves_kernel` for 3-input ops whose
// gradient writes all three input grads (da, db, dc). Two templates:
//
//   * `ternary_backward_kernel`        — unscaled. Functor signature:
//       `__device__ void operator()(T dy, T a, T b, T c,
//                                   T& da, T& db, T& dc) const`.
//     Used by Fma (da=dy·b, db=dy·a, dc=dy) and Clamp (mask × dy).
//
//   * `ternary_backward_scaled_kernel` — scaled. Functor signature:
//       `__device__ void operator()(T dy, T a, T b, T c, float scale,
//                                   T& da, T& db, T& dc) const`.
//     Used by Addcmul (da=dy, db=dy·scale·c, dc=dy·scale·b) and
//     Addcdiv (da=dy, db=dy·scale/c, dc=-dy·scale·b/c²).
//
// Caller convention: all three saved inputs `a`, `b`, `c` are always
// read by the kernel — the Rust side requires them to be supplied even
// for ops where one isn't algebraically referenced (Fma's `c`, Addcmul/
// Addcdiv's `a`). This keeps the ABI uniform across the 4 ops; the
// unused load is a single coalesced read per cell — negligible cost
// versus the 7-pointer launch fixed overhead.
//
// Same launch convention as `binary_backward_saves_kernel` — pure SIMT,
// linear sweep, grid-cap loop for unbounded numel.

template <typename T, typename F>
__global__ void ternary_backward_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ c,
    T* __restrict__ da,
    T* __restrict__ db,
    T* __restrict__ dc,
    int64_t numel,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        T da_out, db_out, dc_out;
        op(dy[i], a[i], b[i], c[i], da_out, db_out, dc_out);
        da[i] = da_out;
        db[i] = db_out;
        dc[i] = dc_out;
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_ternary_backward(
    const T* dy, const T* a, const T* b, const T* c,
    T* da, T* db, T* dc,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    ternary_backward_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        dy, a, b, c, da, db, dc, numel, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void ternary_backward_scaled_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ c,
    T* __restrict__ da,
    T* __restrict__ db,
    T* __restrict__ dc,
    int64_t numel,
    float scale,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        T da_out, db_out, dc_out;
        op(dy[i], a[i], b[i], c[i], scale, da_out, db_out, dc_out);
        da[i] = da_out;
        db[i] = db_out;
        dc[i] = dc_out;
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_ternary_backward_scaled(
    const T* dy, const T* a, const T* b, const T* c,
    T* da, T* db, T* dc,
    int64_t numel,
    float scale,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    ternary_backward_scaled_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        dy, a, b, c, da, db, dc, numel, scale, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Where (heterogeneous-dtype ternary) kernels — contig and strided
// =============================================================================
//
// `y = cond ? a : b` — element-select with a `uint8_t` cond input
// (PyTorch / NumPy bool storage convention: 0 = false, non-zero = true)
// and same-dtype a / b / y. Distinct from the homogeneous-dtype ternary
// family because the first input has a different dtype than the
// others; this kernel takes the cond pointer typed as `const uint8_t*`
// at the launcher boundary.
//
// No functor abstraction here — the op is fixed (`cond ? a : b`), so
// the kernel inlines the body directly. If future heterogeneous-cond
// ops join (e.g., a `masked_fill` variant), they get their own kernel
// template; we don't pretend to share a functor type when the math
// fundamentally differs.

template <typename T>
__global__ void where_pointwise_contig_kernel(
    const uint8_t* __restrict__ cond,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ y,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = cond[i] ? a[i] : b[i];
    }
}

template <typename T>
__host__ inline int32_t launch_where_pointwise_contig(
    const uint8_t* cond, const T* a, const T* b, T* y,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    where_pointwise_contig_kernel<T><<<blocks, kBlock, 0, stream>>>(cond, a, b, y, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__global__ void where_pointwise_strided_kernel(
    const uint8_t* __restrict__ cond,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_cond,
    DimsI64 stride_a,
    DimsI64 stride_b,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_c = 0, off_a = 0, off_b = 0, off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_c += coord * stride_cond.v[d];
            off_a += coord * stride_a.v[d];
            off_b += coord * stride_b.v[d];
            off_y += coord * stride_y.v[d];
        }
        y[off_y] = cond[off_c] ? a[off_a] : b[off_b];
    }
}

template <typename T>
__host__ inline int32_t launch_where_pointwise_strided(
    const uint8_t* cond, const T* a, const T* b, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_cond_host,
    const int64_t* stride_a_host,
    const int64_t* stride_b_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 sc = {}, sa = {}, sb = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sc.v[i]    = stride_cond_host[i];
        sa.v[i]    = stride_a_host[i];
        sb.v[i]    = stride_b_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    where_pointwise_strided_kernel<T><<<blocks, kBlock, 0, stream>>>(
        cond, a, b, y, numel, rank, shape, sc, sa, sb, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Where backward kernel — contig only
// =============================================================================
//
// Forward: `y = cond ? a : b`. Backward (cond is non-differentiable):
//   da[i] = cond[i] ? dy[i] : 0
//   db[i] = cond[i] ? 0     : dy[i]
//
// Pure mask + copy — no arithmetic at all, so output is bit-exact
// against host reference at every dtype. Trailblazer is contig-only:
// caller materializes broadcasted operands before launch.
//
// `cond` is `const uint8_t*` matching the FW convention. `dy`, `da`,
// `db` share dtype `T`.

template <typename T>
__global__ void where_backward_pointwise_contig_kernel(
    const uint8_t* __restrict__ cond,
    const T* __restrict__ dy,
    T* __restrict__ da,
    T* __restrict__ db,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    const T zero = T(0);
    for (int64_t i = tid; i < numel; i += step) {
        T g = dy[i];
        if (cond[i]) {
            da[i] = g;
            db[i] = zero;
        } else {
            da[i] = zero;
            db[i] = g;
        }
    }
}

template <typename T>
__host__ inline int32_t launch_where_backward_pointwise_contig(
    const uint8_t* cond, const T* dy, T* da, T* db,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    where_backward_pointwise_contig_kernel<T><<<blocks, kBlock, 0, stream>>>(
        cond, dy, da, db, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Ternary pointwise kernels — contig and strided
// =============================================================================
//
// Same shape as the binary family but with 3 inputs (a, b, c) and 1
// output (y), all of the same scalar type `T`. `F` is a ternary functor
// with `__device__ T operator()(T, T, T) const`. The strided variant
// uses the same DimsI32 / DimsI64 / MAX_RANK shared infra plus a third
// stride array.

template <typename T, typename F>
__global__ void ternary_pointwise_contig_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ c,
    T* __restrict__ y,
    int64_t numel,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = op(a[i], b[i], c[i]);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_ternary_pointwise_contig(
    const T* a, const T* b, const T* c, T* y,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    ternary_pointwise_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, c, y, numel, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void ternary_pointwise_strided_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ c,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_a,
    DimsI64 stride_b,
    DimsI64 stride_c,
    DimsI64 stride_y,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_a = 0, off_b = 0, off_c = 0, off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_a += coord * stride_a.v[d];
            off_b += coord * stride_b.v[d];
            off_c += coord * stride_c.v[d];
            off_y += coord * stride_y.v[d];
        }
        y[off_y] = op(a[off_a], b[off_b], c[off_c]);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_ternary_pointwise_strided(
    const T* a, const T* b, const T* c, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_a_host,
    const int64_t* stride_b_host,
    const int64_t* stride_c_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 sa = {}, sb = {}, sc = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sa.v[i]    = stride_a_host[i];
        sb.v[i]    = stride_b_host[i];
        sc.v[i]    = stride_c_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    ternary_pointwise_strided_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, c, y, numel, rank, shape, sa, sb, sc, sy, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Binary comparison kernels — contig and strided
// =============================================================================
//
// Sibling of the binary pointwise family above, but the output type is
// fixed to `uint8_t` (PyTorch / NumPy bool storage convention: 0 =
// false, 1 = true). The functor returns `uint8_t`. Same DimsI32 /
// DimsI64 / MAX_RANK shared infra.
//
// Why a separate kernel family: the output type differs from the
// input type, so the kernel signature can't unify with
// `binary_pointwise_*_kernel<T, F>` (which has `T` for both input and
// output). The strided variant follows the same per-axis coord-from-
// linear-index unraveling pattern.

template <typename T, typename F>
__global__ void binary_cmp_pointwise_contig_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    uint8_t* __restrict__ y,
    int64_t numel,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = op(a[i], b[i]);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_binary_cmp_pointwise_contig(
    const T* a, const T* b, uint8_t* y,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    binary_cmp_pointwise_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, y, numel, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void binary_cmp_pointwise_strided_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    uint8_t* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_a,
    DimsI64 stride_b,
    DimsI64 stride_y,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_a = 0, off_b = 0, off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_a += coord * stride_a.v[d];
            off_b += coord * stride_b.v[d];
            off_y += coord * stride_y.v[d];
        }
        y[off_y] = op(a[off_a], b[off_b]);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_binary_cmp_pointwise_strided(
    const T* a, const T* b, uint8_t* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_a_host,
    const int64_t* stride_b_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 sa = {};
    DimsI64 sb = {};
    DimsI64 sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sa.v[i]    = stride_a_host[i];
        sb.v[i]    = stride_b_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    binary_cmp_pointwise_strided_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        a, b, y, numel, rank, shape, sa, sb, sy, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Unary pointwise kernels — contig and strided
// =============================================================================
//
// Same shape as the binary kernels above but 1→1: one input, one output.
// `F` is a unary functor with `__device__ T operator()(T) const`. The
// strided variant uses the same per-axis coord-from-linear-index
// unraveling, with one fewer stride array (no second input). The
// `DimsI32` / `DimsI64` structs and `MAX_RANK` constant are shared
// with the binary path.
//
// Broadcast doesn't really apply to unary (input shape == output shape
// always — a "broadcast" unary would be `f(x[0])` replicated, which is
// trivially a host-side computation). The strided path here handles
// non-contig input / output views (transposed, sliced) but the
// dispatcher requires `x.shape == y.shape`.

template <typename T, typename F>
__global__ void unary_pointwise_contig_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = op(x[i]);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_unary_pointwise_contig(
    const T* x, T* y,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unary_pointwise_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void unary_pointwise_strided_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_x,
    DimsI64 stride_y,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_x = 0, off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_x += coord * stride_x.v[d];
            off_y += coord * stride_y.v[d];
        }
        y[off_y] = op(x[off_x]);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_unary_pointwise_strided(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape = {};
    DimsI64 sx    = {};
    DimsI64 sy    = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unary_pointwise_strided_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, rank, shape, sx, sy, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Gated-activation kernels — Phase 3 Category C′
// =============================================================================
//
// Plan shape: split a rank-N input `x` along `split_dim` into two halves
// `(a, b)` of equal size, output `y = a · gate(b)`. The output's shape
// equals `x.shape` with `shape[split_dim]` halved. One thread per
// **output** cell — for each output coord `c` we read `a` and `b` from
// `x` using contig strides; the b-half lives at `off_x + x_half_offset`
// where `x_half_offset = (x_shape[split_dim]/2) · stride_x[split_dim]`.
//
// Functor signature (FW): `__device__ T operator()(T a, T b) const`,
// computing `a · gate(b)` so f16/bf16 functors can do the f32 detour
// once with both halves in scope.
//
// Functor signature (BW): `__device__ void operator()(T dy, T a, T b,
// T& da_out, T& db_out) const`, where `da_out = dy * gate(b)` and
// `db_out = dy * a * gate'(b)`.
//
// Today: contig-only (callers pass row-major strides). Strided fanout
// follows the binary-strided pattern; defer until needed.

template <typename T, typename F>
__global__ void gated_activation_contig_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    DimsI32 output_shape,
    int32_t split_dim,
    int64_t x_half_offset,
    DimsI64 stride_x,
    DimsI64 stride_y,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    (void)split_dim;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_x_a = 0, off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = output_shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_x_a += coord * stride_x.v[d];
            off_y   += coord * stride_y.v[d];
        }
        int64_t off_x_b = off_x_a + x_half_offset;
        y[off_y] = op(x[off_x_a], x[off_x_b]);
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_gated_activation_contig(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* output_shape_host,
    int32_t split_dim,
    int64_t x_half_offset,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (split_dim < 0 || split_dim >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sx    = {};
    DimsI64 sy    = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = output_shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    gated_activation_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, shape, split_dim, x_half_offset, sx, sy, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename F>
__global__ void gated_activation_backward_contig_kernel(
    const T* __restrict__ x,
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int64_t output_numel,
    int32_t rank,
    DimsI32 output_shape,
    int32_t split_dim,
    int64_t x_half_offset,
    int64_t dx_half_offset,
    DimsI64 stride_x,
    DimsI64 stride_dy,
    DimsI64 stride_dx,
    F op)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    (void)split_dim;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_x_a = 0, off_dy = 0, off_dx_a = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = output_shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_x_a  += coord * stride_x.v[d];
            off_dy   += coord * stride_dy.v[d];
            off_dx_a += coord * stride_dx.v[d];
        }
        int64_t off_x_b  = off_x_a  + x_half_offset;
        int64_t off_dx_b = off_dx_a + dx_half_offset;
        T da_out, db_out;
        op(dy[off_dy], x[off_x_a], x[off_x_b], da_out, db_out);
        dx[off_dx_a] = da_out;
        dx[off_dx_b] = db_out;
    }
}

template <typename T, typename F>
__host__ inline int32_t launch_gated_activation_backward_contig(
    const T* x, const T* dy, T* dx,
    int64_t output_numel,
    int32_t rank,
    const int32_t* output_shape_host,
    int32_t split_dim,
    int64_t x_half_offset,
    int64_t dx_half_offset,
    const int64_t* stride_x_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_dx_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (split_dim < 0 || split_dim >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sx    = {};
    DimsI64 sdy   = {};
    DimsI64 sdx   = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = output_shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sdy.v[i]   = stride_dy_host[i];
        sdx.v[i]   = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    gated_activation_backward_contig_kernel<T, F><<<blocks, kBlock, 0, stream>>>(
        x, dy, dx, output_numel, rank, shape, split_dim,
        x_half_offset, dx_half_offset, sx, sdy, sdx, F{});
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::elementwise

// =============================================================================
// INSTANTIATE macros
// =============================================================================
//
// One macro per op-shape category. The macros emit `extern "C"`
// launcher symbols compatible with the FFI declarations in
// `baracuda-kernels-sys/src/lib.rs`.
//
// Naming convention:
//   baracuda_kernels_<category>_<op>_<dtype>_run
//   baracuda_kernels_<category>_<op>_<dtype>_can_implement
//
// Example: `binary` + `add` + `f32` →
//   baracuda_kernels_binary_add_f32_run
//   baracuda_kernels_binary_add_f32_can_implement

// Emit one binary-pointwise contig launcher pair.
//
// NAME    : symbol body — e.g. `binary_add_f32` (joins between
//           `baracuda_kernels_` and `_run` / `_can_implement`).
// T       : scalar element type (e.g. `float`).
// FUNCTOR : binary functor type with `__device__ T operator()(T, T)`.
#define BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(NAME, T, FUNCTOR)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                            \
        int64_t numel,                                                                            \
        const void* a, const void* b, void* y,                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                         \
        void* stream_ptr)                                                                         \
    {                                                                                             \
        if (numel < 0) return 2;                                                                 \
        if (numel == 0) return 0;                                                                \
        if (a == nullptr || b == nullptr || y == nullptr) return 2;                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                             \
        return baracuda::elementwise::launch_binary_pointwise_contig<T, FUNCTOR>(                \
            static_cast<const T*>(a),                                                            \
            static_cast<const T*>(b),                                                            \
            static_cast<T*>(y),                                                                   \
            numel,                                                                                \
            stream);                                                                              \
    }                                                                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                  \
        int64_t numel,                                                                            \
        const void* /*a*/, const void* /*b*/, const void* /*y*/)                                 \
    {                                                                                             \
        if (numel < 0) return 2;                                                                 \
        return 0;                                                                                 \
    }

// Emit one strided / broadcast binary-pointwise launcher.
//
// Companion to BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE that
// handles non-contiguous operands (broadcast via stride 0; arbitrary
// strided views). The Rust dispatcher picks contig vs strided based on
// `is_contiguous()` at launch time; both launchers are emitted per
// (op, dtype) cell.
//
// NAME    : symbol body, e.g. `binary_add_f32` — joins between
//           `baracuda_kernels_` and `_strided_run`.
// T       : scalar element type.
// FUNCTOR : binary functor type with `__device__ T operator()(T, T)`.
#define BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(NAME, T, FUNCTOR)               \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                 \
        int64_t numel,                                                                         \
        int32_t rank,                                                                          \
        const int32_t* shape,                                                                 \
        const int64_t* stride_a,                                                              \
        const int64_t* stride_b,                                                              \
        const int64_t* stride_y,                                                              \
        const void* a, const void* b, void* y,                                                \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                      \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (numel < 0) return 2;                                                              \
        if (numel == 0) return 0;                                                             \
        if (a == nullptr || b == nullptr || y == nullptr) return 2;                           \
        if (shape == nullptr || stride_a == nullptr || stride_b == nullptr ||                 \
            stride_y == nullptr) return 2;                                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                          \
        return baracuda::elementwise::launch_binary_pointwise_strided<T, FUNCTOR>(            \
            static_cast<const T*>(a),                                                         \
            static_cast<const T*>(b),                                                          \
            static_cast<T*>(y),                                                                \
            numel, rank, shape, stride_a, stride_b, stride_y,                                 \
            stream);                                                                           \
    }

// Emit one ternary pointwise contig launcher pair.
//
// Sibling of BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE for 3-input
// ops (clamp, fma, addcmul, addcdiv). All inputs and the output are
// the same scalar type T.
//
// NAME    : symbol body — e.g. `ternary_clamp_f32`.
// T       : scalar element type.
// FUNCTOR : ternary functor type with `__device__ T operator()(T, T, T)`.
#define BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(NAME, T, FUNCTOR)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* a, const void* b, const void* c, void* y,                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (a == nullptr || b == nullptr || c == nullptr || y == nullptr) return 2;               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_ternary_pointwise_contig<T, FUNCTOR>(                \
            static_cast<const T*>(a), static_cast<const T*>(b),                                   \
            static_cast<const T*>(c), static_cast<T*>(y),                                          \
            numel, stream);                                                                        \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int64_t numel,                                                                             \
        const void* /*a*/, const void* /*b*/, const void* /*c*/, const void* /*y*/)               \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        return 0;                                                                                  \
    }

// Emit one ternary pointwise strided launcher.
#define BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(NAME, T, FUNCTOR)                  \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                     \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_a,                                                                  \
        const int64_t* stride_b,                                                                  \
        const int64_t* stride_c,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* a, const void* b, const void* c, void* y,                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (a == nullptr || b == nullptr || c == nullptr || y == nullptr) return 2;               \
        if (shape == nullptr || stride_a == nullptr || stride_b == nullptr ||                     \
            stride_c == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_ternary_pointwise_strided<T, FUNCTOR>(               \
            static_cast<const T*>(a), static_cast<const T*>(b),                                   \
            static_cast<const T*>(c), static_cast<T*>(y),                                          \
            numel, rank, shape, stride_a, stride_b, stride_c, stride_y, stream);                  \
    }

// Emit one no-save binary backward launcher (Add / Sub family — the
// gradient depends only on `dy`).
//
// NAME    : symbol body — e.g. `binary_add_backward_f32`.
// T       : scalar element type.
// FUNCTOR : functor with `__device__ void operator()(T dy, T& da, T& db) const`.
#define BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(NAME, T, FUNCTOR)                     \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* dy, void* da, void* db,                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || da == nullptr || db == nullptr) return 2;                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_binary_backward_nosave<T, FUNCTOR>(                  \
            static_cast<const T*>(dy), static_cast<T*>(da), static_cast<T*>(db),                  \
            numel, stream);                                                                       \
    }

// Emit one saves-using binary backward launcher (Mul / Div family — the
// gradient references saved forward inputs `a` and `b`).
//
// NAME    : symbol body — e.g. `binary_mul_backward_f32`.
// T       : scalar element type.
// FUNCTOR : functor with `__device__ void operator()(T dy, T a, T b, T& da, T& db) const`.
#define BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(NAME, T, FUNCTOR)                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* dy, const void* a, const void* b, void* da, void* db,                         \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || a == nullptr || b == nullptr ||                                       \
            da == nullptr || db == nullptr) return 2;                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_binary_backward_saves<T, FUNCTOR>(                   \
            static_cast<const T*>(dy),                                                            \
            static_cast<const T*>(a), static_cast<const T*>(b),                                   \
            static_cast<T*>(da), static_cast<T*>(db),                                              \
            numel, stream);                                                                       \
    }

// Emit one unary backward launcher. The kernel takes one saved tensor
// (either `x` or `y` depending on op semantics — see the op's BW formula)
// and writes one gradient `dx`. Functor signature:
//   `__device__ T operator()(T dy, T saved) const`.
//
// NAME    : symbol body — e.g. `unary_sin_backward_f32`.
// T       : scalar element type.
// FUNCTOR : functor type providing the pointwise BW formula.
#define BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(NAME, T, FUNCTOR)                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* dy, const void* saved, void* dx,                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || saved == nullptr || dx == nullptr) return 2;                         \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_unary_backward<T, FUNCTOR>(                          \
            static_cast<const T*>(dy), static_cast<const T*>(saved),                              \
            static_cast<T*>(dx),                                                                  \
            numel, stream);                                                                       \
    }

// Emit one reduce-sum backward launcher. ABI mirrors the binary strided
// launcher shape — caller passes the full dx shape + strides for both
// dy and dx; setting `stride_dy[reduce_axis] = 0` realizes the
// broadcast.
//
// NAME : symbol body — e.g. `reduce_sum_backward_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_REDUCE_SUM_BACKWARD_INSTANTIATE(NAME, T)                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_dx,                                                                 \
        const void* dy, void* dx,                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || dx == nullptr) return 2;                                             \
        if (shape == nullptr || stride_dy == nullptr || stride_dx == nullptr) return 2;           \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_sum_backward<T>(                              \
            static_cast<const T*>(dy), static_cast<T*>(dx),                                       \
            numel, rank, shape, stride_dy, stride_dx, stream);                                    \
    }

// Emit one reduce-mean backward launcher. Adds `inv_extent_d` (a double
// passed by value) to the Sum BW ABI — the kernel scales the broadcast
// dy by it at use.
//
// NAME : symbol body — e.g. `reduce_mean_backward_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_REDUCE_MEAN_BACKWARD_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_dx,                                                                 \
        const void* dy, void* dx,                                                                 \
        double inv_extent,                                                                        \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || dx == nullptr) return 2;                                             \
        if (shape == nullptr || stride_dy == nullptr || stride_dx == nullptr) return 2;           \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_mean_backward<T>(                             \
            static_cast<const T*>(dy), static_cast<T*>(dx),                                       \
            numel, rank, shape, stride_dy, stride_dx, inv_extent, stream);                        \
    }

// Emit one reduce max/min backward launcher. Single kernel handles both
// Max BW and Min BW — caller passes the forward output `y` (max or
// min) and `x[c] == y[c_reduced]` identifies recipient positions.
//
// NAME : symbol body — e.g. `reduce_max_min_backward_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_REDUCE_MAX_MIN_BACKWARD_INSTANTIATE(NAME, T)                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        const void* dy, const void* x, const void* y, void* dx,                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || y == nullptr || dx == nullptr) return 2;             \
        if (shape == nullptr || stride_dy == nullptr || stride_x == nullptr ||                    \
            stride_y == nullptr || stride_dx == nullptr) return 2;                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_max_min_backward<T>(                          \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<const T*>(y), static_cast<T*>(dx),                                        \
            numel, rank, shape, stride_dy, stride_x, stride_y, stride_dx, stream);                \
    }

// Emit one reduce-prod backward launcher. Same ABI shape as
// REDUCE_MAX_MIN_BACKWARD; kernel computes `dy * y / x`.
#define BARACUDA_KERNELS_REDUCE_PROD_BACKWARD_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        const void* dy, const void* x, const void* y, void* dx,                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || y == nullptr || dx == nullptr) return 2;             \
        if (shape == nullptr || stride_dy == nullptr || stride_x == nullptr ||                    \
            stride_y == nullptr || stride_dx == nullptr) return 2;                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_prod_backward<T>(                             \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<const T*>(y), static_cast<T*>(dx),                                        \
            numel, rank, shape, stride_dy, stride_x, stride_y, stride_dx, stream);                \
    }

// Emit one reduce-norm2 backward launcher. Same ABI shape; kernel
// computes `dy * x / y`.
#define BARACUDA_KERNELS_REDUCE_NORM2_BACKWARD_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        const void* dy, const void* x, const void* y, void* dx,                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || y == nullptr || dx == nullptr) return 2;             \
        if (shape == nullptr || stride_dy == nullptr || stride_x == nullptr ||                    \
            stride_y == nullptr || stride_dx == nullptr) return 2;                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_norm2_backward<T>(                            \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<const T*>(y), static_cast<T*>(dx),                                        \
            numel, rank, shape, stride_dy, stride_x, stride_y, stride_dx, stream);                \
    }

// Emit one reduce-logsumexp backward launcher. Same ABI shape as
// REDUCE_NORM2_BACKWARD; kernel computes `dy * exp(x - y)` where
// `y = lse(x)` so `x - y ≤ 0` and the exp is bounded in `(0, 1]`.
#define BARACUDA_KERNELS_REDUCE_LOGSUMEXP_BACKWARD_INSTANTIATE(NAME, T)                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        const void* dy, const void* x, const void* y, void* dx,                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || y == nullptr || dx == nullptr) return 2;             \
        if (shape == nullptr || stride_dy == nullptr || stride_x == nullptr ||                    \
            stride_y == nullptr || stride_dx == nullptr) return 2;                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_logsumexp_backward<T>(                        \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<const T*>(y), static_cast<T*>(dx),                                        \
            numel, rank, shape, stride_dy, stride_x, stride_y, stride_dx, stream);                \
    }

// Emit one Repeat launcher.
#define BARACUDA_KERNELS_REPEAT_INSTANTIATE(NAME, T)                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* input_shape,                                                               \
        const int32_t* output_shape,                                                              \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (input_shape == nullptr || output_shape == nullptr ||                                  \
            stride_x == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_repeat<T>(                                           \
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                     \
            input_shape, output_shape, stride_x, stride_y, stream);                               \
    }

// Emit a Welford reduce (var or std) launcher pair templated on T —
// DoSqrt chooses between variance (false) and std (true). Accumulator
// is `WelfordAcc<T>::type` (float for f32/f16/bf16, double for f64).
#define BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(NAME, T, DO_SQRT)                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* output_shape,                                                              \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        int32_t reduce_axis,                                                                      \
        int32_t reduce_extent,                                                                    \
        int64_t reduce_stride_x,                                                                  \
        int32_t correction,                                                                       \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (output_shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;      \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_welford_axis<T, DO_SQRT>(                     \
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                     \
            output_shape, stride_x, stride_y,                                                     \
            reduce_axis, reduce_extent, reduce_stride_x, correction, stream);                     \
    }

// Emit a Welford reduce backward (Var or Std) launcher templated on T
// — `DO_SQRT` chooses between Var BW (false) and Std BW (true). Same
// dual-save ABI as Prod/Norm2 BW plus the four Welford trailing
// params (reduce_axis, reduce_extent, reduce_stride_x, correction).
#define BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(NAME, T, DO_SQRT)                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        const void* dy, const void* x, const void* y, void* dx,                                   \
        int32_t reduce_axis,                                                                      \
        int32_t reduce_extent,                                                                    \
        int64_t reduce_stride_x,                                                                  \
        int32_t correction,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || dx == nullptr) return 2;                             \
        /* y is consumed only by Std BW (DoSqrt=true). Var BW ignores it. */                      \
        if ((DO_SQRT) && y == nullptr) return 2;                                                  \
        if (shape == nullptr || stride_dy == nullptr || stride_x == nullptr ||                    \
            stride_y == nullptr || stride_dx == nullptr) return 2;                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_reduce_welford_backward<T, DO_SQRT>(                 \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<const T*>(y), static_cast<T*>(dx),                                        \
            numel, rank, shape, stride_dy, stride_x, stride_y, stride_dx,                         \
            reduce_axis, reduce_extent, reduce_stride_x, correction, stream);                     \
    }

// Emit one Flip launcher.
#define BARACUDA_KERNELS_FLIP_INSTANTIATE(NAME, T)                                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int32_t* flip_axes,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (shape == nullptr || flip_axes == nullptr ||                                           \
            stride_x == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_flip<T>(                                             \
            static_cast<const T*>(x), static_cast<T*>(y), numel, rank,                            \
            shape, flip_axes, stride_x, stride_y, stream);                                        \
    }

// Emit one Roll launcher.
#define BARACUDA_KERNELS_ROLL_INSTANTIATE(NAME, T)                                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int32_t* shifts,                                                                    \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (shape == nullptr || shifts == nullptr ||                                              \
            stride_x == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_roll<T>(                                             \
            static_cast<const T*>(x), static_cast<T*>(y), numel, rank,                            \
            shape, shifts, stride_x, stride_y, stream);                                           \
    }

// Emit one ArgReduce-axis launcher.
//
// NAME    : symbol body — e.g. `arg_reduce_argmax_f32` (i64 output) or
//           `arg_reduce_argmax_f32_u32` (u32 output).
// T       : value (input) element type.
// FUNCTOR : ArgmaxPolicy<T> / ArgminPolicy<T>.
// OUT_I   : output index dtype — `int64_t` (default), `uint32_t`, or
//           `int32_t`. Phase 12.2 generalized this from a hard-coded
//           `int64_t*` to support Fuel's preferred `u32` output dtype.
#define BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(NAME, T, FUNCTOR, OUT_I)                     \
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
        return baracuda::elementwise::launch_arg_reduce_axis<T, FUNCTOR, OUT_I>(                  \
            static_cast<const T*>(x), static_cast<OUT_I*>(y), output_numel, rank,                 \
            output_shape, stride_x, stride_y,                                                     \
            reduce_axis, reduce_extent, reduce_stride_x, stream);                                 \
    }

// Emit one Permute launcher.
//
// NAME : symbol body — e.g. `permute_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_PERMUTE_INSTANTIATE(NAME, T)                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t input_numel,                                                                       \
        int32_t rank,                                                                              \
        const int32_t* input_shape,                                                               \
        const int32_t* dims,                                                                      \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (input_numel < 0) return 2;                                                            \
        if (input_numel == 0) return 0;                                                           \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (input_shape == nullptr || dims == nullptr || stride_x == nullptr ||                   \
            stride_y == nullptr) return 2;                                                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_permute<T>(                                          \
            static_cast<const T*>(x), static_cast<T*>(y), input_numel, rank,                      \
            input_shape, dims, stride_x, stride_y, stream);                                       \
    }

// Emit one Concat-2-input launcher.
//
// NAME : symbol body — e.g. `concat2_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_CONCAT2_INSTANTIATE(NAME, T)                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* output_shape,                                                              \
        int32_t concat_dim,                                                                       \
        int32_t split_offset,                                                                     \
        const int64_t* stride_a,                                                                  \
        const int64_t* stride_b,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* a, const void* b, void* y,                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (a == nullptr || b == nullptr || y == nullptr) return 2;                               \
        if (output_shape == nullptr || stride_a == nullptr || stride_b == nullptr ||              \
            stride_y == nullptr) return 2;                                                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_concat2<T>(                                          \
            static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(y),               \
            output_numel, rank, output_shape, concat_dim, split_offset,                           \
            stride_a, stride_b, stride_y, stream);                                                \
    }

// Emit one Pad-constant launcher.
//
// NAME : symbol body — e.g. `pad_constant_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_PAD_CONSTANT_INSTANTIATE(NAME, T)                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* input_shape,                                                               \
        const int32_t* output_shape,                                                              \
        const int32_t* pad_low,                                                                   \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        T value,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (input_shape == nullptr || output_shape == nullptr || pad_low == nullptr ||            \
            stride_x == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_pad_constant<T>(                                     \
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                     \
            input_shape, output_shape, pad_low, stride_x, stride_y,                               \
            value, stream);                                                                       \
    }

// Emit one Pad-reflect launcher. No `value` parameter — pad-region
// values come from the reflected input.
//
// NAME : symbol body — e.g. `pad_reflect_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_PAD_REFLECT_INSTANTIATE(NAME, T)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* input_shape,                                                               \
        const int32_t* output_shape,                                                              \
        const int32_t* pad_low,                                                                   \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (input_shape == nullptr || output_shape == nullptr || pad_low == nullptr ||            \
            stride_x == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_pad_reflect<T>(                                      \
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                     \
            input_shape, output_shape, pad_low, stride_x, stride_y, stream);                      \
    }

// Emit one Pad-replicate launcher. No `value` parameter — pad-region
// values are the clamped (replicated) edge of the input.
#define BARACUDA_KERNELS_PAD_REPLICATE_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* input_shape,                                                               \
        const int32_t* output_shape,                                                              \
        const int32_t* pad_low,                                                                   \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (input_shape == nullptr || output_shape == nullptr || pad_low == nullptr ||            \
            stride_x == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_pad_replicate<T>(                                    \
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                     \
            input_shape, output_shape, pad_low, stride_x, stride_y, stream);                      \
    }

// Emit one Pad-circular launcher. No `value` parameter — pad-region
// values are cyclic wraps from the opposite end of each axis.
#define BARACUDA_KERNELS_PAD_CIRCULAR_INSTANTIATE(NAME, T)                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* input_shape,                                                               \
        const int32_t* output_shape,                                                              \
        const int32_t* pad_low,                                                                   \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (input_shape == nullptr || output_shape == nullptr || pad_low == nullptr ||            \
            stride_x == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_pad_circular<T>(                                     \
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                     \
            input_shape, output_shape, pad_low, stride_x, stride_y, stream);                      \
    }

// Emit one Pad-constant backward (slice) launcher.
//
// `dx = dy[pad_low : pad_low + input_shape]` — pure slice, no math.
// Iterates `input_numel` (dx-coord space). NAME: e.g.
// `pad_constant_backward_f32`. T: scalar element type.
#define BARACUDA_KERNELS_PAD_CONSTANT_BACKWARD_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t input_numel,                                                                       \
        int32_t rank,                                                                              \
        const int32_t* input_shape,                                                               \
        const int32_t* pad_low,                                                                   \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_dx,                                                                 \
        const void* dy, void* dx,                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (input_numel < 0) return 2;                                                            \
        if (input_numel == 0) return 0;                                                           \
        if (dy == nullptr || dx == nullptr) return 2;                                             \
        if (input_shape == nullptr || pad_low == nullptr ||                                       \
            stride_dy == nullptr || stride_dx == nullptr) return 2;                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_pad_constant_backward<T>(                            \
            static_cast<const T*>(dy), static_cast<T*>(dx), input_numel, rank,                    \
            input_shape, pad_low, stride_dy, stride_dx, stream);                                  \
    }

// Emit one Repeat backward (gather-adjoint sum) launcher.
//
// `dx[c_in] = sum_{k} dy[c_in + k * input_shape]` per axis; one thread
// per dx cell loops the repeats grid. NAME: e.g. `repeat_backward_f32`.
// T: scalar element type. f16 / bf16 accumulate in float; f32 / f64 in
// their native dtype.
#define BARACUDA_KERNELS_REPEAT_BACKWARD_INSTANTIATE(NAME, T)                                     \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t input_numel,                                                                       \
        int32_t rank,                                                                              \
        const int32_t* input_shape,                                                               \
        const int32_t* repeats,                                                                   \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_dx,                                                                 \
        const void* dy, void* dx,                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (input_numel < 0) return 2;                                                            \
        if (input_numel == 0) return 0;                                                           \
        if (dy == nullptr || dx == nullptr) return 2;                                             \
        if (input_shape == nullptr || repeats == nullptr ||                                       \
            stride_dy == nullptr || stride_dx == nullptr) return 2;                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_repeat_backward<T>(                                  \
            static_cast<const T*>(dy), static_cast<T*>(dx), input_numel, rank,                    \
            input_shape, repeats, stride_dy, stride_dx, stream);                                  \
    }

// Emit one Concat2 backward (slice-split) launcher.
//
// Backward of `y = cat(a, b, dim=k)`: pure inverse routing. Every dy
// cell maps to exactly one of `da` or `db`. Bit-exact across every
// wired dtype — no arithmetic. Iterates `output_numel` (= dy.numel()).
// NAME : symbol body — e.g. `concat2_backward_f32`. T: scalar element type.
#define BARACUDA_KERNELS_CONCAT2_BACKWARD_INSTANTIATE(NAME, T)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* output_shape,                                                              \
        int32_t concat_dim,                                                                       \
        int32_t split_offset,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_da,                                                                 \
        const int64_t* stride_db,                                                                 \
        const void* dy, void* da, void* db,                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (dy == nullptr || da == nullptr || db == nullptr) return 2;                            \
        if (output_shape == nullptr || stride_dy == nullptr ||                                    \
            stride_da == nullptr || stride_db == nullptr) return 2;                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_concat2_backward<T>(                                 \
            static_cast<const T*>(dy), static_cast<T*>(da), static_cast<T*>(db),                  \
            output_numel, rank, output_shape, concat_dim, split_offset,                           \
            stride_dy, stride_da, stride_db, stream);                                             \
    }

// Emit one Reduce-axis launcher.
//
// NAME    : symbol body — e.g. `reduce_sum_f32`.
// T       : scalar element type.
// FUNCTOR : reduce functor type (with static `T init()` + `T operator()(T, T)`).
#define BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(NAME, T, FUNCTOR)                                \
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
        return baracuda::elementwise::launch_reduce_axis<T, FUNCTOR>(                             \
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                     \
            output_shape, stride_x, stride_y,                                                     \
            reduce_axis, reduce_extent, reduce_stride_x,                                          \
            stream);                                                                              \
    }

// Emit one heterogeneous-output reduce-axis launcher for Any (output: uint8_t Bool).
//
// NAME  : symbol body — e.g. `reduce_any_f32`.
// T_IN  : input scalar element type.
// FUNCTOR : reduce functor with static `uint8_t init()` and
//           `uint8_t operator()(uint8_t acc, T_IN x) const`.
#define BARACUDA_KERNELS_REDUCE_ANY_INSTANTIATE(NAME, T_IN, FUNCTOR)                              \
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
        return baracuda::elementwise::launch_reduce_axis_hetero<T_IN, uint8_t, FUNCTOR>(          \
            static_cast<const T_IN*>(x), static_cast<uint8_t*>(y), output_numel, rank,            \
            output_shape, stride_x, stride_y,                                                     \
            reduce_axis, reduce_extent, reduce_stride_x,                                          \
            stream);                                                                              \
    }

// Emit one heterogeneous-output reduce-axis launcher for All (output: uint8_t Bool).
//
// Same parameter shape as REDUCE_ANY_INSTANTIATE. Macro exists for
// symmetry / readability — the macro body is identical to ANY (they
// only differ in the functor that the caller supplies).
#define BARACUDA_KERNELS_REDUCE_ALL_INSTANTIATE(NAME, T_IN, FUNCTOR)                              \
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
        return baracuda::elementwise::launch_reduce_axis_hetero<T_IN, uint8_t, FUNCTOR>(          \
            static_cast<const T_IN*>(x), static_cast<uint8_t*>(y), output_numel, rank,            \
            output_shape, stride_x, stride_y,                                                     \
            reduce_axis, reduce_extent, reduce_stride_x,                                          \
            stream);                                                                              \
    }

// Emit one heterogeneous-output reduce-axis launcher for CountNonzero
// (output: int64_t — PyTorch `torch.count_nonzero` returns int64).
//
// NAME    : symbol body — e.g. `reduce_count_nonzero_f32`.
// T_IN    : input scalar element type.
// FUNCTOR : reduce functor with static `int64_t init()` and
//           `int64_t operator()(int64_t acc, T_IN x) const`.
#define BARACUDA_KERNELS_REDUCE_COUNT_NONZERO_INSTANTIATE(NAME, T_IN, FUNCTOR)                    \
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
        return baracuda::elementwise::launch_reduce_axis_hetero<T_IN, int64_t, FUNCTOR>(          \
            static_cast<const T_IN*>(x), static_cast<int64_t*>(y), output_numel, rank,            \
            output_shape, stride_x, stride_y,                                                     \
            reduce_axis, reduce_extent, reduce_stride_x,                                          \
            stream);                                                                              \
    }

// Emit one scan-axis launcher.
//
// Like REDUCE_AXIS but the scan axis is *length-preserving* (output
// shape == input shape) and the kernel takes a `reverse` flag.
//
// NAME    : symbol body — e.g. `scan_cumsum_f32`.
// T       : scalar element type.
// FUNCTOR : functor with `init()`, `op(acc, x)`, `finalize(acc, ext)`.
#define BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(NAME, T, FUNCTOR)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        int32_t scan_axis,                                                                        \
        int32_t scan_extent,                                                                      \
        int64_t scan_stride_x,                                                                    \
        int32_t reverse,                                                                          \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_scan_axis<T, FUNCTOR>(                               \
            static_cast<const T*>(x), static_cast<T*>(y), numel, rank,                            \
            shape, stride_x, stride_y,                                                            \
            scan_axis, scan_extent, scan_stride_x, reverse,                                       \
            stream);                                                                              \
    }

// Emit one scan-cumprod backward launcher.
//
// NAME : symbol body — e.g. `scan_cumprod_backward_f32`.
// T    : storage element type.
// ACC  : accumulator type (T itself for f32/f64; float for f16/bf16).
#define BARACUDA_KERNELS_SCAN_CUMPROD_BACKWARD_INSTANTIATE(NAME, T, ACC)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        int32_t scan_axis,                                                                        \
        int32_t scan_extent,                                                                      \
        int32_t reverse,                                                                          \
        const void* dy, const void* x, const void* y, void* dx,                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || y == nullptr || dx == nullptr) return 2;             \
        if (shape == nullptr || stride_dy == nullptr || stride_x == nullptr ||                    \
            stride_y == nullptr || stride_dx == nullptr) return 2;                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_scan_cumprod_backward<T, ACC>(                       \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<const T*>(y), static_cast<T*>(dx),                                        \
            numel, rank, shape, stride_dy, stride_x, stride_y, stride_dx,                         \
            scan_axis, scan_extent, reverse, stream);                                             \
    }

// Emit one scan-cummax-or-cummin backward launcher.
//
// NAME   : symbol body — e.g. `scan_cummax_backward_f32`.
// T      : storage element type.
// ACC    : accumulator type.
// IS_MAX : `true` for cummax, `false` for cummin.
#define BARACUDA_KERNELS_SCAN_CUMMAX_MIN_BACKWARD_INSTANTIATE(NAME, T, ACC, IS_MAX)              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_dx,                                                                 \
        int32_t scan_axis,                                                                        \
        int32_t scan_extent,                                                                      \
        int32_t reverse,                                                                          \
        const void* dy, const void* x, void* dx,                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || dx == nullptr) return 2;                             \
        if (shape == nullptr || stride_dy == nullptr || stride_x == nullptr ||                    \
            stride_dx == nullptr) return 2;                                                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_scan_cummax_min_backward<T, ACC, IS_MAX>(            \
            static_cast<const T*>(dy), static_cast<const T*>(x), static_cast<T*>(dx),             \
            numel, rank, shape, stride_dy, stride_x, stride_dx,                                   \
            scan_axis, scan_extent, reverse, stream);                                             \
    }

// Emit one log-cumsum-exp FW launcher. ABI mirrors
// `BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE` exactly so the Rust dispatcher
// can reach it through the same shape.
//
// NAME : symbol body — e.g. `scan_log_cumsum_exp_f32`.
// T    : storage element type (compute is f32 for f16 / bf16, T for
//        f32 / f64 via the LogCumsumExpDtype trait).
#define BARACUDA_KERNELS_LOG_CUMSUM_EXP_INSTANTIATE(NAME, T)                                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        int32_t scan_axis,                                                                        \
        int32_t scan_extent,                                                                      \
        int64_t scan_stride_x,                                                                    \
        int32_t reverse,                                                                          \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_log_cumsum_exp<T>(                                   \
            static_cast<const T*>(x), static_cast<T*>(y), numel, rank,                            \
            shape, stride_x, stride_y,                                                            \
            scan_axis, scan_extent, scan_stride_x, reverse,                                       \
            stream);                                                                              \
    }

// Emit one log-cumsum-exp BW launcher. ABI mirrors
// `BARACUDA_KERNELS_SCAN_CUMPROD_BACKWARD_INSTANTIATE` (also needs both
// saved x and saved y).
//
// NAME : symbol body — e.g. `scan_log_cumsum_exp_backward_f32`.
// T    : storage element type.
#define BARACUDA_KERNELS_LOG_CUMSUM_EXP_BACKWARD_INSTANTIATE(NAME, T)                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        int32_t scan_axis,                                                                        \
        int32_t scan_extent,                                                                      \
        int32_t reverse,                                                                          \
        const void* dy, const void* x, const void* y, void* dx,                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || y == nullptr || dx == nullptr) return 2;             \
        if (shape == nullptr || stride_dy == nullptr || stride_x == nullptr ||                    \
            stride_y == nullptr || stride_dx == nullptr) return 2;                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_log_cumsum_exp_backward<T>(                          \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<const T*>(y), static_cast<T*>(dx),                                        \
            numel, rank, shape, stride_dy, stride_x, stride_y, stride_dx,                         \
            scan_axis, scan_extent, reverse, stream);                                             \
    }

// Emit one scaled ternary pointwise contig launcher pair.
//
// Companion to BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE for ops
// that take an f32 scalar parameter (Addcmul, Addcdiv). The functor
// is 4-arg: `T operator()(T a, T b, T c, float scale)`.
#define BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(NAME, T, FUNCTOR)               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int64_t numel,                                                                          \
        const void* a, const void* b, const void* c, void* y,                                  \
        float scale,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                       \
    {                                                                                           \
        if (numel < 0) return 2;                                                               \
        if (numel == 0) return 0;                                                              \
        if (a == nullptr || b == nullptr || c == nullptr || y == nullptr) return 2;            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::elementwise::launch_ternary_scaled_pointwise_contig<T, FUNCTOR>(      \
            static_cast<const T*>(a), static_cast<const T*>(b),                                \
            static_cast<const T*>(c), static_cast<T*>(y),                                       \
            numel, scale, stream);                                                              \
    }                                                                                           \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int64_t numel,                                                                          \
        const void* /*a*/, const void* /*b*/, const void* /*c*/, const void* /*y*/)            \
    {                                                                                           \
        if (numel < 0) return 2;                                                               \
        return 0;                                                                               \
    }

// Emit one scaled ternary pointwise strided launcher.
#define BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(NAME, T, FUNCTOR)        \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                  \
        int64_t numel,                                                                          \
        int32_t rank,                                                                           \
        const int32_t* shape,                                                                  \
        const int64_t* stride_a,                                                               \
        const int64_t* stride_b,                                                               \
        const int64_t* stride_c,                                                               \
        const int64_t* stride_y,                                                               \
        const void* a, const void* b, const void* c, void* y,                                  \
        float scale,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                       \
    {                                                                                           \
        if (numel < 0) return 2;                                                               \
        if (numel == 0) return 0;                                                              \
        if (a == nullptr || b == nullptr || c == nullptr || y == nullptr) return 2;            \
        if (shape == nullptr || stride_a == nullptr || stride_b == nullptr ||                  \
            stride_c == nullptr || stride_y == nullptr) return 2;                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::elementwise::launch_ternary_scaled_pointwise_strided<T, FUNCTOR>(     \
            static_cast<const T*>(a), static_cast<const T*>(b),                                \
            static_cast<const T*>(c), static_cast<T*>(y),                                       \
            numel, rank, shape, stride_a, stride_b, stride_c, stride_y, scale, stream);        \
    }

// Emit one ternary backward launcher (Fma / Clamp family — no scale).
//
// Functor signature:
//   `__device__ void operator()(T dy, T a, T b, T c,
//                               T& da, T& db, T& dc) const`.
//
// NAME    : symbol body — e.g. `ternary_fma_backward_f32`.
// T       : scalar element type.
// FUNCTOR : functor with the unscaled BW formula above.
#define BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(NAME, T, FUNCTOR)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int64_t numel,                                                                          \
        const void* dy, const void* a, const void* b, const void* c,                           \
        void* da, void* db, void* dc,                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                       \
    {                                                                                           \
        if (numel < 0) return 2;                                                               \
        if (numel == 0) return 0;                                                              \
        if (dy == nullptr || a == nullptr || b == nullptr || c == nullptr ||                   \
            da == nullptr || db == nullptr || dc == nullptr) return 2;                         \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::elementwise::launch_ternary_backward<T, FUNCTOR>(                     \
            static_cast<const T*>(dy),                                                         \
            static_cast<const T*>(a), static_cast<const T*>(b), static_cast<const T*>(c),      \
            static_cast<T*>(da), static_cast<T*>(db), static_cast<T*>(dc),                     \
            numel, stream);                                                                    \
    }

// Emit one scaled ternary backward launcher (Addcmul / Addcdiv family —
// reads an f32 `scale` between `dc` and the workspace pointer, mirroring
// the FW scaled-ternary ABI).
//
// Functor signature:
//   `__device__ void operator()(T dy, T a, T b, T c, float scale,
//                               T& da, T& db, T& dc) const`.
#define BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(NAME, T, FUNCTOR)                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int64_t numel,                                                                          \
        const void* dy, const void* a, const void* b, const void* c,                           \
        void* da, void* db, void* dc,                                                          \
        float scale,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                       \
    {                                                                                           \
        if (numel < 0) return 2;                                                               \
        if (numel == 0) return 0;                                                              \
        if (dy == nullptr || a == nullptr || b == nullptr || c == nullptr ||                   \
            da == nullptr || db == nullptr || dc == nullptr) return 2;                         \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::elementwise::launch_ternary_backward_scaled<T, FUNCTOR>(              \
            static_cast<const T*>(dy),                                                         \
            static_cast<const T*>(a), static_cast<const T*>(b), static_cast<const T*>(c),      \
            static_cast<T*>(da), static_cast<T*>(db), static_cast<T*>(dc),                     \
            numel, scale, stream);                                                              \
    }

// Emit one where (heterogeneous-cond ternary) launcher pair.
//
// NAME  : symbol body — e.g. `where_f32`.
// T     : value scalar type (cond is always `uint8_t`).
#define BARACUDA_KERNELS_WHERE_INSTANTIATE(NAME, T)                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* cond,                                                                          \
        const void* a, const void* b, void* y,                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (cond == nullptr || a == nullptr || b == nullptr || y == nullptr) return 2;            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_where_pointwise_contig<T>(                           \
            static_cast<const uint8_t*>(cond),                                                    \
            static_cast<const T*>(a),                                                             \
            static_cast<const T*>(b),                                                              \
            static_cast<T*>(y),                                                                    \
            numel, stream);                                                                        \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int64_t numel,                                                                             \
        const void* /*cond*/, const void* /*a*/, const void* /*b*/, const void* /*y*/)            \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_WHERE_INSTANTIATE_STRIDED(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                     \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_cond,                                                               \
        const int64_t* stride_a,                                                                  \
        const int64_t* stride_b,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* cond,                                                                          \
        const void* a, const void* b, void* y,                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (cond == nullptr || a == nullptr || b == nullptr || y == nullptr) return 2;            \
        if (shape == nullptr || stride_cond == nullptr || stride_a == nullptr ||                  \
            stride_b == nullptr || stride_y == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_where_pointwise_strided<T>(                          \
            static_cast<const uint8_t*>(cond),                                                    \
            static_cast<const T*>(a),                                                             \
            static_cast<const T*>(b),                                                              \
            static_cast<T*>(y),                                                                    \
            numel, rank, shape, stride_cond, stride_a, stride_b, stride_y, stream);               \
    }

// Emit one where-backward (heterogeneous-cond ternary BW) launcher.
//
// FW: `y = cond ? a : b`. BW (cond non-differentiable):
//   da = cond ? dy : 0,  db = cond ? 0 : dy.
// Trailblazer is contig-only — no broadcast path.
//
// NAME : symbol body — e.g. `where_backward_f32`.
// T    : value scalar type (cond is always `uint8_t`).
#define BARACUDA_KERNELS_WHERE_BACKWARD_INSTANTIATE(NAME, T)                                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* cond,                                                                          \
        const void* dy,                                                                            \
        void* da, void* db,                                                                        \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (cond == nullptr || dy == nullptr || da == nullptr || db == nullptr) return 2;         \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_where_backward_pointwise_contig<T>(                  \
            static_cast<const uint8_t*>(cond),                                                    \
            static_cast<const T*>(dy),                                                            \
            static_cast<T*>(da),                                                                  \
            static_cast<T*>(db),                                                                  \
            numel, stream);                                                                       \
    }

// Emit one binary comparison contig launcher pair.
//
// Output is fixed to `uint8_t` (0 = false, 1 = true) — the functor
// returns `uint8_t`.
//
// NAME    : symbol body — e.g. `binary_cmp_eq_f32`.
// T       : input scalar element type.
// FUNCTOR : binary functor type with `__device__ uint8_t operator()(T, T)`.
#define BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(NAME, T, FUNCTOR)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* a, const void* b, void* y,                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (a == nullptr || b == nullptr || y == nullptr) return 2;                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_binary_cmp_pointwise_contig<T, FUNCTOR>(             \
            static_cast<const T*>(a),                                                             \
            static_cast<const T*>(b),                                                              \
            static_cast<uint8_t*>(y),                                                              \
            numel, stream);                                                                        \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int64_t numel,                                                                             \
        const void* /*a*/, const void* /*b*/, const void* /*y*/)                                  \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        return 0;                                                                                  \
    }

// Emit one binary comparison strided launcher.
#define BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(NAME, T, FUNCTOR)               \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                     \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_a,                                                                  \
        const int64_t* stride_b,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* a, const void* b, void* y,                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (a == nullptr || b == nullptr || y == nullptr) return 2;                               \
        if (shape == nullptr || stride_a == nullptr || stride_b == nullptr ||                     \
            stride_y == nullptr) return 2;                                                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_binary_cmp_pointwise_strided<T, FUNCTOR>(            \
            static_cast<const T*>(a),                                                             \
            static_cast<const T*>(b),                                                              \
            static_cast<uint8_t*>(y),                                                              \
            numel, rank, shape, stride_a, stride_b, stride_y, stream);                            \
    }

// Emit one unary contig pointwise launcher pair.
//
// Sibling to BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE for unary
// (1→1) ops — `y = f(x)` over fully contiguous tensors.
//
// NAME    : symbol body — e.g. `unary_neg_f32`.
// T       : scalar element type.
// FUNCTOR : unary functor type with `__device__ T operator()(T)`.
#define BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(NAME, T, FUNCTOR)                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_unary_pointwise_contig<T, FUNCTOR>(                  \
            static_cast<const T*>(x), static_cast<T*>(y), numel, stream);                         \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int64_t numel,                                                                             \
        const void* /*x*/, const void* /*y*/)                                                     \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        return 0;                                                                                  \
    }

// Emit one unary strided pointwise launcher.
//
// Sibling to BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED for
// unary (1→1) ops. Handles non-contig input / output views (transposed,
// sliced). Input shape must equal output shape — broadcast is not a
// meaningful unary semantic.
#define BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(NAME, T, FUNCTOR)                    \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                     \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_unary_pointwise_strided<T, FUNCTOR>(                 \
            static_cast<const T*>(x), static_cast<T*>(y),                                         \
            numel, rank, shape, stride_x, stride_y, stream);                                      \
    }

// Emit one gated-activation contig launcher.
//
// Computes `y = a · gate(b)` where `(a, b)` are the two halves of input
// `x` along `split_dim`. One thread per output cell. ABI:
//   `(output_numel, rank, output_shape, split_dim, x_half_offset,
//     stride_x, stride_y, x, y, ws, ws_bytes, stream) -> int32`
//
// NAME    : symbol body — e.g. `gated_swiglu_f32`.
// T       : scalar element type.
// FUNCTOR : functor with `__device__ T operator()(T a, T b) const`.
#define BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(NAME, T, FUNCTOR)                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* output_shape,                                                              \
        int32_t split_dim,                                                                        \
        int64_t x_half_offset,                                                                    \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (output_shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_gated_activation_contig<T, FUNCTOR>(                 \
            static_cast<const T*>(x), static_cast<T*>(y),                                         \
            output_numel, rank, output_shape, split_dim, x_half_offset,                           \
            stride_x, stride_y, stream);                                                          \
    }

// Emit one gated-activation backward contig launcher.
//
// Computes `dx[a_half] = dy · gate(b)`, `dx[b_half] = dy · a · gate'(b)`.
// ABI:
//   `(output_numel, rank, output_shape, split_dim,
//     x_half_offset, dx_half_offset,
//     stride_x, stride_dy, stride_dx,
//     x, dy, dx, ws, ws_bytes, stream) -> int32`
//
// NAME    : symbol body — e.g. `gated_swiglu_backward_f32`.
// T       : scalar element type.
// FUNCTOR : functor with
//   `__device__ void operator()(T dy, T a, T b, T& da_out, T& db_out) const`.
#define BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(NAME, T, FUNCTOR)                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t output_numel,                                                                      \
        int32_t rank,                                                                              \
        const int32_t* output_shape,                                                              \
        int32_t split_dim,                                                                        \
        int64_t x_half_offset,                                                                    \
        int64_t dx_half_offset,                                                                   \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_dx,                                                                 \
        const void* x, const void* dy, void* dx,                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (output_numel < 0) return 2;                                                           \
        if (output_numel == 0) return 0;                                                          \
        if (x == nullptr || dy == nullptr || dx == nullptr) return 2;                             \
        if (output_shape == nullptr || stride_x == nullptr ||                                     \
            stride_dy == nullptr || stride_dx == nullptr) return 2;                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_gated_activation_backward_contig<T, FUNCTOR>(        \
            static_cast<const T*>(x), static_cast<const T*>(dy), static_cast<T*>(dx),             \
            output_numel, rank, output_shape, split_dim,                                          \
            x_half_offset, dx_half_offset,                                                        \
            stride_x, stride_dy, stride_dx, stream);                                              \
    }

// Emit one parameterized unary pointwise launcher. Sibling of
// `UNARY_POINTWISE_INSTANTIATE` with two extra `float` parameters threaded
// through to the functor. Contig only (no strided variant) — the
// trailblazer scope; future param-bearing ops re-emit through this same
// macro.
//
// ABI:
//   `(numel, x, y, p0, p1, ws, ws_bytes, stream) -> int32`
//
// NAME    : symbol body — e.g. `unary_threshold_f32`.
// T       : scalar element type.
// FUNCTOR : functor with `__device__ T operator()(T x, float p0, float p1) const`.
#define BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(NAME, T, FUNCTOR)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* x, void* y,                                                                   \
        float p0, float p1,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_unary_param_pointwise_contig<T, FUNCTOR>(            \
            static_cast<const T*>(x), static_cast<T*>(y), numel, p0, p1, stream);                 \
    }

// Emit one strided parameterized unary launcher (Phase 14.2).
//
// Companion to BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE that handles
// non-contig input / output views. The Rust dispatcher picks contig vs
// strided based on `is_contiguous()` of both operands at launch time;
// both launchers are emitted per (op, dtype) cell.
//
// ABI:
//   `(numel, rank, shape, stride_x, stride_y, x, y, p0, p1,
//     ws, ws_bytes, stream) -> int32`
//
// NAME    : symbol body — e.g. `unary_powi_f32` (joins with `_strided_run`).
// T       : scalar element type.
// FUNCTOR : functor with `__device__ T operator()(T x, float p0, float p1) const`.
#define BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE_STRIDED(NAME, T, FUNCTOR)                        \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                     \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        const void* x, void* y,                                                                   \
        float p0, float p1,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_unary_param_pointwise_strided<T, FUNCTOR>(           \
            static_cast<const T*>(x), static_cast<T*>(y),                                         \
            numel, rank, shape, stride_x, stride_y, p0, p1, stream);                              \
    }

// Emit one parameterized unary backward launcher. The kernel takes the
// saved forward input `x` plus two scalar parameters and writes `dx`.
//
// ABI:
//   `(numel, dy, x, dx, p0, p1, ws, ws_bytes, stream) -> int32`
//
// NAME    : symbol body — e.g. `unary_threshold_backward_f32`.
// T       : scalar element type.
// FUNCTOR : functor with `__device__ T operator()(T dy, T x, float p0, float p1) const`.
#define BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(NAME, T, FUNCTOR)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* dy, const void* x, void* dx,                                                  \
        float p0, float p1,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || dx == nullptr) return 2;                             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_unary_param_backward<T, FUNCTOR>(                    \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<T*>(dx), numel, p0, p1, stream);                                          \
    }

// Emit one strided parameterized unary backward launcher (Phase 14.2).
//
// Companion to BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE for
// non-contig views. Carries three independent stride arrays — `dy`, `x`,
// and `dx` may each be strided differently.
//
// ABI:
//   `(numel, rank, shape, stride_x, stride_dy, stride_dx,
//     x, dy, dx, p0, p1, ws, ws_bytes, stream) -> int32`
//
// NAME    : symbol body — e.g. `unary_powi_backward_f32`.
// T       : scalar element type.
// FUNCTOR : functor with
//   `__device__ T operator()(T dy, T x, float p0, float p1) const`.
#define BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE_STRIDED(NAME, T, FUNCTOR)                \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                     \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_dx,                                                                 \
        const void* x, const void* dy, void* dx,                                                  \
        float p0, float p1,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || x == nullptr || dx == nullptr) return 2;                             \
        if (shape == nullptr || stride_x == nullptr || stride_dy == nullptr ||                    \
            stride_dx == nullptr) return 2;                                                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_unary_param_backward_strided<T, FUNCTOR>(            \
            static_cast<const T*>(dy), static_cast<const T*>(x),                                  \
            static_cast<T*>(dx), numel, rank, shape,                                              \
            stride_x, stride_dy, stride_dx, p0, p1, stream);                                      \
    }

// Emit one parameterized binary pointwise launcher. Sibling of
// `BINARY_POINTWISE_INSTANTIATE` with one extra `float` parameter
// threaded through to the functor. Contig only.
//
// ABI:
//   `(numel, a, b, y, p, ws, ws_bytes, stream) -> int32`
//
// NAME    : symbol body — e.g. `binary_lerp_f32`.
// T       : scalar element type.
// FUNCTOR : functor with `__device__ T operator()(T a, T b, float p) const`.
#define BARACUDA_KERNELS_BINARY_PARAM_INSTANTIATE(NAME, T, FUNCTOR)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* a, const void* b, void* y,                                                    \
        float p,                                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (a == nullptr || b == nullptr || y == nullptr) return 2;                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_binary_param_pointwise_contig<T, FUNCTOR>(           \
            static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(y),               \
            numel, p, stream);                                                                    \
    }

// Emit one parameterized binary backward launcher (no-save variant — the
// gradient is a pure function of `dy` and the scalar param). The functor
// writes both `da` and `db`.
//
// ABI:
//   `(numel, dy, da, db, p, ws, ws_bytes, stream) -> int32`
//
// NAME    : symbol body — e.g. `binary_lerp_backward_f32`.
// T       : scalar element type.
// FUNCTOR : functor with
//   `__device__ void operator()(T dy, float p, T& da, T& db) const`.
#define BARACUDA_KERNELS_BINARY_PARAM_BACKWARD_INSTANTIATE(NAME, T, FUNCTOR)                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* dy, void* da, void* db,                                                       \
        float p,                                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || da == nullptr || db == nullptr) return 2;                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::elementwise::launch_binary_param_backward_nosave<T, FUNCTOR>(            \
            static_cast<const T*>(dy), static_cast<T*>(da), static_cast<T*>(db),                  \
            numel, p, stream);                                                                    \
    }

#endif // BARACUDA_ELEMENTWISE_CUH
