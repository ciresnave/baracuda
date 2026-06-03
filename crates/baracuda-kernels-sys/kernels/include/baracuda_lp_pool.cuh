// baracuda_lp_pool.cuh
//
// Phase 16.2 — LpPool 1d/2d fused bespoke kernels.
//
// PyTorch's `nn.LPPool{1,2}d(p, kernel, stride, ceil_mode)` computes
// the per-window p-norm:
//
//     y[..., i] = (Σ_{k ∈ window(i)} |x[..., i*s + k]|^p)^(1/p)
//
// PyTorch's LPPool has NO padding (in contrast to AvgPool/MaxPool which
// take an explicit `pad` argument). In `ceil_mode` the output extents
// round up — windows whose stride origin lies inside `in` but whose
// trailing edge would overhang the input boundary are *truncated*
// (window clamps to `[start, min(start + kernel, in))`); windows that
// fully overhang are not produced.
//
// Output extent formula:
//   ceil_mode = false: out = floor((in - kernel) / stride) + 1
//   ceil_mode = true : out = ceil ((in - kernel) / stride) + 1
//
// Why a fused kernel: stacking `pow(|x|, p) → avg_pool → pow(·, 1/p)`
// would require a parameterized `Pow(p)` unary plan that doesn't exist
// today (Phase 12's PowI takes integer exponents only) and would pay 3×
// the launch overhead. The fused path also fixes the AvgPool semantics
// gap (LpPool has no divisor — it's a raw sum-of-abs-pow, not an
// average — so multiplying by `kernel_vol` afterward would force an
// extra elementwise pass).
//
// Backward math:
//   y = (Σ_k |x_k|^p)^(1/p)
//   ∂y/∂x_k = |x_k|^(p-1) · sgn(x_k) · y^(1-p)
//   dx_k = dy · |x_k|^(p-1) · sgn(x_k) · y^(1-p)
//
// Edge cases:
//   * y == 0 (all input cells in window are zero): gradient is zero
//     (matches PyTorch convention — would be 0/0 otherwise).
//   * x_k == 0 with p > 1: |x_k|^(p-1) == 0, gradient contribution is
//     zero (well-defined).
//   * x_k == 0 with p == 1: |x_k|^0 == 1 but sgn(0) == 0, so the
//     gradient contribution is zero (well-defined; matches `dy *
//     sgn(x_k)` general formula).
//   * x_k == 0 with p < 1: |x_k|^(p-1) is +inf — we clamp to zero per
//     PyTorch convention.
//   * NaN x_k anywhere in the window: y becomes NaN (correct propagation).
//
// Design:
//   * FW: one thread per output cell. Decompose linear idx into
//     [N, C, (H?), W] coords. For each spatial axis, compute window
//     bounds (start = i*stride; end = min(start + kernel, in)). Loop
//     over the window, accumulating Σ|x|^p in an f32 accumulator
//     (f64 path uses double). Compute y = pow(acc, 1/p). Half/bf16
//     output is downcast at store time.
//   * BW: one thread per *output* cell + atomicAdd scatter into dx
//     (matches the rest of pool BW). Loops over the same window the
//     FW used and `atomicAdd_via_cas` for half/bf16 via the existing
//     `baracuda::atomic::add` helper from `baracuda_atomic.cuh`.
//
// Status codes mirror the rest of the family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   5 internal launch error.

#ifndef BARACUDA_LP_POOL_CUH
#define BARACUDA_LP_POOL_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_atomic.cuh"

namespace baracuda { namespace lp_pool {

// =============================================================================
// Dtype helpers — uniform load / store + f32-detour accumulator for half/bf16.
// =============================================================================

template <typename T> struct AccOf { using type = float; };
template <> struct AccOf<double> { using type = double; };

template <typename T>
__device__ __forceinline__ typename AccOf<T>::type load_acc(const T* p) {
    return static_cast<typename AccOf<T>::type>(*p);
}
template <>
__device__ __forceinline__ float load_acc<__half>(const __half* p) {
    return __half2float(*p);
}
template <>
__device__ __forceinline__ float load_acc<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename T>
__device__ __forceinline__ T store_from_acc(typename AccOf<T>::type v) {
    return static_cast<T>(v);
}
template <>
__device__ __forceinline__ __half store_from_acc<__half>(float v) {
    return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 store_from_acc<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// `pow` overload in the accumulator type (float or double).
__device__ __forceinline__ float  acc_pow(float  base, float  exp_) { return __powf(base, exp_); }
__device__ __forceinline__ double acc_pow(double base, double exp_) { return pow(base, exp_); }
__device__ __forceinline__ float  acc_fabs(float  v) { return fabsf(v); }
__device__ __forceinline__ double acc_fabs(double v) { return fabs(v); }

// =============================================================================
// 1-D FW kernel
// =============================================================================

template <typename T>
__global__ void lp_pool1d_fw_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int32_t N, int32_t C, int32_t L_in, int32_t L_out,
    int32_t kernel, int32_t stride,
    float norm_p)
{
    using Acc = typename AccOf<T>::type;
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)L_out;
    if (tid >= total) return;

    int32_t ol = (int32_t)(tid % (int64_t)L_out);
    int64_t tmp = tid / (int64_t)L_out;
    int32_t ci = (int32_t)(tmp % (int64_t)C);
    int32_t ni = (int32_t)(tmp / (int64_t)C);

    int32_t l_start = ol * stride;
    int32_t l_end = l_start + kernel;
    if (l_end > L_in) l_end = L_in;

    const Acc p = static_cast<Acc>(norm_p);
    const Acc inv_p = (Acc)1 / p;

    Acc acc = (Acc)0;
    const T* base = x + ((int64_t)ni * C + ci) * (int64_t)L_in;
    for (int32_t l = l_start; l < l_end; ++l) {
        Acc v = load_acc<T>(base + l);
        Acc av = acc_fabs(v);
        acc += acc_pow(av, p);
    }
    Acc out = (acc == (Acc)0) ? (Acc)0 : acc_pow(acc, inv_p);
    y[tid] = store_from_acc<T>(out);
}

// =============================================================================
// 1-D BW kernel — atomicAdd scatter from each output cell.
// =============================================================================

template <typename T>
__global__ void lp_pool1d_bw_kernel(
    const T* __restrict__ x,
    const T* __restrict__ y,
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int32_t N, int32_t C, int32_t L_in, int32_t L_out,
    int32_t kernel, int32_t stride,
    float norm_p)
{
    using Acc = typename AccOf<T>::type;
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)L_out;
    if (tid >= total) return;

    int32_t ol = (int32_t)(tid % (int64_t)L_out);
    int64_t tmp = tid / (int64_t)L_out;
    int32_t ci = (int32_t)(tmp % (int64_t)C);
    int32_t ni = (int32_t)(tmp / (int64_t)C);

    int32_t l_start = ol * stride;
    int32_t l_end = l_start + kernel;
    if (l_end > L_in) l_end = L_in;

    Acc y_v  = load_acc<T>(y  + tid);
    Acc dy_v = load_acc<T>(dy + tid);

    // y == 0 → gradient zero (definition).
    if (y_v == (Acc)0) return;

    const Acc p = static_cast<Acc>(norm_p);
    // y^(1-p) — for the common p=1 case this is 1.0 exactly and the
    // multiplier simplifies. For p>1 it stays finite.
    Acc y_pow = acc_pow(y_v, (Acc)1 - p);
    Acc coeff = dy_v * y_pow;

    T*  base_dx = dx + ((int64_t)ni * C + ci) * (int64_t)L_in;
    const T* base_x = x + ((int64_t)ni * C + ci) * (int64_t)L_in;

    for (int32_t l = l_start; l < l_end; ++l) {
        Acc xv = load_acc<T>(base_x + l);
        Acc ax = acc_fabs(xv);
        // sgn(x): x>0 → 1, x<0 → -1, x==0 → 0.
        Acc sg = (xv > (Acc)0) ? (Acc)1 : ((xv < (Acc)0) ? (Acc)-1 : (Acc)0);
        if (sg == (Acc)0) continue;  // contribution exactly zero
        // For p<1 and ax==0, |x|^(p-1) → inf; sg==0 already filtered.
        Acc ax_pm1 = acc_pow(ax, p - (Acc)1);
        // Numerical-safety clamp: if NaN/inf slip through (shouldn't,
        // given sg==0 filter), drop the contribution.
        if (!isfinite((double)ax_pm1)) continue;
        Acc contrib = coeff * ax_pm1 * sg;
        T t = store_from_acc<T>(contrib);
        baracuda::atomic::add(base_dx + l, t);
    }
}

// =============================================================================
// 2-D FW kernel
// =============================================================================

template <typename T>
__global__ void lp_pool2d_fw_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int32_t N, int32_t C, int32_t H_in, int32_t W_in,
    int32_t H_out, int32_t W_out,
    int32_t kh, int32_t kw, int32_t sh, int32_t sw,
    float norm_p)
{
    using Acc = typename AccOf<T>::type;
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)H_out * (int64_t)W_out;
    if (tid >= total) return;

    int32_t ow = (int32_t)(tid % (int64_t)W_out);
    int64_t tmp = tid / (int64_t)W_out;
    int32_t oh = (int32_t)(tmp % (int64_t)H_out);
    tmp /= (int64_t)H_out;
    int32_t ci = (int32_t)(tmp % (int64_t)C);
    int32_t ni = (int32_t)(tmp / (int64_t)C);

    int32_t h_start = oh * sh;
    int32_t h_end = h_start + kh;
    if (h_end > H_in) h_end = H_in;
    int32_t w_start = ow * sw;
    int32_t w_end = w_start + kw;
    if (w_end > W_in) w_end = W_in;

    const Acc p = static_cast<Acc>(norm_p);
    const Acc inv_p = (Acc)1 / p;

    Acc acc = (Acc)0;
    const T* base = x + ((int64_t)ni * C + ci) * (int64_t)H_in * (int64_t)W_in;
    for (int32_t h = h_start; h < h_end; ++h) {
        const T* row = base + (int64_t)h * (int64_t)W_in;
        for (int32_t w = w_start; w < w_end; ++w) {
            Acc v = load_acc<T>(row + w);
            Acc av = acc_fabs(v);
            acc += acc_pow(av, p);
        }
    }
    Acc out = (acc == (Acc)0) ? (Acc)0 : acc_pow(acc, inv_p);
    y[tid] = store_from_acc<T>(out);
}

// =============================================================================
// 2-D BW kernel — atomicAdd scatter from each output cell.
// =============================================================================

template <typename T>
__global__ void lp_pool2d_bw_kernel(
    const T* __restrict__ x,
    const T* __restrict__ y,
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int32_t N, int32_t C, int32_t H_in, int32_t W_in,
    int32_t H_out, int32_t W_out,
    int32_t kh, int32_t kw, int32_t sh, int32_t sw,
    float norm_p)
{
    using Acc = typename AccOf<T>::type;
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)H_out * (int64_t)W_out;
    if (tid >= total) return;

    int32_t ow = (int32_t)(tid % (int64_t)W_out);
    int64_t tmp = tid / (int64_t)W_out;
    int32_t oh = (int32_t)(tmp % (int64_t)H_out);
    tmp /= (int64_t)H_out;
    int32_t ci = (int32_t)(tmp % (int64_t)C);
    int32_t ni = (int32_t)(tmp / (int64_t)C);

    int32_t h_start = oh * sh;
    int32_t h_end = h_start + kh;
    if (h_end > H_in) h_end = H_in;
    int32_t w_start = ow * sw;
    int32_t w_end = w_start + kw;
    if (w_end > W_in) w_end = W_in;

    Acc y_v  = load_acc<T>(y  + tid);
    Acc dy_v = load_acc<T>(dy + tid);

    if (y_v == (Acc)0) return;

    const Acc p = static_cast<Acc>(norm_p);
    Acc y_pow = acc_pow(y_v, (Acc)1 - p);
    Acc coeff = dy_v * y_pow;

    T*       base_dx = dx + ((int64_t)ni * C + ci) * (int64_t)H_in * (int64_t)W_in;
    const T* base_x  = x  + ((int64_t)ni * C + ci) * (int64_t)H_in * (int64_t)W_in;

    for (int32_t h = h_start; h < h_end; ++h) {
        const T* x_row  = base_x  + (int64_t)h * (int64_t)W_in;
        T*       dx_row = base_dx + (int64_t)h * (int64_t)W_in;
        for (int32_t w = w_start; w < w_end; ++w) {
            Acc xv = load_acc<T>(x_row + w);
            Acc ax = acc_fabs(xv);
            Acc sg = (xv > (Acc)0) ? (Acc)1 : ((xv < (Acc)0) ? (Acc)-1 : (Acc)0);
            if (sg == (Acc)0) continue;
            Acc ax_pm1 = acc_pow(ax, p - (Acc)1);
            if (!isfinite((double)ax_pm1)) continue;
            Acc contrib = coeff * ax_pm1 * sg;
            T t = store_from_acc<T>(contrib);
            baracuda::atomic::add(dx_row + w, t);
        }
    }
}

// =============================================================================
// Output-dim helpers.
// =============================================================================

__host__ __device__ __forceinline__
int32_t compute_out_dim(int32_t in_dim, int32_t kernel, int32_t stride, int32_t ceil_mode) {
    if (kernel > in_dim) return 0;
    int32_t diff = in_dim - kernel;
    int32_t out;
    if (ceil_mode != 0) {
        out = (diff + stride - 1) / stride + 1;
        // ceil_mode may produce a window whose stride origin lies
        // strictly inside `in` but whose start is at the boundary;
        // PyTorch's contract is "windows that start in the padded
        // region are skipped." Since we have no padding the only
        // skip case is when the start sits *at* in (out of range),
        // which can't happen given `diff = in - kernel >= 0`.
    } else {
        out = diff / stride + 1;
    }
    return out;
}

// =============================================================================
// 1-D launcher
// =============================================================================

template <typename T>
__host__ inline int32_t launch_lp_pool1d_fw(
    const void* x, void* y,
    int32_t batch, int32_t channels, int32_t L_in,
    int32_t kernel, int32_t stride, int32_t L_out,
    float norm_p,
    cudaStream_t stream)
{
    if (x == nullptr || y == nullptr) return 2;
    if (batch <= 0 || channels <= 0 || L_in <= 0) return 2;
    if (kernel <= 0 || stride <= 0) return 2;
    if (L_out <= 0) return 2;
    if (norm_p <= 0.0f) return 3;

    int64_t total = (int64_t)batch * (int64_t)channels * (int64_t)L_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    int64_t blocks64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks64 > 2147483647LL ? 2147483647LL : blocks64);
    if (blocks <= 0) blocks = 1;
    lp_pool1d_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        static_cast<const T*>(x), static_cast<T*>(y),
        batch, channels, L_in, L_out, kernel, stride, norm_p);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_lp_pool1d_bw(
    const void* x, const void* y, const void* dy, void* dx,
    int32_t batch, int32_t channels, int32_t L_in,
    int32_t kernel, int32_t stride, int32_t L_out,
    float norm_p,
    cudaStream_t stream)
{
    if (x == nullptr || y == nullptr || dy == nullptr || dx == nullptr) return 2;
    if (batch <= 0 || channels <= 0 || L_in <= 0) return 2;
    if (kernel <= 0 || stride <= 0 || L_out <= 0) return 2;
    if (norm_p <= 0.0f) return 3;

    // Caller is responsible for zeroing dx first (atomicAdd scatter).
    int64_t total = (int64_t)batch * (int64_t)channels * (int64_t)L_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    int64_t blocks64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks64 > 2147483647LL ? 2147483647LL : blocks64);
    if (blocks <= 0) blocks = 1;
    lp_pool1d_bw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        static_cast<const T*>(x), static_cast<const T*>(y),
        static_cast<const T*>(dy), static_cast<T*>(dx),
        batch, channels, L_in, L_out, kernel, stride, norm_p);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// 2-D launcher
// =============================================================================

template <typename T>
__host__ inline int32_t launch_lp_pool2d_fw(
    const void* x, void* y,
    int32_t batch, int32_t channels, int32_t H_in, int32_t W_in,
    int32_t kh, int32_t kw, int32_t sh, int32_t sw,
    int32_t H_out, int32_t W_out,
    float norm_p,
    cudaStream_t stream)
{
    if (x == nullptr || y == nullptr) return 2;
    if (batch <= 0 || channels <= 0 || H_in <= 0 || W_in <= 0) return 2;
    if (kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0) return 2;
    if (H_out <= 0 || W_out <= 0) return 2;
    if (norm_p <= 0.0f) return 3;

    int64_t total = (int64_t)batch * (int64_t)channels * (int64_t)H_out * (int64_t)W_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    int64_t blocks64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks64 > 2147483647LL ? 2147483647LL : blocks64);
    if (blocks <= 0) blocks = 1;
    lp_pool2d_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        static_cast<const T*>(x), static_cast<T*>(y),
        batch, channels, H_in, W_in, H_out, W_out, kh, kw, sh, sw, norm_p);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_lp_pool2d_bw(
    const void* x, const void* y, const void* dy, void* dx,
    int32_t batch, int32_t channels, int32_t H_in, int32_t W_in,
    int32_t kh, int32_t kw, int32_t sh, int32_t sw,
    int32_t H_out, int32_t W_out,
    float norm_p,
    cudaStream_t stream)
{
    if (x == nullptr || y == nullptr || dy == nullptr || dx == nullptr) return 2;
    if (batch <= 0 || channels <= 0 || H_in <= 0 || W_in <= 0) return 2;
    if (kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0) return 2;
    if (H_out <= 0 || W_out <= 0) return 2;
    if (norm_p <= 0.0f) return 3;

    int64_t total = (int64_t)batch * (int64_t)channels * (int64_t)H_out * (int64_t)W_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    int64_t blocks64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks64 > 2147483647LL ? 2147483647LL : blocks64);
    if (blocks <= 0) blocks = 1;
    lp_pool2d_bw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        static_cast<const T*>(x), static_cast<const T*>(y),
        static_cast<const T*>(dy), static_cast<T*>(dx),
        batch, channels, H_in, W_in, H_out, W_out, kh, kw, sh, sw, norm_p);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::lp_pool

// Emit one LpPool1d FW launcher per-dtype.
#define BARACUDA_KERNELS_LP_POOL_1D_FW_INSTANTIATE(NAME, T)                                          \
    extern "C" int32_t baracuda_kernels_lp_pool_1d_##NAME##_run(                                      \
        const void* x, void* y,                                                                       \
        int32_t batch, int32_t channels, int32_t l_in,                                                \
        int32_t kernel, int32_t stride, int32_t l_out,                                                \
        float norm_p,                                                                                 \
        int32_t /* ceil_mode_unused — already baked into l_out by the safe layer */,                  \
        void* stream_ptr)                                                                             \
    {                                                                                                  \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::lp_pool::launch_lp_pool1d_fw<T>(                                             \
            x, y, batch, channels, l_in, kernel, stride, l_out, norm_p, stream);                      \
    }                                                                                                  \
    extern "C" int32_t baracuda_kernels_lp_pool_1d_##NAME##_can_implement(                            \
        const void* /*x*/, const void* /*y*/,                                                         \
        int32_t batch, int32_t channels, int32_t l_in,                                                \
        int32_t kernel, int32_t stride, int32_t l_out,                                                \
        float /*norm_p*/,                                                                             \
        int32_t /*ceil_mode_unused*/)                                                                 \
    {                                                                                                  \
        if (batch < 0 || channels < 0 || l_in < 0 || l_out < 0) return 2;                             \
        if (kernel <= 0 || stride <= 0) return 2;                                                     \
        return 0;                                                                                      \
    }

#define BARACUDA_KERNELS_LP_POOL_1D_BW_INSTANTIATE(NAME, T)                                          \
    extern "C" int32_t baracuda_kernels_lp_pool_1d_##NAME##_backward_run(                             \
        const void* x, const void* y, const void* dy, void* dx,                                       \
        int32_t batch, int32_t channels, int32_t l_in,                                                \
        int32_t kernel, int32_t stride, int32_t l_out,                                                \
        float norm_p,                                                                                 \
        int32_t /* ceil_mode_unused */,                                                                \
        void* stream_ptr)                                                                             \
    {                                                                                                  \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::lp_pool::launch_lp_pool1d_bw<T>(                                             \
            x, y, dy, dx, batch, channels, l_in, kernel, stride, l_out, norm_p, stream);              \
    }                                                                                                  \
    extern "C" int32_t baracuda_kernels_lp_pool_1d_##NAME##_backward_can_implement(                   \
        const void* /*x*/, const void* /*y*/, const void* /*dy*/, const void* /*dx*/,                 \
        int32_t batch, int32_t channels, int32_t l_in,                                                \
        int32_t kernel, int32_t stride, int32_t l_out,                                                \
        float /*norm_p*/,                                                                             \
        int32_t /*ceil_mode_unused*/)                                                                 \
    {                                                                                                  \
        if (batch < 0 || channels < 0 || l_in < 0 || l_out < 0) return 2;                             \
        if (kernel <= 0 || stride <= 0) return 2;                                                     \
        return 0;                                                                                      \
    }

#define BARACUDA_KERNELS_LP_POOL_2D_FW_INSTANTIATE(NAME, T)                                          \
    extern "C" int32_t baracuda_kernels_lp_pool_2d_##NAME##_run(                                      \
        const void* x, void* y,                                                                       \
        int32_t batch, int32_t channels, int32_t h_in, int32_t w_in,                                  \
        int32_t kh, int32_t kw, int32_t sh, int32_t sw,                                               \
        int32_t h_out, int32_t w_out,                                                                 \
        float norm_p,                                                                                 \
        int32_t /* ceil_mode_unused */,                                                                \
        void* stream_ptr)                                                                             \
    {                                                                                                  \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::lp_pool::launch_lp_pool2d_fw<T>(                                             \
            x, y, batch, channels, h_in, w_in, kh, kw, sh, sw, h_out, w_out, norm_p, stream);         \
    }                                                                                                  \
    extern "C" int32_t baracuda_kernels_lp_pool_2d_##NAME##_can_implement(                            \
        const void* /*x*/, const void* /*y*/,                                                         \
        int32_t batch, int32_t channels, int32_t h_in, int32_t w_in,                                  \
        int32_t kh, int32_t kw, int32_t sh, int32_t sw,                                               \
        int32_t h_out, int32_t w_out,                                                                 \
        float /*norm_p*/,                                                                             \
        int32_t /*ceil_mode_unused*/)                                                                 \
    {                                                                                                  \
        if (batch < 0 || channels < 0 || h_in < 0 || w_in < 0 || h_out < 0 || w_out < 0) return 2;   \
        if (kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0) return 2;                                       \
        return 0;                                                                                      \
    }

#define BARACUDA_KERNELS_LP_POOL_2D_BW_INSTANTIATE(NAME, T)                                          \
    extern "C" int32_t baracuda_kernels_lp_pool_2d_##NAME##_backward_run(                             \
        const void* x, const void* y, const void* dy, void* dx,                                       \
        int32_t batch, int32_t channels, int32_t h_in, int32_t w_in,                                  \
        int32_t kh, int32_t kw, int32_t sh, int32_t sw,                                               \
        int32_t h_out, int32_t w_out,                                                                 \
        float norm_p,                                                                                 \
        int32_t /* ceil_mode_unused */,                                                                \
        void* stream_ptr)                                                                             \
    {                                                                                                  \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::lp_pool::launch_lp_pool2d_bw<T>(                                             \
            x, y, dy, dx, batch, channels, h_in, w_in, kh, kw, sh, sw, h_out, w_out, norm_p, stream); \
    }                                                                                                  \
    extern "C" int32_t baracuda_kernels_lp_pool_2d_##NAME##_backward_can_implement(                   \
        const void* /*x*/, const void* /*y*/, const void* /*dy*/, const void* /*dx*/,                 \
        int32_t batch, int32_t channels, int32_t h_in, int32_t w_in,                                  \
        int32_t kh, int32_t kw, int32_t sh, int32_t sw,                                               \
        int32_t h_out, int32_t w_out,                                                                 \
        float /*norm_p*/,                                                                             \
        int32_t /*ceil_mode_unused*/)                                                                 \
    {                                                                                                  \
        if (batch < 0 || channels < 0 || h_in < 0 || w_in < 0 || h_out < 0 || w_out < 0) return 2;   \
        if (kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0) return 2;                                       \
        return 0;                                                                                      \
    }

#endif // BARACUDA_LP_POOL_CUH
