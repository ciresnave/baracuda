// baracuda_attention.cuh
//
// Templated kernels and INSTANTIATE macros for the attention op family
// (Phase 6 Category K of the comprehensive plan).
//
// Today's wiring (Milestone 6.1 — positional encodings, FW + BW × 4 FP
// dtypes: f32, f16, bf16, f64):
//
//   - RoPE   (Rotary Position Embedding)
//            For Q/K of shape [B, H, S, D] (D must be even), each pair
//            (2i, 2i+1) is rotated by θ_i = pos · base^(-2i/D):
//                y[2i]   = x[2i]   · cos(θ_i) - x[2i+1] · sin(θ_i)
//                y[2i+1] = x[2i+1] · cos(θ_i) + x[2i]   · sin(θ_i)
//            BW reverses the sign of θ (rotation is orthogonal):
//                dx[2i]   = dy[2i]   · cos(θ_i) + dy[2i+1] · sin(θ_i)
//                dx[2i+1] = dy[2i+1] · cos(θ_i) - dy[2i]   · sin(θ_i)
//            `positions` is an optional i64[S] override; when null, the
//            kernel uses pos = s (the seq-len index).
//
//   - ALiBi  (Attention with Linear Biases)
//            For attention scores A of shape [B, H, Q, K]:
//                y[b, h, i, j] = A[b, h, i, j] + slope[h] · (j - i)
//            BW: dA = dY (pass-through), plus
//                dslope[h] = Σ_{b, i, j} dY[b, h, i, j] · (j - i)
//            computed by one block per head, deterministic warp-shuffle
//            reduction (no atomicAdd).
//
// f16 / bf16 ALWAYS detour through f32 for trig and reductions. f64 uses
// native double.
//
// Status codes match the elementwise family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

#ifndef BARACUDA_ATTENTION_CUH
#define BARACUDA_ATTENTION_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace attention {

// =============================================================================
// dtype helpers — f32 detour for half / bf16, native otherwise.
// =============================================================================

template <typename T>
__device__ __forceinline__ float load_as_f32(T x) { return (float)x; }

template <>
__device__ __forceinline__ float load_as_f32<__half>(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float load_as_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T store_from_f32(float v) { return (T)v; }

template <>
__device__ __forceinline__ __half store_from_f32<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// =============================================================================
// RoPE FW kernel
// =============================================================================
//
// Input  `x`        : T[B, H, S, D] contiguous (row-major).
// Output `y`        : T[B, H, S, D] contiguous (row-major).
// Optional `pos`    : i64[S] of position indices. May be null — then the
//                     kernel uses `pos[s] = s`.
// `base`            : f32 rotary base (default 10000.0).
// `pos_default_flag`: 0 = use positions[s]; non-zero = default to s.
//
// One thread per output cell. Total cells = B*H*S*D. For each cell at
// (b, h, s, d) the thread computes the pair index `p = d / 2` and the
// inverse-frequency exponent `freq = base^(-(2*p)/D)`, then θ = pos · freq.
// Depending on parity of `d`, we read either `(x[2p], x[2p+1])` or the
// same pair (using `d-1` if `d` is odd).

template <typename T>
__global__ void rope_fp_kernel(
    const T* __restrict__ x,
    const int64_t* __restrict__ positions,
    T* __restrict__ y,
    int32_t batch,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    float base,
    int32_t pos_default_flag)
{
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float inv_d = 1.0f / (float)head_dim;
    for (int64_t lin = tid; lin < total; lin += step) {
        // Unravel row-major (B, H, S, D)
        int32_t d = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest = lin / (int64_t)head_dim;
        int32_t s = (int32_t)(rest % (int64_t)seq);
        rest /= (int64_t)seq;
        // b, h unused individually — we just need the row index without d.
        int32_t pair = d >> 1;
        int32_t d_even = pair << 1;
        bool is_high = (d & 1) != 0;
        float pos_f;
        if (pos_default_flag != 0) {
            pos_f = (float)s;
        } else {
            pos_f = (float)positions[s];
        }
        // freq = base^(-(2 * pair) / D)
        float exponent = -(float)d_even * inv_d;
        float freq = __powf(base, exponent);
        float theta = pos_f * freq;
        float c = __cosf(theta);
        float si = __sinf(theta);
        // Linear offsets of the pair partners
        int64_t base_off = lin - (int64_t)d;
        int64_t off_e = base_off + (int64_t)d_even;
        int64_t off_o = off_e + 1;
        float x_e = load_as_f32<T>(x[off_e]);
        float x_o = load_as_f32<T>(x[off_o]);
        float out;
        if (!is_high) {
            // d even slot: y[2i] = x[2i] · cos - x[2i+1] · sin
            out = x_e * c - x_o * si;
        } else {
            // d odd slot: y[2i+1] = x[2i+1] · cos + x[2i] · sin
            out = x_o * c + x_e * si;
        }
        y[lin] = store_from_f32<T>(out);
    }
}

template <>
__global__ void rope_fp_kernel<double>(
    const double* __restrict__ x,
    const int64_t* __restrict__ positions,
    double* __restrict__ y,
    int32_t batch,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    float base,
    int32_t pos_default_flag)
{
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_d = 1.0 / (double)head_dim;
    double base_d = (double)base;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest = lin / (int64_t)head_dim;
        int32_t s = (int32_t)(rest % (int64_t)seq);
        rest /= (int64_t)seq;
        int32_t pair = d >> 1;
        int32_t d_even = pair << 1;
        bool is_high = (d & 1) != 0;
        double pos_d;
        if (pos_default_flag != 0) {
            pos_d = (double)s;
        } else {
            pos_d = (double)positions[s];
        }
        double exponent = -(double)d_even * inv_d;
        double freq = pow(base_d, exponent);
        double theta = pos_d * freq;
        double c = cos(theta);
        double si = sin(theta);
        int64_t base_off = lin - (int64_t)d;
        int64_t off_e = base_off + (int64_t)d_even;
        int64_t off_o = off_e + 1;
        double x_e = x[off_e];
        double x_o = x[off_o];
        double out;
        if (!is_high) {
            out = x_e * c - x_o * si;
        } else {
            out = x_o * c + x_e * si;
        }
        y[lin] = out;
    }
}

template <typename T>
__host__ inline int32_t launch_rope_fp(
    const T* x, const int64_t* positions, T* y,
    int32_t batch, int32_t heads, int32_t seq, int32_t head_dim,
    float base, int32_t pos_default_flag,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || seq < 0 || head_dim < 0) return 2;
    if (head_dim % 2 != 0) return 2;
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    rope_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, positions, y, batch, heads, seq, head_dim, base, pos_default_flag);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// RoPE BW kernel
// =============================================================================
//
// Same shape, swapped trig signs (rotation by -θ).

template <typename T>
__global__ void rope_backward_fp_kernel(
    const T* __restrict__ dy,
    const int64_t* __restrict__ positions,
    T* __restrict__ dx,
    int32_t batch,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    float base,
    int32_t pos_default_flag)
{
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float inv_d = 1.0f / (float)head_dim;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest = lin / (int64_t)head_dim;
        int32_t s = (int32_t)(rest % (int64_t)seq);
        rest /= (int64_t)seq;
        int32_t pair = d >> 1;
        int32_t d_even = pair << 1;
        bool is_high = (d & 1) != 0;
        float pos_f;
        if (pos_default_flag != 0) {
            pos_f = (float)s;
        } else {
            pos_f = (float)positions[s];
        }
        float exponent = -(float)d_even * inv_d;
        float freq = __powf(base, exponent);
        float theta = pos_f * freq;
        float c = __cosf(theta);
        float si = __sinf(theta);
        int64_t base_off = lin - (int64_t)d;
        int64_t off_e = base_off + (int64_t)d_even;
        int64_t off_o = off_e + 1;
        float dy_e = load_as_f32<T>(dy[off_e]);
        float dy_o = load_as_f32<T>(dy[off_o]);
        float out;
        if (!is_high) {
            // dx[2i] = dy[2i] · cos + dy[2i+1] · sin
            out = dy_e * c + dy_o * si;
        } else {
            // dx[2i+1] = dy[2i+1] · cos - dy[2i] · sin
            out = dy_o * c - dy_e * si;
        }
        dx[lin] = store_from_f32<T>(out);
    }
}

template <>
__global__ void rope_backward_fp_kernel<double>(
    const double* __restrict__ dy,
    const int64_t* __restrict__ positions,
    double* __restrict__ dx,
    int32_t batch,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    float base,
    int32_t pos_default_flag)
{
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_d = 1.0 / (double)head_dim;
    double base_d = (double)base;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest = lin / (int64_t)head_dim;
        int32_t s = (int32_t)(rest % (int64_t)seq);
        rest /= (int64_t)seq;
        int32_t pair = d >> 1;
        int32_t d_even = pair << 1;
        bool is_high = (d & 1) != 0;
        double pos_d;
        if (pos_default_flag != 0) {
            pos_d = (double)s;
        } else {
            pos_d = (double)positions[s];
        }
        double exponent = -(double)d_even * inv_d;
        double freq = pow(base_d, exponent);
        double theta = pos_d * freq;
        double c = cos(theta);
        double si = sin(theta);
        int64_t base_off = lin - (int64_t)d;
        int64_t off_e = base_off + (int64_t)d_even;
        int64_t off_o = off_e + 1;
        double dy_e = dy[off_e];
        double dy_o = dy[off_o];
        double out;
        if (!is_high) {
            out = dy_e * c + dy_o * si;
        } else {
            out = dy_o * c - dy_e * si;
        }
        dx[lin] = out;
    }
}

template <typename T>
__host__ inline int32_t launch_rope_backward_fp(
    const T* dy, const int64_t* positions, T* dx,
    int32_t batch, int32_t heads, int32_t seq, int32_t head_dim,
    float base, int32_t pos_default_flag,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || seq < 0 || head_dim < 0) return 2;
    if (head_dim % 2 != 0) return 2;
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    rope_backward_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, positions, dx, batch, heads, seq, head_dim, base, pos_default_flag);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// RoPE strided FW / BW — Phase 14.4
// =============================================================================
//
// Outer dims (batch, heads, seq) may have arbitrary signed-i64 strides;
// the innermost `head_dim` axis MUST have stride 1 because RoPE rotates
// adjacent pairs (2i, 2i+1) which must sit next to each other in memory.
// The Rust plan layer enforces that contract before crossing the FFI.
//
// One thread per output cell. The thread unravels its linear index into
// (b, h, s, d) coords, then uses (stride_x_b, stride_x_h, stride_x_s)
// plus a literal +1 / +0 on the head_dim axis to compute the read and
// write byte offsets.

template <typename T>
__global__ void rope_strided_fp_kernel(
    const T* __restrict__ x,
    const int64_t* __restrict__ positions,
    T* __restrict__ y,
    int32_t batch,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    int64_t stride_x_b, int64_t stride_x_h, int64_t stride_x_s,
    int64_t stride_y_b, int64_t stride_y_h, int64_t stride_y_s,
    float base,
    int32_t pos_default_flag)
{
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float inv_d = 1.0f / (float)head_dim;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest = lin / (int64_t)head_dim;
        int32_t s = (int32_t)(rest % (int64_t)seq);
        rest /= (int64_t)seq;
        int32_t h = (int32_t)(rest % (int64_t)heads);
        int32_t b = (int32_t)(rest / (int64_t)heads);
        int32_t pair = d >> 1;
        int32_t d_even = pair << 1;
        bool is_high = (d & 1) != 0;
        float pos_f;
        if (pos_default_flag != 0) {
            pos_f = (float)s;
        } else {
            pos_f = (float)positions[s];
        }
        float exponent = -(float)d_even * inv_d;
        float freq = __powf(base, exponent);
        float theta = pos_f * freq;
        float c = __cosf(theta);
        float si = __sinf(theta);
        int64_t off_x_outer = (int64_t)b * stride_x_b
                             + (int64_t)h * stride_x_h
                             + (int64_t)s * stride_x_s;
        int64_t off_y_outer = (int64_t)b * stride_y_b
                             + (int64_t)h * stride_y_h
                             + (int64_t)s * stride_y_s;
        // head_dim stride is implicit 1 — pair partners sit at d_even and d_even+1.
        int64_t off_x_e = off_x_outer + (int64_t)d_even;
        int64_t off_x_o = off_x_e + 1;
        int64_t off_y   = off_y_outer + (int64_t)d;
        float x_e = load_as_f32<T>(x[off_x_e]);
        float x_o = load_as_f32<T>(x[off_x_o]);
        float out;
        if (!is_high) {
            out = x_e * c - x_o * si;
        } else {
            out = x_o * c + x_e * si;
        }
        y[off_y] = store_from_f32<T>(out);
    }
}

template <>
__global__ void rope_strided_fp_kernel<double>(
    const double* __restrict__ x,
    const int64_t* __restrict__ positions,
    double* __restrict__ y,
    int32_t batch,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    int64_t stride_x_b, int64_t stride_x_h, int64_t stride_x_s,
    int64_t stride_y_b, int64_t stride_y_h, int64_t stride_y_s,
    float base,
    int32_t pos_default_flag)
{
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_d = 1.0 / (double)head_dim;
    double base_d = (double)base;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest = lin / (int64_t)head_dim;
        int32_t s = (int32_t)(rest % (int64_t)seq);
        rest /= (int64_t)seq;
        int32_t h = (int32_t)(rest % (int64_t)heads);
        int32_t b = (int32_t)(rest / (int64_t)heads);
        int32_t pair = d >> 1;
        int32_t d_even = pair << 1;
        bool is_high = (d & 1) != 0;
        double pos_d;
        if (pos_default_flag != 0) {
            pos_d = (double)s;
        } else {
            pos_d = (double)positions[s];
        }
        double exponent = -(double)d_even * inv_d;
        double freq = pow(base_d, exponent);
        double theta = pos_d * freq;
        double c = cos(theta);
        double si = sin(theta);
        int64_t off_x_outer = (int64_t)b * stride_x_b
                             + (int64_t)h * stride_x_h
                             + (int64_t)s * stride_x_s;
        int64_t off_y_outer = (int64_t)b * stride_y_b
                             + (int64_t)h * stride_y_h
                             + (int64_t)s * stride_y_s;
        int64_t off_x_e = off_x_outer + (int64_t)d_even;
        int64_t off_x_o = off_x_e + 1;
        int64_t off_y   = off_y_outer + (int64_t)d;
        double x_e = x[off_x_e];
        double x_o = x[off_x_o];
        double out;
        if (!is_high) {
            out = x_e * c - x_o * si;
        } else {
            out = x_o * c + x_e * si;
        }
        y[off_y] = out;
    }
}

template <typename T>
__host__ inline int32_t launch_rope_strided_fp(
    const T* x, const int64_t* positions, T* y,
    int32_t batch, int32_t heads, int32_t seq, int32_t head_dim,
    int64_t stride_x_b, int64_t stride_x_h, int64_t stride_x_s,
    int64_t stride_y_b, int64_t stride_y_h, int64_t stride_y_s,
    float base, int32_t pos_default_flag,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || seq < 0 || head_dim < 0) return 2;
    if (head_dim % 2 != 0) return 2;
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    rope_strided_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, positions, y, batch, heads, seq, head_dim,
        stride_x_b, stride_x_h, stride_x_s,
        stride_y_b, stride_y_h, stride_y_s,
        base, pos_default_flag);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// RoPE BW strided — same shape, swapped trig signs.

template <typename T>
__global__ void rope_backward_strided_fp_kernel(
    const T* __restrict__ dy,
    const int64_t* __restrict__ positions,
    T* __restrict__ dx,
    int32_t batch,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    int64_t stride_dy_b, int64_t stride_dy_h, int64_t stride_dy_s,
    int64_t stride_dx_b, int64_t stride_dx_h, int64_t stride_dx_s,
    float base,
    int32_t pos_default_flag)
{
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float inv_d = 1.0f / (float)head_dim;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest = lin / (int64_t)head_dim;
        int32_t s = (int32_t)(rest % (int64_t)seq);
        rest /= (int64_t)seq;
        int32_t h = (int32_t)(rest % (int64_t)heads);
        int32_t b = (int32_t)(rest / (int64_t)heads);
        int32_t pair = d >> 1;
        int32_t d_even = pair << 1;
        bool is_high = (d & 1) != 0;
        float pos_f;
        if (pos_default_flag != 0) {
            pos_f = (float)s;
        } else {
            pos_f = (float)positions[s];
        }
        float exponent = -(float)d_even * inv_d;
        float freq = __powf(base, exponent);
        float theta = pos_f * freq;
        float c = __cosf(theta);
        float si = __sinf(theta);
        int64_t off_dy_outer = (int64_t)b * stride_dy_b
                              + (int64_t)h * stride_dy_h
                              + (int64_t)s * stride_dy_s;
        int64_t off_dx_outer = (int64_t)b * stride_dx_b
                              + (int64_t)h * stride_dx_h
                              + (int64_t)s * stride_dx_s;
        int64_t off_dy_e = off_dy_outer + (int64_t)d_even;
        int64_t off_dy_o = off_dy_e + 1;
        int64_t off_dx   = off_dx_outer + (int64_t)d;
        float dy_e = load_as_f32<T>(dy[off_dy_e]);
        float dy_o = load_as_f32<T>(dy[off_dy_o]);
        float out;
        if (!is_high) {
            out = dy_e * c + dy_o * si;
        } else {
            out = dy_o * c - dy_e * si;
        }
        dx[off_dx] = store_from_f32<T>(out);
    }
}

template <>
__global__ void rope_backward_strided_fp_kernel<double>(
    const double* __restrict__ dy,
    const int64_t* __restrict__ positions,
    double* __restrict__ dx,
    int32_t batch,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    int64_t stride_dy_b, int64_t stride_dy_h, int64_t stride_dy_s,
    int64_t stride_dx_b, int64_t stride_dx_h, int64_t stride_dx_s,
    float base,
    int32_t pos_default_flag)
{
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_d = 1.0 / (double)head_dim;
    double base_d = (double)base;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest = lin / (int64_t)head_dim;
        int32_t s = (int32_t)(rest % (int64_t)seq);
        rest /= (int64_t)seq;
        int32_t h = (int32_t)(rest % (int64_t)heads);
        int32_t b = (int32_t)(rest / (int64_t)heads);
        int32_t pair = d >> 1;
        int32_t d_even = pair << 1;
        bool is_high = (d & 1) != 0;
        double pos_d;
        if (pos_default_flag != 0) {
            pos_d = (double)s;
        } else {
            pos_d = (double)positions[s];
        }
        double exponent = -(double)d_even * inv_d;
        double freq = pow(base_d, exponent);
        double theta = pos_d * freq;
        double c = cos(theta);
        double si = sin(theta);
        int64_t off_dy_outer = (int64_t)b * stride_dy_b
                              + (int64_t)h * stride_dy_h
                              + (int64_t)s * stride_dy_s;
        int64_t off_dx_outer = (int64_t)b * stride_dx_b
                              + (int64_t)h * stride_dx_h
                              + (int64_t)s * stride_dx_s;
        int64_t off_dy_e = off_dy_outer + (int64_t)d_even;
        int64_t off_dy_o = off_dy_e + 1;
        int64_t off_dx   = off_dx_outer + (int64_t)d;
        double dy_e = dy[off_dy_e];
        double dy_o = dy[off_dy_o];
        double out;
        if (!is_high) {
            out = dy_e * c + dy_o * si;
        } else {
            out = dy_o * c - dy_e * si;
        }
        dx[off_dx] = out;
    }
}

template <typename T>
__host__ inline int32_t launch_rope_backward_strided_fp(
    const T* dy, const int64_t* positions, T* dx,
    int32_t batch, int32_t heads, int32_t seq, int32_t head_dim,
    int64_t stride_dy_b, int64_t stride_dy_h, int64_t stride_dy_s,
    int64_t stride_dx_b, int64_t stride_dx_h, int64_t stride_dx_s,
    float base, int32_t pos_default_flag,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || seq < 0 || head_dim < 0) return 2;
    if (head_dim % 2 != 0) return 2;
    int64_t total = (int64_t)batch * heads * seq * head_dim;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    rope_backward_strided_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, positions, dx, batch, heads, seq, head_dim,
        stride_dy_b, stride_dy_h, stride_dy_s,
        stride_dx_b, stride_dx_h, stride_dx_s,
        base, pos_default_flag);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// ALiBi FW kernel
// =============================================================================
//
// scores  `A`    : T[B, H, Q, K] contiguous (row-major).
// slopes         : T[H]
// output  `Y`    : T[B, H, Q, K]
// Y[b, h, i, j] = A[b, h, i, j] + slope[h] · (j - i)

template <typename T>
__global__ void alibi_fp_kernel(
    const T* __restrict__ scores,
    const T* __restrict__ slopes,
    T* __restrict__ y,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len)
{
    int64_t total = (int64_t)batch * heads * q_len * k_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t j = (int32_t)(lin % (int64_t)k_len);
        int64_t rest = lin / (int64_t)k_len;
        int32_t i = (int32_t)(rest % (int64_t)q_len);
        rest /= (int64_t)q_len;
        int32_t h = (int32_t)(rest % (int64_t)heads);
        // b unused — slopes only depend on h, bias depends on (i, j).
        float a_v = load_as_f32<T>(scores[lin]);
        float sl  = load_as_f32<T>(slopes[h]);
        float delta = (float)j - (float)i;
        float out = a_v + sl * delta;
        y[lin] = store_from_f32<T>(out);
    }
}

template <>
__global__ void alibi_fp_kernel<double>(
    const double* __restrict__ scores,
    const double* __restrict__ slopes,
    double* __restrict__ y,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len)
{
    int64_t total = (int64_t)batch * heads * q_len * k_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t j = (int32_t)(lin % (int64_t)k_len);
        int64_t rest = lin / (int64_t)k_len;
        int32_t i = (int32_t)(rest % (int64_t)q_len);
        rest /= (int64_t)q_len;
        int32_t h = (int32_t)(rest % (int64_t)heads);
        double a_v = scores[lin];
        double sl  = slopes[h];
        double delta = (double)j - (double)i;
        y[lin] = a_v + sl * delta;
    }
}

template <typename T>
__host__ inline int32_t launch_alibi_fp(
    const T* scores, const T* slopes, T* y,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0) return 2;
    int64_t total = (int64_t)batch * heads * q_len * k_len;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    alibi_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        scores, slopes, y, batch, heads, q_len, k_len);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// ALiBi BW — dA (pass-through copy) + dslope (one-block-per-head reduction)
// =============================================================================

template <typename T>
__global__ void alibi_backward_da_kernel(
    const T* __restrict__ dy,
    T* __restrict__ da,
    int64_t total)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        da[lin] = dy[lin];
    }
}

// One block per head h ∈ [0, heads). Threads stride over (b, i, j) flat
// index in [0, batch * q_len * k_len) and accumulate dy[lin] * (j - i)
// into an f32 partial (f64 path uses double). Warp-shuffle + smem
// reduction; lone surviving thread writes dslope[h].

template <typename T>
__global__ void alibi_backward_dslope_kernel(
    const T* __restrict__ dy,
    T* __restrict__ dslope,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len)
{
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsize = (int)blockDim.x;
    int64_t bqk = (int64_t)batch * q_len * k_len;
    int64_t qk = (int64_t)q_len * k_len;
    float partial = 0.0f;
    // Each (b, i, j) cell contributes dy[b, h, i, j] · (j - i)
    for (int64_t lin = (int64_t)tid; lin < bqk; lin += (int64_t)bsize) {
        int32_t j = (int32_t)(lin % (int64_t)k_len);
        int64_t r = lin / (int64_t)k_len;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        int64_t b = r / (int64_t)q_len;
        // Full flat offset: ((b * heads + h) * q_len + i) * k_len + j
        int64_t off = ((b * (int64_t)heads + (int64_t)h) * (int64_t)q_len + (int64_t)i)
                      * (int64_t)k_len + (int64_t)j;
        float dy_v = load_as_f32<T>(dy[off]);
        float delta = (float)j - (float)i;
        partial += dy_v * delta;
    }
    __shared__ float smem[32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) smem[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (bsize + 31) >> 5;
        float v = (lane < n_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) {
            dslope[h] = store_from_f32<T>(v);
        }
    }
}

template <>
__global__ void alibi_backward_dslope_kernel<double>(
    const double* __restrict__ dy,
    double* __restrict__ dslope,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len)
{
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsize = (int)blockDim.x;
    int64_t bqk = (int64_t)batch * q_len * k_len;
    double partial = 0.0;
    for (int64_t lin = (int64_t)tid; lin < bqk; lin += (int64_t)bsize) {
        int32_t j = (int32_t)(lin % (int64_t)k_len);
        int64_t r = lin / (int64_t)k_len;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        int64_t b = r / (int64_t)q_len;
        int64_t off = ((b * (int64_t)heads + (int64_t)h) * (int64_t)q_len + (int64_t)i)
                      * (int64_t)k_len + (int64_t)j;
        double dy_v = dy[off];
        double delta = (double)j - (double)i;
        partial += dy_v * delta;
    }
    __shared__ double smem[32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) smem[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (bsize + 31) >> 5;
        double v = (lane < n_warps) ? smem[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) {
            dslope[h] = v;
        }
    }
}

template <typename T>
__host__ inline int32_t launch_alibi_backward_fp(
    const T* dy, T* da, T* dslope,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0) return 2;
    int64_t total = (int64_t)batch * heads * q_len * k_len;
    if (total == 0) return 0;
    // dA = dY copy (optional — caller may pass `da == nullptr` if it
    // only wants `dslope`).
    if (da != nullptr) {
        constexpr int kBlock = 256;
        constexpr int64_t kMaxBlocks = 65535;
        int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        alibi_backward_da_kernel<T><<<blocks, kBlock, 0, stream>>>(dy, da, total);
        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess) return 5;
    }
    if (dslope != nullptr) {
        constexpr int kBlockRed = 256;
        alibi_backward_dslope_kernel<T><<<(int)heads, kBlockRed, 0, stream>>>(
            dy, dslope, batch, heads, q_len, k_len);
        cudaError_t err2 = cudaGetLastError();
        if (err2 != cudaSuccess) return 5;
    }
    return 0;
}

// =============================================================================
// KV-cache append (Milestone 6.5)
// =============================================================================
//
// Decoder-inference helper. At each autoregressive step the model
// produces one (or `new_len`) new (K, V) row(s) per batch sample and
// per head, which need to be appended into running caches:
//
//   K_cache : [B, H, L_max, D_k]
//   V_cache : [B, H, L_max, D_v]
//
// The destination row in the cache is determined per sample by the
// `cache_offsets[b]` vector — the next slot to fill for that sample.
// Ragged-batch inference is supported because each sample carries its
// own offset.
//
// Op semantics:
//
//   for b, h, l_new, d in [B, H, L_new, D_k]:
//       K_cache[b, h, cache_offsets[b] + l_new, d] = K_new[b, h, l_new, d]
//
//   for b, h, l_new, d in [B, H, L_new, D_v]:
//       V_cache[b, h, cache_offsets[b] + l_new, d] = V_new[b, h, l_new, d]
//
// Pure copy — bit-exact across every dtype. Boundary safety: if
// `cache_offsets[b] + l_new >= max_cache_len` for a given cell, the
// thread silently skips that store (caller's responsibility to size
// the cache). No BW — KV-cache is an inference-time op.

template <typename T>
__global__ void kv_cache_append_kernel(
    const T* __restrict__ src,
    const int64_t* __restrict__ cache_offsets,
    T* __restrict__ dst,
    int32_t batch,
    int32_t heads,
    int32_t new_len,
    int32_t max_cache_len,
    int32_t head_dim)
{
    int64_t total = (int64_t)batch * heads * new_len * head_dim;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d        = (int32_t)(lin % (int64_t)head_dim);
        int64_t rest1    = lin / (int64_t)head_dim;
        int32_t new_pos  = (int32_t)(rest1 % (int64_t)new_len);
        int64_t rest2    = rest1 / (int64_t)new_len;
        int32_t h        = (int32_t)(rest2 % (int64_t)heads);
        int32_t b        = (int32_t)(rest2 / (int64_t)heads);
        int64_t cache_pos = cache_offsets[b] + (int64_t)new_pos;
        if (cache_pos < 0 || cache_pos >= (int64_t)max_cache_len) continue;
        int64_t src_off = (((int64_t)b * (int64_t)heads + (int64_t)h) * (int64_t)new_len
                            + (int64_t)new_pos) * (int64_t)head_dim + (int64_t)d;
        int64_t dst_off = (((int64_t)b * (int64_t)heads + (int64_t)h) * (int64_t)max_cache_len
                            + cache_pos) * (int64_t)head_dim + (int64_t)d;
        dst[dst_off] = src[src_off];
    }
}

template <typename T>
__host__ inline int32_t launch_kv_cache_append(
    const T* k_new, const T* v_new, const int64_t* cache_offsets,
    T* k_cache, T* v_cache,
    int32_t batch, int32_t heads, int32_t new_len,
    int32_t max_cache_len, int32_t d_k, int32_t d_v,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || new_len < 0 || max_cache_len < 0 ||
        d_k < 0 || d_v < 0) return 2;
    // Empty problem — nothing to do (zero new_len or zero batch / heads).
    if (batch == 0 || heads == 0 || new_len == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    // K copy
    if (d_k > 0) {
        int64_t total_k = (int64_t)batch * heads * new_len * d_k;
        int64_t blocks_i64 = (total_k + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        kv_cache_append_kernel<T><<<blocks, kBlock, 0, stream>>>(
            k_new, cache_offsets, k_cache,
            batch, heads, new_len, max_cache_len, d_k);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    // V copy
    if (d_v > 0) {
        int64_t total_v = (int64_t)batch * heads * new_len * d_v;
        int64_t blocks_i64 = (total_v + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        kv_cache_append_kernel<T><<<blocks, kBlock, 0, stream>>>(
            v_new, cache_offsets, v_cache,
            batch, heads, new_len, max_cache_len, d_v);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    return 0;
}

// =============================================================================
// RoPE apply (precomputed cos/sin tables) — Phase 36 (Fuel ask Gap 2)
// =============================================================================
//
// Unlike `rope_fp_kernel` which derives θ internally from
// `pos · base^(-2i/D)`, the apply variant takes caller-supplied
// `cos` / `sin` tables. This is the LLaMA-style extended-context API
// (YaRN / NTK / dynamic-scaling schedules pre-bake the trig values).
//
// Flat layout — `x`, `y` are `[bh, td]` (bh = outer batch * heads;
// td = seq * head_dim per (batch, head)). One thread per output cell.
//
// cos / sin layout — each holds `td/2` per `bh` row when `stride_b ==
// td/2` (per-row cos/sin), or a single shared `[td/2]` table when
// `stride_b == 0` (shared across all rows). The cos/sin table is
// indexed as `cos[bh_row * stride_b + s * (d/2) + pair]`.
//
// The kernel mirrors `rope_fp_kernel` cell-by-cell semantics; only the
// source of `cos(θ)` / `sin(θ)` differs (table lookup vs `__powf` +
// `__cosf` / `__sinf`). f16 / bf16 detour through f32 for the
// arithmetic (same as the base RoPE kernel).

template <typename T>
__global__ void rope_apply_fp_kernel(
    const T* __restrict__ x,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    T* __restrict__ y,
    int32_t bh,
    int32_t td,
    int32_t d,
    int32_t stride_b)
{
    int64_t total = (int64_t)bh * (int64_t)td;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int32_t half_d = d >> 1;
    int32_t seq = td / d;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t dim_idx = (int32_t)(lin % (int64_t)d);
        int64_t rest    = lin / (int64_t)d;
        int32_t s       = (int32_t)(rest % (int64_t)seq);
        int64_t bh_row  = rest / (int64_t)seq;
        int32_t pair    = dim_idx >> 1;
        int32_t d_even  = pair << 1;
        bool is_high    = (dim_idx & 1) != 0;
        int64_t cs_off  = (int64_t)bh_row * (int64_t)stride_b
                        + (int64_t)s * (int64_t)half_d
                        + (int64_t)pair;
        float c = cos_tab[cs_off];
        float si = sin_tab[cs_off];
        int64_t base_off = lin - (int64_t)dim_idx;
        int64_t off_e = base_off + (int64_t)d_even;
        int64_t off_o = off_e + 1;
        float x_e = load_as_f32<T>(x[off_e]);
        float x_o = load_as_f32<T>(x[off_o]);
        float out;
        if (!is_high) {
            out = x_e * c - x_o * si;
        } else {
            out = x_o * c + x_e * si;
        }
        y[lin] = store_from_f32<T>(out);
    }
}

template <>
__global__ void rope_apply_fp_kernel<double>(
    const double* __restrict__ x,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    double* __restrict__ y,
    int32_t bh,
    int32_t td,
    int32_t d,
    int32_t stride_b)
{
    int64_t total = (int64_t)bh * (int64_t)td;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int32_t half_d = d >> 1;
    int32_t seq = td / d;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t dim_idx = (int32_t)(lin % (int64_t)d);
        int64_t rest    = lin / (int64_t)d;
        int32_t s       = (int32_t)(rest % (int64_t)seq);
        int64_t bh_row  = rest / (int64_t)seq;
        int32_t pair    = dim_idx >> 1;
        int32_t d_even  = pair << 1;
        bool is_high    = (dim_idx & 1) != 0;
        int64_t cs_off  = (int64_t)bh_row * (int64_t)stride_b
                        + (int64_t)s * (int64_t)half_d
                        + (int64_t)pair;
        double c  = (double)cos_tab[cs_off];
        double si = (double)sin_tab[cs_off];
        int64_t base_off = lin - (int64_t)dim_idx;
        int64_t off_e = base_off + (int64_t)d_even;
        int64_t off_o = off_e + 1;
        double x_e = x[off_e];
        double x_o = x[off_o];
        double out;
        if (!is_high) {
            out = x_e * c - x_o * si;
        } else {
            out = x_o * c + x_e * si;
        }
        y[lin] = out;
    }
}

template <typename T>
__host__ inline int32_t launch_rope_apply_fp(
    const T* x,
    const float* cos_tab,
    const float* sin_tab,
    T* y,
    int32_t bh, int32_t td, int32_t d, int32_t stride_b,
    cudaStream_t stream)
{
    if (bh < 0 || td < 0 || d < 0 || stride_b < 0) return 2;
    if (d == 0) return 2;
    if (d % 2 != 0) return 2;
    if (td % d != 0) return 2;
    int64_t total = (int64_t)bh * (int64_t)td;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    rope_apply_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, cos_tab, sin_tab, y, bh, td, d, stride_b);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// RoPE apply BW — orthogonal rotation reverse (swap trig signs). Same
// cos/sin table layout as FW; the caller passes the SAME tables (no
// negation needed at the table layer because we flip the sign in the
// kernel arithmetic).

template <typename T>
__global__ void rope_apply_backward_fp_kernel(
    const T* __restrict__ dy,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    T* __restrict__ dx,
    int32_t bh,
    int32_t td,
    int32_t d,
    int32_t stride_b)
{
    int64_t total = (int64_t)bh * (int64_t)td;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int32_t half_d = d >> 1;
    int32_t seq = td / d;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t dim_idx = (int32_t)(lin % (int64_t)d);
        int64_t rest    = lin / (int64_t)d;
        int32_t s       = (int32_t)(rest % (int64_t)seq);
        int64_t bh_row  = rest / (int64_t)seq;
        int32_t pair    = dim_idx >> 1;
        int32_t d_even  = pair << 1;
        bool is_high    = (dim_idx & 1) != 0;
        int64_t cs_off  = (int64_t)bh_row * (int64_t)stride_b
                        + (int64_t)s * (int64_t)half_d
                        + (int64_t)pair;
        float c = cos_tab[cs_off];
        float si = sin_tab[cs_off];
        int64_t base_off = lin - (int64_t)dim_idx;
        int64_t off_e = base_off + (int64_t)d_even;
        int64_t off_o = off_e + 1;
        float dy_e = load_as_f32<T>(dy[off_e]);
        float dy_o = load_as_f32<T>(dy[off_o]);
        float out;
        if (!is_high) {
            // dx[2i]   = dy[2i]   · cos + dy[2i+1] · sin
            out = dy_e * c + dy_o * si;
        } else {
            // dx[2i+1] = dy[2i+1] · cos - dy[2i] · sin
            out = dy_o * c - dy_e * si;
        }
        dx[lin] = store_from_f32<T>(out);
    }
}

template <>
__global__ void rope_apply_backward_fp_kernel<double>(
    const double* __restrict__ dy,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    double* __restrict__ dx,
    int32_t bh,
    int32_t td,
    int32_t d,
    int32_t stride_b)
{
    int64_t total = (int64_t)bh * (int64_t)td;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int32_t half_d = d >> 1;
    int32_t seq = td / d;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t dim_idx = (int32_t)(lin % (int64_t)d);
        int64_t rest    = lin / (int64_t)d;
        int32_t s       = (int32_t)(rest % (int64_t)seq);
        int64_t bh_row  = rest / (int64_t)seq;
        int32_t pair    = dim_idx >> 1;
        int32_t d_even  = pair << 1;
        bool is_high    = (dim_idx & 1) != 0;
        int64_t cs_off  = (int64_t)bh_row * (int64_t)stride_b
                        + (int64_t)s * (int64_t)half_d
                        + (int64_t)pair;
        double c  = (double)cos_tab[cs_off];
        double si = (double)sin_tab[cs_off];
        int64_t base_off = lin - (int64_t)dim_idx;
        int64_t off_e = base_off + (int64_t)d_even;
        int64_t off_o = off_e + 1;
        double dy_e = dy[off_e];
        double dy_o = dy[off_o];
        double out;
        if (!is_high) {
            out = dy_e * c + dy_o * si;
        } else {
            out = dy_o * c - dy_e * si;
        }
        dx[lin] = out;
    }
}

template <typename T>
__host__ inline int32_t launch_rope_apply_backward_fp(
    const T* dy,
    const float* cos_tab,
    const float* sin_tab,
    T* dx,
    int32_t bh, int32_t td, int32_t d, int32_t stride_b,
    cudaStream_t stream)
{
    if (bh < 0 || td < 0 || d < 0 || stride_b < 0) return 2;
    if (d == 0) return 2;
    if (d % 2 != 0) return 2;
    if (td % d != 0) return 2;
    int64_t total = (int64_t)bh * (int64_t)td;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    rope_apply_backward_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, cos_tab, sin_tab, dx, bh, td, d, stride_b);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::attention

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launcher symbols per dtype.
// =============================================================================

#define BARACUDA_KERNELS_ROPE_INSTANTIATE(NAME, T)                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                           \
        int32_t batch,                                                                          \
        int32_t heads,                                                                          \
        int32_t seq,                                                                            \
        int32_t head_dim,                                                                       \
        float base,                                                                             \
        int32_t pos_default_flag,                                                               \
        const void* x,                                                                          \
        const void* positions,                                                                  \
        void* y,                                                                                \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (batch < 0 || heads < 0 || seq < 0 || head_dim < 0) return 2;                        \
        if (head_dim % 2 != 0) return 2;                                                        \
        int64_t total = (int64_t)batch * heads * seq * head_dim;                                \
        if (total == 0) return 0;                                                               \
        if (x == nullptr || y == nullptr) return 2;                                             \
        if (pos_default_flag == 0 && positions == nullptr) return 2;                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_rope_fp<T>(                                          \
            static_cast<const T*>(x),                                                           \
            static_cast<const int64_t*>(positions),                                             \
            static_cast<T*>(y),                                                                 \
            batch, heads, seq, head_dim, base, pos_default_flag,                                \
            stream);                                                                            \
    }

#define BARACUDA_KERNELS_ROPE_BACKWARD_INSTANTIATE(NAME, T)                                     \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                           \
        int32_t batch,                                                                          \
        int32_t heads,                                                                          \
        int32_t seq,                                                                            \
        int32_t head_dim,                                                                       \
        float base,                                                                             \
        int32_t pos_default_flag,                                                               \
        const void* dy,                                                                         \
        const void* positions,                                                                  \
        void* dx,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (batch < 0 || heads < 0 || seq < 0 || head_dim < 0) return 2;                        \
        if (head_dim % 2 != 0) return 2;                                                        \
        int64_t total = (int64_t)batch * heads * seq * head_dim;                                \
        if (total == 0) return 0;                                                               \
        if (dy == nullptr || dx == nullptr) return 2;                                           \
        if (pos_default_flag == 0 && positions == nullptr) return 2;                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_rope_backward_fp<T>(                                 \
            static_cast<const T*>(dy),                                                          \
            static_cast<const int64_t*>(positions),                                             \
            static_cast<T*>(dx),                                                                \
            batch, heads, seq, head_dim, base, pos_default_flag,                                \
            stream);                                                                            \
    }

// Strided sibling INSTANTIATEs — Phase 14.4. One symbol per dtype:
// `baracuda_kernels_rope_<dtype>_strided_run` and
// `baracuda_kernels_rope_backward_<dtype>_strided_run`. The innermost
// `head_dim` axis is implicitly stride=1 (caller-enforced); only the
// three outer dims (batch, heads, seq) carry signed-i64 strides.
#define BARACUDA_KERNELS_ROPE_STRIDED_INSTANTIATE(NAME, T)                                      \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                   \
        int32_t batch,                                                                          \
        int32_t heads,                                                                          \
        int32_t seq,                                                                            \
        int32_t head_dim,                                                                       \
        int64_t stride_x_b, int64_t stride_x_h, int64_t stride_x_s,                             \
        int64_t stride_y_b, int64_t stride_y_h, int64_t stride_y_s,                             \
        float base,                                                                             \
        int32_t pos_default_flag,                                                               \
        const void* x,                                                                          \
        const void* positions,                                                                  \
        void* y,                                                                                \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (batch < 0 || heads < 0 || seq < 0 || head_dim < 0) return 2;                        \
        if (head_dim % 2 != 0) return 2;                                                        \
        int64_t total = (int64_t)batch * heads * seq * head_dim;                                \
        if (total == 0) return 0;                                                               \
        if (x == nullptr || y == nullptr) return 2;                                             \
        if (pos_default_flag == 0 && positions == nullptr) return 2;                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_rope_strided_fp<T>(                                  \
            static_cast<const T*>(x),                                                           \
            static_cast<const int64_t*>(positions),                                             \
            static_cast<T*>(y),                                                                 \
            batch, heads, seq, head_dim,                                                        \
            stride_x_b, stride_x_h, stride_x_s,                                                 \
            stride_y_b, stride_y_h, stride_y_s,                                                 \
            base, pos_default_flag,                                                             \
            stream);                                                                            \
    }

#define BARACUDA_KERNELS_ROPE_BACKWARD_STRIDED_INSTANTIATE(NAME, T)                             \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                   \
        int32_t batch,                                                                          \
        int32_t heads,                                                                          \
        int32_t seq,                                                                            \
        int32_t head_dim,                                                                       \
        int64_t stride_dy_b, int64_t stride_dy_h, int64_t stride_dy_s,                          \
        int64_t stride_dx_b, int64_t stride_dx_h, int64_t stride_dx_s,                          \
        float base,                                                                             \
        int32_t pos_default_flag,                                                               \
        const void* dy,                                                                         \
        const void* positions,                                                                  \
        void* dx,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (batch < 0 || heads < 0 || seq < 0 || head_dim < 0) return 2;                        \
        if (head_dim % 2 != 0) return 2;                                                        \
        int64_t total = (int64_t)batch * heads * seq * head_dim;                                \
        if (total == 0) return 0;                                                               \
        if (dy == nullptr || dx == nullptr) return 2;                                           \
        if (pos_default_flag == 0 && positions == nullptr) return 2;                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_rope_backward_strided_fp<T>(                         \
            static_cast<const T*>(dy),                                                          \
            static_cast<const int64_t*>(positions),                                             \
            static_cast<T*>(dx),                                                                \
            batch, heads, seq, head_dim,                                                        \
            stride_dy_b, stride_dy_h, stride_dy_s,                                              \
            stride_dx_b, stride_dx_h, stride_dx_s,                                              \
            base, pos_default_flag,                                                             \
            stream);                                                                            \
    }

// RoPE apply INSTANTIATE — Phase 36 (Fuel ask Gap 2). Coexists with the
// `BARACUDA_KERNELS_ROPE_INSTANTIATE` symbols (which generate θ
// internally); the apply variant accepts caller-supplied cos/sin tables.
// Signature:
//   (bh, td, d, stride_b, x, cos, sin, y, ws, ws_bytes, stream)
//
// `bh`       = outer = batch * heads
// `td`       = seq * head_dim per (batch, head)
// `d`        = head_dim (must be even)
// `stride_b` = 0 if cos/sin shared across all bh rows;
//              td/2 if per-row cos/sin
// `cos`/`sin` always stored as `float` (f32) regardless of the operand
//   dtype — the same convention as Fuel's Vulkan rope: trig tables are
//   bake-time precomputed in f32. f16 / bf16 detour through f32 inside
//   the kernel for the multiplies; f64 promotes the f32 tables to
//   double at load time.
#define BARACUDA_KERNELS_ROPE_APPLY_INSTANTIATE(NAME, T)                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                           \
        int32_t bh,                                                                             \
        int32_t td,                                                                             \
        int32_t d,                                                                              \
        int32_t stride_b,                                                                       \
        const void* x,                                                                          \
        const void* cos_tab,                                                                    \
        const void* sin_tab,                                                                    \
        void* y,                                                                                \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (bh < 0 || td < 0 || d < 0 || stride_b < 0) return 2;                                \
        if (d == 0) return 2;                                                                   \
        if (d % 2 != 0) return 2;                                                               \
        if (td % d != 0) return 2;                                                              \
        int64_t total = (int64_t)bh * (int64_t)td;                                              \
        if (total == 0) return 0;                                                               \
        if (x == nullptr || y == nullptr) return 2;                                             \
        if (cos_tab == nullptr || sin_tab == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_rope_apply_fp<T>(                                    \
            static_cast<const T*>(x),                                                           \
            static_cast<const float*>(cos_tab),                                                 \
            static_cast<const float*>(sin_tab),                                                 \
            static_cast<T*>(y),                                                                 \
            bh, td, d, stride_b, stream);                                                       \
    }                                                                                            \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                 \
        int32_t bh, int32_t td, int32_t d, int32_t stride_b)                                    \
    {                                                                                            \
        if (bh < 0 || td < 0 || d < 0 || stride_b < 0) return 2;                                \
        if (d == 0) return 2;                                                                   \
        if (d % 2 != 0) return 2;                                                               \
        if (td % d != 0) return 2;                                                              \
        return 0;                                                                               \
    }

#define BARACUDA_KERNELS_ROPE_APPLY_BACKWARD_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                           \
        int32_t bh,                                                                             \
        int32_t td,                                                                             \
        int32_t d,                                                                              \
        int32_t stride_b,                                                                       \
        const void* dy,                                                                         \
        const void* cos_tab,                                                                    \
        const void* sin_tab,                                                                    \
        void* dx,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (bh < 0 || td < 0 || d < 0 || stride_b < 0) return 2;                                \
        if (d == 0) return 2;                                                                   \
        if (d % 2 != 0) return 2;                                                               \
        if (td % d != 0) return 2;                                                              \
        int64_t total = (int64_t)bh * (int64_t)td;                                              \
        if (total == 0) return 0;                                                               \
        if (dy == nullptr || dx == nullptr) return 2;                                           \
        if (cos_tab == nullptr || sin_tab == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_rope_apply_backward_fp<T>(                           \
            static_cast<const T*>(dy),                                                          \
            static_cast<const float*>(cos_tab),                                                 \
            static_cast<const float*>(sin_tab),                                                 \
            static_cast<T*>(dx),                                                                \
            bh, td, d, stride_b, stream);                                                       \
    }                                                                                            \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                 \
        int32_t bh, int32_t td, int32_t d, int32_t stride_b)                                    \
    {                                                                                            \
        if (bh < 0 || td < 0 || d < 0 || stride_b < 0) return 2;                                \
        if (d == 0) return 2;                                                                   \
        if (d % 2 != 0) return 2;                                                               \
        if (td % d != 0) return 2;                                                              \
        return 0;                                                                               \
    }

#define BARACUDA_KERNELS_ALIBI_INSTANTIATE(NAME, T)                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                           \
        int32_t batch,                                                                          \
        int32_t heads,                                                                          \
        int32_t q_len,                                                                          \
        int32_t k_len,                                                                          \
        const void* scores,                                                                     \
        const void* slopes,                                                                     \
        void* y,                                                                                \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0) return 2;                         \
        int64_t total = (int64_t)batch * heads * q_len * k_len;                                 \
        if (total == 0) return 0;                                                               \
        if (scores == nullptr || slopes == nullptr || y == nullptr) return 2;                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_alibi_fp<T>(                                         \
            static_cast<const T*>(scores),                                                      \
            static_cast<const T*>(slopes),                                                      \
            static_cast<T*>(y),                                                                 \
            batch, heads, q_len, k_len,                                                         \
            stream);                                                                            \
    }

#define BARACUDA_KERNELS_ALIBI_BACKWARD_INSTANTIATE(NAME, T)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                           \
        int32_t batch,                                                                          \
        int32_t heads,                                                                          \
        int32_t q_len,                                                                          \
        int32_t k_len,                                                                          \
        const void* dy,                                                                         \
        void* da,                                                                               \
        void* dslope,                                                                           \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0) return 2;                         \
        int64_t total = (int64_t)batch * heads * q_len * k_len;                                 \
        if (total == 0) return 0;                                                               \
        if (dy == nullptr) return 2;                                                            \
        if (da == nullptr && dslope == nullptr) return 2;                                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_alibi_backward_fp<T>(                                \
            static_cast<const T*>(dy),                                                          \
            static_cast<T*>(da),                                                                \
            static_cast<T*>(dslope),                                                            \
            batch, heads, q_len, k_len,                                                         \
            stream);                                                                            \
    }

#define BARACUDA_KERNELS_KV_CACHE_APPEND_INSTANTIATE(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                           \
        int32_t batch,                                                                          \
        int32_t heads,                                                                          \
        int32_t new_len,                                                                        \
        int32_t max_cache_len,                                                                  \
        int32_t d_k,                                                                            \
        int32_t d_v,                                                                            \
        const void* k_new,                                                                      \
        const void* v_new,                                                                      \
        const void* cache_offsets,                                                              \
        void* k_cache,                                                                          \
        void* v_cache,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        if (batch < 0 || heads < 0 || new_len < 0 || max_cache_len < 0 ||                       \
            d_k < 0 || d_v < 0) return 2;                                                       \
        if (batch == 0 || heads == 0 || new_len == 0) return 0;                                 \
        if (k_new == nullptr || v_new == nullptr || cache_offsets == nullptr) return 2;         \
        if (k_cache == nullptr || v_cache == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::attention::launch_kv_cache_append<T>(                                  \
            static_cast<const T*>(k_new),                                                       \
            static_cast<const T*>(v_new),                                                       \
            static_cast<const int64_t*>(cache_offsets),                                         \
            static_cast<T*>(k_cache),                                                           \
            static_cast<T*>(v_cache),                                                           \
            batch, heads, new_len, max_cache_len, d_k, d_v,                                     \
            stream);                                                                            \
    }

#endif // BARACUDA_ATTENTION_CUH
