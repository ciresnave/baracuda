// baracuda-kernels Phase 50 — causal-conv1d backward (FP types).
//
// Hand-port of Tri Dao's causal-conv1d BW (BSD-3-Clause).
// See `vendor/causal-conv1d/VENDOR.md` for upstream attribution.
//
// Inputs (from FW):
//   x      : [B, C, L]  — the original input.
//   weight : [C, W]     — the filter.
//   bias   : [C] or null
//   y_pre  : [B, C, L] or null — the pre-activation output (only
//            needed when `use_silu`). When null and use_silu=1, the
//            kernel recomputes pre-act = conv(x) + bias on the fly.
//   dy     : [B, C, L]  — gradient w.r.t. output (post-activation).
//
// Outputs:
//   dx     : [B, C, L]  — gradient w.r.t. input.
//   dw     : [C, W]     — gradient w.r.t. filter (atomicAdd accumulated).
//   db     : [C] or null — gradient w.r.t. bias (atomicAdd accumulated).
//
// Derivation:
//   y_pre[b, c, t] = sum_{k=0..W-1} w[c, k] * x[b, c, t - (W-1-k)]
//                    + bias[c]
//   y[b, c, t] = act(y_pre)  -- SiLU or identity
//
//   dy_pre = dy * act'(y_pre)
//   dx[b, c, t']  = sum over k where t - (W-1-k) == t', i.e. k = W-1 - (t - t'),
//                   summed over output positions t = t' + (W-1-k):
//                   dx[b, c, t'] = sum_{k=0..W-1} w[c, k] * dy_pre[b, c, t' + (W-1-k)]
//                   IF t' + (W-1-k) < L, else 0.
//   dw[c, k] = sum_{b, t} x_padded[b, c, t - (W-1-k)] * dy_pre[b, c, t]
//   db[c]    = sum_{b, t} dy_pre[b, c, t]
//
// Implementation: one thread per (b, c, t) output cell. The dx
// contribution per thread is local (W reads of dy_pre, W reads of w,
// one write to dx). The dw / db contributions are accumulated with
// atomicAdd. Backward is therefore deterministic in dx but not in
// dw / db (FP atomicAdd order-dependent).
//
// db / dw atomicAdd: f32 / f64 / f16 / bf16 atomicAdd are all supported
// natively on sm_80+ (no atomicCAS dance needed).

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace causal_conv1d_bw {

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

// SiLU + derivative.
__device__ __forceinline__ float silu_grad_f32(float x) {
    float sigma = 1.0f / (1.0f + __expf(-x));
    float silu  = x * sigma;
    return sigma + silu * (1.0f - sigma);
}

__device__ __forceinline__ double silu_grad_f64(double x) {
    double sigma = 1.0 / (1.0 + exp(-x));
    double silu  = x * sigma;
    return sigma + silu * (1.0 - sigma);
}

// AtomicAdd dispatchers (f16/bf16 handled natively on sm_80+).
template <typename T>
__device__ __forceinline__ void atomic_add_T(T* addr, float v) {
    atomicAdd(addr, (T)v);
}

template <>
__device__ __forceinline__ void atomic_add_T<__half>(__half* addr, float v) {
    atomicAdd(addr, __float2half(v));
}

template <>
__device__ __forceinline__ void atomic_add_T<__nv_bfloat16>(__nv_bfloat16* addr, float v) {
    atomicAdd(addr, __float2bfloat16(v));
}

template <>
__device__ __forceinline__ void atomic_add_T<double>(double* addr, float v) {
    atomicAdd(addr, (double)v);
}

__device__ __forceinline__ void atomic_add_T_double(double* addr, double v) {
    atomicAdd(addr, v);
}

// =========================================================================
// BW kernel — FP version
// =========================================================================
//
// One thread per (b, c, t) output cell. The thread:
//   1. Recomputes pre-activation if needed (cheap, W <= 4 reads).
//   2. Multiplies dy by SiLU' if use_silu.
//   3. Accumulates dx (local), dw / db (atomicAdd).
//
// Note: the dx accumulation is local because each output position t
// only contributes to W input positions (one per filter tap). We
// iterate over the filter taps and add the matching contributions.

template <typename T, int W>
__global__ void causal_conv1d_bwd_kernel(
    const T* __restrict__ x,         // [B, C, L]
    const T* __restrict__ weight,    // [C, W]
    const T* __restrict__ bias,      // [C] or null
    const T* __restrict__ dy,        // [B, C, L]
    T* __restrict__ dx,              // [B, C, L]
    T* __restrict__ dw,              // [C, W]  (must be zero-init by caller)
    T* __restrict__ db,              // [C] or null  (must be zero-init by caller)
    int32_t batch,
    int32_t channels,
    int32_t seqlen,
    int32_t use_silu)
{
    const int64_t total = (int64_t)batch * channels * seqlen;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int32_t t = (int32_t)(tid % seqlen);
    const int32_t c = (int32_t)((tid / seqlen) % channels);
    const int32_t b = (int32_t)(tid / ((int64_t)seqlen * channels));
    const int64_t bc_off = ((int64_t)b * channels + c) * seqlen;

    // Pre-load weight tap into regs.
    float w_f32[W];
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        w_f32[k] = load_as_f32<T>(weight[c * W + k]);
    }

    // Compute dy_pre = dy * silu_grad(pre_act) when use_silu, else dy.
    float dy_post = load_as_f32<T>(dy[bc_off + t]);
    float dy_pre  = dy_post;
    if (use_silu) {
        // Recompute pre-act on the fly (W reads of x + one bias).
        float pre = 0.0f;
        #pragma unroll
        for (int k = 0; k < W; ++k) {
            int32_t xi = t - (W - 1 - k);
            if (xi >= 0) {
                pre += w_f32[k] * load_as_f32<T>(x[bc_off + xi]);
            }
        }
        if (bias != nullptr) pre += load_as_f32<T>(bias[c]);
        dy_pre = dy_post * silu_grad_f32(pre);
    }

    // dx contribution: dx[b, c, t'] += sum_{k} w[c, k] * dy_pre[b, c, t' + (W-1-k)]
    // From this thread's viewpoint (we own dy_pre at output position t),
    // it contributes to dx[b, c, t - (W-1-k)] for each k.
    // To make this race-free we instead compute dx[b, c, t] directly by
    // summing over all output positions that contribute to this input
    // position. Since we use 1 thread per (b, c, t), we let t be the
    // INPUT index for dx (and recompute the relevant dy_pre values).
    //
    // Actually it's simpler to use this thread's t as the DX index:
    //   dx[b, c, t] = sum_{k=0..W-1} w[c, k] * dy_pre[b, c, t + (W-1-k)]
    //                  (clipped to t' < L)
    // For dy_pre recompute under use_silu we'd need to know the pre-act
    // at the OTHER positions (t + (W-1-k)), not just at t. To keep this
    // O(W) per thread when use_silu, we use a two-pass scheme: first
    // pass writes dy_pre to a scratch buffer; second pass reads it.
    //
    // To avoid that complexity, we use a simpler split-kernel pipeline:
    //   pass 1: compute dy_pre[b, c, t] (write to scratch or in-place
    //           on dy if caller permits).
    //   pass 2: compute dx, atomic dw, atomic db using dy_pre.
    //
    // For this single-kernel path we'll RECOMPUTE the W pre-acts we
    // need (cheap; total work is O(W^2) per output cell).

    float dx_f32 = 0.0f;
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        int32_t t_out = t + (W - 1 - k);
        if (t_out < seqlen) {
            // dy_pre at output position t_out.
            float dy_post_k = load_as_f32<T>(dy[bc_off + t_out]);
            float dy_pre_k  = dy_post_k;
            if (use_silu) {
                float pre_k = 0.0f;
                #pragma unroll
                for (int j = 0; j < W; ++j) {
                    int32_t xi = t_out - (W - 1 - j);
                    if (xi >= 0) {
                        pre_k += w_f32[j] * load_as_f32<T>(x[bc_off + xi]);
                    }
                }
                if (bias != nullptr) pre_k += load_as_f32<T>(bias[c]);
                dy_pre_k = dy_post_k * silu_grad_f32(pre_k);
            }
            dx_f32 += w_f32[k] * dy_pre_k;
        }
    }
    dx[bc_off + t] = store_from_f32<T>(dx_f32);

    // dw / db: atomic accumulation per (c, k) / per c.
    // Contribution from this output position t:
    //   dw[c, k] += x_padded[b, c, t - (W-1-k)] * dy_pre[t]
    //   db[c]    += dy_pre[t]
    // (Note: db is accumulated once per (b, c, t) thread; we use the
    // dy_pre value we already computed up top.)
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        int32_t xi = t - (W - 1 - k);
        if (xi >= 0) {
            float x_f32 = load_as_f32<T>(x[bc_off + xi]);
            atomic_add_T<T>(&dw[c * W + k], x_f32 * dy_pre);
        }
    }
    if (db != nullptr) {
        atomic_add_T<T>(&db[c], dy_pre);
    }
}

// f64 specialisation
template <int W>
__global__ void causal_conv1d_bwd_kernel_f64(
    const double* __restrict__ x,
    const double* __restrict__ weight,
    const double* __restrict__ bias,
    const double* __restrict__ dy,
    double* __restrict__ dx,
    double* __restrict__ dw,
    double* __restrict__ db,
    int32_t batch,
    int32_t channels,
    int32_t seqlen,
    int32_t use_silu)
{
    const int64_t total = (int64_t)batch * channels * seqlen;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int32_t t = (int32_t)(tid % seqlen);
    const int32_t c = (int32_t)((tid / seqlen) % channels);
    const int32_t b = (int32_t)(tid / ((int64_t)seqlen * channels));
    const int64_t bc_off = ((int64_t)b * channels + c) * seqlen;

    double w_f64[W];
    #pragma unroll
    for (int k = 0; k < W; ++k) w_f64[k] = weight[c * W + k];

    double dy_post = dy[bc_off + t];
    double dy_pre  = dy_post;
    if (use_silu) {
        double pre = 0.0;
        #pragma unroll
        for (int k = 0; k < W; ++k) {
            int32_t xi = t - (W - 1 - k);
            if (xi >= 0) pre += w_f64[k] * x[bc_off + xi];
        }
        if (bias != nullptr) pre += bias[c];
        dy_pre = dy_post * silu_grad_f64(pre);
    }

    double dx_d = 0.0;
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        int32_t t_out = t + (W - 1 - k);
        if (t_out < seqlen) {
            double dy_post_k = dy[bc_off + t_out];
            double dy_pre_k  = dy_post_k;
            if (use_silu) {
                double pre_k = 0.0;
                #pragma unroll
                for (int j = 0; j < W; ++j) {
                    int32_t xi = t_out - (W - 1 - j);
                    if (xi >= 0) pre_k += w_f64[j] * x[bc_off + xi];
                }
                if (bias != nullptr) pre_k += bias[c];
                dy_pre_k = dy_post_k * silu_grad_f64(pre_k);
            }
            dx_d += w_f64[k] * dy_pre_k;
        }
    }
    dx[bc_off + t] = dx_d;

    #pragma unroll
    for (int k = 0; k < W; ++k) {
        int32_t xi = t - (W - 1 - k);
        if (xi >= 0) {
            atomic_add_T_double(&dw[c * W + k], x[bc_off + xi] * dy_pre);
        }
    }
    if (db != nullptr) {
        atomic_add_T_double(&db[c], dy_pre);
    }
}

template <typename T>
int32_t launch_causal_conv1d_bwd(
    const T* x, const T* weight, const T* bias, const T* dy,
    T* dx, T* dw, T* db,
    int32_t batch, int32_t channels, int32_t seqlen, int32_t width,
    int32_t use_silu, cudaStream_t stream)
{
    if (width < 2 || width > 4) return 3;
    if (batch == 0 || channels == 0 || seqlen == 0) return 0;

    // Caller is responsible for zero-init of dw and db before launch.
    // (Documented in the Rust plan layer.)

    const int64_t total = (int64_t)batch * channels * seqlen;
    const int32_t threads_per_block = 256;
    const int64_t blocks = (total + threads_per_block - 1) / threads_per_block;
    if (blocks > (int64_t)0x7FFFFFFF) return 3;
    dim3 grid((unsigned)blocks);
    dim3 block(threads_per_block);

    if constexpr (sizeof(T) == sizeof(double)) {
        if (width == 2) causal_conv1d_bwd_kernel_f64<2><<<grid, block, 0, stream>>>(
            (const double*)x, (const double*)weight, (const double*)bias, (const double*)dy,
            (double*)dx, (double*)dw, (double*)db,
            batch, channels, seqlen, use_silu);
        else if (width == 3) causal_conv1d_bwd_kernel_f64<3><<<grid, block, 0, stream>>>(
            (const double*)x, (const double*)weight, (const double*)bias, (const double*)dy,
            (double*)dx, (double*)dw, (double*)db,
            batch, channels, seqlen, use_silu);
        else /* 4 */ causal_conv1d_bwd_kernel_f64<4><<<grid, block, 0, stream>>>(
            (const double*)x, (const double*)weight, (const double*)bias, (const double*)dy,
            (double*)dx, (double*)dw, (double*)db,
            batch, channels, seqlen, use_silu);
    } else {
        if (width == 2) causal_conv1d_bwd_kernel<T, 2><<<grid, block, 0, stream>>>(
            x, weight, bias, dy, dx, dw, db, batch, channels, seqlen, use_silu);
        else if (width == 3) causal_conv1d_bwd_kernel<T, 3><<<grid, block, 0, stream>>>(
            x, weight, bias, dy, dx, dw, db, batch, channels, seqlen, use_silu);
        else /* 4 */ causal_conv1d_bwd_kernel<T, 4><<<grid, block, 0, stream>>>(
            x, weight, bias, dy, dx, dw, db, batch, channels, seqlen, use_silu);
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}}  // namespace baracuda::causal_conv1d_bw

#define BARACUDA_CAUSAL_CONV1D_BWD_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_backward_run(                                \
        int32_t batch, int32_t channels, int32_t seqlen, int32_t width,                       \
        int32_t use_silu,                                                                     \
        const void* x, const void* weight, const void* bias, const void* dy,                  \
        void* dx, void* dw, void* db,                                                         \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                      \
        void* stream_ptr)                                                                     \
    {                                                                                          \
        if (batch < 0 || channels < 0 || seqlen < 0) return 2;                                \
        if (width < 2 || width > 4) return 3;                                                 \
        if (batch == 0 || channels == 0 || seqlen == 0) return 0;                             \
        if (x == nullptr || weight == nullptr || dy == nullptr) return 2;                     \
        if (dx == nullptr || dw == nullptr) return 2;                                         \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                          \
        return baracuda::causal_conv1d_bw::launch_causal_conv1d_bwd<T>(                       \
            static_cast<const T*>(x),                                                         \
            static_cast<const T*>(weight),                                                    \
            static_cast<const T*>(bias),                                                      \
            static_cast<const T*>(dy),                                                        \
            static_cast<T*>(dx),                                                              \
            static_cast<T*>(dw),                                                              \
            static_cast<T*>(db),                                                              \
            batch, channels, seqlen, width, use_silu, stream);                                \
    }

BARACUDA_CAUSAL_CONV1D_BWD_INSTANTIATE(causal_conv1d_f32,  float)
BARACUDA_CAUSAL_CONV1D_BWD_INSTANTIATE(causal_conv1d_f16,  __half)
BARACUDA_CAUSAL_CONV1D_BWD_INSTANTIATE(causal_conv1d_bf16, __nv_bfloat16)
BARACUDA_CAUSAL_CONV1D_BWD_INSTANTIATE(causal_conv1d_f64,  double)
