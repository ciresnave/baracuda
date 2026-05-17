// baracuda_sdpa.cuh
//
// Templated kernels and INSTANTIATE macros for naive Scaled Dot-Product
// Attention (SDPA) — Phase 6 Milestone 6.2 of Category K.
//
// FW formula (PyTorch `F.scaled_dot_product_attention` baseline that
// materializes the full attention matrix):
//
//     scores = (Q @ K^T) * scale          shape [B, H, Q, K]
//     scores += mask (optional)           same shape (broadcast-ready)
//     scores += causal mask (optional)    upper-tri positions → -inf
//     attn   = row-softmax(scores)        shape [B, H, Q, K]   (saved for BW)
//     y      = attn @ V                   shape [B, H, Q, D_v]
//
// BW (given upstream dy of shape [B, H, Q, D_v]):
//     dV     = attn^T @ dy                shape [B, H, K, D_v]
//     dattn  = dy @ V^T                   shape [B, H, Q, K]
//     dscores= softmax_bw(attn, dattn)    row-wise, reuses the standard
//                                          softmax-BW formula
//                                          dscores[i] = attn[i]·(dattn[i] − Σ_j attn[j]·dattn[j])
//     dQ     = dscores @ K * scale        shape [B, H, Q, D_k]
//     dK     = dscores^T @ Q * scale      shape [B, H, K, D_k]
//
// Naive three-kernel pipeline (FW) / five-kernel pipeline (BW). All
// kernels iterate one thread per output cell — no tiling, no FA-style
// fusion. f16 / bf16 accumulate in f32 throughout; f64 uses native double.
//
// Status codes match the family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.
//
// The `scores` tensor (FW intermediate) is reused as `attn` after the
// row-softmax kernel rewrites it in-place. Caller allocates it as the
// `attn` argument — no separate workspace is needed beyond that.

#ifndef BARACUDA_SDPA_CUH
#define BARACUDA_SDPA_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace sdpa {

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
// Kernel 1: scores = Q @ K^T * scale  (+ mask, + causal mask)
// =============================================================================
//
// One thread per output cell (b, h, i, j) ∈ [B, H, Q, K]. Each thread
// computes a `d_k`-length dot product Σ_d Q[b, h, i, d] · K[b, h, j, d]
// and multiplies by `scale`. If a per-cell mask tensor is present it is
// added; if `is_causal != 0` and j > i, the cell is set to -INF (which
// makes that contribution drop out of the softmax).

template <typename T>
__global__ void sdpa_scores_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ mask,   // may be null
    T* __restrict__ scores,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    float scale,
    int32_t is_causal,
    int32_t has_mask)
{
    int64_t total = (int64_t)batch * heads * q_len * k_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t j = (int32_t)(lin % (int64_t)k_len);
        int64_t r = lin / (int64_t)k_len;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        r /= (int64_t)q_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        // Q row base: [b, h, i, :] of length d_k
        int64_t q_base = (((int64_t)b * heads + h) * q_len + i) * d_k;
        // K row base: [b, h, j, :]
        int64_t k_base = (((int64_t)b * heads + h) * k_len + j) * d_k;
        float acc = 0.0f;
        for (int32_t d = 0; d < d_k; ++d) {
            float qv = load_as_f32<T>(q[q_base + d]);
            float kv = load_as_f32<T>(k[k_base + d]);
            acc += qv * kv;
        }
        float out = acc * scale;
        if (has_mask != 0) {
            out += load_as_f32<T>(mask[lin]);
        }
        if (is_causal != 0 && j > i) {
            out = -INFINITY;
        }
        scores[lin] = store_from_f32<T>(out);
    }
}

template <>
__global__ void sdpa_scores_kernel<double>(
    const double* __restrict__ q,
    const double* __restrict__ k,
    const double* __restrict__ mask,
    double* __restrict__ scores,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    float scale,
    int32_t is_causal,
    int32_t has_mask)
{
    int64_t total = (int64_t)batch * heads * q_len * k_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double scale_d = (double)scale;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t j = (int32_t)(lin % (int64_t)k_len);
        int64_t r = lin / (int64_t)k_len;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        r /= (int64_t)q_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        int64_t q_base = (((int64_t)b * heads + h) * q_len + i) * d_k;
        int64_t k_base = (((int64_t)b * heads + h) * k_len + j) * d_k;
        double acc = 0.0;
        for (int32_t d = 0; d < d_k; ++d) {
            acc += q[q_base + d] * k[k_base + d];
        }
        double out = acc * scale_d;
        if (has_mask != 0) {
            out += mask[lin];
        }
        if (is_causal != 0 && j > i) {
            out = -INFINITY;
        }
        scores[lin] = out;
    }
}

// =============================================================================
// Kernel 2: row-softmax along last axis (the K axis of [B, H, Q, K])
// =============================================================================
//
// One thread per row (b, h, i). Standard numerically-stable two-pass
// softmax — find row max, sum of exp, then write the per-cell output.
// Writes in-place when scores == attn (the launcher always passes the
// same buffer).

template <typename T>
__global__ void sdpa_row_softmax_kernel(
    const T* __restrict__ scores,
    T* __restrict__ attn,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len)
{
    int64_t total_rows = (int64_t)batch * heads * q_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t row = tid; row < total_rows; row += step) {
        int64_t base = row * (int64_t)k_len;
        // Pass 1a — row max.
        float m = -INFINITY;
        for (int32_t j = 0; j < k_len; ++j) {
            float v = load_as_f32<T>(scores[base + j]);
            if (v > m) m = v;
        }
        // Edge case: all -inf (e.g. fully masked row under causal mask
        // when q_len > k_len in some recipe). Avoid div-by-zero by
        // emitting an all-zero output (PyTorch matches this when the
        // softmax sees no finite entries — the result is implementation
        // -defined NaN; zeros are a safer trailblazer choice).
        bool row_all_neg_inf = !isfinite(m);
        // Pass 1b — sum of exp.
        float s = 0.0f;
        if (!row_all_neg_inf) {
            for (int32_t j = 0; j < k_len; ++j) {
                s += expf(load_as_f32<T>(scores[base + j]) - m);
            }
        }
        // Pass 2 — per-cell write.
        if (row_all_neg_inf || s == 0.0f) {
            for (int32_t j = 0; j < k_len; ++j) {
                attn[base + j] = store_from_f32<T>(0.0f);
            }
        } else {
            float inv_s = 1.0f / s;
            for (int32_t j = 0; j < k_len; ++j) {
                float v = expf(load_as_f32<T>(scores[base + j]) - m) * inv_s;
                attn[base + j] = store_from_f32<T>(v);
            }
        }
    }
}

template <>
__global__ void sdpa_row_softmax_kernel<double>(
    const double* __restrict__ scores,
    double* __restrict__ attn,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len)
{
    int64_t total_rows = (int64_t)batch * heads * q_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t row = tid; row < total_rows; row += step) {
        int64_t base = row * (int64_t)k_len;
        double m = -INFINITY;
        for (int32_t j = 0; j < k_len; ++j) {
            double v = scores[base + j];
            if (v > m) m = v;
        }
        bool row_all_neg_inf = !isfinite(m);
        double s = 0.0;
        if (!row_all_neg_inf) {
            for (int32_t j = 0; j < k_len; ++j) {
                s += exp(scores[base + j] - m);
            }
        }
        if (row_all_neg_inf || s == 0.0) {
            for (int32_t j = 0; j < k_len; ++j) {
                attn[base + j] = 0.0;
            }
        } else {
            double inv_s = 1.0 / s;
            for (int32_t j = 0; j < k_len; ++j) {
                attn[base + j] = exp(scores[base + j] - m) * inv_s;
            }
        }
    }
}

// =============================================================================
// Kernel 3: y = attn @ V
// =============================================================================
//
// One thread per output cell (b, h, i, dv) ∈ [B, H, Q, D_v]. Sums
// Σ_k attn[b, h, i, k] · V[b, h, k, dv].

template <typename T>
__global__ void sdpa_out_kernel(
    const T* __restrict__ attn,
    const T* __restrict__ v,
    T* __restrict__ y,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_v)
{
    int64_t total = (int64_t)batch * heads * q_len * d_v;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t dv = (int32_t)(lin % (int64_t)d_v);
        int64_t r = lin / (int64_t)d_v;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        r /= (int64_t)q_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        // attn row: [b, h, i, :] length k_len
        int64_t a_base = (((int64_t)b * heads + h) * q_len + i) * k_len;
        // V column: V[b, h, k, dv] across k
        int64_t v_base = ((int64_t)b * heads + h) * k_len * d_v + dv;
        float acc = 0.0f;
        for (int32_t kk = 0; kk < k_len; ++kk) {
            float av = load_as_f32<T>(attn[a_base + kk]);
            float vv = load_as_f32<T>(v[v_base + (int64_t)kk * d_v]);
            acc += av * vv;
        }
        y[lin] = store_from_f32<T>(acc);
    }
}

template <>
__global__ void sdpa_out_kernel<double>(
    const double* __restrict__ attn,
    const double* __restrict__ v,
    double* __restrict__ y,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_v)
{
    int64_t total = (int64_t)batch * heads * q_len * d_v;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t dv = (int32_t)(lin % (int64_t)d_v);
        int64_t r = lin / (int64_t)d_v;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        r /= (int64_t)q_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        int64_t a_base = (((int64_t)b * heads + h) * q_len + i) * k_len;
        int64_t v_base = ((int64_t)b * heads + h) * k_len * d_v + dv;
        double acc = 0.0;
        for (int32_t kk = 0; kk < k_len; ++kk) {
            acc += attn[a_base + kk] * v[v_base + (int64_t)kk * d_v];
        }
        y[lin] = acc;
    }
}

// =============================================================================
// FW launcher — fires 3 kernels.
// =============================================================================

template <typename T>
__host__ inline int32_t launch_sdpa_fp(
    const T* q, const T* k, const T* v, const T* mask,
    T* attn, T* y,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    int32_t d_k, int32_t d_v,
    float scale, int32_t is_causal, int32_t has_mask,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;
    int64_t total_scores = (int64_t)batch * heads * q_len * k_len;
    int64_t total_y      = (int64_t)batch * heads * q_len * d_v;
    if (total_scores == 0 || total_y == 0) return 0;

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;

    // Kernel 1 — scores
    {
        int64_t bi = (total_scores + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(bi > kMaxBlocks ? kMaxBlocks : bi);
        if (blocks <= 0) blocks = 1;
        sdpa_scores_kernel<T><<<blocks, kBlock, 0, stream>>>(
            q, k, mask, attn, batch, heads, q_len, k_len, d_k,
            scale, is_causal, has_mask);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    // Kernel 2 — row-softmax (in-place on attn)
    {
        int64_t total_rows = (int64_t)batch * heads * q_len;
        int64_t bi = (total_rows + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(bi > kMaxBlocks ? kMaxBlocks : bi);
        if (blocks <= 0) blocks = 1;
        sdpa_row_softmax_kernel<T><<<blocks, kBlock, 0, stream>>>(
            attn, attn, batch, heads, q_len, k_len);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    // Kernel 3 — y = attn @ V
    {
        int64_t bi = (total_y + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(bi > kMaxBlocks ? kMaxBlocks : bi);
        if (blocks <= 0) blocks = 1;
        sdpa_out_kernel<T><<<blocks, kBlock, 0, stream>>>(
            attn, v, y, batch, heads, q_len, k_len, d_v);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    return 0;
}

// =============================================================================
// BW kernels
// =============================================================================
//
// Kernel B1: dV = attn^T @ dy. One thread per (b, h, k, dv).
//   dV[b, h, k, dv] = Σ_i attn[b, h, i, k] · dy[b, h, i, dv]

template <typename T>
__global__ void sdpa_dV_kernel(
    const T* __restrict__ attn,
    const T* __restrict__ dy,
    T* __restrict__ dV,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_v)
{
    int64_t total = (int64_t)batch * heads * k_len * d_v;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t dv = (int32_t)(lin % (int64_t)d_v);
        int64_t r = lin / (int64_t)d_v;
        int32_t kk = (int32_t)(r % (int64_t)k_len);
        r /= (int64_t)k_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        // attn[b, h, i, kk] across i
        int64_t a_col_base = (((int64_t)b * heads + h) * q_len) * k_len + kk;
        // dy[b, h, i, dv] across i
        int64_t dy_col_base = (((int64_t)b * heads + h) * q_len) * d_v + dv;
        float acc = 0.0f;
        for (int32_t i = 0; i < q_len; ++i) {
            float a = load_as_f32<T>(attn[a_col_base + (int64_t)i * k_len]);
            float d = load_as_f32<T>(dy[dy_col_base + (int64_t)i * d_v]);
            acc += a * d;
        }
        dV[lin] = store_from_f32<T>(acc);
    }
}

template <>
__global__ void sdpa_dV_kernel<double>(
    const double* __restrict__ attn,
    const double* __restrict__ dy,
    double* __restrict__ dV,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_v)
{
    int64_t total = (int64_t)batch * heads * k_len * d_v;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t dv = (int32_t)(lin % (int64_t)d_v);
        int64_t r = lin / (int64_t)d_v;
        int32_t kk = (int32_t)(r % (int64_t)k_len);
        r /= (int64_t)k_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        int64_t a_col_base = (((int64_t)b * heads + h) * q_len) * k_len + kk;
        int64_t dy_col_base = (((int64_t)b * heads + h) * q_len) * d_v + dv;
        double acc = 0.0;
        for (int32_t i = 0; i < q_len; ++i) {
            acc += attn[a_col_base + (int64_t)i * k_len] * dy[dy_col_base + (int64_t)i * d_v];
        }
        dV[lin] = acc;
    }
}

// Kernel B2: dattn = dy @ V^T. One thread per (b, h, i, kk).
//   dattn[b, h, i, kk] = Σ_dv dy[b, h, i, dv] · V[b, h, kk, dv]

template <typename T>
__global__ void sdpa_dattn_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ v,
    T* __restrict__ dattn,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_v)
{
    int64_t total = (int64_t)batch * heads * q_len * k_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t kk = (int32_t)(lin % (int64_t)k_len);
        int64_t r = lin / (int64_t)k_len;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        r /= (int64_t)q_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        int64_t dy_row = (((int64_t)b * heads + h) * q_len + i) * d_v;
        int64_t v_row  = (((int64_t)b * heads + h) * k_len + kk) * d_v;
        float acc = 0.0f;
        for (int32_t dv = 0; dv < d_v; ++dv) {
            float dyv = load_as_f32<T>(dy[dy_row + dv]);
            float vv  = load_as_f32<T>(v[v_row + dv]);
            acc += dyv * vv;
        }
        dattn[lin] = store_from_f32<T>(acc);
    }
}

template <>
__global__ void sdpa_dattn_kernel<double>(
    const double* __restrict__ dy,
    const double* __restrict__ v,
    double* __restrict__ dattn,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_v)
{
    int64_t total = (int64_t)batch * heads * q_len * k_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t kk = (int32_t)(lin % (int64_t)k_len);
        int64_t r = lin / (int64_t)k_len;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        r /= (int64_t)q_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        int64_t dy_row = (((int64_t)b * heads + h) * q_len + i) * d_v;
        int64_t v_row  = (((int64_t)b * heads + h) * k_len + kk) * d_v;
        double acc = 0.0;
        for (int32_t dv = 0; dv < d_v; ++dv) {
            acc += dy[dy_row + dv] * v[v_row + dv];
        }
        dattn[lin] = acc;
    }
}

// Kernel B3: dscores = softmax_bw(attn, dattn) — applied per row of K.
//   dot[i]      = Σ_j attn[i, j] · dattn[i, j]
//   dscores[i,j] = attn[i, j] · (dattn[i, j] − dot[i])
//
// One thread per row (b, h, i). Each thread does one pass to compute
// `dot`, then a second pass to write per-cell dscores. Same shape pair
// as the FW row-softmax.

template <typename T>
__global__ void sdpa_dscores_kernel(
    const T* __restrict__ attn,
    const T* __restrict__ dattn,
    T* __restrict__ dscores,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len)
{
    int64_t total_rows = (int64_t)batch * heads * q_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t row = tid; row < total_rows; row += step) {
        int64_t base = row * (int64_t)k_len;
        float dot = 0.0f;
        for (int32_t j = 0; j < k_len; ++j) {
            float a = load_as_f32<T>(attn[base + j]);
            float d = load_as_f32<T>(dattn[base + j]);
            dot += a * d;
        }
        for (int32_t j = 0; j < k_len; ++j) {
            float a = load_as_f32<T>(attn[base + j]);
            float d = load_as_f32<T>(dattn[base + j]);
            dscores[base + j] = store_from_f32<T>(a * (d - dot));
        }
    }
}

template <>
__global__ void sdpa_dscores_kernel<double>(
    const double* __restrict__ attn,
    const double* __restrict__ dattn,
    double* __restrict__ dscores,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len)
{
    int64_t total_rows = (int64_t)batch * heads * q_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t row = tid; row < total_rows; row += step) {
        int64_t base = row * (int64_t)k_len;
        double dot = 0.0;
        for (int32_t j = 0; j < k_len; ++j) {
            dot += attn[base + j] * dattn[base + j];
        }
        for (int32_t j = 0; j < k_len; ++j) {
            double a = attn[base + j];
            double d = dattn[base + j];
            dscores[base + j] = a * (d - dot);
        }
    }
}

// Kernel B4: dQ = dscores @ K * scale. One thread per (b, h, i, d).
//   dQ[b, h, i, d] = scale · Σ_kk dscores[b, h, i, kk] · K[b, h, kk, d]

template <typename T>
__global__ void sdpa_dQ_kernel(
    const T* __restrict__ dscores,
    const T* __restrict__ k_in,
    T* __restrict__ dQ,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    float scale)
{
    int64_t total = (int64_t)batch * heads * q_len * d_k;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)d_k);
        int64_t r = lin / (int64_t)d_k;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        r /= (int64_t)q_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        int64_t ds_row = (((int64_t)b * heads + h) * q_len + i) * k_len;
        // K[b, h, kk, d] across kk
        int64_t k_col_base = ((int64_t)b * heads + h) * k_len * d_k + d;
        float acc = 0.0f;
        for (int32_t kk = 0; kk < k_len; ++kk) {
            float ds = load_as_f32<T>(dscores[ds_row + kk]);
            float kv = load_as_f32<T>(k_in[k_col_base + (int64_t)kk * d_k]);
            acc += ds * kv;
        }
        dQ[lin] = store_from_f32<T>(acc * scale);
    }
}

template <>
__global__ void sdpa_dQ_kernel<double>(
    const double* __restrict__ dscores,
    const double* __restrict__ k_in,
    double* __restrict__ dQ,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    float scale)
{
    int64_t total = (int64_t)batch * heads * q_len * d_k;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double scale_d = (double)scale;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)d_k);
        int64_t r = lin / (int64_t)d_k;
        int32_t i = (int32_t)(r % (int64_t)q_len);
        r /= (int64_t)q_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        int64_t ds_row = (((int64_t)b * heads + h) * q_len + i) * k_len;
        int64_t k_col_base = ((int64_t)b * heads + h) * k_len * d_k + d;
        double acc = 0.0;
        for (int32_t kk = 0; kk < k_len; ++kk) {
            acc += dscores[ds_row + kk] * k_in[k_col_base + (int64_t)kk * d_k];
        }
        dQ[lin] = acc * scale_d;
    }
}

// Kernel B5: dK = dscores^T @ Q * scale. One thread per (b, h, kk, d).
//   dK[b, h, kk, d] = scale · Σ_i dscores[b, h, i, kk] · Q[b, h, i, d]

template <typename T>
__global__ void sdpa_dK_kernel(
    const T* __restrict__ dscores,
    const T* __restrict__ q_in,
    T* __restrict__ dK,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    float scale)
{
    int64_t total = (int64_t)batch * heads * k_len * d_k;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)d_k);
        int64_t r = lin / (int64_t)d_k;
        int32_t kk = (int32_t)(r % (int64_t)k_len);
        r /= (int64_t)k_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        // dscores[b, h, i, kk] across i
        int64_t ds_col_base = (((int64_t)b * heads + h) * q_len) * k_len + kk;
        // Q[b, h, i, d] across i
        int64_t q_col_base = (((int64_t)b * heads + h) * q_len) * d_k + d;
        float acc = 0.0f;
        for (int32_t i = 0; i < q_len; ++i) {
            float ds = load_as_f32<T>(dscores[ds_col_base + (int64_t)i * k_len]);
            float qv = load_as_f32<T>(q_in[q_col_base + (int64_t)i * d_k]);
            acc += ds * qv;
        }
        dK[lin] = store_from_f32<T>(acc * scale);
    }
}

template <>
__global__ void sdpa_dK_kernel<double>(
    const double* __restrict__ dscores,
    const double* __restrict__ q_in,
    double* __restrict__ dK,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    float scale)
{
    int64_t total = (int64_t)batch * heads * k_len * d_k;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double scale_d = (double)scale;
    for (int64_t lin = tid; lin < total; lin += step) {
        int32_t d = (int32_t)(lin % (int64_t)d_k);
        int64_t r = lin / (int64_t)d_k;
        int32_t kk = (int32_t)(r % (int64_t)k_len);
        r /= (int64_t)k_len;
        int32_t h = (int32_t)(r % (int64_t)heads);
        int32_t b = (int32_t)(r / (int64_t)heads);
        int64_t ds_col_base = (((int64_t)b * heads + h) * q_len) * k_len + kk;
        int64_t q_col_base = (((int64_t)b * heads + h) * q_len) * d_k + d;
        double acc = 0.0;
        for (int32_t i = 0; i < q_len; ++i) {
            acc += dscores[ds_col_base + (int64_t)i * k_len] * q_in[q_col_base + (int64_t)i * d_k];
        }
        dK[lin] = acc * scale_d;
    }
}

// =============================================================================
// BW launcher — fires 5 kernels. The caller-allocated `dscores` workspace
// doubles as `dattn` storage (B2 writes dattn, B3 overwrites in place).
// =============================================================================

template <typename T>
__host__ inline int32_t launch_sdpa_backward_fp(
    const T* q, const T* k_in, const T* v, const T* attn, const T* dy,
    T* dscores_ws,   // [B, H, Q, K] scratch — reused as dattn then dscores
    T* dQ, T* dK, T* dV,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    int32_t d_k, int32_t d_v,
    float scale,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;
    int64_t total_attn = (int64_t)batch * heads * q_len * k_len;
    if (total_attn == 0) return 0;

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;

    auto grid_for = [&](int64_t total) {
        int64_t bi = (total + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(bi > kMaxBlocks ? kMaxBlocks : bi);
        if (blocks <= 0) blocks = 1;
        return blocks;
    };

    // B1 — dV
    {
        int64_t total = (int64_t)batch * heads * k_len * d_v;
        if (total > 0) {
            sdpa_dV_kernel<T><<<grid_for(total), kBlock, 0, stream>>>(
                attn, dy, dV, batch, heads, q_len, k_len, d_v);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return 5;
        }
    }
    // B2 — dattn = dy @ V^T (written into dscores_ws)
    {
        sdpa_dattn_kernel<T><<<grid_for(total_attn), kBlock, 0, stream>>>(
            dy, v, dscores_ws, batch, heads, q_len, k_len, d_v);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    // B3 — dscores = softmax_bw(attn, dattn). In-place rewrite of dscores_ws.
    {
        int64_t total_rows = (int64_t)batch * heads * q_len;
        sdpa_dscores_kernel<T><<<grid_for(total_rows), kBlock, 0, stream>>>(
            attn, dscores_ws, dscores_ws, batch, heads, q_len, k_len);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    // B4 — dQ
    {
        int64_t total = (int64_t)batch * heads * q_len * d_k;
        if (total > 0) {
            sdpa_dQ_kernel<T><<<grid_for(total), kBlock, 0, stream>>>(
                dscores_ws, k_in, dQ, batch, heads, q_len, k_len, d_k, scale);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return 5;
        }
    }
    // B5 — dK
    {
        int64_t total = (int64_t)batch * heads * k_len * d_k;
        if (total > 0) {
            sdpa_dK_kernel<T><<<grid_for(total), kBlock, 0, stream>>>(
                dscores_ws, q, dK, batch, heads, q_len, k_len, d_k, scale);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return 5;
        }
    }
    return 0;
}

} } // namespace baracuda::sdpa

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launcher symbols per dtype.
// =============================================================================

#define BARACUDA_KERNELS_SDPA_INSTANTIATE(NAME, T)                                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                               \
        int32_t batch,                                                                              \
        int32_t heads,                                                                              \
        int32_t q_len,                                                                              \
        int32_t k_len,                                                                              \
        int32_t d_k,                                                                                \
        int32_t d_v,                                                                                \
        float scale,                                                                                \
        int32_t is_causal,                                                                          \
        int32_t has_mask,                                                                           \
        const void* q,                                                                              \
        const void* k,                                                                              \
        const void* v,                                                                              \
        const void* mask,                                                                           \
        void* attn,                                                                                 \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                                \
        if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;       \
        int64_t total_attn = (int64_t)batch * heads * q_len * k_len;                                \
        int64_t total_y    = (int64_t)batch * heads * q_len * d_v;                                  \
        if (total_attn == 0 || total_y == 0) return 0;                                              \
        if (q == nullptr || k == nullptr || v == nullptr || attn == nullptr || y == nullptr)        \
            return 2;                                                                                \
        if (has_mask != 0 && mask == nullptr) return 2;                                             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::sdpa::launch_sdpa_fp<T>(                                                   \
            static_cast<const T*>(q),                                                               \
            static_cast<const T*>(k),                                                               \
            static_cast<const T*>(v),                                                               \
            static_cast<const T*>(mask),                                                            \
            static_cast<T*>(attn),                                                                  \
            static_cast<T*>(y),                                                                     \
            batch, heads, q_len, k_len, d_k, d_v,                                                   \
            scale, is_causal, has_mask,                                                             \
            stream);                                                                                \
    }

#define BARACUDA_KERNELS_SDPA_BACKWARD_INSTANTIATE(NAME, T)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                               \
        int32_t batch,                                                                              \
        int32_t heads,                                                                              \
        int32_t q_len,                                                                              \
        int32_t k_len,                                                                              \
        int32_t d_k,                                                                                \
        int32_t d_v,                                                                                \
        float scale,                                                                                \
        const void* q,                                                                              \
        const void* k,                                                                              \
        const void* v,                                                                              \
        const void* attn,                                                                           \
        const void* dy,                                                                             \
        void* dscores_ws,                                                                           \
        void* dQ,                                                                                   \
        void* dK,                                                                                   \
        void* dV,                                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                                \
        if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;       \
        int64_t total_attn = (int64_t)batch * heads * q_len * k_len;                                \
        if (total_attn == 0) return 0;                                                              \
        if (q == nullptr || k == nullptr || v == nullptr || attn == nullptr || dy == nullptr)       \
            return 2;                                                                                \
        if (dscores_ws == nullptr) return 2;                                                        \
        if (dQ == nullptr || dK == nullptr || dV == nullptr) return 2;                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::sdpa::launch_sdpa_backward_fp<T>(                                          \
            static_cast<const T*>(q),                                                               \
            static_cast<const T*>(k),                                                               \
            static_cast<const T*>(v),                                                               \
            static_cast<const T*>(attn),                                                            \
            static_cast<const T*>(dy),                                                              \
            static_cast<T*>(dscores_ws),                                                            \
            static_cast<T*>(dQ),                                                                    \
            static_cast<T*>(dK),                                                                    \
            static_cast<T*>(dV),                                                                    \
            batch, heads, q_len, k_len, d_k, d_v,                                                   \
            scale,                                                                                  \
            stream);                                                                                \
    }

#endif // BARACUDA_SDPA_CUH
