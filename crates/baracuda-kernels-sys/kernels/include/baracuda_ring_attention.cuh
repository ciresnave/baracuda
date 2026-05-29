// baracuda_ring_attention.cuh
//
// Sequence-parallel Ring Attention kernel (Phase 56, Tier 1).
//
// Algorithm:
//   Liu, Yan & Abbeel — "Ring Attention with Blockwise Transformers for
//   Near-Infinite Context" (arXiv:2310.01889 / NeurIPS 2023). Apache-2.0
//   reference at https://github.com/lhao499/RingAttention (JAX).
//
// This is a clean-room hand-port of the CUDA equivalent — no upstream
// source vendored. Algorithm credit + license attribution kept in the
// `RingAttentionPlan` Rust module documentation.
//
// =============================================================================
// What the kernel does (PER RANK / PER ROTATION STEP)
// =============================================================================
//
//   Inputs (per rank):
//     q:        T[B, H, Q_local,  D]    Q-slice owned by this rank
//     k_local:  T[B, H, K_chunk,  D]    K-chunk currently resident
//     v_local:  T[B, H, K_chunk,  D]    V-chunk currently resident
//
//   In/out (per rank, persistent across rotation steps):
//     o_acc:    f32[B, H, Q_local, D]   running output accumulator (P @ V partial sums)
//     m_acc:    f32[B, H, Q_local]      running per-row max (online-softmax state)
//     l_acc:    f32[B, H, Q_local]      running per-row denom (online-softmax state)
//
//   This kernel performs ONE step's contribution: it computes the partial
//   attention of Q[my_slice] against the resident K/V chunk and folds
//   the result into (o_acc, m_acc, l_acc) via standard online-softmax
//   reconstruction (Tri Dao 2022 / Milnikov-Davis 2018 streaming softmax).
//
//   After `world_size` kernel launches (one per rotation step) the
//   plan's `finalize` kernel divides o_acc by l_acc and writes the
//   final y in the operand dtype.
//
// =============================================================================
// Causal masking under ring rotation
// =============================================================================
//
//   In a ring with world_size P, rank `r` owns Q-slice
//   `[r * Q_local, (r+1) * Q_local)` (global indices). At rotation step
//   `s`, the resident K/V chunk's global index range is
//   `[((r - s + P) % P) * K_chunk, ...)`. The kernel takes the
//   absolute base-indices (`q_global_base`, `k_global_base`) as launch
//   parameters and applies the standard `q_idx < k_idx → mask` rule
//   on global indices. This means whole-block early-exit is possible
//   (kernel skips chunks whose global k-base is beyond every owned
//   q's global index in causal mode).
//
// =============================================================================
// Tier 1 constraints
// =============================================================================
//
//   - dtype ∈ {__half, __nv_bfloat16}; f32/f64 deferred (Tier 2).
//   - head_dim == 128 (Tier 1; the kernel reads it as a runtime parameter
//     but the launcher rejects anything else for clarity).
//   - Br = Bc = 64 (matches Phase 6.6 FlashAttention tile shape).
//   - No GQA broadcast (deferred to Tier 2).
//   - No arbitrary additive mask (deferred to Tier 2).
//
// =============================================================================

#ifndef BARACUDA_RING_ATTENTION_CUH
#define BARACUDA_RING_ATTENTION_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace ring_attention {

// Tile geometry — matches Phase 6.6 FlashAttention conventions so the
// degenerate (world_size == 1) case is numerically comparable to
// FlashSdpaPlan.
constexpr int kBr = 64;
constexpr int kBc = 64;
constexpr int kThreadsPerBlock = 128;
constexpr int kHeadDim = 128;       // Tier 1 fixes head_dim = 128

// ---------- dtype helpers (f32 detour for f16/bf16) ----------

template <typename T>
__device__ __forceinline__ float load_f32(T x) { return (float)x; }

template <>
__device__ __forceinline__ float load_f32<__half>(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float load_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T store_f32(float v) { return (T)v; }

template <>
__device__ __forceinline__ __half store_f32<__half>(float v) { return __float2half(v); }

template <>
__device__ __forceinline__ __nv_bfloat16 store_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// =============================================================================
// Step kernel — folds one K/V chunk's contribution into the persistent
// (o_acc, m_acc, l_acc) state. One block per (b, h, q_block).
//
// Shared memory layout (per block):
//   sQ   : T[Br * D]
//   sK   : T[Bc * D]
//   sV   : T[Bc * D]
//   sS   : float[Br * Bc]       — score tile then probability tile (in-place)
//   sM   : float[Br]            — running max (loaded from m_acc on entry)
//   sL   : float[Br]            — running denom (loaded from l_acc on entry)
//   sO   : float[Br * D]        — running output (loaded from o_acc on entry)
// =============================================================================

template <typename T>
__global__ void ring_attention_step_kernel(
    const T* __restrict__ q,           // [B, H, Q_local, D]
    const T* __restrict__ k_local,     // [B, H, K_chunk, D]
    const T* __restrict__ v_local,     // [B, H, K_chunk, D]
    float* __restrict__ o_acc,         // [B, H, Q_local, D]
    float* __restrict__ m_acc,         // [B, H, Q_local]
    float* __restrict__ l_acc,         // [B, H, Q_local]
    int32_t batch,
    int32_t heads,
    int32_t q_local,
    int32_t k_chunk,
    int32_t d,
    int32_t q_global_base,             // global Q index of this rank's slice
    int32_t k_global_base,             // global K index of the resident chunk
    float   scale,
    int32_t is_causal)
{
    extern __shared__ unsigned char smem_raw[];
    unsigned char* sp = smem_raw;
    T*     sQ = reinterpret_cast<T*>(sp);     sp += sizeof(T)     * kBr * d;
    T*     sK = reinterpret_cast<T*>(sp);     sp += sizeof(T)     * kBc * d;
    T*     sV = reinterpret_cast<T*>(sp);     sp += sizeof(T)     * kBc * d;
    float* sS = reinterpret_cast<float*>(sp); sp += sizeof(float) * kBr * kBc;
    float* sM = reinterpret_cast<float*>(sp); sp += sizeof(float) * kBr;
    float* sL = reinterpret_cast<float*>(sp); sp += sizeof(float) * kBr;
    float* sO = reinterpret_cast<float*>(sp); sp += sizeof(float) * kBr * d;

    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;
    const int q_base = qb * kBr;
    const int br_eff = (q_base + kBr <= q_local) ? kBr : (q_local - q_base);
    if (br_eff <= 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int64_t bh_off = ((int64_t)b * heads + h);
    const T* q_base_ptr = q + bh_off * q_local * d + (int64_t)q_base * d;
    const T* k_full     = k_local + bh_off * k_chunk * d;
    const T* v_full     = v_local + bh_off * k_chunk * d;
    float* o_base = o_acc + bh_off * q_local * d + (int64_t)q_base * d;
    float* m_base = m_acc + bh_off * q_local + q_base;
    float* l_base = l_acc + bh_off * q_local + q_base;

    // Load Q tile.
    {
        int total = br_eff * d;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d;
            int c = idx - r * d;
            sQ[r * d + c] = q_base_ptr[(int64_t)r * d + c];
        }
    }
    // Load persistent (m_acc, l_acc) for this q-block.
    for (int r = tid; r < br_eff; r += nthreads) {
        sM[r] = m_base[r];
        sL[r] = l_base[r];
    }
    // Load persistent o_acc tile.
    {
        int total = br_eff * d;
        for (int idx = tid; idx < total; idx += nthreads) {
            sO[idx] = o_base[idx];
        }
    }
    __syncthreads();

    // Iterate K/V tiles within the resident chunk.
    const int num_kb = (k_chunk + kBc - 1) / kBc;
    for (int kb = 0; kb < num_kb; ++kb) {
        const int kbase_local = kb * kBc;
        const int bc_eff = (kbase_local + kBc <= k_chunk) ? kBc : (k_chunk - kbase_local);
        const int kbase_global = k_global_base + kbase_local;
        const int qmax_global = q_global_base + q_base + br_eff - 1;

        // Causal early-out: if the entire resident tile is past every
        // q's global index, no q here attends to any k here.
        if (is_causal != 0 && kbase_global > qmax_global) {
            continue;
        }

        // Load K, V tiles.
        {
            int total = bc_eff * d;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d;
                int c = idx - r * d;
                sK[r * d + c] = k_full[((int64_t)(kbase_local + r)) * d + c];
            }
        }
        {
            int total = bc_eff * d;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d;
                int c = idx - r * d;
                sV[r * d + c] = v_full[((int64_t)(kbase_local + r)) * d + c];
            }
        }
        __syncthreads();

        // S = Q · K^T · scale, with causal mask on GLOBAL indices.
        {
            int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) {
                    sS[idx] = -INFINITY;
                    continue;
                }
                float acc = 0.0f;
                #pragma unroll 4
                for (int dd = 0; dd < d; ++dd) {
                    float qv = load_f32<T>(sQ[r * d + dd]);
                    float kv = load_f32<T>(sK[c * d + dd]);
                    acc += qv * kv;
                }
                float v = acc * scale;
                int q_idx_abs = q_global_base + q_base + r;
                int k_idx_abs = kbase_global + c;
                if (is_causal != 0 && k_idx_abs > q_idx_abs) {
                    v = -INFINITY;
                }
                sS[idx] = v;
            }
        }
        __syncthreads();

        // Per-row online softmax update — folds this tile's contribution
        // into the persistent (m, l, O) state.
        //
        // For each row r (one thread per row, br_eff ≤ 64 ≤ nthreads):
        //   m_local = max over c of S[r, :bc_eff]
        //   m_new   = max(m_old, m_local)
        //   alpha   = exp(m_old - m_new)
        //   P[r, :] = exp(S[r, :] - m_new) (in-place over sS)
        //   l_new   = alpha * l_old + sum(P[r, :])
        //   O[r, :] = alpha * O[r, :] + P[r, :] @ V[:, :]
        __shared__ float sAlpha[kBr];
        if (tid < br_eff) {
            int r = tid;
            float mloc = -INFINITY;
            for (int c = 0; c < bc_eff; ++c) {
                float v = sS[r * kBc + c];
                if (v > mloc) mloc = v;
            }
            float mold = sM[r];
            float mnew = (mold > mloc) ? mold : mloc;
            float alpha;
            float l_local = 0.0f;
            if (!isfinite(mnew)) {
                // Entire history + this tile all-masked → leave state untouched.
                alpha = 1.0f;
                for (int c = 0; c < bc_eff; ++c) sS[r * kBc + c] = 0.0f;
                sAlpha[r] = alpha;
            } else {
                if (isfinite(mold)) {
                    alpha = expf(mold - mnew);
                } else {
                    alpha = 0.0f;       // first finite contribution → wipe accumulators
                }
                for (int c = 0; c < bc_eff; ++c) {
                    float s = sS[r * kBc + c];
                    float p = isfinite(s) ? expf(s - mnew) : 0.0f;
                    sS[r * kBc + c] = p;
                    l_local += p;
                }
                sAlpha[r] = alpha;
                sM[r] = mnew;
                sL[r] = alpha * sL[r] + l_local;
            }
            // Zero out padded columns explicitly.
            for (int c = bc_eff; c < kBc; ++c) sS[r * kBc + c] = 0.0f;
        }
        // Zero out P rows beyond br_eff so the matmul below doesn't pull garbage.
        {
            int total_pad = kBr * kBc;
            for (int idx = tid; idx < total_pad; idx += nthreads) {
                int r = idx / kBc;
                if (r >= br_eff) sS[idx] = 0.0f;
            }
        }
        __syncthreads();

        // O ← alpha * O + P @ V.
        {
            int total = kBr * d;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d;
                if (r >= br_eff) continue;
                int dv = idx - r * d;
                float acc = 0.0f;
                for (int c = 0; c < bc_eff; ++c) {
                    float p = sS[r * kBc + c];
                    float vv = load_f32<T>(sV[c * d + dv]);
                    acc += p * vv;
                }
                float a = sAlpha[r];
                sO[r * d + dv] = a * sO[r * d + dv] + acc;
            }
        }
        __syncthreads();
    } // end kb

    // Flush updated state back to global memory for the next step.
    for (int r = tid; r < br_eff; r += nthreads) {
        m_base[r] = sM[r];
        l_base[r] = sL[r];
    }
    {
        int total = br_eff * d;
        for (int idx = tid; idx < total; idx += nthreads) {
            o_base[idx] = sO[idx];
        }
    }
}

// =============================================================================
// Finalize kernel — after all rotation steps, divide o_acc by l_acc
// and emit y in the operand dtype; emit LSE = m + log(l) if requested.
// One block per (b, h, q_block).
// =============================================================================

template <typename T>
__global__ void ring_attention_finalize_kernel(
    const float* __restrict__ o_acc,    // [B, H, Q_local, D]
    const float* __restrict__ m_acc,    // [B, H, Q_local]
    const float* __restrict__ l_acc,    // [B, H, Q_local]
    T* __restrict__ y,                  // [B, H, Q_local, D]
    T* __restrict__ lse_out,            // [B, H, Q_local] or nullptr
    int32_t batch,
    int32_t heads,
    int32_t q_local,
    int32_t d)
{
    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;
    const int q_base = qb * kBr;
    const int br_eff = (q_base + kBr <= q_local) ? kBr : (q_local - q_base);
    if (br_eff <= 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int64_t bh_off = ((int64_t)b * heads + h);
    const float* o_base = o_acc + bh_off * q_local * d + (int64_t)q_base * d;
    const float* m_base = m_acc + bh_off * q_local + q_base;
    const float* l_base = l_acc + bh_off * q_local + q_base;
    T* y_base = y + bh_off * q_local * d + (int64_t)q_base * d;
    T* lse_base = (lse_out != nullptr) ? (lse_out + bh_off * q_local + q_base) : nullptr;

    int total = br_eff * d;
    for (int idx = tid; idx < total; idx += nthreads) {
        int r = idx / d;
        int dv = idx - r * d;
        float l = l_base[r];
        float yv;
        if (l > 0.0f && isfinite(l)) {
            yv = o_base[r * d + dv] / l;
        } else {
            yv = 0.0f;
        }
        y_base[(int64_t)r * d + dv] = store_f32<T>(yv);
    }
    if (lse_base != nullptr && tid < br_eff) {
        int r = tid;
        float l = l_base[r];
        float v = (l > 0.0f && isfinite(l)) ? (m_base[r] + logf(l)) : -INFINITY;
        lse_base[r] = store_f32<T>(v);
    }
}

// =============================================================================
// Init kernel — zero o_acc, set m_acc = -INF, l_acc = 0.
// =============================================================================
__global__ void ring_attention_init_kernel(
    float* o_acc,
    float* m_acc,
    float* l_acc,
    int64_t o_len,
    int64_t ml_len)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < o_len; i += step) o_acc[i] = 0.0f;
    for (int64_t i = tid; i < ml_len; i += step) {
        m_acc[i] = -INFINITY;
        l_acc[i] = 0.0f;
    }
}

// =============================================================================
// Host-side helpers
// =============================================================================

template <typename T>
__host__ inline size_t ring_step_smem_bytes(int d) {
    size_t s = 0;
    s += sizeof(T)     * kBr * d;      // sQ
    s += sizeof(T)     * kBc * d;      // sK
    s += sizeof(T)     * kBc * d;      // sV
    s += sizeof(float) * kBr * kBc;    // sS
    s += sizeof(float) * kBr;          // sM
    s += sizeof(float) * kBr;          // sL
    s += sizeof(float) * kBr * d;      // sO
    return s;
}

inline __host__ size_t ring_attention_workspace_bytes_host(
    int64_t batch, int64_t heads, int64_t q_local, int64_t d)
{
    int64_t o_bytes  = batch * heads * q_local * d * (int64_t)sizeof(float);
    int64_t ml_bytes = batch * heads * q_local      * (int64_t)sizeof(float);
    // o_acc (f32) + m_acc (f32) + l_acc (f32)
    return (size_t)(o_bytes + ml_bytes * 2);
}

template <typename T>
__host__ inline int32_t launch_ring_step(
    const T* q, const T* k_local, const T* v_local,
    float* o_acc, float* m_acc, float* l_acc,
    int32_t batch, int32_t heads, int32_t q_local, int32_t k_chunk, int32_t d,
    int32_t q_global_base, int32_t k_global_base,
    float scale, int32_t is_causal,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_local < 0 || k_chunk < 0 || d < 0) return 2;
    if (d != kHeadDim) return 3;                  // Tier 1: head_dim == 128
    int64_t total = (int64_t)batch * heads * q_local;
    if (total == 0) return 0;
    if (k_chunk == 0) return 0;                   // no contribution this step
    int num_qb = (q_local + kBr - 1) / kBr;
    if (num_qb <= 0) return 0;
    dim3 grid((unsigned)num_qb, (unsigned)heads, (unsigned)batch);
    dim3 block((unsigned)kThreadsPerBlock);
    size_t smem = ring_step_smem_bytes<T>(d);
    if (smem > 48 * 1024) {
        cudaError_t serr = cudaFuncSetAttribute(
            (const void*)ring_attention_step_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        if (serr != cudaSuccess) return 1000 + (int32_t)serr;
    }
    ring_attention_step_kernel<T><<<grid, block, smem, stream>>>(
        q, k_local, v_local,
        o_acc, m_acc, l_acc,
        batch, heads, q_local, k_chunk, d,
        q_global_base, k_global_base, scale, is_causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

template <typename T>
__host__ inline int32_t launch_ring_finalize(
    const float* o_acc, const float* m_acc, const float* l_acc,
    T* y, T* lse_out,
    int32_t batch, int32_t heads, int32_t q_local, int32_t d,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_local < 0 || d < 0) return 2;
    int64_t total = (int64_t)batch * heads * q_local * d;
    if (total == 0) return 0;
    int num_qb = (q_local + kBr - 1) / kBr;
    if (num_qb <= 0) return 0;
    dim3 grid((unsigned)num_qb, (unsigned)heads, (unsigned)batch);
    dim3 block((unsigned)kThreadsPerBlock);
    ring_attention_finalize_kernel<T><<<grid, block, 0, stream>>>(
        o_acc, m_acc, l_acc, y, lse_out, batch, heads, q_local, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

__host__ inline int32_t launch_ring_init(
    float* o_acc, float* m_acc, float* l_acc,
    int64_t o_len, int64_t ml_len,
    cudaStream_t stream)
{
    if (o_len == 0 && ml_len == 0) return 0;
    constexpr int kBlock = 256;
    int64_t total = (o_len > ml_len) ? o_len : ml_len;
    int64_t bi = (total + kBlock - 1) / kBlock;
    constexpr int64_t kMaxBlocks = 65535;
    int blocks = static_cast<int>(bi > kMaxBlocks ? kMaxBlocks : bi);
    if (blocks <= 0) blocks = 1;
    ring_attention_init_kernel<<<blocks, kBlock, 0, stream>>>(
        o_acc, m_acc, l_acc, o_len, ml_len);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

}} // namespace baracuda::ring_attention

// =============================================================================
// C-ABI INSTANTIATE macros — one per dtype.
//
// Layout: three entry points per dtype family
//   baracuda_kernels_ring_attention_<dt>_step_run
//   baracuda_kernels_ring_attention_<dt>_finalize_run
// Plus one dtype-independent init helper:
//   baracuda_kernels_ring_attention_init_run
// =============================================================================

#define BARACUDA_KERNELS_RING_ATTENTION_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_ring_attention_##NAME##_step_run(                     \
        int32_t batch, int32_t heads, int32_t q_local, int32_t k_chunk, int32_t d,            \
        int32_t q_global_base, int32_t k_global_base,                                         \
        float scale, int32_t is_causal,                                                       \
        const void* q, const void* k_local, const void* v_local,                              \
        void* o_acc, void* m_acc, void* l_acc,                                                \
        void* stream_ptr)                                                                     \
    {                                                                                          \
        if (batch < 0 || heads < 0 || q_local < 0 || k_chunk < 0 || d < 0) return 2;          \
        if (q == nullptr || k_local == nullptr || v_local == nullptr) return 2;               \
        if (o_acc == nullptr || m_acc == nullptr || l_acc == nullptr) return 2;               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                          \
        return baracuda::ring_attention::launch_ring_step<T>(                                 \
            static_cast<const T*>(q),                                                         \
            static_cast<const T*>(k_local),                                                   \
            static_cast<const T*>(v_local),                                                   \
            static_cast<float*>(o_acc),                                                       \
            static_cast<float*>(m_acc),                                                       \
            static_cast<float*>(l_acc),                                                       \
            batch, heads, q_local, k_chunk, d,                                                \
            q_global_base, k_global_base,                                                     \
            scale, is_causal, stream);                                                        \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_ring_attention_##NAME##_finalize_run(                 \
        int32_t batch, int32_t heads, int32_t q_local, int32_t d,                             \
        const void* o_acc, const void* m_acc, const void* l_acc,                              \
        void* y, void* lse,                                                                   \
        void* stream_ptr)                                                                     \
    {                                                                                          \
        if (batch < 0 || heads < 0 || q_local < 0 || d < 0) return 2;                         \
        if (o_acc == nullptr || m_acc == nullptr || l_acc == nullptr) return 2;               \
        if (y == nullptr) return 2;                                                           \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                          \
        return baracuda::ring_attention::launch_ring_finalize<T>(                             \
            static_cast<const float*>(o_acc),                                                 \
            static_cast<const float*>(m_acc),                                                 \
            static_cast<const float*>(l_acc),                                                 \
            static_cast<T*>(y),                                                               \
            static_cast<T*>(lse),                                                             \
            batch, heads, q_local, d, stream);                                                \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_ring_attention_##NAME##_step_can_implement(           \
        int32_t batch, int32_t heads, int32_t q_local, int32_t k_chunk, int32_t d)            \
    {                                                                                          \
        if (batch < 0 || heads < 0 || q_local < 0 || k_chunk < 0 || d < 0) return 2;          \
        if (d != baracuda::ring_attention::kHeadDim) return 3;                                \
        return 0;                                                                              \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_ring_attention_##NAME##_finalize_can_implement(       \
        int32_t batch, int32_t heads, int32_t q_local, int32_t d)                             \
    {                                                                                          \
        if (batch < 0 || heads < 0 || q_local < 0 || d < 0) return 2;                         \
        if (d != baracuda::ring_attention::kHeadDim) return 3;                                \
        return 0;                                                                              \
    }

#endif // BARACUDA_RING_ATTENTION_CUH
