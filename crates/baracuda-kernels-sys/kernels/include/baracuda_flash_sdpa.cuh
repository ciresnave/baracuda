// baracuda_flash_sdpa.cuh
//
// Templated kernels and INSTANTIATE macros for Flash Attention SDPA
// (Phase 6 Milestone 6.6 of Category K — the memory-efficient tiled
// fused alternative to the naive 3-kernel SDPA in
// `baracuda_sdpa.cuh`). Algorithm follows Tri Dao 2022
// (https://arxiv.org/abs/2205.14135).
//
// FW (online-softmax tiled fusion). For each query block Qᵢ of size
// `[Br, d_k]`:
//   m_i = -inf, l_i = 0, O_i = 0
//   for each key block Kⱼ, Vⱼ of size `[Bc, d_k]`, `[Bc, d_v]`:
//     S_ij = Qᵢ · Kⱼ^T · scale  (apply causal mask if applicable)
//     m_new = max(m_i, rowmax(S_ij))
//     P_ij = exp(S_ij − m_new[:, None])
//     α = exp(m_i − m_new)
//     l_new = α · l_i + rowsum(P_ij)
//     O_i = α[:, None] · O_i + P_ij @ Vⱼ
//     m_i, l_i = m_new, l_new
//   O_i = O_i / l_i[:, None]
//   L_i = m_i + log(l_i)              (saved log-sum-exp for BW)
//
// BW (3-kernel deterministic pipeline). For each query Qᵢ:
//   D_i = rowsum(dy_i ⊙ y_i)         (kernel 1: compute D, no Q-blocking)
//   dQ_i = 0
//   for each key block Kⱼ, Vⱼ:        (kernel 2: outer = Qᵢ)
//     S_ij  = Qᵢ · Kⱼ^T · scale (+ causal mask)
//     P_ij  = exp(S_ij − L_i[:, None])
//     dP_ij = dy_i · Vⱼ^T
//     dS_ij = P_ij ⊙ (dP_ij − D_i[:, None])
//     dQ_i += dS_ij · Kⱼ · scale
//   for each query block Qᵢ:         (kernel 3: outer = Kⱼ)
//     dV_j += P_ij^T · dy_i
//     dK_j += dS_ij^T · Qᵢ · scale
//
// Trailblazer constraints:
//   Br = Bc = 64 fixed.
//   d_k = d_v ≤ 128.
//   f16 / bf16 accumulate in f32 throughout.
//   One CUDA block per (batch, head, q_block) for FW + dQ.
//   One CUDA block per (batch, head, k_block) for dK/dV.
//   No atomicAdd: dK / dV are computed by the kernel that owns the
//   k-block, so each output cell is written by exactly one block.
//
// Status codes match the family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

#ifndef BARACUDA_FLASH_SDPA_CUH
#define BARACUDA_FLASH_SDPA_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace flash_sdpa {

// =============================================================================
// Tile geometry.
//
// Tile shape is per-dtype because f64 BW at the f32-tile size (64×64) blows
// past sm_89's 99 KiB per-block dynamic SMEM cap. Concretely the dQ BW
// kernel at d_k=32 needs:
//   f32 64×64:  ~92 KiB  (fits, with the 99 KiB opt-in carveout)
//   f64 64×64: ~145 KiB  (does NOT fit)
//   f64 32×32: ~56  KiB  (fits comfortably)
// f32/f16/bf16 keep 64×64 — they already validated at that size in
// Milestone 6.6. Only f64 takes the smaller tile.
// =============================================================================

template <typename T> struct TileShape {
    static constexpr int Br = 64;
    static constexpr int Bc = 64;
};
template <> struct TileShape<double> {
    static constexpr int Br = 32;
    static constexpr int Bc = 32;
};

constexpr int kMaxD = 128;          // d_k = d_v upper bound
constexpr int kThreadsPerBlock = 128;

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

// Compute-type promotion: f16/bf16/f32 → float, f64 → double.
template <typename T> struct ComputeType { using type = float; };
template <> struct ComputeType<double>   { using type = double; };

// Compute-precision-preserving load/store. For f16/bf16 these go through
// f32 (no native FMA in compute precision); for f32 they're identity;
// for f64 they pass double through without losing precision.
template <typename T>
__device__ __forceinline__ typename ComputeType<T>::type load_ct(T x) {
    return (typename ComputeType<T>::type)x;
}
template <>
__device__ __forceinline__ float load_ct<__half>(__half x) { return __half2float(x); }
template <>
__device__ __forceinline__ float load_ct<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T store_ct(typename ComputeType<T>::type v) { return (T)v; }
template <>
__device__ __forceinline__ __half store_ct<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 store_ct<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// Cooperative thread-block reduce of a per-row partial — produces the
// per-row max or sum across the Bc dimension. We just inline two-pass
// shared-memory reductions inside each kernel rather than factor out a
// helper, since the row layout is bespoke to each tile pass.

// =============================================================================
// FW kernel — one block per (b, h, q_block). Computes y[b, h, qbase..qbase+Br]
// and L[b, h, qbase..qbase+Br] over the full K range using online softmax.
// =============================================================================
//
// Shared memory layout (per block):
//   sQ : T[Br * d_k]
//   sK : T[Bc * d_k]
//   sV : T[Bc * d_v]
//   sS : float[Br * Bc]    (scores / probs tile in compute precision)
//   sO : float[Br * d_v]   (running output accumulator)
//   sM : float[Br]         (running row max)
//   sL : float[Br]         (running row denom)
//
// All threads in the block cooperate. We use a "thread-per-output-cell"
// pattern for the two matmul tiles: thread index spans Br*Bc / Br*d_v
// outputs, looped via grid-stride within the block when threads <
// outputs. This is the simplest correct mapping (sub-optimal — Milestone
// 11 will tile the matmuls properly).

template <typename T>
__global__ void flash_sdpa_fw_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ y,
    T* __restrict__ lse,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    int32_t d_v,
    float scale,
    int32_t is_causal)
{
    using CT = typename ComputeType<T>::type;
    static constexpr int kBr = TileShape<T>::Br;
    static constexpr int kBc = TileShape<T>::Bc;

    extern __shared__ unsigned char smem_raw[];
    unsigned char* sp = smem_raw;
    T* sQ = reinterpret_cast<T*>(sp);                 sp += sizeof(T) * kBr * d_k;
    T* sK = reinterpret_cast<T*>(sp);                 sp += sizeof(T) * kBc * d_k;
    T* sV = reinterpret_cast<T*>(sp);                 sp += sizeof(T) * kBc * d_v;
    CT* sS = reinterpret_cast<CT*>(sp);               sp += sizeof(CT) * kBr * kBc;
    CT* sO = reinterpret_cast<CT*>(sp);               sp += sizeof(CT) * kBr * d_v;
    CT* sM = reinterpret_cast<CT*>(sp);               sp += sizeof(CT) * kBr;
    CT* sL = reinterpret_cast<CT*>(sp);               sp += sizeof(CT) * kBr;

    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;
    const int qbase = qb * kBr;
    const int br_eff = (qbase + kBr <= q_len) ? kBr : (q_len - qbase);
    if (br_eff <= 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Base offsets (in element units).
    const int64_t bh_off = ((int64_t)b * heads + h);
    const T* q_base = q + bh_off * q_len * d_k + (int64_t)qbase * d_k;
    const T* k_full = k + bh_off * k_len * d_k;
    const T* v_full = v + bh_off * k_len * d_v;
    T*       y_base = y + bh_off * q_len * d_v + (int64_t)qbase * d_v;
    T*       lse_base = lse + bh_off * q_len + qbase;

    // Load Q tile [br_eff, d_k] into sQ.
    {
        int total = br_eff * d_k;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_k;
            int c = idx - r * d_k;
            sQ[r * d_k + c] = q_base[(int64_t)r * d_k + c];
        }
    }
    // Init sO, sM, sL.
    {
        int total_o = br_eff * d_v;
        for (int idx = tid; idx < total_o; idx += nthreads) {
            sO[idx] = (CT)0;
        }
        for (int r = tid; r < br_eff; r += nthreads) {
            sM[r] = -(CT)INFINITY;
            sL[r] = (CT)0;
        }
    }
    __syncthreads();

    const CT scale_ct = (CT)scale;
    const int num_kb = (k_len + kBc - 1) / kBc;
    for (int kb = 0; kb < num_kb; ++kb) {
        const int kbase = kb * kBc;
        const int bc_eff = (kbase + kBc <= k_len) ? kBc : (k_len - kbase);

        // Causal early-out: every query row in this q-block has q-index
        // in [qbase, qbase + br_eff - 1]. The minimum allowed k is up to
        // q_max = qbase + br_eff - 1. If kbase > q_max, this entire block
        // is masked out for every row. Skip.
        if (is_causal != 0 && kbase > (qbase + br_eff - 1)) {
            continue;
        }

        // Load K tile [bc_eff, d_k] into sK.
        {
            int total = bc_eff * d_k;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_k;
                int c = idx - r * d_k;
                sK[r * d_k + c] = k_full[((int64_t)(kbase + r)) * d_k + c];
            }
        }
        // Load V tile [bc_eff, d_v] into sV.
        {
            int total = bc_eff * d_v;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_v;
                int c = idx - r * d_v;
                sV[r * d_v + c] = v_full[((int64_t)(kbase + r)) * d_v + c];
            }
        }
        __syncthreads();

        // S_ij = Qᵢ · Kⱼ^T · scale, shape [br_eff, bc_eff]. Apply causal
        // and pad masks (set masked cells to -INF for both reasons).
        {
            int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) {
                    sS[idx] = -(CT)INFINITY;
                    continue;
                }
                CT acc = (CT)0;
                #pragma unroll 4
                for (int d = 0; d < d_k; ++d) {
                    CT qv = load_ct<T>(sQ[r * d_k + d]);
                    CT kv = load_ct<T>(sK[c * d_k + d]);
                    acc += qv * kv;
                }
                CT v = acc * scale_ct;
                int q_idx_abs = qbase + r;
                int k_idx_abs = kbase + c;
                if (is_causal != 0 && k_idx_abs > q_idx_abs) {
                    v = -(CT)INFINITY;
                }
                sS[idx] = v;
            }
        }
        __syncthreads();

        // Per-row online softmax update. One row per thread (br_eff ≤ 64,
        // nthreads = 128 ≥ 64 so each row picks up one thread).
        // Each thread that owns a row:
        //   1. find m_local = max over c in [0, bc_eff)
        //   2. m_new = max(m_i, m_local)
        //   3. compute α = exp(m_i − m_new)
        //   4. write P_ij = exp(S_ij − m_new) in-place over sS
        //   5. l_local = sum P_ij
        //   6. l_new = α · l_i + l_local
        //   7. write back m_i = m_new, l_i = l_new
        // Then a __syncthreads, then the per-cell sO update (which reads
        // sM[r] for α — but we've already absorbed α into l, what's left
        // is sO ← α·sO + P·V. We need α per row. Save α in sL temporarily?
        // Cleaner: introduce a small per-row α buffer alongside sM/sL.
        //
        // Implementation choice: stash α in sM after the update (sM gets
        // replaced with m_new in step 7), so we use a separate small
        // shared-memory slot `sAlpha[Br]` allocated below sM.
        //
        // We re-use sM[r] as α between phase A (row reduction) and phase B
        // (O update), then update sM[r] = m_new at end of phase B's read.
        // No — simpler: have one thread per row compute and store α in
        // sL (we save l_new = α·l_i + l_local into a *temporary* register,
        // and sL[r] keeps the *old* l_i until the O-update reads it as α
        // via a separate alpha array). To keep the code clean we just
        // declare a small shared-memory `alpha` array below.

        __shared__ CT sAlpha[kBr];

        if (tid < br_eff) {
            int r = tid;
            // Row max over bc_eff valid cells (the padding cells were
            // -INF by construction so they naturally drop out of the max).
            CT mloc = -(CT)INFINITY;
            for (int c = 0; c < bc_eff; ++c) {
                CT v = sS[r * kBc + c];
                if (v > mloc) mloc = v;
            }
            CT mold = sM[r];
            CT mnew = (mold > mloc) ? mold : mloc;
            // Edge case: if this tile is entirely masked out AND we
            // haven't seen any finite value yet, mnew is -inf. Just skip
            // this tile (alpha = 1, no P contribution).
            CT alpha;
            CT l_local = (CT)0;
            if (!isfinite((float)mnew)) {
                // Both sides -inf → leave m_i, l_i, O_i unchanged.
                alpha = (CT)1;
                // Force all P cells to 0 for downstream O update.
                for (int c = 0; c < bc_eff; ++c) {
                    sS[r * kBc + c] = (CT)0;
                }
                sAlpha[r] = alpha;
                // Don't update sM[r] / sL[r] — keep them as is.
            } else {
                if (isfinite((float)mold)) {
                    alpha = (CT)exp((double)(mold - mnew));
                } else {
                    alpha = (CT)0;        // first finite tile — wipe l_i.
                }
                for (int c = 0; c < bc_eff; ++c) {
                    CT v = sS[r * kBc + c];
                    CT p = (isfinite((float)v)) ? (CT)exp((double)(v - mnew)) : (CT)0;
                    sS[r * kBc + c] = p;
                    l_local += p;
                }
                sAlpha[r] = alpha;
                sM[r] = mnew;
                sL[r] = alpha * sL[r] + l_local;
            }
            // Zero out padded columns explicitly so the matmul below
            // doesn't pull garbage.
            for (int c = bc_eff; c < kBc; ++c) {
                sS[r * kBc + c] = (CT)0;
            }
        }
        // Zero out P rows ≥ br_eff (padding rows) — those don't affect
        // sO anyway since we'll guard sO writes by r < br_eff, but the
        // matmul below assumes the full kBr × kBc P tile, so we need P
        // = 0 in those rows for safety.
        {
            int total_pad = kBr * kBc;
            for (int idx = tid; idx < total_pad; idx += nthreads) {
                int r = idx / kBc;
                if (r >= br_eff) sS[idx] = (CT)0;
            }
        }
        __syncthreads();

        // sO ← α[:, None] · sO + sS · sV   (P @ V), shape [br_eff, d_v].
        {
            int total = kBr * d_v;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_v;
                if (r >= br_eff) continue;
                int dv = idx - r * d_v;
                CT acc = (CT)0;
                for (int c = 0; c < bc_eff; ++c) {
                    CT p = sS[r * kBc + c];
                    CT vv = load_ct<T>(sV[c * d_v + dv]);
                    acc += p * vv;
                }
                CT alpha = sAlpha[r];
                sO[r * d_v + dv] = alpha * sO[r * d_v + dv] + acc;
            }
        }
        __syncthreads();
    } // end kb

    // Finalize. y = sO / sL[:, None]. lse = sM + log(sL). Rows with
    // all-masked input emit zero output and lse = -inf.
    {
        int total = br_eff * d_v;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_v;
            int dv = idx - r * d_v;
            CT l = sL[r];
            CT yv;
            if (l > (CT)0 && isfinite((float)l)) {
                yv = sO[r * d_v + dv] / l;
            } else {
                yv = (CT)0;
            }
            y_base[(int64_t)r * d_v + dv] = store_ct<T>(yv);
        }
    }
    if (tid < br_eff) {
        int r = tid;
        CT l = sL[r];
        CT lse_v;
        if (l > (CT)0 && isfinite((float)l)) {
            lse_v = sM[r] + (CT)log((double)l);
        } else {
            lse_v = -(CT)INFINITY;
        }
        lse_base[r] = store_ct<T>(lse_v);
    }
}

// f64 specialization mirrors the f32 path but with double compute. We
// just rely on the template above with CT = double; the specialization
// is implicit through `ComputeType<double>`. No explicit override needed
// here because all CT-typed operations apply to both float and double.

// =============================================================================
// FW launcher
// =============================================================================

template <typename T>
__host__ inline size_t flash_fw_smem_bytes(int d_k, int d_v) {
    using CT = typename ComputeType<T>::type;
    constexpr int kBr = TileShape<T>::Br;
    constexpr int kBc = TileShape<T>::Bc;
    size_t sQ = sizeof(T) * kBr * d_k;
    size_t sK = sizeof(T) * kBc * d_k;
    size_t sV = sizeof(T) * kBc * d_v;
    size_t sS = sizeof(CT) * kBr * kBc;
    size_t sO = sizeof(CT) * kBr * d_v;
    size_t sM = sizeof(CT) * kBr;
    size_t sL = sizeof(CT) * kBr;
    return sQ + sK + sV + sS + sO + sM + sL;
}

template <typename T>
__host__ inline int32_t launch_flash_sdpa_fp(
    const T* q, const T* k_in, const T* v_in,
    T* y, T* lse,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    int32_t d_k, int32_t d_v,
    float scale, int32_t is_causal,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;
    if (d_k > kMaxD || d_v > kMaxD) return 3;
    if (d_k != d_v) return 3;     // trailblazer enforcement
    int64_t total_y = (int64_t)batch * heads * q_len * d_v;
    if (total_y == 0) return 0;
    constexpr int kBr = TileShape<T>::Br;
    int num_qb = (q_len + kBr - 1) / kBr;
    if (num_qb <= 0) return 0;
    dim3 grid((unsigned)num_qb, (unsigned)heads, (unsigned)batch);
    dim3 block((unsigned)kThreadsPerBlock);
    size_t smem = flash_fw_smem_bytes<T>(d_k, d_v);
    // Opt into the full sm_80+ dynamic SMEM budget. Default carveout is
    // 48 KiB on most archs; this kernel can exceed that for f32/f64.
    if (smem > 48 * 1024) {
        cudaError_t serr = cudaFuncSetAttribute(
            (const void*)flash_sdpa_fw_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        if (serr != cudaSuccess) return 1000 + (int32_t)serr;
    }
    flash_sdpa_fw_kernel<T><<<grid, block, smem, stream>>>(
        q, k_in, v_in, y, lse, batch, heads, q_len, k_len,
        d_k, d_v, scale, is_causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

// =============================================================================
// BW kernel 1: D = rowsum(dy ⊙ y). One block per (b, h, q_block); the
// block reduces D over the d_v dimension for each row of the block.
//
// We don't tile D — d_v ≤ 128 fits comfortably in shared memory and the
// reduction is trivial.
// =============================================================================

template <typename T>
__global__ void flash_sdpa_bw_D_kernel(
    const T* __restrict__ y,
    const T* __restrict__ dy,
    T* __restrict__ D,        // shape [B, H, Q]
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t d_v)
{
    using CT = typename ComputeType<T>::type;
    int64_t total = (int64_t)batch * heads * q_len;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t row = tid; row < total; row += step) {
        int64_t base = row * (int64_t)d_v;
        CT acc = (CT)0;
        for (int dv = 0; dv < d_v; ++dv) {
            CT yv  = load_ct<T>(y[base + dv]);
            CT dyv = load_ct<T>(dy[base + dv]);
            acc += yv * dyv;
        }
        D[row] = store_ct<T>(acc);
    }
}

// =============================================================================
// BW kernel 2: dQ. One block per (b, h, q_block). Iterates over all key
// blocks. Each q-block produces its complete dQ slice (no race).
// =============================================================================

template <typename T>
__global__ void flash_sdpa_bw_dQ_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_in,
    const T* __restrict__ v_in,
    const T* __restrict__ dy,
    const T* __restrict__ lse,
    const T* __restrict__ D,
    T* __restrict__ dQ,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    int32_t d_v,
    float scale,
    int32_t is_causal)
{
    using CT = typename ComputeType<T>::type;
    static constexpr int kBr = TileShape<T>::Br;
    static constexpr int kBc = TileShape<T>::Bc;

    extern __shared__ unsigned char smem_raw[];
    unsigned char* sp = smem_raw;
    T* sQ  = reinterpret_cast<T*>(sp);   sp += sizeof(T) * kBr * d_k;
    T* sK  = reinterpret_cast<T*>(sp);   sp += sizeof(T) * kBc * d_k;
    T* sV  = reinterpret_cast<T*>(sp);   sp += sizeof(T) * kBc * d_v;
    T* sDy = reinterpret_cast<T*>(sp);   sp += sizeof(T) * kBr * d_v;
    CT* sS  = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr * kBc;   // P then dS
    CT* sDP = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr * kBc;
    CT* sDQ = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr * d_k;
    CT* sLse = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr;
    CT* sD   = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr;

    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;
    const int qbase = qb * kBr;
    const int br_eff = (qbase + kBr <= q_len) ? kBr : (q_len - qbase);
    if (br_eff <= 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int64_t bh_off = ((int64_t)b * heads + h);
    const T* q_base  = q  + bh_off * q_len * d_k + (int64_t)qbase * d_k;
    const T* dy_base = dy + bh_off * q_len * d_v + (int64_t)qbase * d_v;
    const T* lse_base = lse + bh_off * q_len + qbase;
    const T* D_base   = D   + bh_off * q_len + qbase;
    const T* k_full = k_in + bh_off * k_len * d_k;
    const T* v_full = v_in + bh_off * k_len * d_v;
    T*       dQ_base = dQ + bh_off * q_len * d_k + (int64_t)qbase * d_k;

    // Load Q, dy, lse, D for this q-block.
    {
        int total = br_eff * d_k;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_k;
            int c = idx - r * d_k;
            sQ[r * d_k + c] = q_base[(int64_t)r * d_k + c];
        }
    }
    {
        int total = br_eff * d_v;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_v;
            int c = idx - r * d_v;
            sDy[r * d_v + c] = dy_base[(int64_t)r * d_v + c];
        }
    }
    for (int r = tid; r < br_eff; r += nthreads) {
        sLse[r] = load_ct<T>(lse_base[r]);
        sD[r]   = load_ct<T>(D_base[r]);
    }
    // Init sDQ to 0.
    {
        int total = br_eff * d_k;
        for (int idx = tid; idx < total; idx += nthreads) {
            sDQ[idx] = (CT)0;
        }
    }
    __syncthreads();

    const CT scale_ct = (CT)scale;
    const int num_kb = (k_len + kBc - 1) / kBc;
    for (int kb = 0; kb < num_kb; ++kb) {
        const int kbase = kb * kBc;
        const int bc_eff = (kbase + kBc <= k_len) ? kBc : (k_len - kbase);

        // Causal early-out: entire block masked if kbase > q_max.
        if (is_causal != 0 && kbase > (qbase + br_eff - 1)) {
            continue;
        }

        // Load K, V tiles.
        {
            int total = bc_eff * d_k;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_k;
                int c = idx - r * d_k;
                sK[r * d_k + c] = k_full[((int64_t)(kbase + r)) * d_k + c];
            }
        }
        {
            int total = bc_eff * d_v;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_v;
                int c = idx - r * d_v;
                sV[r * d_v + c] = v_full[((int64_t)(kbase + r)) * d_v + c];
            }
        }
        __syncthreads();

        // P_ij = exp(S_ij − lse). Skip masked / pad cells.
        {
            int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) { sS[idx] = (CT)0; continue; }
                CT acc = (CT)0;
                #pragma unroll 4
                for (int d = 0; d < d_k; ++d) {
                    CT qv = load_ct<T>(sQ[r * d_k + d]);
                    CT kv = load_ct<T>(sK[c * d_k + d]);
                    acc += qv * kv;
                }
                CT s = acc * scale_ct;
                int q_idx_abs = qbase + r;
                int k_idx_abs = kbase + c;
                bool masked = (is_causal != 0 && k_idx_abs > q_idx_abs);
                CT p;
                if (masked) {
                    p = (CT)0;
                } else {
                    CT le = sLse[r];
                    if (!isfinite((float)le)) p = (CT)0;
                    else p = (CT)exp((double)(s - le));
                }
                sS[idx] = p;
            }
        }
        __syncthreads();

        // dP_ij = dy_i · Vⱼ^T, shape [br_eff, bc_eff].
        {
            int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) { sDP[idx] = (CT)0; continue; }
                CT acc = (CT)0;
                #pragma unroll 4
                for (int dv = 0; dv < d_v; ++dv) {
                    CT dyv = load_ct<T>(sDy[r * d_v + dv]);
                    CT vv  = load_ct<T>(sV[c * d_v + dv]);
                    acc += dyv * vv;
                }
                sDP[idx] = acc;
            }
        }
        __syncthreads();

        // dS = P ⊙ (dP − D[:, None]). Write back into sS (overwrites P).
        {
            int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) { sS[idx] = (CT)0; continue; }
                CT p  = sS[idx];
                CT dp = sDP[idx];
                CT d  = sD[r];
                sS[idx] = p * (dp - d);
            }
        }
        __syncthreads();

        // dQ_i += dS · Kⱼ · scale.
        {
            int total = kBr * d_k;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_k;
                if (r >= br_eff) continue;
                int d = idx - r * d_k;
                CT acc = (CT)0;
                for (int c = 0; c < bc_eff; ++c) {
                    CT ds = sS[r * kBc + c];
                    CT kv = load_ct<T>(sK[c * d_k + d]);
                    acc += ds * kv;
                }
                sDQ[r * d_k + d] += acc * scale_ct;
            }
        }
        __syncthreads();
    } // end kb

    // Flush sDQ to global.
    {
        int total = br_eff * d_k;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_k;
            int d = idx - r * d_k;
            dQ_base[(int64_t)r * d_k + d] = store_ct<T>(sDQ[r * d_k + d]);
        }
    }
}

// =============================================================================
// BW kernel 3: dK / dV. One block per (b, h, k_block). Iterates over all
// query blocks. Each k-block writes its complete dK_j and dV_j slices.
// =============================================================================

template <typename T>
__global__ void flash_sdpa_bw_dKdV_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_in,
    const T* __restrict__ v_in,
    const T* __restrict__ dy,
    const T* __restrict__ lse,
    const T* __restrict__ D,
    T* __restrict__ dK,
    T* __restrict__ dV,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    int32_t d_v,
    float scale,
    int32_t is_causal)
{
    using CT = typename ComputeType<T>::type;
    static constexpr int kBr = TileShape<T>::Br;
    static constexpr int kBc = TileShape<T>::Bc;

    extern __shared__ unsigned char smem_raw[];
    unsigned char* sp = smem_raw;
    T* sQ  = reinterpret_cast<T*>(sp);   sp += sizeof(T) * kBr * d_k;
    T* sK  = reinterpret_cast<T*>(sp);   sp += sizeof(T) * kBc * d_k;
    T* sV  = reinterpret_cast<T*>(sp);   sp += sizeof(T) * kBc * d_v;
    T* sDy = reinterpret_cast<T*>(sp);   sp += sizeof(T) * kBr * d_v;
    CT* sS  = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr * kBc;
    CT* sDP = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr * kBc;
    CT* sDK = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBc * d_k;
    CT* sDV = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBc * d_v;
    CT* sLse = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr;
    CT* sD   = reinterpret_cast<CT*>(sp); sp += sizeof(CT) * kBr;

    const int kb = blockIdx.x;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;
    const int kbase = kb * kBc;
    const int bc_eff = (kbase + kBc <= k_len) ? kBc : (k_len - kbase);
    if (bc_eff <= 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int64_t bh_off = ((int64_t)b * heads + h);
    const T* k_base = k_in + bh_off * k_len * d_k + (int64_t)kbase * d_k;
    const T* v_base = v_in + bh_off * k_len * d_v + (int64_t)kbase * d_v;
    T* dK_base = dK + bh_off * k_len * d_k + (int64_t)kbase * d_k;
    T* dV_base = dV + bh_off * k_len * d_v + (int64_t)kbase * d_v;
    const T* q_full   = q  + bh_off * q_len * d_k;
    const T* dy_full  = dy + bh_off * q_len * d_v;
    const T* lse_full = lse + bh_off * q_len;
    const T* D_full   = D   + bh_off * q_len;

    // Load K, V tiles for this k-block.
    {
        int total = bc_eff * d_k;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_k;
            int c = idx - r * d_k;
            sK[r * d_k + c] = k_base[(int64_t)r * d_k + c];
        }
    }
    {
        int total = bc_eff * d_v;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_v;
            int c = idx - r * d_v;
            sV[r * d_v + c] = v_base[(int64_t)r * d_v + c];
        }
    }
    // Init sDK, sDV.
    {
        int total_k = bc_eff * d_k;
        for (int idx = tid; idx < total_k; idx += nthreads) sDK[idx] = (CT)0;
        int total_v = bc_eff * d_v;
        for (int idx = tid; idx < total_v; idx += nthreads) sDV[idx] = (CT)0;
    }
    __syncthreads();

    const CT scale_ct = (CT)scale;
    const int num_qb = (q_len + kBr - 1) / kBr;
    for (int qb = 0; qb < num_qb; ++qb) {
        const int qbase = qb * kBr;
        const int br_eff = (qbase + kBr <= q_len) ? kBr : (q_len - qbase);

        // Causal early-out for this (q_block, k_block) pair: if every
        // q-index in [qbase, qbase + br_eff - 1] is < kbase, the entire
        // tile is masked out → contributes nothing.
        if (is_causal != 0 && (qbase + br_eff - 1) < kbase) {
            continue;
        }

        // Load Q, dy, lse, D for this q-block.
        {
            int total = br_eff * d_k;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_k;
                int c = idx - r * d_k;
                sQ[r * d_k + c] = q_full[((int64_t)(qbase + r)) * d_k + c];
            }
        }
        {
            int total = br_eff * d_v;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_v;
                int c = idx - r * d_v;
                sDy[r * d_v + c] = dy_full[((int64_t)(qbase + r)) * d_v + c];
            }
        }
        for (int r = tid; r < br_eff; r += nthreads) {
            sLse[r] = load_ct<T>(lse_full[qbase + r]);
            sD[r]   = load_ct<T>(D_full[qbase + r]);
        }
        __syncthreads();

        // P, dP. Same as in dQ kernel. Tile shape [br_eff, bc_eff].
        {
            int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) { sS[idx] = (CT)0; sDP[idx] = (CT)0; continue; }
                CT acc = (CT)0;
                #pragma unroll 4
                for (int d = 0; d < d_k; ++d) {
                    CT qv = load_ct<T>(sQ[r * d_k + d]);
                    CT kv = load_ct<T>(sK[c * d_k + d]);
                    acc += qv * kv;
                }
                CT s = acc * scale_ct;
                int q_idx_abs = qbase + r;
                int k_idx_abs = kbase + c;
                bool masked = (is_causal != 0 && k_idx_abs > q_idx_abs);
                CT p;
                if (masked) {
                    p = (CT)0;
                } else {
                    CT le = sLse[r];
                    if (!isfinite((float)le)) p = (CT)0;
                    else p = (CT)exp((double)(s - le));
                }
                sS[idx] = p;
                CT acc_dp = (CT)0;
                #pragma unroll 4
                for (int dv = 0; dv < d_v; ++dv) {
                    CT dyv = load_ct<T>(sDy[r * d_v + dv]);
                    CT vv  = load_ct<T>(sV[c * d_v + dv]);
                    acc_dp += dyv * vv;
                }
                sDP[idx] = acc_dp;
            }
        }
        __syncthreads();

        // dV_j += P^T @ dy_i  → use sS = P (before dS overwrite).
        // dV_j[c, dv] += Σ_r P[r, c] · dy[r, dv]
        {
            int total = kBc * d_v;
            for (int idx = tid; idx < total; idx += nthreads) {
                int c = idx / d_v;
                if (c >= bc_eff) continue;
                int dv = idx - c * d_v;
                CT acc = (CT)0;
                for (int r = 0; r < br_eff; ++r) {
                    CT p = sS[r * kBc + c];
                    CT dyv = load_ct<T>(sDy[r * d_v + dv]);
                    acc += p * dyv;
                }
                sDV[c * d_v + dv] += acc;
            }
        }
        __syncthreads();

        // dS = P ⊙ (dP − D[:, None]). Overwrite sS in-place.
        {
            int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) { sS[idx] = (CT)0; continue; }
                CT p  = sS[idx];
                CT dp = sDP[idx];
                CT d  = sD[r];
                sS[idx] = p * (dp - d);
            }
        }
        __syncthreads();

        // dK_j += dS^T @ Q_i · scale.
        // dK_j[c, d] += scale · Σ_r dS[r, c] · Q[r, d]
        {
            int total = kBc * d_k;
            for (int idx = tid; idx < total; idx += nthreads) {
                int c = idx / d_k;
                if (c >= bc_eff) continue;
                int d = idx - c * d_k;
                CT acc = (CT)0;
                for (int r = 0; r < br_eff; ++r) {
                    CT ds = sS[r * kBc + c];
                    CT qv = load_ct<T>(sQ[r * d_k + d]);
                    acc += ds * qv;
                }
                sDK[c * d_k + d] += acc * scale_ct;
            }
        }
        __syncthreads();
    } // end qb

    // Flush sDK, sDV to global.
    {
        int total = bc_eff * d_k;
        for (int idx = tid; idx < total; idx += nthreads) {
            int c = idx / d_k;
            int d = idx - c * d_k;
            dK_base[(int64_t)c * d_k + d] = store_ct<T>(sDK[c * d_k + d]);
        }
    }
    {
        int total = bc_eff * d_v;
        for (int idx = tid; idx < total; idx += nthreads) {
            int c = idx / d_v;
            int dv = idx - c * d_v;
            dV_base[(int64_t)c * d_v + dv] = store_ct<T>(sDV[c * d_v + dv]);
        }
    }
}

// =============================================================================
// BW launcher
// =============================================================================

template <typename T>
__host__ inline size_t flash_bw_dQ_smem_bytes(int d_k, int d_v) {
    using CT = typename ComputeType<T>::type;
    constexpr int kBr = TileShape<T>::Br;
    constexpr int kBc = TileShape<T>::Bc;
    size_t s = 0;
    s += sizeof(T) * kBr * d_k;          // sQ
    s += sizeof(T) * kBc * d_k;          // sK
    s += sizeof(T) * kBc * d_v;          // sV
    s += sizeof(T) * kBr * d_v;          // sDy
    s += sizeof(CT) * kBr * kBc;         // sS (P / dS)
    s += sizeof(CT) * kBr * kBc;         // sDP
    s += sizeof(CT) * kBr * d_k;         // sDQ
    s += sizeof(CT) * kBr;               // sLse
    s += sizeof(CT) * kBr;               // sD
    return s;
}

template <typename T>
__host__ inline size_t flash_bw_dKdV_smem_bytes(int d_k, int d_v) {
    using CT = typename ComputeType<T>::type;
    constexpr int kBr = TileShape<T>::Br;
    constexpr int kBc = TileShape<T>::Bc;
    size_t s = 0;
    s += sizeof(T) * kBr * d_k;          // sQ
    s += sizeof(T) * kBc * d_k;          // sK
    s += sizeof(T) * kBc * d_v;          // sV
    s += sizeof(T) * kBr * d_v;          // sDy
    s += sizeof(CT) * kBr * kBc;         // sS (P / dS)
    s += sizeof(CT) * kBr * kBc;         // sDP
    s += sizeof(CT) * kBc * d_k;         // sDK
    s += sizeof(CT) * kBc * d_v;         // sDV
    s += sizeof(CT) * kBr;               // sLse
    s += sizeof(CT) * kBr;               // sD
    return s;
}

template <typename T>
__host__ inline int32_t launch_flash_sdpa_backward_fp(
    const T* q, const T* k_in, const T* v_in,
    const T* y, const T* lse, const T* dy,
    T* D_ws,           // [B, H, Q] scratch
    T* dQ, T* dK, T* dV,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    int32_t d_k, int32_t d_v,
    float scale, int32_t is_causal,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;
    if (d_k > kMaxD || d_v > kMaxD) return 3;
    if (d_k != d_v) return 3;
    // Tile shape is dtype-dependent (see TileShape<T> above). f32/f16/bf16
    // tile at 64×64; f64 tiles at 32×32 so the BW dQ SMEM (~56 KiB at
    // d_k=32) fits under sm_89's 99 KiB per-block cap.
    constexpr int kBr = TileShape<T>::Br;
    constexpr int kBc = TileShape<T>::Bc;
    int64_t total_qk = (int64_t)batch * heads * q_len * k_len;
    if (total_qk == 0) return 0;

    // K1: D = rowsum(y ⊙ dy). One thread per (b, h, q). Reuse a thread-
    // per-row grid-stride loop.
    {
        constexpr int kBlock = 256;
        constexpr int64_t kMaxBlocks = 65535;
        int64_t total = (int64_t)batch * heads * q_len;
        int64_t bi = (total + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(bi > kMaxBlocks ? kMaxBlocks : bi);
        if (blocks <= 0) blocks = 1;
        flash_sdpa_bw_D_kernel<T><<<blocks, kBlock, 0, stream>>>(
            y, dy, D_ws, batch, heads, q_len, d_v);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 1000 + (int32_t)err;
    }
    // K2: dQ.
    {
        int num_qb = (q_len + kBr - 1) / kBr;
        if (num_qb > 0) {
            dim3 grid((unsigned)num_qb, (unsigned)heads, (unsigned)batch);
            dim3 block((unsigned)kThreadsPerBlock);
            size_t smem = flash_bw_dQ_smem_bytes<T>(d_k, d_v);
            if (smem > 48 * 1024) {
                cudaError_t serr = cudaFuncSetAttribute(
                    (const void*)flash_sdpa_bw_dQ_kernel<T>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)smem);
                if (serr != cudaSuccess) return 1000 + (int32_t)serr;
            }
            flash_sdpa_bw_dQ_kernel<T><<<grid, block, smem, stream>>>(
                q, k_in, v_in, dy, lse, D_ws, dQ,
                batch, heads, q_len, k_len, d_k, d_v, scale, is_causal);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return 1000 + (int32_t)err;
        }
    }
    // K3: dK, dV.
    {
        int num_kb = (k_len + kBc - 1) / kBc;
        if (num_kb > 0) {
            dim3 grid((unsigned)num_kb, (unsigned)heads, (unsigned)batch);
            dim3 block((unsigned)kThreadsPerBlock);
            size_t smem = flash_bw_dKdV_smem_bytes<T>(d_k, d_v);
            if (smem > 48 * 1024) {
                cudaError_t serr = cudaFuncSetAttribute(
                    (const void*)flash_sdpa_bw_dKdV_kernel<T>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)smem);
                if (serr != cudaSuccess) return 1000 + (int32_t)serr;
            }
            flash_sdpa_bw_dKdV_kernel<T><<<grid, block, smem, stream>>>(
                q, k_in, v_in, dy, lse, D_ws, dK, dV,
                batch, heads, q_len, k_len, d_k, d_v, scale, is_causal);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return 1000 + (int32_t)err;
        }
    }
    return 0;
}

} } // namespace baracuda::flash_sdpa

// =============================================================================
// INSTANTIATE macros
// =============================================================================

#define BARACUDA_KERNELS_FLASH_SDPA_INSTANTIATE(NAME, T)                                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                               \
        int32_t batch,                                                                              \
        int32_t heads,                                                                              \
        int32_t q_len,                                                                              \
        int32_t k_len,                                                                              \
        int32_t d_k,                                                                                \
        int32_t d_v,                                                                                \
        float scale,                                                                                \
        int32_t is_causal,                                                                          \
        const void* q,                                                                              \
        const void* k,                                                                              \
        const void* v,                                                                              \
        void* y,                                                                                    \
        void* lse,                                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                                \
        if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;       \
        int64_t total_y = (int64_t)batch * heads * q_len * d_v;                                     \
        if (total_y == 0) return 0;                                                                 \
        if (q == nullptr || k == nullptr || v == nullptr) return 2;                                 \
        if (y == nullptr || lse == nullptr) return 2;                                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::flash_sdpa::launch_flash_sdpa_fp<T>(                                       \
            static_cast<const T*>(q),                                                               \
            static_cast<const T*>(k),                                                               \
            static_cast<const T*>(v),                                                               \
            static_cast<T*>(y),                                                                     \
            static_cast<T*>(lse),                                                                   \
            batch, heads, q_len, k_len, d_k, d_v,                                                   \
            scale, is_causal,                                                                       \
            stream);                                                                                \
    }

#define BARACUDA_KERNELS_FLASH_SDPA_BACKWARD_INSTANTIATE(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                               \
        int32_t batch,                                                                              \
        int32_t heads,                                                                              \
        int32_t q_len,                                                                              \
        int32_t k_len,                                                                              \
        int32_t d_k,                                                                                \
        int32_t d_v,                                                                                \
        float scale,                                                                                \
        int32_t is_causal,                                                                          \
        const void* q,                                                                              \
        const void* k,                                                                              \
        const void* v,                                                                              \
        const void* y,                                                                              \
        const void* lse,                                                                            \
        const void* dy,                                                                             \
        void* d_ws,                                                                                 \
        void* dQ,                                                                                   \
        void* dK,                                                                                   \
        void* dV,                                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                                \
        if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;       \
        int64_t total = (int64_t)batch * heads * q_len * k_len;                                     \
        if (total == 0) return 0;                                                                   \
        if (q == nullptr || k == nullptr || v == nullptr) return 2;                                 \
        if (y == nullptr || lse == nullptr || dy == nullptr) return 2;                              \
        if (d_ws == nullptr) return 2;                                                              \
        if (dQ == nullptr || dK == nullptr || dV == nullptr) return 2;                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::flash_sdpa::launch_flash_sdpa_backward_fp<T>(                              \
            static_cast<const T*>(q),                                                               \
            static_cast<const T*>(k),                                                               \
            static_cast<const T*>(v),                                                               \
            static_cast<const T*>(y),                                                               \
            static_cast<const T*>(lse),                                                             \
            static_cast<const T*>(dy),                                                              \
            static_cast<T*>(d_ws),                                                                  \
            static_cast<T*>(dQ),                                                                    \
            static_cast<T*>(dK),                                                                    \
            static_cast<T*>(dV),                                                                    \
            batch, heads, q_len, k_len, d_k, d_v,                                                   \
            scale, is_causal,                                                                       \
            stream);                                                                                \
    }

#endif // BARACUDA_FLASH_SDPA_CUH
