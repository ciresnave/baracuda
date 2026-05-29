// baracuda_attn_arbmask.cuh
//
// Phase 51 — arbitrary additive-mask attention forward kernel.
//
// Same online-softmax algorithm as baracuda_flash_sdpa.cuh but adds an
// f32 additive bias `mask[B, H, Q, K]` to S = QK^T * scale *before* the
// row max/softmax. This unlocks:
//
//   * Speculative-decoding tree-attention masks (EAGLE / Medusa style)
//   * MoE expert masking
//   * Prefix-LM attention
//   * Sliding-window with attention sinks
//
// The mask is **always f32** regardless of element type — additive bias
// precision should be independent of QKV precision, and a single mask
// dtype simplifies the FFI surface dramatically (4 dtype × 1 mask vs
// 4 × 4). Use -INFINITY in mask cells to force exact suppression.
//
// Tile geometry mirrors `baracuda_flash_sdpa.cuh` (Br = Bc = 64;
// f64 path uses 32×32 to fit sm_89's 99 KiB dynamic SMEM cap). The
// kernel reuses the same `load_ct` / `store_ct` / `ComputeType`
// utilities — we just include the existing flash header.
//
// Layout contract (rank-4, contiguous, row-major):
//   Q   : [B, H, Q, D_k]
//   K   : [B, H, K, D_k]
//   V   : [B, H, K, D_v]
//   y   : [B, H, Q, D_v]
//   lse : [B, H, Q]
//   mask: [B, H, Q, K]  (f32; additive)
//
// Per Phase 51 Tier-1 scope:
//   * FW only (BW deferred to Tier 2 — same as the FA2 vendor).
//   * `is_causal` interacts with `mask` additively: when both, the
//     causal cells already at -INF stay -INF after the add (FP add).
//
// Status codes match the family:
//   0 success, 2 invalid problem, 3 unsupported,
//   1000+e launch-failure (e = cudaError_t).

#ifndef BARACUDA_ATTN_ARBMASK_CUH
#define BARACUDA_ATTN_ARBMASK_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_flash_sdpa.cuh"  // TileShape, ComputeType, load_ct, store_ct

namespace baracuda { namespace attn_arbmask {

using ::baracuda::flash_sdpa::ComputeType;
using ::baracuda::flash_sdpa::TileShape;
using ::baracuda::flash_sdpa::kMaxD;
using ::baracuda::flash_sdpa::kThreadsPerBlock;
using ::baracuda::flash_sdpa::load_ct;
using ::baracuda::flash_sdpa::store_ct;

// =============================================================================
// FW kernel — one block per (b, h, q_block). Identical structure to
// `flash_sdpa_fw_kernel` with one addition: after computing S = QK^T·scale
// (and applying the causal mask), we add the f32 `mask[b, h, q, k]` cell.
// =============================================================================

template <typename T>
__global__ void attn_arbmask_fw_kernel(
    const T*     __restrict__ q,
    const T*     __restrict__ k,
    const T*     __restrict__ v,
    const float* __restrict__ mask,   // [B, H, Q, K], f32 additive
    T*           __restrict__ y,
    T*           __restrict__ lse,
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
    T*  sQ = reinterpret_cast<T*>(sp);   sp += sizeof(T)  * kBr * d_k;
    T*  sK = reinterpret_cast<T*>(sp);   sp += sizeof(T)  * kBc * d_k;
    T*  sV = reinterpret_cast<T*>(sp);   sp += sizeof(T)  * kBc * d_v;
    CT* sS = reinterpret_cast<CT*>(sp);  sp += sizeof(CT) * kBr * kBc;
    CT* sO = reinterpret_cast<CT*>(sp);  sp += sizeof(CT) * kBr * d_v;
    CT* sM = reinterpret_cast<CT*>(sp);  sp += sizeof(CT) * kBr;
    CT* sL = reinterpret_cast<CT*>(sp);  sp += sizeof(CT) * kBr;

    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;
    const int qbase = qb * kBr;
    const int br_eff = (qbase + kBr <= q_len) ? kBr : (q_len - qbase);
    if (br_eff <= 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int64_t bh_off = ((int64_t)b * heads + h);
    const T* q_base = q + bh_off * q_len * d_k + (int64_t)qbase * d_k;
    const T* k_full = k + bh_off * k_len * d_k;
    const T* v_full = v + bh_off * k_len * d_v;
    T*       y_base = y + bh_off * q_len * d_v + (int64_t)qbase * d_v;
    T*       lse_base = lse + bh_off * q_len + qbase;
    // Mask base for this (b, h, q_block) — note mask is f32, not T.
    const float* mask_base = mask + bh_off * (int64_t)q_len * (int64_t)k_len
                                + (int64_t)qbase * (int64_t)k_len;

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

        // Causal early-out: every query row has q-index in
        // [qbase, qbase + br_eff - 1]. If kbase > q_max for all rows,
        // every cell here is causal-masked AND the additive mask may
        // not flip that (we don't know cheaply). Just do the work —
        // the causal cells get -INF and the additive mask add stays
        // -INF (a + -INF == -INF for finite a; -INF + -INF == -INF).
        //
        // NB: unlike `flash_sdpa_fw_kernel`, we do NOT skip causal
        // tiles, because the additive mask could in principle add a
        // *negative* number that we'd want to track. Today the
        // additive mask only adds, so -INF stays -INF — the early-out
        // would still be correct. We keep the early-out for symmetry
        // with the bespoke kernel but only when no mask path matters:
        // the tile is fully `-INF` after causal AND after mask.add.
        // Since we can't cheaply prove "mask is finite here", skip the
        // early-out entirely.

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

        // S_ij = Qᵢ · Kⱼ^T · scale  +  mask[b, h, q, k]
        // Padding (r >= br_eff or c >= bc_eff) → -INF.
        // Causal mask  (k_idx_abs > q_idx_abs) → -INF.
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
                CT s = acc * scale_ct;
                int q_idx_abs = qbase + r;
                int k_idx_abs = kbase + c;
                if (is_causal != 0 && k_idx_abs > q_idx_abs) {
                    s = -(CT)INFINITY;
                } else {
                    // Add the f32 additive mask. -INF in mask forces
                    // exact suppression. The cast to CT happens after
                    // the load — for f64 we widen, for f32/f16/bf16
                    // (CT == float) the cast is a no-op.
                    float m = mask_base[(int64_t)r * (int64_t)k_len + k_idx_abs];
                    s = s + (CT)m;
                }
                sS[idx] = s;
            }
        }
        __syncthreads();

        // Per-row online softmax update (identical to flash_sdpa).
        __shared__ CT sAlpha[kBr];

        if (tid < br_eff) {
            int r = tid;
            CT mloc = -(CT)INFINITY;
            for (int c = 0; c < bc_eff; ++c) {
                CT v = sS[r * kBc + c];
                if (v > mloc) mloc = v;
            }
            CT mold = sM[r];
            CT mnew = (mold > mloc) ? mold : mloc;
            CT alpha;
            CT l_local = (CT)0;
            if (!isfinite((float)mnew)) {
                alpha = (CT)1;
                for (int c = 0; c < bc_eff; ++c) {
                    sS[r * kBc + c] = (CT)0;
                }
                sAlpha[r] = alpha;
            } else {
                if (isfinite((float)mold)) {
                    alpha = (CT)exp((double)(mold - mnew));
                } else {
                    alpha = (CT)0;
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
            for (int c = bc_eff; c < kBc; ++c) {
                sS[r * kBc + c] = (CT)0;
            }
        }
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

    // Finalize: y = sO / sL[:, None].  lse = sM + log(sL).
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

// =============================================================================
// FW launcher
// =============================================================================

template <typename T>
__host__ inline size_t arbmask_fw_smem_bytes(int d_k, int d_v) {
    using CT = typename ComputeType<T>::type;
    constexpr int kBr = TileShape<T>::Br;
    constexpr int kBc = TileShape<T>::Bc;
    size_t sQ = sizeof(T)  * kBr * d_k;
    size_t sK = sizeof(T)  * kBc * d_k;
    size_t sV = sizeof(T)  * kBc * d_v;
    size_t sS = sizeof(CT) * kBr * kBc;
    size_t sO = sizeof(CT) * kBr * d_v;
    size_t sM = sizeof(CT) * kBr;
    size_t sL = sizeof(CT) * kBr;
    return sQ + sK + sV + sS + sO + sM + sL;
}

template <typename T>
__host__ inline int32_t launch_attn_arbmask_fp(
    const T* q, const T* k_in, const T* v_in,
    const float* mask,
    T* y, T* lse,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    int32_t d_k, int32_t d_v,
    float scale, int32_t is_causal,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;
    if (d_k > kMaxD || d_v > kMaxD) return 3;
    if (d_k != d_v) return 3;
    int64_t total_y = (int64_t)batch * heads * q_len * d_v;
    if (total_y == 0) return 0;
    constexpr int kBr = TileShape<T>::Br;
    int num_qb = (q_len + kBr - 1) / kBr;
    if (num_qb <= 0) return 0;
    dim3 grid((unsigned)num_qb, (unsigned)heads, (unsigned)batch);
    dim3 block((unsigned)kThreadsPerBlock);
    size_t smem = arbmask_fw_smem_bytes<T>(d_k, d_v);
    if (smem > 48 * 1024) {
        cudaError_t serr = cudaFuncSetAttribute(
            (const void*)attn_arbmask_fw_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        if (serr != cudaSuccess) return 1000 + (int32_t)serr;
    }
    attn_arbmask_fw_kernel<T><<<grid, block, smem, stream>>>(
        q, k_in, v_in, mask, y, lse, batch, heads, q_len, k_len,
        d_k, d_v, scale, is_causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

// =============================================================================
// C-ABI INSTANTIATE macro — emits an `extern "C"` symbol named
// `baracuda_kernels_<NAME>_arbmask_run` plus an `_arbmask_can_implement`
// host gate. Matches the conventions of the bespoke flash_sdpa family.
// =============================================================================

#define BARACUDA_KERNELS_ATTN_ARBMASK_INSTANTIATE(NAME, T)                            \
    extern "C" int32_t baracuda_kernels_##NAME##_arbmask_run(                         \
        int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,                   \
        int32_t d_k, int32_t d_v, float scale, int32_t is_causal,                     \
        const void* q, const void* k, const void* v,                                  \
        const void* mask,                                                             \
        void* y, void* lse,                                                           \
        void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)           \
    {                                                                                 \
        return ::baracuda::attn_arbmask::launch_attn_arbmask_fp<T>(                   \
            reinterpret_cast<const T*>(q),                                            \
            reinterpret_cast<const T*>(k),                                            \
            reinterpret_cast<const T*>(v),                                            \
            reinterpret_cast<const float*>(mask),                                     \
            reinterpret_cast<T*>(y),                                                  \
            reinterpret_cast<T*>(lse),                                                \
            batch, heads, q_len, k_len, d_k, d_v, scale, is_causal,                   \
            reinterpret_cast<cudaStream_t>(stream));                                  \
    }                                                                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_arbmask_can_implement(               \
        int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,                   \
        int32_t d_k, int32_t d_v, int32_t /*is_causal*/)                              \
    {                                                                                 \
        if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0)   \
            return 2;                                                                 \
        if (d_k > ::baracuda::flash_sdpa::kMaxD ||                                    \
            d_v > ::baracuda::flash_sdpa::kMaxD) return 3;                            \
        if (d_k != d_v) return 3;                                                     \
        return 0;                                                                     \
    }

} } // namespace baracuda::attn_arbmask

#endif // BARACUDA_ATTN_ARBMASK_CUH
