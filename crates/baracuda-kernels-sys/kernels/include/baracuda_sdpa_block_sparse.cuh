// baracuda_sdpa_block_sparse.cuh
//
// Phase 54 — BlockSparse Attention FW kernel (clean-room hand-port
// of facebookresearch/xformers `components/attention/blocksparse.py`
// algorithm reference; see `vendor/xformers/VENDOR.md` for attribution).
//
// Block-sparse SDPA: the attention mask is a per-block boolean pattern
// `[B, H, num_blocks_q × num_blocks_k]`. Only the (q_block, k_block)
// pairs marked `true` participate in the QK^T matmul + softmax
// accumulation; masked blocks are **skipped entirely** (no compute, no
// memory traffic for K/V). This is the differentiator from baracuda's
// existing arbitrary additive-mask path (Phase 51) which still computes
// every (Q, K) pair and just adds a bias.
//
// Algorithm:
//   For each (b, h, qb):
//     Initialize m = -inf, l = 0, O = 0  (online softmax state)
//     For each kb in 0..num_blocks_k:
//       If block_pattern[b, h, qb, kb] == 0: skip
//       Apply causal early-out: if is_causal and kb_start > q_end: skip
//       Load K[kb], V[kb] tiles
//       Run one Flash-style tile of online softmax (same as Phase 6.6)
//     Finalize: y = O / l, lse = m + log(l)
//
// Tile geometry: `block_size` is supplied at launch time (typical
// values: 32, 64, 128). The kernel uses one CUDA block per (b, h, qb)
// and one CUDA thread block iterates the active k-blocks sequentially
// (no parallelism across k-blocks, same as Flash). `block_size` ==
// `Br == Bc`.
//
// Layout contract (rank-4, contiguous, row-major):
//   Q             : [B, H, Q_len, D_k]
//   K             : [B, H, K_len, D_k]
//   V             : [B, H, K_len, D_v]
//   y             : [B, H, Q_len, D_v]
//   lse           : [B, H, Q_len]
//   block_pattern : [B, H, num_blocks_q * num_blocks_k]  (uint8_t, row-major)
//
// `num_blocks_q = ceil(Q_len / block_size)`,
// `num_blocks_k = ceil(K_len / block_size)`. The pattern is stored as
// `uint8_t` (1 byte per block-pair) — small and easy to set from Rust.
//
// Status codes:
//   0 success, 2 invalid problem, 3 unsupported,
//   1000+e launch failure (e = cudaError_t).

#ifndef BARACUDA_SDPA_BLOCK_SPARSE_CUH
#define BARACUDA_SDPA_BLOCK_SPARSE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_flash_sdpa.cuh"  // ComputeType, load_ct, store_ct

namespace baracuda { namespace sdpa_block_sparse {

using ::baracuda::flash_sdpa::ComputeType;
using ::baracuda::flash_sdpa::load_ct;
using ::baracuda::flash_sdpa::store_ct;

// Max block_size we support in the Tier-1 trailblazer. Constrained by
// the dynamic SMEM budget at d_k = d_v = 128 (the upper bound).
// Per-block SMEM (f32 compute path):
//   sQ:  Br * d_k  * sizeof(T)         = 128 * 128 * 2 = 32 KiB (f16)
//   sK:  Bc * d_k  * sizeof(T)         = 128 * 128 * 2 = 32 KiB (f16)
//   sV:  Bc * d_v  * sizeof(T)         = 128 * 128 * 2 = 32 KiB (f16)
//   sS:  Br * Bc   * sizeof(float)     = 128 * 128 * 4 = 64 KiB
//   sO:  Br * d_v  * sizeof(float)     = 128 * 128 * 4 = 64 KiB
// Total ≈ 224 KiB at Br=Bc=128 — exceeds sm_89's 99 KiB cap.
// Cap block_size at 64 for the trailblazer; 128 deferred to Tier-2
// (would need register-tiled matmul + warp-shuffle softmax).
constexpr int kBlockSparseMaxBlockSize = 64;
constexpr int kBlockSparseMaxD         = 128;
constexpr int kThreadsPerBlock         = 128;

// =============================================================================
// FW kernel — one block per (b, h, q_block). Iterates only the active
// k-blocks per the supplied `block_pattern` and runs the online-softmax
// tile loop on each.
// =============================================================================

template <typename T>
__global__ void sdpa_block_sparse_fw_kernel(
    const T*       __restrict__ q,
    const T*       __restrict__ k,
    const T*       __restrict__ v,
    const uint8_t* __restrict__ block_pattern,  // [B, H, nbq * nbk]
    T*             __restrict__ y,
    T*             __restrict__ lse,
    int32_t batch,
    int32_t heads,
    int32_t q_len,
    int32_t k_len,
    int32_t d_k,
    int32_t d_v,
    int32_t block_size,
    int32_t num_blocks_q,
    int32_t num_blocks_k,
    float   scale,
    int32_t is_causal)
{
    using CT = typename ComputeType<T>::type;

    // Compile-time upper bound for the SMEM allocation; runtime block_size
    // is always ≤ this.
    constexpr int kMaxBs = kBlockSparseMaxBlockSize;
    const int Br = block_size;
    const int Bc = block_size;

    extern __shared__ unsigned char smem_raw[];
    unsigned char* sp = smem_raw;
    T*  sQ = reinterpret_cast<T*>(sp);   sp += sizeof(T)  * (size_t)Br * d_k;
    T*  sK = reinterpret_cast<T*>(sp);   sp += sizeof(T)  * (size_t)Bc * d_k;
    T*  sV = reinterpret_cast<T*>(sp);   sp += sizeof(T)  * (size_t)Bc * d_v;
    CT* sS = reinterpret_cast<CT*>(sp);  sp += sizeof(CT) * (size_t)Br * Bc;
    CT* sO = reinterpret_cast<CT*>(sp);  sp += sizeof(CT) * (size_t)Br * d_v;
    CT* sM = reinterpret_cast<CT*>(sp);  sp += sizeof(CT) * (size_t)Br;
    CT* sL = reinterpret_cast<CT*>(sp);  sp += sizeof(CT) * (size_t)Br;

    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;
    const int qbase = qb * Br;
    const int br_eff = (qbase + Br <= q_len) ? Br : (q_len - qbase);
    if (br_eff <= 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Base offsets (element units).
    const int64_t bh_off = ((int64_t)b * heads + h);
    const T* q_base   = q + bh_off * q_len * d_k + (int64_t)qbase * d_k;
    const T* k_full   = k + bh_off * k_len * d_k;
    const T* v_full   = v + bh_off * k_len * d_v;
    T*       y_base   = y + bh_off * q_len * d_v + (int64_t)qbase * d_v;
    T*       lse_base = lse + bh_off * q_len + qbase;

    // Block-pattern row for this (b, h, qb).
    const int64_t bp_row_off =
        (((int64_t)b * heads + h) * num_blocks_q + qb) * num_blocks_k;
    const uint8_t* bp_row = block_pattern + bp_row_off;

    // Load Q tile.
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

    for (int kb = 0; kb < num_blocks_k; ++kb) {
        // Per-block sparsity gate. Pattern is uint8_t; treat 0 as
        // "skip", anything else as "compute".
        if (bp_row[kb] == 0) continue;

        const int kbase = kb * Bc;
        const int bc_eff = (kbase + Bc <= k_len) ? Bc : (k_len - kbase);
        if (bc_eff <= 0) continue;

        // Causal early-out: if every row in this q-block has q-index
        // < kbase, the entire k-block is masked.
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

        // S = Q · K^T · scale, applying causal and pad masks.
        {
            int total = Br * Bc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / Bc;
                int c = idx - r * Bc;
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
                CT vcell = acc * scale_ct;
                int q_idx_abs = qbase + r;
                int k_idx_abs = kbase + c;
                if (is_causal != 0 && k_idx_abs > q_idx_abs) {
                    vcell = -(CT)INFINITY;
                }
                sS[idx] = vcell;
            }
        }
        __syncthreads();

        // Per-row online softmax update + α stash.
        __shared__ CT sAlpha[kMaxBs];

        if (tid < br_eff) {
            int r = tid;
            CT mloc = -(CT)INFINITY;
            for (int c = 0; c < bc_eff; ++c) {
                CT vc = sS[r * Bc + c];
                if (vc > mloc) mloc = vc;
            }
            CT mold = sM[r];
            CT mnew = (mold > mloc) ? mold : mloc;
            CT alpha;
            CT l_local = (CT)0;
            if (!isfinite((float)mnew)) {
                alpha = (CT)1;
                for (int c = 0; c < bc_eff; ++c) {
                    sS[r * Bc + c] = (CT)0;
                }
                sAlpha[r] = alpha;
            } else {
                if (isfinite((float)mold)) {
                    alpha = (CT)exp((double)(mold - mnew));
                } else {
                    alpha = (CT)0;
                }
                for (int c = 0; c < bc_eff; ++c) {
                    CT vc = sS[r * Bc + c];
                    CT p = (isfinite((float)vc))
                           ? (CT)exp((double)(vc - mnew))
                           : (CT)0;
                    sS[r * Bc + c] = p;
                    l_local += p;
                }
                sAlpha[r] = alpha;
                sM[r] = mnew;
                sL[r] = alpha * sL[r] + l_local;
            }
            for (int c = bc_eff; c < Bc; ++c) {
                sS[r * Bc + c] = (CT)0;
            }
        }
        // Zero padding rows.
        {
            int total_pad = Br * Bc;
            for (int idx = tid; idx < total_pad; idx += nthreads) {
                int r = idx / Bc;
                if (r >= br_eff) sS[idx] = (CT)0;
            }
        }
        __syncthreads();

        // O ← α · O + P · V
        {
            int total = Br * d_v;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_v;
                if (r >= br_eff) continue;
                int dv = idx - r * d_v;
                CT acc = (CT)0;
                for (int c = 0; c < bc_eff; ++c) {
                    CT p = sS[r * Bc + c];
                    CT vv = load_ct<T>(sV[c * d_v + dv]);
                    acc += p * vv;
                }
                CT alpha = sAlpha[r];
                sO[r * d_v + dv] = alpha * sO[r * d_v + dv] + acc;
            }
        }
        __syncthreads();
    } // end kb

    // Finalize.
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
__host__ inline size_t block_sparse_fw_smem_bytes(int block_size, int d_k, int d_v) {
    using CT = typename ComputeType<T>::type;
    size_t Br = (size_t)block_size;
    size_t Bc = (size_t)block_size;
    size_t sQ = sizeof(T)  * Br * d_k;
    size_t sK = sizeof(T)  * Bc * d_k;
    size_t sV = sizeof(T)  * Bc * d_v;
    size_t sS = sizeof(CT) * Br * Bc;
    size_t sO = sizeof(CT) * Br * d_v;
    size_t sM = sizeof(CT) * Br;
    size_t sL = sizeof(CT) * Br;
    return sQ + sK + sV + sS + sO + sM + sL;
}

template <typename T>
__host__ inline int32_t launch_sdpa_block_sparse_fp(
    const T* q, const T* k_in, const T* v_in,
    const uint8_t* block_pattern,
    T* y, T* lse,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    int32_t d_k, int32_t d_v,
    int32_t block_size,
    float scale, int32_t is_causal,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) {
        return 2;
    }
    if (block_size <= 0 || block_size > kBlockSparseMaxBlockSize) return 3;
    // block_size must be a power of 2 or at least divide the tile-loop
    // assumption cleanly. We don't strictly require a power of 2 (the
    // ceil-div handles tails) but the caller is responsible for the
    // block_pattern's row layout matching ceil(q_len / block_size).
    if (d_k > kBlockSparseMaxD || d_v > kBlockSparseMaxD) return 3;
    if (d_k != d_v) return 3;     // trailblazer: matched K/V head dim
    int64_t total_y = (int64_t)batch * heads * q_len * d_v;
    if (total_y == 0) return 0;

    int num_blocks_q = (q_len + block_size - 1) / block_size;
    int num_blocks_k = (k_len + block_size - 1) / block_size;
    if (num_blocks_q <= 0 || num_blocks_k <= 0) return 0;

    dim3 grid((unsigned)num_blocks_q, (unsigned)heads, (unsigned)batch);
    dim3 block((unsigned)kThreadsPerBlock);
    size_t smem = block_sparse_fw_smem_bytes<T>(block_size, d_k, d_v);
    if (smem > 48 * 1024) {
        cudaError_t serr = cudaFuncSetAttribute(
            (const void*)sdpa_block_sparse_fw_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        if (serr != cudaSuccess) return 1000 + (int32_t)serr;
    }
    sdpa_block_sparse_fw_kernel<T><<<grid, block, smem, stream>>>(
        q, k_in, v_in, block_pattern, y, lse,
        batch, heads, q_len, k_len, d_k, d_v,
        block_size, num_blocks_q, num_blocks_k,
        scale, is_causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

} } // namespace baracuda::sdpa_block_sparse

// =============================================================================
// INSTANTIATE macro — emits the C-ABI _run and _can_implement symbols
// for a given (name, T) pair.
// =============================================================================

#define BARACUDA_KERNELS_SDPA_BLOCK_SPARSE_INSTANTIATE(NAME, T)                  \
    extern "C" int32_t baracuda_kernels_##NAME##_block_sparse_run(               \
        int32_t batch, int32_t heads,                                             \
        int32_t q_len, int32_t k_len,                                             \
        int32_t d_k, int32_t d_v,                                                 \
        int32_t block_size,                                                       \
        float scale, int32_t is_causal,                                           \
        const void* q, const void* k, const void* v,                              \
        const void* block_pattern,                                                \
        void* y, void* lse,                                                       \
        void* /*workspace*/, uint64_t /*workspace_bytes*/,                        \
        void* stream)                                                             \
    {                                                                             \
        return ::baracuda::sdpa_block_sparse::launch_sdpa_block_sparse_fp<T>(    \
            reinterpret_cast<const T*>(q),                                        \
            reinterpret_cast<const T*>(k),                                        \
            reinterpret_cast<const T*>(v),                                        \
            reinterpret_cast<const uint8_t*>(block_pattern),                      \
            reinterpret_cast<T*>(y),                                              \
            reinterpret_cast<T*>(lse),                                            \
            batch, heads, q_len, k_len, d_k, d_v,                                 \
            block_size, scale, is_causal,                                         \
            reinterpret_cast<cudaStream_t>(stream));                              \
    }                                                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_block_sparse_can_implement(     \
        int32_t batch, int32_t heads,                                             \
        int32_t q_len, int32_t k_len,                                             \
        int32_t d_k, int32_t d_v,                                                 \
        int32_t block_size)                                                       \
    {                                                                             \
        if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0                      \
            || d_k < 0 || d_v < 0) return 2;                                      \
        if (block_size <= 0                                                       \
            || block_size > ::baracuda::sdpa_block_sparse::kBlockSparseMaxBlockSize) \
            return 3;                                                             \
        if (d_k > ::baracuda::sdpa_block_sparse::kBlockSparseMaxD                 \
            || d_v > ::baracuda::sdpa_block_sparse::kBlockSparseMaxD) return 3;   \
        if (d_k != d_v) return 3;                                                 \
        return 0;                                                                 \
    }

#endif // BARACUDA_SDPA_BLOCK_SPARSE_CUH
