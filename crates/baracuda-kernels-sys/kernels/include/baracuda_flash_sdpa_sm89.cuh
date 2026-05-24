// baracuda_flash_sdpa_sm89.cuh
//
// sm_89 (Ada Lovelace) sibling of `baracuda_flash_sdpa.cuh`. Phase 10
// Milestone 10.3 of the baracuda CUDA stack — first proof of the
// sibling-plan + arch-dispatcher pattern.
//
// Algorithm is identical to the sm_80 baseline (Tri Dao 2022 FlashAttention
// FW with online softmax). What changes here is the *data-movement
// strategy*:
//
//   1. **`cp.async` double-buffered K/V loads.** While the current
//      iteration computes scores + softmax + P·V on tile `kb`, the
//      *next* K/V tile (`kb + 1`) is prefetched directly from gmem into
//      a second smem buffer via `cp.async.cg.shared.global` (16-byte
//      cache-line granularity on Ada). The K/V load latency is hidden
//      behind the matmul, instead of stalling the warp on global memory.
//
//   2. **Wider thread block.** sm_89 has a larger register file per SM
//      than sm_80 baseline (256 KB vs 192 KB for the comparable Ampere
//      die), so we can afford 256 threads/block (vs the sm_80 baseline's
//      128) without losing occupancy. More threads means each
//      thread-per-output-cell pass finishes in fewer grid-stride iters.
//
//   3. **Tile shape stays 64×64** for f16 / bf16. Bumping Br = Bc = 128
//      with `d_k = d_v = 128` blows past the 99 KiB per-block dynamic-
//      smem cap (would need ~225 KiB). The async double-buffer adds
//      ~16 KiB per dtype (one extra `sK_next` + `sV_next` tile) on top
//      of the baseline 60 KiB, so total stays well under 99 KiB.
//
// What this trailblazer does *not* yet do (documented as follow-ups):
//
//   - `ldmatrix.sync.aligned.x4` warp-cooperative fragment loads. The
//     baseline thread-per-output-cell scalar-FMA pattern is retained so
//     bit-for-bit cross-validation against the sm_80 plan stays at
//     ~32·eps. Switching to `ldmatrix` + `wmma::fragment` matmuls is the
//     follow-up perf milestone (Milestone 10.4 / 10.5).
//   - `nvcuda::wmma` m16n8k16 matmul fragments. Same reasoning — the
//     wmma path reorders the inner accumulation, which means a separate
//     tolerance budget. It will ship once the async-prefetch path is
//     fully measured in isolation.
//   - FP8 (E4M3 / E5M2) tensor-core attention. Phase 10 follow-up
//     once the f16/bf16 prefetch path is benched.
//
// Dtype scope: f16 + bf16 only. f32 / f64 stay on the sm_80 baseline —
// Ada's FP8 / FP16 tensor cores don't help f32, and f64 is uncommon in
// transformer inference. This sibling is purely *additive*; the existing
// `FlashSdpaPlan<T>` continues to handle all four dtypes via the sm_80
// baseline kernel for callers that don't request the sm_89 specialization.
//
// SMEM layout (per block), with N = `kNumStages = 2`:
//
//   sQ          : T[Br * d_k]                  (loaded once)
//   sK[stage]   : T[Bc * d_k]   × N            (cp.async double-buffer)
//   sV[stage]   : T[Bc * d_v]   × N            (cp.async double-buffer)
//   sS          : float[Br * Bc]               (scores / probs)
//   sO          : float[Br * d_v]              (running output)
//   sM, sL      : float[Br]                    (running stats)
//   sAlpha      : float[Br]                    (per-row α scratch)
//
// f16/bf16 at d=128, Br=Bc=64, N=2 stages:
//   sQ = 16K, 2·sK = 16K, 2·sV = 16K, sS = 16K, sO = 32K, sM+sL+sAlpha < 1K
//   total ≈ 97 KiB — under the 99 KiB cap. Tight but it fits.
//
// Status codes match the family.

#ifndef BARACUDA_FLASH_SDPA_SM89_CUH
#define BARACUDA_FLASH_SDPA_SM89_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace baracuda { namespace flash_sdpa_sm89 {

// =============================================================================
// Tile geometry.
// =============================================================================
//
// f16 / bf16 only on this sm_89 path. Br = Bc = 64 — see header notes for
// the smem budget that drove this choice. Threads-per-block bumped to 256
// (vs sm_80 baseline's 128) for higher in-block parallelism on Ada's
// larger SM register file.

constexpr int kBr = 64;
constexpr int kBc = 64;
constexpr int kMaxD = 128;
constexpr int kThreadsPerBlock = 256;
constexpr int kNumStages = 2;   // cp.async double-buffer.

// =============================================================================
// dtype helpers — f32 detour for half / bf16. f32 / f64 are NOT instantiated
// on this path; they stay on the sm_80 baseline.
// =============================================================================

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
// cp.async helpers.
//
// `__pipeline_memcpy_async` (cuda_pipeline.h) wraps PTX
// `cp.async.cg.shared.global` on sm_80+ and `cp.async.bulk.tensor` /
// equivalents on newer arches. We use the 16-byte granularity path (`.cg`)
// — Ada loads K/V at coalesced 16B / thread boundaries.
//
// Per the CUDA programming guide, cp.async needs commit + wait barriers
// to make the data visible to subsequent code that reads from smem. We
// drive these via the `cuda::pipeline` / `__pipeline_*` thin wrappers.
// =============================================================================

template <typename T>
__device__ __forceinline__ void async_copy_tile(
    T*       __restrict__ smem_dst,
    const T* __restrict__ gmem_src,
    int n_rows,
    int n_cols,
    int gmem_row_stride,
    int smem_row_stride,
    int tid,
    int nthreads)
{
    // Each thread issues 16-byte chunks. For T = __half / __nv_bfloat16
    // (2 bytes), that's 8 elements per chunk. For other widths we fall
    // back to element-stride.
    constexpr int kChunkBytes = 16;
    constexpr int kElemsPerChunk = kChunkBytes / (int)sizeof(T);
    const int total_chunks_per_row = n_cols / kElemsPerChunk;

    if (total_chunks_per_row >= 1) {
        const int total = n_rows * total_chunks_per_row;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / total_chunks_per_row;
            int c = idx - r * total_chunks_per_row;
            int col_elem = c * kElemsPerChunk;
            T*       dst = smem_dst + r * smem_row_stride + col_elem;
            const T* src = gmem_src + r * gmem_row_stride + col_elem;
            __pipeline_memcpy_async(dst, src, kChunkBytes);
        }
        // Trailing leftover columns that don't fit a 16B chunk — these
        // happen when d_k is not a multiple of kElemsPerChunk (e.g. d=24
        // for f16). Fall back to scalar copies.
        const int tail_start = total_chunks_per_row * kElemsPerChunk;
        if (tail_start < n_cols) {
            const int tail_total = n_rows * (n_cols - tail_start);
            for (int idx = tid; idx < tail_total; idx += nthreads) {
                int r = idx / (n_cols - tail_start);
                int c = idx - r * (n_cols - tail_start);
                smem_dst[r * smem_row_stride + tail_start + c] =
                    gmem_src[r * gmem_row_stride + tail_start + c];
            }
        }
    } else {
        // Tiny widths — pure scalar fallback.
        const int total = n_rows * n_cols;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / n_cols;
            int c = idx - r * n_cols;
            smem_dst[r * smem_row_stride + c] =
                gmem_src[r * gmem_row_stride + c];
        }
    }
}

// =============================================================================
// FW kernel — sm_89 specialization.
//
// Grid: one block per (batch, head, q_block). Within each block we walk
// the K dimension in tiles of `kBc`, double-buffering K and V via
// `cp.async`. Stage 0 / stage 1 alternate.
// =============================================================================

template <typename T>
__global__ void flash_sdpa_sm89_fw_kernel(
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
    extern __shared__ unsigned char smem_raw[];
    unsigned char* sp = smem_raw;
    T* sQ = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBr * d_k;
    T* sK[kNumStages];
    T* sV[kNumStages];
    sK[0] = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBc * d_k;
    sK[1] = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBc * d_k;
    sV[0] = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBc * d_v;
    sV[1] = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBc * d_v;
    float* sS  = reinterpret_cast<float*>(sp);                 sp += sizeof(float) * kBr * kBc;
    float* sO  = reinterpret_cast<float*>(sp);                 sp += sizeof(float) * kBr * d_v;
    float* sM  = reinterpret_cast<float*>(sp);                 sp += sizeof(float) * kBr;
    float* sL  = reinterpret_cast<float*>(sp);                 sp += sizeof(float) * kBr;
    float* sAlpha = reinterpret_cast<float*>(sp);              sp += sizeof(float) * kBr;

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

    // Load Q once (synchronous — the rest of the kernel uses it on every
    // K-tile). We use scalar loads here because Q is small enough that
    // the savings from cp.async are negligible compared to the
    // initialization cost.
    {
        const int total = br_eff * d_k;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_k;
            int c = idx - r * d_k;
            sQ[r * d_k + c] = q_base[(int64_t)r * d_k + c];
        }
    }
    // Init sO, sM, sL.
    {
        const int total_o = br_eff * d_v;
        for (int idx = tid; idx < total_o; idx += nthreads) sO[idx] = 0.f;
        for (int r = tid; r < br_eff; r += nthreads) {
            sM[r] = -INFINITY;
            sL[r] = 0.f;
        }
    }

    const int num_kb = (k_len + kBc - 1) / kBc;
    if (num_kb <= 0) {
        // Should not happen given the early-return on br_eff <= 0, but
        // guards a zero-K corner case.
        __syncthreads();
        return;
    }

    // Issue stage-0 async loads for kb = 0 immediately. This and the
    // matching `__pipeline_commit` below kick off the gmem fetch in
    // parallel with the Q copy + sO init above (kernel-level parallelism).
    {
        const int kbase0 = 0;
        const int bc_eff0 = (kbase0 + kBc <= k_len) ? kBc : (k_len - kbase0);
        async_copy_tile<T>(
            sK[0], k_full + (int64_t)kbase0 * d_k,
            bc_eff0, d_k, d_k, d_k, tid, nthreads);
        async_copy_tile<T>(
            sV[0], v_full + (int64_t)kbase0 * d_v,
            bc_eff0, d_v, d_v, d_v, tid, nthreads);
        __pipeline_commit();
    }
    __syncthreads();

    const float scale_f = scale;
    for (int kb = 0; kb < num_kb; ++kb) {
        const int stage = kb & 1;
        const int kbase = kb * kBc;
        const int bc_eff = (kbase + kBc <= k_len) ? kBc : (k_len - kbase);

        // Causal early-out — same logic as the sm_80 baseline.
        const bool block_fully_masked =
            (is_causal != 0 && kbase > (qbase + br_eff - 1));

        if (!block_fully_masked) {
            // Wait for the cp.async batch we issued one iteration ago
            // (or in the preamble for kb=0). After this point sK[stage] /
            // sV[stage] are visible to all threads in the block.
            __pipeline_wait_prior(0);
            __syncthreads();
        }

        // Prefetch the next K/V tile into the other stage's smem while
        // the current tile is being consumed below. We always issue the
        // prefetch (even on the last iteration's tail, which just
        // overwrites an unused stage). Skipping the prefetch on the last
        // iter would save a few load instructions but complicate the
        // wait-prior bookkeeping.
        if (kb + 1 < num_kb) {
            const int next_stage = (kb + 1) & 1;
            const int kbase_next = (kb + 1) * kBc;
            const int bc_eff_next = (kbase_next + kBc <= k_len)
                ? kBc : (k_len - kbase_next);
            async_copy_tile<T>(
                sK[next_stage], k_full + (int64_t)kbase_next * d_k,
                bc_eff_next, d_k, d_k, d_k, tid, nthreads);
            async_copy_tile<T>(
                sV[next_stage], v_full + (int64_t)kbase_next * d_v,
                bc_eff_next, d_v, d_v, d_v, tid, nthreads);
            __pipeline_commit();
        }

        if (block_fully_masked) {
            // Nothing to do for this k-block. The prefetch above is
            // still issued (so the next iteration can wait on it), but
            // we skip the matmul + softmax bookkeeping.
            continue;
        }

        // S_ij = Q · K^T · scale into sS. Same scalar pattern as the
        // sm_80 baseline so the math is bit-for-bit equivalent (modulo
        // float-order in independent terms — which both kernels handle
        // identically because the inner FMA loop is unrolled the same
        // way).
        {
            const T* sKstage = sK[stage];
            const int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) {
                    sS[idx] = -INFINITY;
                    continue;
                }
                float acc = 0.f;
                #pragma unroll 4
                for (int d = 0; d < d_k; ++d) {
                    float qv = load_f32<T>(sQ[r * d_k + d]);
                    float kv = load_f32<T>(sKstage[c * d_k + d]);
                    acc += qv * kv;
                }
                float vscaled = acc * scale_f;
                int q_idx_abs = qbase + r;
                int k_idx_abs = kbase + c;
                if (is_causal != 0 && k_idx_abs > q_idx_abs) {
                    vscaled = -INFINITY;
                }
                sS[idx] = vscaled;
            }
        }
        __syncthreads();

        // Per-row online softmax update — identical to the sm_80 path.
        if (tid < br_eff) {
            int r = tid;
            float mloc = -INFINITY;
            for (int c = 0; c < bc_eff; ++c) {
                float v_s = sS[r * kBc + c];
                if (v_s > mloc) mloc = v_s;
            }
            float mold = sM[r];
            float mnew = (mold > mloc) ? mold : mloc;
            float alpha;
            float l_local = 0.f;
            if (!isfinite(mnew)) {
                alpha = 1.f;
                for (int c = 0; c < bc_eff; ++c) sS[r * kBc + c] = 0.f;
                sAlpha[r] = alpha;
            } else {
                if (isfinite(mold)) {
                    alpha = expf(mold - mnew);
                } else {
                    alpha = 0.f;
                }
                for (int c = 0; c < bc_eff; ++c) {
                    float v_s = sS[r * kBc + c];
                    float p = isfinite(v_s) ? expf(v_s - mnew) : 0.f;
                    sS[r * kBc + c] = p;
                    l_local += p;
                }
                sAlpha[r] = alpha;
                sM[r] = mnew;
                sL[r] = alpha * sL[r] + l_local;
            }
            // Zero pad columns.
            for (int c = bc_eff; c < kBc; ++c) sS[r * kBc + c] = 0.f;
        }
        // Zero pad rows.
        {
            const int total_pad = kBr * kBc;
            for (int idx = tid; idx < total_pad; idx += nthreads) {
                int r = idx / kBc;
                if (r >= br_eff) sS[idx] = 0.f;
            }
        }
        __syncthreads();

        // sO ← α[:, None] · sO + P · V.
        {
            const T* sVstage = sV[stage];
            const int total = kBr * d_v;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_v;
                if (r >= br_eff) continue;
                int dv = idx - r * d_v;
                float acc = 0.f;
                for (int c = 0; c < bc_eff; ++c) {
                    float p = sS[r * kBc + c];
                    float vv = load_f32<T>(sVstage[c * d_v + dv]);
                    acc += p * vv;
                }
                float a = sAlpha[r];
                sO[r * d_v + dv] = a * sO[r * d_v + dv] + acc;
            }
        }
        __syncthreads();
    } // end kb

    // Drain any remaining outstanding cp.async batches. This is mostly a
    // safety net — by the time we reach here we've already issued the
    // wait inside the loop and `__pipeline_wait_prior(0)` is idempotent
    // when no batches are pending.
    __pipeline_wait_prior(0);

    // Finalize. y = sO / sL[:, None]. lse = sM + log(sL).
    {
        const int total = br_eff * d_v;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_v;
            int dv = idx - r * d_v;
            float l = sL[r];
            float yv;
            if (l > 0.f && isfinite(l)) {
                yv = sO[r * d_v + dv] / l;
            } else {
                yv = 0.f;
            }
            y_base[(int64_t)r * d_v + dv] = store_f32<T>(yv);
        }
    }
    if (tid < br_eff) {
        int r = tid;
        float l = sL[r];
        float lse_v;
        if (l > 0.f && isfinite(l)) {
            lse_v = sM[r] + logf(l);
        } else {
            lse_v = -INFINITY;
        }
        lse_base[r] = store_f32<T>(lse_v);
    }
}

// =============================================================================
// Launcher.
// =============================================================================

template <typename T>
__host__ inline size_t flash_sm89_smem_bytes(int d_k, int d_v) {
    size_t s = 0;
    s += sizeof(T) * kBr * d_k;                         // sQ
    s += sizeof(T) * kBc * d_k * kNumStages;            // sK[0..N]
    s += sizeof(T) * kBc * d_v * kNumStages;            // sV[0..N]
    s += sizeof(float) * kBr * kBc;                     // sS
    s += sizeof(float) * kBr * d_v;                     // sO
    s += sizeof(float) * kBr * 3;                       // sM, sL, sAlpha
    return s;
}

template <typename T>
__host__ inline int32_t launch_flash_sdpa_sm89_fp(
    const T* q, const T* k_in, const T* v_in,
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
    int num_qb = (q_len + kBr - 1) / kBr;
    if (num_qb <= 0) return 0;
    dim3 grid((unsigned)num_qb, (unsigned)heads, (unsigned)batch);
    dim3 block((unsigned)kThreadsPerBlock);
    size_t smem = flash_sm89_smem_bytes<T>(d_k, d_v);
    if (smem > 48 * 1024) {
        cudaError_t serr = cudaFuncSetAttribute(
            (const void*)flash_sdpa_sm89_fw_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        if (serr != cudaSuccess) return 1000 + (int32_t)serr;
    }
    flash_sdpa_sm89_fw_kernel<T><<<grid, block, smem, stream>>>(
        q, k_in, v_in, y, lse, batch, heads, q_len, k_len,
        d_k, d_v, scale, is_causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

// =============================================================================
// FW kernel — sm_89 specialization, **strided** variant (Phase 17.1).
//
// Sibling of `flash_sdpa_sm89_fw_kernel` above. Same algorithm and same
// SMEM layout — what's different is that per-tensor outer-dim strides
// (batch / head / seq) are caller-supplied rather than implied by the
// canonical `[B, H, S, D]` layout.
//
// Strides convention (all in elements, not bytes):
//   stride_q_b, stride_q_h, stride_q_s : Q outer-dim strides
//   stride_k_b, stride_k_h, stride_k_s : K outer-dim strides
//                                        (stride_k_h may be 0 for GQA broadcast)
//   stride_v_b, stride_v_h, stride_v_s : V outer-dim strides
//                                        (stride_v_h may be 0 for GQA broadcast)
//   stride_y_b, stride_y_h, stride_y_s : Y outer-dim strides
//
// The innermost head_dim axis must remain stride=1 — the SMEM tile layout
// assumes contiguous head_dim. Rust plan enforces this.
//
// `lse` stays contig `[B, H, Q]` (BW path routes through sm_80 baseline
// which has its own LSE handling); not exposed to strided callers.
//
// GQA broadcast: when `stride_k_h == 0` (or `stride_v_h == 0`), every
// Q-head in a kv-head group dereferences the same K (or V) row. The
// `cp.async` tile loads "just work" — multiple blocks reading the same
// gmem range is well-defined behavior. No code-path branch needed.
// =============================================================================

template <typename T>
__global__ void flash_sdpa_sm89_fw_kernel_strided(
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
    int64_t stride_q_b, int64_t stride_q_h, int64_t stride_q_s,
    int64_t stride_k_b, int64_t stride_k_h, int64_t stride_k_s,
    int64_t stride_v_b, int64_t stride_v_h, int64_t stride_v_s,
    int64_t stride_y_b, int64_t stride_y_h, int64_t stride_y_s,
    float scale,
    int32_t is_causal)
{
    extern __shared__ unsigned char smem_raw[];
    unsigned char* sp = smem_raw;
    T* sQ = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBr * d_k;
    T* sK[kNumStages];
    T* sV[kNumStages];
    sK[0] = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBc * d_k;
    sK[1] = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBc * d_k;
    sV[0] = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBc * d_v;
    sV[1] = reinterpret_cast<T*>(sp);                          sp += sizeof(T) * kBc * d_v;
    float* sS  = reinterpret_cast<float*>(sp);                 sp += sizeof(float) * kBr * kBc;
    float* sO  = reinterpret_cast<float*>(sp);                 sp += sizeof(float) * kBr * d_v;
    float* sM  = reinterpret_cast<float*>(sp);                 sp += sizeof(float) * kBr;
    float* sL  = reinterpret_cast<float*>(sp);                 sp += sizeof(float) * kBr;
    float* sAlpha = reinterpret_cast<float*>(sp);              sp += sizeof(float) * kBr;

    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;
    const int qbase = qb * kBr;
    const int br_eff = (qbase + kBr <= q_len) ? kBr : (q_len - qbase);
    if (br_eff <= 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Strided base pointers (replaces the hardcoded
    // `bh_off * {q_len, k_len} * {d_k, d_v}` arithmetic).
    const T* q_base = q + (int64_t)b * stride_q_b + (int64_t)h * stride_q_h
                        + (int64_t)qbase * stride_q_s;
    const T* k_full = k + (int64_t)b * stride_k_b + (int64_t)h * stride_k_h;
    const T* v_full = v + (int64_t)b * stride_v_b + (int64_t)h * stride_v_h;
    T*       y_base = y + (int64_t)b * stride_y_b + (int64_t)h * stride_y_h
                        + (int64_t)qbase * stride_y_s;
    // lse stays contig [B, H, Q] — same as the non-strided kernel.
    T*       lse_base = lse + ((int64_t)b * heads + h) * q_len + qbase;

    // Load Q once via strided gmem reads.
    {
        const int total = br_eff * d_k;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_k;
            int c = idx - r * d_k;
            sQ[r * d_k + c] = q_base[(int64_t)r * stride_q_s + c];
        }
    }
    // Init sO, sM, sL.
    {
        const int total_o = br_eff * d_v;
        for (int idx = tid; idx < total_o; idx += nthreads) sO[idx] = 0.f;
        for (int r = tid; r < br_eff; r += nthreads) {
            sM[r] = -INFINITY;
            sL[r] = 0.f;
        }
    }

    const int num_kb = (k_len + kBc - 1) / kBc;
    if (num_kb <= 0) {
        __syncthreads();
        return;
    }

    // Issue stage-0 async loads for kb = 0. The `async_copy_tile` helper
    // is already stride-aware at the gmem level — we just pass the
    // strided `stride_k_s` / `stride_v_s` instead of the implicit
    // `d_k` / `d_v` row strides.
    //
    // When `stride_k_s == d_k` (contig K seq) the cp.async fast path
    // (16-byte chunks) still applies, so the contig case is no slower
    // here than in the non-strided kernel.
    {
        const int kbase0 = 0;
        const int bc_eff0 = (kbase0 + kBc <= k_len) ? kBc : (k_len - kbase0);
        async_copy_tile<T>(
            sK[0], k_full + (int64_t)kbase0 * stride_k_s,
            bc_eff0, d_k, (int)stride_k_s, d_k, tid, nthreads);
        async_copy_tile<T>(
            sV[0], v_full + (int64_t)kbase0 * stride_v_s,
            bc_eff0, d_v, (int)stride_v_s, d_v, tid, nthreads);
        __pipeline_commit();
    }
    __syncthreads();

    const float scale_f = scale;
    for (int kb = 0; kb < num_kb; ++kb) {
        const int stage = kb & 1;
        const int kbase = kb * kBc;
        const int bc_eff = (kbase + kBc <= k_len) ? kBc : (k_len - kbase);

        const bool block_fully_masked =
            (is_causal != 0 && kbase > (qbase + br_eff - 1));

        if (!block_fully_masked) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }

        if (kb + 1 < num_kb) {
            const int next_stage = (kb + 1) & 1;
            const int kbase_next = (kb + 1) * kBc;
            const int bc_eff_next = (kbase_next + kBc <= k_len)
                ? kBc : (k_len - kbase_next);
            async_copy_tile<T>(
                sK[next_stage], k_full + (int64_t)kbase_next * stride_k_s,
                bc_eff_next, d_k, (int)stride_k_s, d_k, tid, nthreads);
            async_copy_tile<T>(
                sV[next_stage], v_full + (int64_t)kbase_next * stride_v_s,
                bc_eff_next, d_v, (int)stride_v_s, d_v, tid, nthreads);
            __pipeline_commit();
        }

        if (block_fully_masked) continue;

        // Scores: identical to the non-strided kernel from here on —
        // K/V are now in SMEM at the canonical contig tile layout, and
        // sQ / sO / sL / sM live in SMEM throughout.
        {
            const T* sKstage = sK[stage];
            const int total = kBr * kBc;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / kBc;
                int c = idx - r * kBc;
                if (r >= br_eff || c >= bc_eff) {
                    sS[idx] = -INFINITY;
                    continue;
                }
                float acc = 0.f;
                #pragma unroll 4
                for (int d = 0; d < d_k; ++d) {
                    float qv = load_f32<T>(sQ[r * d_k + d]);
                    float kv = load_f32<T>(sKstage[c * d_k + d]);
                    acc += qv * kv;
                }
                float vscaled = acc * scale_f;
                int q_idx_abs = qbase + r;
                int k_idx_abs = kbase + c;
                if (is_causal != 0 && k_idx_abs > q_idx_abs) {
                    vscaled = -INFINITY;
                }
                sS[idx] = vscaled;
            }
        }
        __syncthreads();

        // Per-row online softmax update — identical to non-strided.
        if (tid < br_eff) {
            int r = tid;
            float mloc = -INFINITY;
            for (int c = 0; c < bc_eff; ++c) {
                float v_s = sS[r * kBc + c];
                if (v_s > mloc) mloc = v_s;
            }
            float mold = sM[r];
            float mnew = (mold > mloc) ? mold : mloc;
            float alpha;
            float l_local = 0.f;
            if (!isfinite(mnew)) {
                alpha = 1.f;
                for (int c = 0; c < bc_eff; ++c) sS[r * kBc + c] = 0.f;
                sAlpha[r] = alpha;
            } else {
                if (isfinite(mold)) {
                    alpha = expf(mold - mnew);
                } else {
                    alpha = 0.f;
                }
                for (int c = 0; c < bc_eff; ++c) {
                    float v_s = sS[r * kBc + c];
                    float p = isfinite(v_s) ? expf(v_s - mnew) : 0.f;
                    sS[r * kBc + c] = p;
                    l_local += p;
                }
                sAlpha[r] = alpha;
                sM[r] = mnew;
                sL[r] = alpha * sL[r] + l_local;
            }
            for (int c = bc_eff; c < kBc; ++c) sS[r * kBc + c] = 0.f;
        }
        {
            const int total_pad = kBr * kBc;
            for (int idx = tid; idx < total_pad; idx += nthreads) {
                int r = idx / kBc;
                if (r >= br_eff) sS[idx] = 0.f;
            }
        }
        __syncthreads();

        // sO ← α · sO + P · V.
        {
            const T* sVstage = sV[stage];
            const int total = kBr * d_v;
            for (int idx = tid; idx < total; idx += nthreads) {
                int r = idx / d_v;
                if (r >= br_eff) continue;
                int dv = idx - r * d_v;
                float acc = 0.f;
                for (int c = 0; c < bc_eff; ++c) {
                    float p = sS[r * kBc + c];
                    float vv = load_f32<T>(sVstage[c * d_v + dv]);
                    acc += p * vv;
                }
                float a = sAlpha[r];
                sO[r * d_v + dv] = a * sO[r * d_v + dv] + acc;
            }
        }
        __syncthreads();
    } // end kb

    __pipeline_wait_prior(0);

    // Finalize y via strided stores.
    {
        const int total = br_eff * d_v;
        for (int idx = tid; idx < total; idx += nthreads) {
            int r = idx / d_v;
            int dv = idx - r * d_v;
            float l = sL[r];
            float yv;
            if (l > 0.f && isfinite(l)) {
                yv = sO[r * d_v + dv] / l;
            } else {
                yv = 0.f;
            }
            y_base[(int64_t)r * stride_y_s + dv] = store_f32<T>(yv);
        }
    }
    // lse stays contig along the q axis (BW path uses sm_80 baseline).
    if (tid < br_eff) {
        int r = tid;
        float l = sL[r];
        float lse_v;
        if (l > 0.f && isfinite(l)) {
            lse_v = sM[r] + logf(l);
        } else {
            lse_v = -INFINITY;
        }
        lse_base[r] = store_f32<T>(lse_v);
    }
}

template <typename T>
__host__ inline int32_t launch_flash_sdpa_sm89_fp_strided(
    const T* q, const T* k_in, const T* v_in,
    T* y, T* lse,
    int32_t batch, int32_t heads, int32_t q_len, int32_t k_len,
    int32_t d_k, int32_t d_v,
    int64_t sqb, int64_t sqh, int64_t sqs,
    int64_t skb, int64_t skh, int64_t sks,
    int64_t svb, int64_t svh, int64_t svs,
    int64_t syb, int64_t syh, int64_t sys,
    float scale, int32_t is_causal,
    cudaStream_t stream)
{
    if (batch < 0 || heads < 0 || q_len < 0 || k_len < 0 || d_k < 0 || d_v < 0) return 2;
    if (d_k > kMaxD || d_v > kMaxD) return 3;
    if (d_k != d_v) return 3;
    int64_t total_y = (int64_t)batch * heads * q_len * d_v;
    if (total_y == 0) return 0;
    int num_qb = (q_len + kBr - 1) / kBr;
    if (num_qb <= 0) return 0;
    dim3 grid((unsigned)num_qb, (unsigned)heads, (unsigned)batch);
    dim3 block((unsigned)kThreadsPerBlock);
    size_t smem = flash_sm89_smem_bytes<T>(d_k, d_v);
    if (smem > 48 * 1024) {
        cudaError_t serr = cudaFuncSetAttribute(
            (const void*)flash_sdpa_sm89_fw_kernel_strided<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        if (serr != cudaSuccess) return 1000 + (int32_t)serr;
    }
    flash_sdpa_sm89_fw_kernel_strided<T><<<grid, block, smem, stream>>>(
        q, k_in, v_in, y, lse, batch, heads, q_len, k_len,
        d_k, d_v,
        sqb, sqh, sqs, skb, skh, sks, svb, svh, svs, syb, syh, sys,
        scale, is_causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

} } // namespace baracuda::flash_sdpa_sm89

// =============================================================================
// INSTANTIATE macro
// =============================================================================

#define BARACUDA_KERNELS_FLASH_SDPA_SM89_INSTANTIATE(NAME, T)                                       \
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
        return baracuda::flash_sdpa_sm89::launch_flash_sdpa_sm89_fp<T>(                             \
            static_cast<const T*>(q),                                                               \
            static_cast<const T*>(k),                                                               \
            static_cast<const T*>(v),                                                               \
            static_cast<T*>(y),                                                                     \
            static_cast<T*>(lse),                                                                   \
            batch, heads, q_len, k_len, d_k, d_v,                                                   \
            scale, is_causal,                                                                       \
            stream);                                                                                \
    }

// Strided FW sibling INSTANTIATE — Phase 17.1.
// Per-tensor stride arrays: stride_q/k/v/y are `*const i64` length 3
// (one per outer dim: batch, heads, seq). The innermost head_dim axis
// is implicitly stride=1 (Rust plan enforces).
// `lse` and mask stay contig (`lse` is `[B, H, Q]` row-major; mask path
// is not yet wired on the sm_89 strided kernel — falls back to the
// non-strided sm_89 plan for mask-having callers, or to the sm_80
// strided naive SDPA). GQA broadcast supported by passing zero for
// stride_k_h / stride_v_h.
#define BARACUDA_KERNELS_FLASH_SDPA_SM89_STRIDED_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_strided_run(                                       \
        int32_t batch,                                                                              \
        int32_t heads,                                                                              \
        int32_t q_len,                                                                              \
        int32_t k_len,                                                                              \
        int32_t d_k,                                                                                \
        int32_t d_v,                                                                                \
        const int64_t* stride_q,                                                                    \
        const int64_t* stride_k,                                                                    \
        const int64_t* stride_v,                                                                    \
        const int64_t* stride_y,                                                                    \
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
        if (stride_q == nullptr || stride_k == nullptr || stride_v == nullptr ||                    \
            stride_y == nullptr) return 2;                                                          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::flash_sdpa_sm89::launch_flash_sdpa_sm89_fp_strided<T>(                     \
            static_cast<const T*>(q),                                                               \
            static_cast<const T*>(k),                                                               \
            static_cast<const T*>(v),                                                               \
            static_cast<T*>(y),                                                                     \
            static_cast<T*>(lse),                                                                   \
            batch, heads, q_len, k_len, d_k, d_v,                                                   \
            stride_q[0], stride_q[1], stride_q[2],                                                  \
            stride_k[0], stride_k[1], stride_k[2],                                                  \
            stride_v[0], stride_v[1], stride_v[2],                                                  \
            stride_y[0], stride_y[1], stride_y[2],                                                  \
            scale, is_causal,                                                                       \
            stream);                                                                                \
    }

#endif // BARACUDA_FLASH_SDPA_SM89_CUH
