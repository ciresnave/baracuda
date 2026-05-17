// baracuda_batched_ormqr_wy.cuh
//
// WY-blocked batched-`ormqr` (Milestone 6.17). Companion to
// `baracuda_batched_ormqr.cuh` (Milestone 6.14, reflector-by-reflector
// GEMV-rates path). This header provides:
//
//  1. A T-build kernel. For each block of `nb` consecutive Householder
//     reflectors emitted by `cublas{S,D}geqrfBatched`, builds the upper-
//     triangular block-reflector matrix `T [nb, nb]` per LAPACK's
//     DLARFT recipe so that
//         H_0 · H_1 · ... · H_{nb-1} = I - V · T · V^T,
//     where `V = [v_0 | v_1 | ... | v_{nb-1}]` is the strict-lower
//     triangle of packed-A columns with an implicit-1 at each diagonal.
//
//  2. A V-extraction kernel. cuBLAS GEMM cannot consume an "implicit-1"
//     packed-A matrix, so we materialize a dense `V [B, M, nb]` per block
//     into caller-provided workspace: set 0 above the diagonal, 1 on the
//     diagonal, and copy the packed-A strict lower below.
//
// Application of each block reflector then reduces to three cuBLAS
// strided-batched GEMMs (executed at the safe-plan layer):
//     W := V^T · C       (nb × N)
//     W := T · W         (nb × N)
//     C := C - V · W     (M × N)
//
// Scope (trailblazer): Side = Left, op ∈ {N, T}, dtype ∈ {f32, f64}.
// Right + complex variants are reserved.
//
// Status codes: 0 success, 2 invalid problem, 5 launch failure.

#ifndef BARACUDA_BATCHED_ORMQR_WY_CUH
#define BARACUDA_BATCHED_ORMQR_WY_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace linalg {

// WY block size — number of reflectors fused per block-reflector. 32 is
// the sweet spot for sm_80+ (one warp worth of reflectors per block;
// keeps T at 32×32 which fits comfortably in shared memory for both
// f32 and f64).
constexpr int kWyNb = 32;

// =============================================================================
// T-build kernel — one CUDA block per (batch_slot, block_index_k).
//
// Inputs:
//   A_packed [B, M, K] column-major — cublas{S,D}geqrfBatched output.
//   tau      [B, K]              — Householder scalars.
//
// Output:
//   T [B, num_blocks, nb, nb]    — upper-triangular per-block reflector
//                                  matrices, contiguous in column-major.
//
// `block_start = blockIdx.y * nb`; the block reflector groups reflectors
// `[block_start, min(block_start + nb, K))`. If the last block is partial
// (`K % nb != 0`), the trailing rows/cols of T are written as zero so
// the GEMM at the apply step stays well-defined (the corresponding
// columns of V are also zero — extract_v zeros them out).
//
// Algorithm — LAPACK DLARFT (forward direction):
//     T[0, 0] = -tau_{block_start}
//     For k = 1 .. nb-1:
//         t[j] = Σ_{r=block_start+k}^{M-1} V[r, j] · V[r, k]      (j < k)
//                where V[r, j] = A_packed[r, block_start + j]
//         T[0..k, k] = T[0..k, 0..k] · (-tau_{block_start+k} · t[0..k])
//         T[k, k] = -tau_{block_start+k}
//
// The k-loop is sequential within the block (column k depends on
// columns 0..k-1). Threads within the block cooperate on the per-column
// dot-products + the small upper-triangular matvec.
// =============================================================================

template <typename T, typename A>
__global__ void batched_ormqr_wy_build_t_kernel(
    const T* __restrict__ A_packed,    // [B, M, K] column-major
    const T* __restrict__ tau,          // [B, K]
    T* __restrict__ T_out,              // [B, num_blocks, nb, nb] col-major upper
    int M, int K, int nb)
{
    const int b = blockIdx.x;
    const int blk = blockIdx.y;
    const int block_start = blk * nb;
    if (block_start >= K) return;
    const int block_end = (block_start + nb < K) ? (block_start + nb) : K;
    const int block_k   = block_end - block_start;  // <= nb (last block may be short)

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const T* Ab    = A_packed + (int64_t)b * (int64_t)M * (int64_t)K;
    const T* taub  = tau      + (int64_t)b * (int64_t)K;
    T*       Tb    = T_out    + ((int64_t)b * (int64_t)gridDim.y + (int64_t)blk)
                                * (int64_t)nb * (int64_t)nb;

    // Shared memory layout:
    //   t_smem [nb]         — current column of intermediate dot-products
    //   red_smem[block_size] — tree-reduction scratch
    extern __shared__ unsigned char smem_raw[];
    A* t_smem   = reinterpret_cast<A*>(smem_raw);
    A* red_smem = t_smem + nb;

    // Zero the whole T_block so the partial-last-block case leaves a
    // well-defined (zero-padded) matrix.
    for (int idx = tid; idx < nb * nb; idx += block_size) {
        Tb[idx] = (T)0;
    }
    __syncthreads();

    if (block_k == 0) return;

    // T[0, 0] = +tau_{block_start}. Per LAPACK DLARFT: for a single
    // reflector, Q = I - V·T·V^T = I - τ_0·v_0·v_0^T = H_0 requires
    // T[0,0] = +τ_0 (NOT -τ_0). The off-diagonal entries get their
    // negative sign from the `-τ_k` scaling in Step 2 below.
    if (tid == 0) {
        Tb[0] = (T)((A)taub[block_start]);
    }
    __syncthreads();

    for (int k = 1; k < block_k; ++k) {
        // ----- Step 1: t[j] = Σ_{r=block_start+k}^{M-1} V[r, j] · V[r, k]
        //               for j ∈ [0, k).
        // V[r, j] = A_packed[r, block_start + j] (the implicit-1 at the
        // diagonal v_j[block_start+j] contributes 0 here because for
        // j < k we always have block_start+k > block_start+j so r is
        // strictly below v_j's diagonal — only the explicit packed-A
        // entries matter).
        const T* v_k_col = Ab + (int64_t)(block_start + k) * (int64_t)M;
        const int diag_k = block_start + k;
        for (int j = 0; j < k; ++j) {
            const T* v_j_col = Ab + (int64_t)(block_start + j) * (int64_t)M;
            A partial = (A)0;
            for (int r = diag_k + tid; r < M; r += block_size) {
                A vj = (A)v_j_col[r];
                // v_k[diag_k] is the implicit Householder 1; the packed-A
                // location at (diag_k, diag_k) actually holds R[k,k]. For
                // any other row, v_k[r] is the explicit strict-lower
                // packed value.
                A vk = (r == diag_k) ? (A)1 : (A)v_k_col[r];
                partial += vj * vk;
            }
            // tree reduction
            red_smem[tid] = partial;
            __syncthreads();
            for (int stride = block_size / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    red_smem[tid] = red_smem[tid] + red_smem[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                t_smem[j] = red_smem[0];
            }
            __syncthreads();
        }

        // ----- Step 2: T[0..k, k] = T[0..k, 0..k] · (-tau_k · t[0..k])
        // T is column-major upper-triangular in storage; T[i, j] at
        // offset j*nb + i. Only entries with i ≤ j are valid; below the
        // diagonal is zero from the initial fill above.
        A neg_tau_k = -(A)taub[block_start + k];
        for (int i = tid; i < k; i += block_size) {
            A sum = (A)0;
            // T[i, j] for j ∈ [i, k): upper-triangular, so j starts at i.
            for (int j = i; j < k; ++j) {
                A t_ij = (A)Tb[(int64_t)j * (int64_t)nb + (int64_t)i];
                sum += t_ij * (neg_tau_k * t_smem[j]);
            }
            Tb[(int64_t)k * (int64_t)nb + (int64_t)i] = (T)sum;
        }
        // T[k, k] = +tau_{block_start+k}. See LAPACK DLARFT note above.
        if (tid == 0) {
            Tb[(int64_t)k * (int64_t)nb + (int64_t)k] = (T)(-neg_tau_k);
        }
        __syncthreads();
    }
}

// =============================================================================
// V-extraction kernel — materialize the dense V [B, M, nb] for one
// block of reflectors into a contiguous scratch buffer suitable for
// cuBLAS GEMM consumption.
//
// V[b, r, j] for block `blk` (block_start = blk * nb):
//   r < block_start:                 0
//   r == block_start + j:            1   (implicit Householder unit)
//   r > block_start + j, r < M:      A_packed[b, r, block_start + j]
//   r >= M:                          n/a (clamped by M)
// For j >= block_k (partial last block):
//   V[b, *, j] = 0   so the GEMM contribution is zero.
//
// One CUDA block per (batch_slot, output_column j). Threads stride over
// rows.
// =============================================================================

template <typename T>
__global__ void batched_ormqr_wy_extract_v_kernel(
    const T* __restrict__ A_packed,    // [B, M, K] column-major
    T* __restrict__ V_out,              // [B, M, nb] column-major for this block
    int M, int K, int nb, int block_start, int block_k)
{
    const int b = blockIdx.x;
    const int j = blockIdx.y;
    if (j >= nb) return;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const T* Ab = A_packed + (int64_t)b * (int64_t)M * (int64_t)K;
    T*       Vb = V_out    + (int64_t)b * (int64_t)M * (int64_t)nb;

    if (j >= block_k) {
        // Zero-pad partial-last-block columns.
        for (int r = tid; r < M; r += block_size) {
            Vb[(int64_t)j * (int64_t)M + (int64_t)r] = (T)0;
        }
        return;
    }

    const int diag = block_start + j;
    for (int r = tid; r < M; r += block_size) {
        T val;
        if (r < diag) {
            val = (T)0;
        } else if (r == diag) {
            val = (T)1;
        } else {
            // r > diag — strict-lower of column (block_start+j) of A_packed.
            val = Ab[(int64_t)diag * (int64_t)M + (int64_t)r];
        }
        Vb[(int64_t)j * (int64_t)M + (int64_t)r] = val;
    }
}

// =============================================================================
// Host launchers.
// =============================================================================

template <typename T, typename A>
__host__ inline int32_t launch_batched_ormqr_wy_build_t(
    const T* A_packed,
    const T* tau,
    T* T_out,
    int batch, int M, int K, int nb,
    int num_blocks,
    cudaStream_t stream)
{
    if (batch < 0 || M < 0 || K < 0 || nb <= 0 || num_blocks < 0) return 2;
    if (batch == 0 || M == 0 || K == 0 || num_blocks == 0) return 0;
    if (A_packed == nullptr || tau == nullptr || T_out == nullptr) return 2;

    // 128 threads — comfortable for the nb=32 inner loop; round to a
    // power of two so the tree reduction halves cleanly.
    int threads = 128;
    int t = 1;
    while (t * 2 <= threads) t *= 2;
    threads = t;

    size_t smem_bytes = (size_t)(nb + threads) * sizeof(A);

    dim3 grid((unsigned)batch, (unsigned)num_blocks, 1);
    dim3 block((unsigned)threads, 1, 1);

    batched_ormqr_wy_build_t_kernel<T, A><<<grid, block, smem_bytes, stream>>>(
        A_packed, tau, T_out, M, K, nb);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 5;
    return 0;
}

template <typename T>
__host__ inline int32_t launch_batched_ormqr_wy_extract_v(
    const T* A_packed,
    T* V_out,
    int batch, int M, int K, int nb, int block_start, int block_k,
    cudaStream_t stream)
{
    if (batch < 0 || M < 0 || K < 0 || nb <= 0) return 2;
    if (block_start < 0 || block_k < 0 || block_k > nb) return 2;
    if (batch == 0 || M == 0 || nb == 0) return 0;
    if (A_packed == nullptr || V_out == nullptr) return 2;

    int threads = 128;
    dim3 grid((unsigned)batch, (unsigned)nb, 1);
    dim3 block((unsigned)threads, 1, 1);
    batched_ormqr_wy_extract_v_kernel<T><<<grid, block, 0, stream>>>(
        A_packed, V_out, M, K, nb, block_start, block_k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 5;
    return 0;
}

} } // namespace baracuda::linalg

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launcher symbols per dtype.
// =============================================================================

#define BARACUDA_KERNELS_BATCHED_ORMQR_WY_BUILD_T_INSTANTIATE(NAME, T, ACC)                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t batch,                                                                          \
        int32_t M,                                                                              \
        int32_t K,                                                                              \
        int32_t nb,                                                                             \
        int32_t num_blocks,                                                                     \
        const void* a_packed,                                                                   \
        const void* tau,                                                                        \
        void* t_out,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::linalg::launch_batched_ormqr_wy_build_t<T, ACC>(                       \
            static_cast<const T*>(a_packed),                                                    \
            static_cast<const T*>(tau),                                                         \
            static_cast<T*>(t_out),                                                              \
            batch, M, K, nb, num_blocks, stream);                                                \
    }

#define BARACUDA_KERNELS_BATCHED_ORMQR_WY_EXTRACT_V_INSTANTIATE(NAME, T)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t batch,                                                                          \
        int32_t M,                                                                              \
        int32_t K,                                                                              \
        int32_t nb,                                                                             \
        int32_t block_start,                                                                    \
        int32_t block_k,                                                                        \
        const void* a_packed,                                                                   \
        void* v_out,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                            \
        return baracuda::linalg::launch_batched_ormqr_wy_extract_v<T>(                          \
            static_cast<const T*>(a_packed),                                                    \
            static_cast<T*>(v_out),                                                              \
            batch, M, K, nb, block_start, block_k, stream);                                      \
    }

#endif // BARACUDA_BATCHED_ORMQR_WY_CUH
