// baracuda_batched_ormqr.cuh
//
// Bespoke batched-`ormqr` kernel for the linalg family (Milestone 6.14).
//
// Algorithm — apply Householder-encoded Q (or Q^T) from the LEFT to a
// stack of independent C matrices, one batch slot per CUDA block. The
// packed input matches cuBLAS / cuSOLVER `geqrf` output:
//
//   A_packed[b]  : column-major [M, K] — strict lower triangle holds the
//                  Householder reflectors `v_i`, with an *implicit* 1 at
//                  position `i` and zeros above.
//   tau[b]       : length K — Householder scalars.
//
// For Side = Left, Op = N ("apply Q"):
//   Q = H_{K-1} · ... · H_1 · H_0 (each H_i = I - tau_i · v_i · v_i^T).
//   To compute  C := Q · C, apply reflectors from innermost first, i.e.
//   iterate i = K-1, K-2, ..., 0 — each step:
//       u_j   = tau_i · ( Σ_{r=i}^{M-1} v_i[r] · C[r, j] )    for every j
//       C[r, j] -= v_i[r] · u_j                                for r ≥ i
//
// For Side = Left, Op = T ("apply Q^T"):
//   Q^T = H_0 · H_1 · ... · H_{K-1} (each H_i is symmetric).
//   Same step body — only the iteration order flips (i = 0, 1, ..., K-1).
//
// Kernel layout:
//   gridDim.x  = batch_size
//   blockDim.x = KBLOCK threads (256 by default; the threads cooperate on
//                each reflector — reducing across rows of M and updating
//                across columns of N).
//
// Per-reflector dynamic shared memory:
//   u: length N — the projection coefficient `tau_i · (v^T C)` for every
//      column of C. Computed by a block-stride reduction over rows of
//      `v` × `C`, then broadcast back to every thread for the outer-
//      product update.
//
// Scope: Side = Left, Op ∈ {N, T}, dtypes ∈ {f32, f64}. Right-side and
// complex variants are deferred (the FFI surface leaves room). The
// kernel is correctness-first — O(K·M·N) work per slot, GEMV-rates not
// GEMM-rates. WY blocking is a future-milestone optimization.
//
// Status codes follow the rest of the kernel library:
//   0 success, 2 invalid problem, 5 internal launch failure.

#ifndef BARACUDA_BATCHED_ORMQR_CUH
#define BARACUDA_BATCHED_ORMQR_CUH

#include <cstddef>
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>

namespace baracuda { namespace linalg {

// Op tag — matches cuBLAS CUBLAS_OP_{N,T,C} (0, 1, 2). The `C` op tag
// is the conjugate-transpose variant, only valid for complex dtypes.
constexpr int32_t BARACUDA_ORMQR_OP_N = 0;
constexpr int32_t BARACUDA_ORMQR_OP_T = 1;
constexpr int32_t BARACUDA_ORMQR_OP_C = 2;

// Side tag — matches cuBLAS CUBLAS_SIDE_{LEFT, RIGHT}.
constexpr int32_t BARACUDA_ORMQR_SIDE_LEFT  = 0;
constexpr int32_t BARACUDA_ORMQR_SIDE_RIGHT = 1;

// =============================================================================
// Element-arithmetic helpers — templated on T so a single kernel body works
// for real (`float` / `double`) and complex (`cuFloatComplex` /
// `cuDoubleComplex`) storage. For real T, `conj_T` is the identity; for
// complex T it returns the conjugate. `mul_T` / `add_T` / `sub_T` /
// `zero_T` route through the standard cuComplex intrinsics for complex T
// and plain operators for real T.
// =============================================================================

template <typename T>
__device__ __forceinline__ T mul_T(T a, T b) { return a * b; }

template <typename T>
__device__ __forceinline__ T add_T(T a, T b) { return a + b; }

template <typename T>
__device__ __forceinline__ T sub_T(T a, T b) { return a - b; }

template <typename T>
__device__ __forceinline__ T conj_T(T a) { return a; }

template <typename T>
__device__ __forceinline__ T zero_T() { return (T)0; }

template <typename T>
__device__ __forceinline__ T one_T() { return (T)1; }

// ----- cuFloatComplex specializations ----------------------------------------

template <>
__device__ __forceinline__ cuFloatComplex mul_T<cuFloatComplex>(cuFloatComplex a, cuFloatComplex b) {
    return cuCmulf(a, b);
}
template <>
__device__ __forceinline__ cuFloatComplex add_T<cuFloatComplex>(cuFloatComplex a, cuFloatComplex b) {
    return cuCaddf(a, b);
}
template <>
__device__ __forceinline__ cuFloatComplex sub_T<cuFloatComplex>(cuFloatComplex a, cuFloatComplex b) {
    return cuCsubf(a, b);
}
template <>
__device__ __forceinline__ cuFloatComplex conj_T<cuFloatComplex>(cuFloatComplex a) {
    return cuConjf(a);
}
template <>
__device__ __forceinline__ cuFloatComplex zero_T<cuFloatComplex>() {
    return make_cuFloatComplex(0.0f, 0.0f);
}
template <>
__device__ __forceinline__ cuFloatComplex one_T<cuFloatComplex>() {
    return make_cuFloatComplex(1.0f, 0.0f);
}

// ----- cuDoubleComplex specializations ---------------------------------------

template <>
__device__ __forceinline__ cuDoubleComplex mul_T<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
}
template <>
__device__ __forceinline__ cuDoubleComplex add_T<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCadd(a, b);
}
template <>
__device__ __forceinline__ cuDoubleComplex sub_T<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCsub(a, b);
}
template <>
__device__ __forceinline__ cuDoubleComplex conj_T<cuDoubleComplex>(cuDoubleComplex a) {
    return cuConj(a);
}
template <>
__device__ __forceinline__ cuDoubleComplex zero_T<cuDoubleComplex>() {
    return make_cuDoubleComplex(0.0, 0.0);
}
template <>
__device__ __forceinline__ cuDoubleComplex one_T<cuDoubleComplex>() {
    return make_cuDoubleComplex(1.0, 0.0);
}

// Thread-block size — fixed (one block per batch slot, threads divide up
// the columns of C for the per-reflector projection / outer-product
// update). 256 is comfortable for small/medium N; for very small N the
// effective parallelism is bounded by N anyway.
constexpr int kOrmqrBlock = 256;

// =============================================================================
// Block-stride sum reduction in shared memory. `tid` is threadIdx.x and
// `block_size == blockDim.x`. Caller passes a length-`block_size`
// scratch array in shared memory. Returns the reduced value in thread 0
// (other threads get an indeterminate value). Uses a tree reduction in
// shared memory — correctness-first, not perf-optimal. NOTE: caller must
// `__syncthreads()` before the next use of the scratch array.
// =============================================================================

template <typename A>
__device__ inline A block_sum_reduce(A val, A* scratch, int tid, int block_size) {
    scratch[tid] = val;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = add_T<A>(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

// =============================================================================
// Apply ONE Householder reflector i to every column of C for this batch
// slot — LEFT side: `C := H_i · C` where `H_i = I - tau_i · v_i · v_i^H`
// (the `^H` reduces to `^T` for real T because `conj_T` is the identity).
//
//   - v_col      : pointer to A_packed[b, :, i] (column i of the packed
//                  input). The implicit 1 at row i is handled by the
//                  read accessor.
//   - tau_i      : Householder scalar for reflector i. For op = C
//                  (conjugate transpose) the caller passes `conj(tau_i)`.
//   - C          : pointer to C[b] (column-major, ldc = M).
//   - i          : index of the reflector (0 ≤ i < K).
//   - M, N       : dimensions of C.
//   - u_smem     : length-N shared-memory scratch (`u[j]` per column).
//   - red_smem   : length-blockDim.x shared-memory scratch for the
//                  per-column projection reductions.
//
// Conjugation placement (Left): `v^H · C` on the reduction side, plain
// `v` on the rank-1 update side. For real T `conj_T` is a no-op so the
// existing real path is unchanged.
// =============================================================================

template <typename T, typename A>
__device__ inline void apply_reflector(
    const T* __restrict__ v_col,      // length M (only rows ≥ i used)
    A tau_i,
    T* __restrict__ C,                // [M, N] column-major
    int i,
    int M,
    int N,
    A* u_smem,
    A* red_smem)
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Phase 1 — compute u[j] = tau_i · (Σ_{r=i}^{M-1} conj(v[r]) · C[r, j])
    // for every column j ∈ [0, N). One column at a time; threads in
    // the block split the rows.
    for (int j = 0; j < N; ++j) {
        A partial = zero_T<A>();
        for (int r = i + tid; r < M; r += block_size) {
            // v[i] is implicitly 1; v[r > i] is the strict-lower-tri
            // entry from the packed A column.
            A vr = (r == i) ? one_T<A>() : (A)v_col[r];
            A cr = (A)C[(int64_t)j * (int64_t)M + (int64_t)r];
            partial = add_T<A>(partial, mul_T<A>(conj_T<A>(vr), cr));
        }
        A sum = block_sum_reduce<A>(partial, red_smem, tid, block_size);
        if (tid == 0) {
            u_smem[j] = mul_T<A>(tau_i, sum);
        }
        __syncthreads();
    }

    // Phase 2 — outer-product update:
    //   C[r, j] -= v[r] · u[j]   for r ∈ [i, M), j ∈ [0, N).
    // Threads divide up the (r, j) pairs by column (every thread strides
    // across r within a column) so reads of u[j] are uniform.
    for (int j = 0; j < N; ++j) {
        A uj = u_smem[j];
        for (int r = i + tid; r < M; r += block_size) {
            A vr = (r == i) ? one_T<A>() : (A)v_col[r];
            int64_t off = (int64_t)j * (int64_t)M + (int64_t)r;
            A cur = (A)C[off];
            C[off] = (T)sub_T<A>(cur, mul_T<A>(vr, uj));
        }
    }
    __syncthreads();
}

// =============================================================================
// Apply ONE Householder reflector i to every row of C for this batch slot
// — RIGHT side: `C := C · H_i = C - tau_i · (C · v_i) · v_i^H`.
//
// Axes are swapped relative to the left-side variant: the reduction is
// across the column dimension (per row of C, sum over j), and the update
// is rank-1 across rows × columns of v.
//
// Conjugation placement (Right): plain `v` on the reduction side, `v^H`
// on the rank-1 update side. (Opposite of Left.) For real T this is a
// no-op via `conj_T`.
//
//   - v_col      : pointer to A_packed[b, :, i] (column i of the packed
//                  input). Shape semantics for Right: `v` lives in the
//                  N-axis (because Q is now N × N), so `v_col` is length
//                  N and only entries [i, N) are read.
//   - tau_i      : Householder scalar (caller passes `conj(tau_i)` for
//                  op = C).
//   - C          : pointer to C[b] (column-major, ldc = M).
//   - i          : reflector index (0 ≤ i < K = N).
//   - M, N       : dimensions of C.
//   - u_smem     : length-M shared-memory scratch (`u[r]` per row).
//   - red_smem   : length-blockDim.x shared-memory scratch.
// =============================================================================

template <typename T, typename A>
__device__ inline void apply_reflector_right(
    const T* __restrict__ v_col,      // length N (only entries ≥ i used)
    A tau_i,
    T* __restrict__ C,                // [M, N] column-major
    int i,
    int M,
    int N,
    A* u_smem,
    A* red_smem)
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Phase 1 — compute u[r] = tau_i · (Σ_{j=i}^{N-1} C[r, j] · v[j])
    // for every row r ∈ [0, M). One row at a time; threads in the block
    // split the columns (j-loop).
    for (int r = 0; r < M; ++r) {
        A partial = zero_T<A>();
        for (int j = i + tid; j < N; j += block_size) {
            A vj = (j == i) ? one_T<A>() : (A)v_col[j];
            A cr = (A)C[(int64_t)j * (int64_t)M + (int64_t)r];
            partial = add_T<A>(partial, mul_T<A>(cr, vj));
        }
        A sum = block_sum_reduce<A>(partial, red_smem, tid, block_size);
        if (tid == 0) {
            u_smem[r] = mul_T<A>(tau_i, sum);
        }
        __syncthreads();
    }

    // Phase 2 — rank-1 update:
    //   C[r, j] -= u[r] · conj(v[j])   for r ∈ [0, M), j ∈ [i, N).
    // Each thread iterates over a strided slice of columns; for each
    // column it touches every row.
    for (int j = i + tid; j < N; j += block_size) {
        A vj = (j == i) ? one_T<A>() : (A)v_col[j];
        A vj_conj = conj_T<A>(vj);
        for (int r = 0; r < M; ++r) {
            int64_t off = (int64_t)j * (int64_t)M + (int64_t)r;
            A cur = (A)C[off];
            C[off] = (T)sub_T<A>(cur, mul_T<A>(u_smem[r], vj_conj));
        }
    }
    __syncthreads();
}

// =============================================================================
// Batched-`ormqr` / `unmqr` kernel. One block per batch slot.
//
// Iteration direction and conjugation of tau depend on (Side, Op) per the
// LAPACK contract:
//
//   | Side  | Op | Iteration       | tau adjustment |
//   |-------|----|-----------------|----------------|
//   | Left  | N  | K-1, K-2, …, 0  | tau_i          |
//   | Left  | T  | 0, 1, …, K-1    | tau_i          |
//   | Left  | C  | 0, 1, …, K-1    | conj(tau_i)    |
//   | Right | N  | 0, 1, …, K-1    | tau_i          |
//   | Right | T  | K-1, K-2, …, 0  | tau_i          |
//   | Right | C  | K-1, K-2, …, 0  | conj(tau_i)    |
//
// For Side = Left, `v_col` lives in the M-axis (Q is M × M, packed in
// `A_packed [B, M, K]`). For Side = Right, the packed input is square
// `[B, N, N]` (K = N) and `v_col` lives in the N-axis. The host launcher
// validates the shape contract.
// =============================================================================

template <typename T, typename A>
__global__ void batched_ormqr_kernel(
    const T* __restrict__ A_packed,    // Left: [B, M, K] col-major (stride M·K);
                                        // Right: [B, N, N] col-major (stride N·N).
    const T* __restrict__ tau,          // [B, K]; per-slot stride = K
    T* __restrict__ C,                  // [B, M, N] column-major; per-slot stride = M·N
    int M, int N, int K,
    int side,                           // 0 = Left, 1 = Right
    int op)                             // 0 = N, 1 = T, 2 = C (conjugate-transpose)
{
    const int b = blockIdx.x;
    const int64_t a_slot_stride = (side == BARACUDA_ORMQR_SIDE_LEFT)
        ? (int64_t)M * (int64_t)K
        : (int64_t)N * (int64_t)N;
    const int v_leading = (side == BARACUDA_ORMQR_SIDE_LEFT) ? M : N;
    const T* Ab    = A_packed + (int64_t)b * a_slot_stride;
    const T* taub  = tau      + (int64_t)b * (int64_t)K;
    T*       Cb    = C        + (int64_t)b * (int64_t)M * (int64_t)N;

    extern __shared__ unsigned char shared_raw[];
    A* u_smem   = reinterpret_cast<A*>(shared_raw);
    // For Left the per-reflector projection vector has length N (one cell per
    // column of C). For Right it has length M (one cell per row of C). The
    // host launcher allocates max(M, N).
    const int u_len = (side == BARACUDA_ORMQR_SIDE_LEFT) ? N : M;
    A* red_smem = u_smem + u_len;  // length blockDim.x

    // Iteration order per the (Side, Op) table above.
    bool forward;
    if (side == BARACUDA_ORMQR_SIDE_LEFT) {
        forward = (op != BARACUDA_ORMQR_OP_N);    // T / C go forward; N goes reverse.
    } else {
        forward = (op == BARACUDA_ORMQR_OP_N);    // N goes forward; T / C go reverse.
    }
    const bool conjugate_tau = (op == BARACUDA_ORMQR_OP_C);

    auto step = [&] (int i) {
        const T* v_col = Ab + (int64_t)i * (int64_t)v_leading;
        A tau_i = (A)taub[i];
        if (conjugate_tau) tau_i = conj_T<A>(tau_i);
        if (side == BARACUDA_ORMQR_SIDE_LEFT) {
            apply_reflector<T, A>(v_col, tau_i, Cb, i, M, N, u_smem, red_smem);
        } else {
            apply_reflector_right<T, A>(v_col, tau_i, Cb, i, M, N, u_smem, red_smem);
        }
    };
    if (forward) {
        for (int i = 0; i < K; ++i) step(i);
    } else {
        for (int i = K - 1; i >= 0; --i) step(i);
    }
}

// =============================================================================
// Host-side launcher. `side` ∈ {0 (Left), 1 (Right)};
// `op` ∈ {0 (N), 1 (T), 2 (C)}.
// =============================================================================

template <typename T, typename A>
__host__ inline int32_t launch_batched_ormqr(
    const T* A_packed,
    const T* tau,
    T* C,
    int batch, int M, int N, int K,
    int side,
    int op,
    cudaStream_t stream)
{
    if (batch < 0 || M < 0 || N < 0 || K < 0) return 2;
    if (side != BARACUDA_ORMQR_SIDE_LEFT && side != BARACUDA_ORMQR_SIDE_RIGHT) return 2;
    if (side == BARACUDA_ORMQR_SIDE_LEFT) {
        if (K > M) return 2;                   // K ≤ M (LAPACK contract: Q is M×M)
    } else {
        if (K > N) return 2;                   // K ≤ N (LAPACK contract: Q is N×N)
    }
    if (batch == 0 || M == 0 || N == 0 || K == 0) return 0;  // nothing to do
    if (A_packed == nullptr || tau == nullptr || C == nullptr) return 2;
    if (op != BARACUDA_ORMQR_OP_N &&
        op != BARACUDA_ORMQR_OP_T &&
        op != BARACUDA_ORMQR_OP_C) return 2;

    int threads = kOrmqrBlock;
    // Round threads down to the next power of two so the tree reduction
    // halves cleanly. (kOrmqrBlock is 256 = 2^8 already, so this is the
    // identity for any block ≥ 256; we keep it explicit for future tuning.)
    int t = 1;
    while (t * 2 <= threads) t *= 2;
    threads = t;

    // Dynamic shared memory: u-vector (length N for Left, M for Right) +
    // reduction scratch[blockDim.x], both as type `A`.
    int u_len = (side == BARACUDA_ORMQR_SIDE_LEFT) ? N : M;
    size_t smem_bytes = (size_t)(u_len + threads) * sizeof(A);

    dim3 grid((unsigned)batch, 1, 1);
    dim3 block((unsigned)threads, 1, 1);

    batched_ormqr_kernel<T, A><<<grid, block, smem_bytes, stream>>>(
        A_packed, tau, C, M, N, K, side, op);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 5;
    return 0;
}

// =============================================================================
// Batched-QR-materialize helpers (Milestone 6.14, Piece 2).
//
// (a) Copy upper triangle of the packed A into a fresh R buffer, zeroing
//     the strict lower triangle. R has shape [B, K, N] where K = min(M, N);
//     A_packed is [B, M, N] column-major.
//
//     One block per (batch_slot, output_column). Threads in the block
//     stride over output rows. Output cell (i, j) gets A_packed[i, j] if
//     i ≤ j else 0.
//
// (b) Stage an identity matrix [B, M, M] into a fresh Q buffer. One block
//     per (batch_slot, output_column). Threads stride over rows. The
//     caller follows this with `BatchedOrmqrPlan` (Left, op=N) to apply
//     Q to the identity in place.
// =============================================================================

template <typename T>
__global__ void batched_qr_materialize_r_kernel(
    const T* __restrict__ A_packed,    // [B, M, N] column-major
    T* __restrict__ R,                  // [B, K, N] column-major; K = min(M, N)
    int M, int N, int K)
{
    int b = blockIdx.x;
    int j = blockIdx.y;                  // column of R
    if (j >= N) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const T* Ab = A_packed + (int64_t)b * (int64_t)M * (int64_t)N;
    T*       Rb = R        + (int64_t)b * (int64_t)K * (int64_t)N;

    for (int i = tid; i < K; i += block_size) {
        T val = (i <= j) ? Ab[(int64_t)j * (int64_t)M + (int64_t)i] : (T)0;
        Rb[(int64_t)j * (int64_t)K + (int64_t)i] = val;
    }
}

template <typename T>
__global__ void batched_qr_materialize_identity_kernel(
    T* __restrict__ Q,                  // [B, M, M] column-major
    int M)
{
    int b = blockIdx.x;
    int j = blockIdx.y;                  // column of Q
    if (j >= M) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    T* Qb = Q + (int64_t)b * (int64_t)M * (int64_t)M;
    for (int i = tid; i < M; i += block_size) {
        T val = (i == j) ? (T)1 : (T)0;
        Qb[(int64_t)j * (int64_t)M + (int64_t)i] = val;
    }
}

template <typename T>
__host__ inline int32_t launch_batched_qr_materialize_r(
    const T* A_packed,
    T* R,
    int batch, int M, int N, int K,
    cudaStream_t stream)
{
    if (batch < 0 || M < 0 || N < 0 || K < 0) return 2;
    if (K > M || K > N) return 2;
    if (batch == 0 || M == 0 || N == 0 || K == 0) return 0;
    if (A_packed == nullptr || R == nullptr) return 2;

    int threads = 128;
    if (threads > K) {
        // Clamp to next power of two ≥ K, ≤ 128. For very small K we
        // still launch the minimum useful block so the kernel doesn't
        // launch with zero threads.
        threads = 32;
    }

    dim3 grid((unsigned)batch, (unsigned)N, 1);
    dim3 block((unsigned)threads, 1, 1);
    batched_qr_materialize_r_kernel<T><<<grid, block, 0, stream>>>(
        A_packed, R, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 5;
    return 0;
}

template <typename T>
__host__ inline int32_t launch_batched_qr_materialize_identity(
    T* Q,
    int batch, int M,
    cudaStream_t stream)
{
    if (batch < 0 || M < 0) return 2;
    if (batch == 0 || M == 0) return 0;
    if (Q == nullptr) return 2;
    int threads = 128;
    dim3 grid((unsigned)batch, (unsigned)M, 1);
    dim3 block((unsigned)threads, 1, 1);
    batched_qr_materialize_identity_kernel<T><<<grid, block, 0, stream>>>(Q, M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 5;
    return 0;
}

} } // namespace baracuda::linalg

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launcher symbols per dtype.
// =============================================================================

#define BARACUDA_KERNELS_BATCHED_ORMQR_INSTANTIATE(NAME, T, ACC)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                            \
        int32_t batch,                                                                            \
        int32_t M,                                                                                \
        int32_t N,                                                                                \
        int32_t K,                                                                                \
        int32_t side,                                                                             \
        int32_t op,                                                                               \
        const void* a_packed,                                                                     \
        const void* tau,                                                                          \
        void* c,                                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::linalg::launch_batched_ormqr<T, ACC>(                                    \
            static_cast<const T*>(a_packed),                                                      \
            static_cast<const T*>(tau),                                                           \
            static_cast<T*>(c),                                                                   \
            batch, M, N, K, side, op, stream);                                                    \
    }                                                                                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                  \
        int32_t batch, int32_t M, int32_t N, int32_t K, int32_t side, int32_t op)                \
    {                                                                                              \
        if (batch < 0 || M < 0 || N < 0 || K < 0) return 2;                                       \
        if (side != 0 && side != 1) return 2;     /* 0 = Left, 1 = Right */                       \
        if (op < 0 || op > 2) return 2;            /* 0 = N, 1 = T, 2 = C */                       \
        if (K > M) return 2;                                                                       \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_BATCHED_QR_MATERIALIZE_R_INSTANTIATE(NAME, T)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                            \
        int32_t batch,                                                                            \
        int32_t M,                                                                                \
        int32_t N,                                                                                \
        int32_t K,                                                                                \
        const void* a_packed,                                                                     \
        void* r,                                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::linalg::launch_batched_qr_materialize_r<T>(                              \
            static_cast<const T*>(a_packed),                                                      \
            static_cast<T*>(r),                                                                   \
            batch, M, N, K, stream);                                                              \
    }                                                                                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                  \
        int32_t batch, int32_t M, int32_t N, int32_t K)                                          \
    {                                                                                              \
        if (batch < 0 || M < 0 || N < 0 || K < 0) return 2;                                       \
        if (K > M || K > N) return 2;                                                              \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_BATCHED_QR_MATERIALIZE_IDENTITY_INSTANTIATE(NAME, T)                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                            \
        int32_t batch,                                                                            \
        int32_t M,                                                                                \
        void* q,                                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::linalg::launch_batched_qr_materialize_identity<T>(                       \
            static_cast<T*>(q), batch, M, stream);                                                \
    }                                                                                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                  \
        int32_t batch, int32_t M)                                                                 \
    {                                                                                              \
        if (batch < 0 || M < 0) return 2;                                                          \
        return 0;                                                                                  \
    }

#endif // BARACUDA_BATCHED_ORMQR_CUH
