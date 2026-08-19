// SPDX-FileCopyrightText: 2026 Eric Evans  (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemv_dense.cu — capture-safe dense m=1 GEMV (f32 / f16 / bf16).
//
// A plain custom GEMV with NO vendor internal state, so a captured CUDA
// graph replays byte-identical by construction. It exists to unblock
// Fuel's CUDA-graph `CapturedRun`: the cuBLAS-backed `gemm_dense`
// facade is faster and batched, but cuBLAS is not capture-safe (it may
// select a different internal implementation between launches). On the
// capture path only, Fuel routes the decode-step matmuls (all m == 1)
// through this kernel instead.
//
// Semantics (RRR, row-major, per batch slot `g ∈ [0, batch)`):
//   D[g, 0, j] = alpha * Σ_{kk=0..k} A[g, 0, kk] * B[g, kk, j]
//              + beta * D[g, 0, j]        for j ∈ [0, n)
//   A: row vector length k (lda unused at m == 1 but honored),
//   B: [k, n] row-major, B[kk, j] = B[kk * ldb + j],
//   D: row vector length n (ldd unused at m == 1 but honored).
// Per-slot base offsets are `ptr + g * stride`; `stride_b == 0` makes
// every slot read the same B (the GQA rhs broadcast). Accumulation is
// f32 for all three dtypes — f32 is true IEEE (NOT TF32); f16 / bf16
// widen to f32 on load and narrow once on store.
//
// Design: one thread owns one output column j, with a serial K-loop
// (thread-per-column, not warp-per-column). Because each column is
// computed by exactly one thread and the reduction order is fixed, the
// result is independent of block / grid shape and every launch is
// bit-identical — the property cuBLAS cannot promise. Naive is "fast
// enough" for the decode regime (m == 1, memory-bound).
//
// The kernel / launcher mechanics mirror the NF4 GEMV family
// (kernels/quantize/nf4_launcher.cu); the FFI parameter list is a
// literal symbol-swap of the strided-batched gemm_dense_*_run signature
// (src/gemm_dense_cublas_facade.rs) so this drops in behind the same
// binding table.
//
// Status codes (matches the rest of baracuda-kernels-sys):
//   0 = success
//   2 = invalid problem (bad extents / layout / leading dims / strides /
//       null pointers)
//   5 = internal kernel error (launch failure)

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

// RRR is the only layout this kernel implements (row-major A / B / D).
constexpr int32_t GEMV_LAYOUT_RRR = 0;

// Status codes.
constexpr int32_t GEMV_OK = 0;
constexpr int32_t GEMV_INVALID = 2;

inline int32_t status_from_launch(cudaError_t err) {
    return (err != cudaSuccess) ? 5 : 0;
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// -----------------------------------------------------------------------------
// f32-accumulate load / store casts. Mirrors the NF4 GEMV family's
// `load_act_nf4<T>` (vendor/bitsandbytes/src/nf4_kernel.cuh) but kept
// self-contained here so this always-on TU carries no dependency on the
// feature-gated bitsandbytes vendor header. f32 is the identity; f16 /
// bf16 widen to f32 on load and narrow on store, so the whole reduction
// runs in f32.
// -----------------------------------------------------------------------------
template <typename T> struct gemv_acc_cast;

template <> struct gemv_acc_cast<float> {
    __device__ __forceinline__ static float load(const float * p) { return *p; }
    __device__ __forceinline__ static float store(float v) { return v; }
};
template <> struct gemv_acc_cast<__half> {
    __device__ __forceinline__ static float load(const __half * p) {
        return __half2float(*p);
    }
    __device__ __forceinline__ static __half store(float v) {
        return __float2half(v);
    }
};
template <> struct gemv_acc_cast<__nv_bfloat16> {
    __device__ __forceinline__ static float load(const __nv_bfloat16 * p) {
        return __bfloat162float(*p);
    }
    __device__ __forceinline__ static __nv_bfloat16 store(float v) {
        return __float2bfloat16(v);
    }
};

// -----------------------------------------------------------------------------
// Host-side shape / layout validation shared by `*_run` and
// `*_can_implement`. Pointer nullness is checked separately in `*_run`
// (after the empty-problem early-out).
// -----------------------------------------------------------------------------
int32_t gemv_validate(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout,
    int64_t lda, int64_t ldb, int64_t ldd, int64_t stride_d)
{
    if (m < 0 || n < 0 || k < 0 || batch < 0) { return GEMV_INVALID; }
    // This family is the m == 1 GEMV specialization: m == 0 is a valid
    // empty problem, m > 1 belongs on the batched gemm_dense path.
    if (m > 1) { return GEMV_INVALID; }
    if (layout != GEMV_LAYOUT_RRR) { return GEMV_INVALID; }
    // RRR leading-dim minimums (elements): A [m, k] ld ≥ k, B [k, n]
    // ld ≥ n, D [m, n] ld ≥ n. lda / ldd are honored though unused at
    // m == 1 (single A / D row → their row strides never apply).
    // Unlike the cuBLAS facade there is no i32 ld ceiling: the kernel
    // indexes B with a 64-bit `kk * ldb + j` (k · n can exceed 2^31 at
    // the vocab / long-context shapes).
    const int64_t min_lda = (k > 0) ? static_cast<int64_t>(k) : 1;
    const int64_t min_ld_n = (n > 0) ? static_cast<int64_t>(n) : 1;
    if (lda < min_lda || ldb < min_ld_n || ldd < min_ld_n) { return GEMV_INVALID; }
    // batch > 1 with a zero output stride overlaps every slot's D.
    if (batch > 1 && stride_d == 0) { return GEMV_INVALID; }
    return GEMV_OK;
}

// -----------------------------------------------------------------------------
// GEMV M=1 kernel. grid.x tiles the N output columns, grid.y is one
// plane per batch slot; blockDim.x threads each own one column.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void gemv_dense_m1_kernel(
    int n, int k,
    float alpha, float beta,
    const T * __restrict__ a, int64_t stride_a,
    const T * __restrict__ b, int64_t ldb, int64_t stride_b,
    T       * __restrict__ d, int64_t stride_d)
{
    const int slot = blockIdx.y;
    // Per-slot base offsets (elements). stride_b == 0 ⇒ every slot reads
    // the same B (GQA rhs broadcast) — the `slot * 0` term is 0 for all
    // slots, so the broadcast needs no special case.
    const T * __restrict__ a_row = a + static_cast<int64_t>(slot) * stride_a; // A: [1, k]
    const T * __restrict__ b_mat = b + static_cast<int64_t>(slot) * stride_b; // B: [k, n]
    T       * __restrict__ d_row = d + static_cast<int64_t>(slot) * stride_d; // D: [1, n]

    // Grid-stride over j keeps the launch grid bounded for large n (up to
    // ~128k) while leaving each column computed by exactly one thread —
    // the reduction order is fixed and shape-independent, so a captured
    // graph replays byte-identical.
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < n;
         j += gridDim.x * blockDim.x)
    {
        // Serial reduction over K in f32 (true IEEE for f32; f16 / bf16
        // widen via gemv_acc_cast<T>::load). At a fixed kk, adjacent
        // threads read B[kk * ldb + j] / B[kk * ldb + j + 1] → coalesced
        // across the warp; a_row[kk] is a uniform broadcast load.
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
            float av = gemv_acc_cast<T>::load(&a_row[kk]);
            float bv = gemv_acc_cast<T>::load(&b_mat[static_cast<int64_t>(kk) * ldb + j]);
            acc += av * bv;
        }
        float out = alpha * acc;
        // beta == 0 is write-only (matches the BLAS / cuBLAS contract: D
        // may be uninitialized or hold NaN and must be ignored). Only
        // read D back when beta actually contributes.
        if (beta != 0.0f) {
            out += beta * gemv_acc_cast<T>::load(&d_row[j]);
        }
        d_row[j] = gemv_acc_cast<T>::store(out);
    }
}

// -----------------------------------------------------------------------------
// GEMV M=1 launcher template.
// -----------------------------------------------------------------------------
template <typename T>
int32_t launch_gemv_dense_m1(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout,
    float alpha, float beta,
    const void * a, int64_t lda, int64_t stride_a,
    const void * b, int64_t ldb, int64_t stride_b,
    void * d, int64_t ldd, int64_t stride_d,
    void * workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
    const int32_t st = gemv_validate(m, n, k, batch, layout, lda, ldb, ldd, stride_d);
    if (st != GEMV_OK) { return st; }
    // Empty problem: nothing to launch (matches gemm_dense's
    // m == 0 / n == 0 / batch == 0 early-out).
    if (m == 0 || n == 0 || batch == 0) { return GEMV_OK; }
    if (!a || !b || !d) { return GEMV_INVALID; }

    // lda / ldd are honored by gemv_validate (RRR leading-dim minimums)
    // but never indexed: at m == 1 there is a single A row and a single
    // D row, so their row strides never apply.
    (void)lda;
    (void)ldd;
    // workspace / workspace_bytes are UNUSED: at m == 1 there is no
    // split-K and no multi-row staging, so no scratch is needed. Accepted
    // for signature parity with gemm_dense_*_run (documented no-op, not
    // silently dropped).
    (void)workspace;
    (void)workspace_bytes;

    constexpr int BLOCK = 256;
    dim3 block(BLOCK, 1, 1);
    // grid.y is one plane per batch slot (= n_heads for the GQA
    // scores / ·v matmuls) — far below the 65535 gridDim.y limit; a
    // pathological over-large batch surfaces cleanly as launch status 5
    // rather than corrupting output.
    dim3 grid(ceil_div(n, BLOCK), batch, 1);

    gemv_dense_m1_kernel<T><<<grid, block, 0, stream>>>(
        n, k, alpha, beta,
        static_cast<const T *>(a), stride_a,
        static_cast<const T *>(b), ldb, stride_b,
        static_cast<T *>(d), stride_d);
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

// =============================================================================
// FFI surface — dense GEMV M=1.
// =============================================================================
//
// D[g, 0, :] = alpha * A[g, 0, :] · B[g] + beta * D[g, 0, :]  (RRR).
//
// Signature is a literal symbol-swap of gemm_dense_*_run's strided-batch
// entry (src/gemm_dense_cublas_facade.rs): alpha / beta are f32 for all
// three dtype SKUs; strides are element counts; `stride_b == 0`
// broadcasts B across slots (GQA); `workspace` / `workspace_bytes` are
// reserved-and-ignored; `layout` must be 0 (RRR).

extern "C" int32_t baracuda_kernels_gemv_dense_m1_f32_run(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout,
    float alpha, float beta,
    const void * a, int64_t lda, int64_t stride_a,
    const void * b, int64_t ldb, int64_t stride_b,
    void * d, int64_t ldd, int64_t stride_d,
    void * workspace, size_t workspace_bytes,
    void * stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_gemv_dense_m1<float>(
        m, n, k, batch, layout, alpha, beta,
        a, lda, stride_a, b, ldb, stride_b, d, ldd, stride_d,
        workspace, workspace_bytes, stream);
}

extern "C" int32_t baracuda_kernels_gemv_dense_m1_f16_run(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout,
    float alpha, float beta,
    const void * a, int64_t lda, int64_t stride_a,
    const void * b, int64_t ldb, int64_t stride_b,
    void * d, int64_t ldd, int64_t stride_d,
    void * workspace, size_t workspace_bytes,
    void * stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_gemv_dense_m1<__half>(
        m, n, k, batch, layout, alpha, beta,
        a, lda, stride_a, b, ldb, stride_b, d, ldd, stride_d,
        workspace, workspace_bytes, stream);
}

extern "C" int32_t baracuda_kernels_gemv_dense_m1_bf16_run(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout,
    float alpha, float beta,
    const void * a, int64_t lda, int64_t stride_a,
    const void * b, int64_t ldb, int64_t stride_b,
    void * d, int64_t ldd, int64_t stride_d,
    void * workspace, size_t workspace_bytes,
    void * stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_gemv_dense_m1<__nv_bfloat16>(
        m, n, k, batch, layout, alpha, beta,
        a, lda, stride_a, b, ldb, stride_b, d, ldd, stride_d,
        workspace, workspace_bytes, stream);
}

// =============================================================================
// _can_implement companions. Signature mirrors
// gemm_dense_*_can_implement (m, n, k, batch, layout, lda, ldb, ldd,
// stride_a, stride_b, stride_d); stride_a / stride_b are accepted
// unconditionally (any value, including 0-broadcast). 0 = OK, 2 = INVALID.
// =============================================================================

extern "C" int32_t baracuda_kernels_gemv_dense_m1_f32_can_implement(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout,
    int64_t lda, int64_t ldb, int64_t ldd,
    int64_t stride_a, int64_t stride_b, int64_t stride_d)
{
    (void)stride_a;
    (void)stride_b;
    return gemv_validate(m, n, k, batch, layout, lda, ldb, ldd, stride_d);
}

extern "C" int32_t baracuda_kernels_gemv_dense_m1_f16_can_implement(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout,
    int64_t lda, int64_t ldb, int64_t ldd,
    int64_t stride_a, int64_t stride_b, int64_t stride_d)
{
    (void)stride_a;
    (void)stride_b;
    return gemv_validate(m, n, k, batch, layout, lda, ldb, ldd, stride_d);
}

extern "C" int32_t baracuda_kernels_gemv_dense_m1_bf16_can_implement(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout,
    int64_t lda, int64_t ldb, int64_t ldd,
    int64_t stride_a, int64_t stride_b, int64_t stride_d)
{
    (void)stride_a;
    (void)stride_b;
    return gemv_validate(m, n, k, batch, layout, lda, ldb, ldd, stride_d);
}

// =============================================================================
// _workspace_size companions. Always 0 — no split-K / staging at m == 1.
// =============================================================================

extern "C" size_t baracuda_kernels_gemv_dense_m1_f32_workspace_size(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout)
{
    (void)m; (void)n; (void)k; (void)batch; (void)layout;
    return 0;
}

extern "C" size_t baracuda_kernels_gemv_dense_m1_f16_workspace_size(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout)
{
    (void)m; (void)n; (void)k; (void)batch; (void)layout;
    return 0;
}

extern "C" size_t baracuda_kernels_gemv_dense_m1_bf16_workspace_size(
    int32_t m, int32_t n, int32_t k, int32_t batch, int32_t layout)
{
    (void)m; (void)n; (void)k; (void)batch; (void)layout;
    return 0;
}
