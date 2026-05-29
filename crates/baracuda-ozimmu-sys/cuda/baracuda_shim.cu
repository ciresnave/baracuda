// baracuda_shim.cu — bridge between the C++ ozIMMU surface (under
// `mtk::ozimmu::`) and the flat C ABI consumed by
// `baracuda-ozimmu-sys/src/lib.rs`.
//
// Two responsibilities:
//
//   1. Supply `mtk::ozimmu::cublasCreate_org` /
//      `mtk::ozimmu::cublasDestroy_org`. Upstream ozIMMU declared
//      these as indirection hooks so the LD_PRELOAD shim could
//      forward to the "real" cuBLAS. baracuda statically links
//      cuBLAS through `baracuda-cublas-sys` — there is no LD_PRELOAD
//      path — so the wrappers are one-liners that call cuBLAS
//      directly.
//   2. Expose the `baracuda_ozimmu_*` flat-C entry points the Rust
//      side calls into. Bypasses bindgen for the upstream C++ header
//      (which exposes `std::vector<std::tuple<...>>` in the
//      `gemm_list_t` overload — bindgen-friendly only at the cost of
//      dragging the whole `std` namespace into the generated
//      bindings). The flat surface only exposes what
//      `baracuda-ozimmu::Handle` actually calls.

#include <cstddef>
#include <cublas_v2.h>

#include "handle.hpp"
#include <ozimmu/ozimmu.hpp>

// ---------------------------------------------------------------------------
// (1) cuBLAS lifecycle hooks the upstream header declares.
// ---------------------------------------------------------------------------

cublasStatus_t mtk::ozimmu::cublasCreate_org(cublasHandle_t *handle_ptr) {
    return ::cublasCreate_v2(handle_ptr);
}

cublasStatus_t mtk::ozimmu::cublasDestroy_org(cublasHandle_t handle_ptr) {
    return ::cublasDestroy_v2(handle_ptr);
}

// ---------------------------------------------------------------------------
// (2) Flat C ABI for the Rust safe wrapper.
// ---------------------------------------------------------------------------

extern "C" {

/// `mtk::ozimmu::create` wrapped to take an int for the malloc mode
/// (`0` = sync, `1` = async). Returns 0 on success, non-zero on
/// failure; the underlying C++ throws a `std::runtime_error` on the
/// cudaMalloc-failed path, which baracuda's call sites catch via the
/// `extern "C"` boundary (rethrowing across the boundary is UB; we
/// instead convert to a return code in `mtk::ozimmu::create`).
int baracuda_ozimmu_create(mtk::ozimmu::handle_t *out_handle,
                           int malloc_mode_async) {
    const auto mm = malloc_mode_async ? mtk::ozimmu::malloc_async
                                      : mtk::ozimmu::malloc_sync;
    try {
        return mtk::ozimmu::create(out_handle, mm);
    } catch (...) {
        return -1;
    }
}

/// `mtk::ozimmu::destroy`. Safe with a null handle (the inner
/// implementation no-ops on null).
int baracuda_ozimmu_destroy(mtk::ozimmu::handle_t handle) {
    try {
        return mtk::ozimmu::destroy(handle);
    } catch (...) {
        return -1;
    }
}

/// Bind a CUDA stream to the handle. Subsequent `dgemm` launches
/// enqueue on `stream`; the cuBLAS handle is re-bound at the same time.
void baracuda_ozimmu_set_cuda_stream(mtk::ozimmu::handle_t handle,
                                     cudaStream_t stream) {
    mtk::ozimmu::set_cuda_stream(handle, stream);
}

/// Pre-grow the working-memory scratch.
std::size_t baracuda_ozimmu_reallocate_working_memory_bytes(
    mtk::ozimmu::handle_t handle, std::size_t size_in_bytes) {
    return mtk::ozimmu::reallocate_working_memory(handle, size_in_bytes);
}

/// FP64 real GEMM via the Ozaki scheme.
///
/// `op_a` / `op_b` follow the `mtk::ozimmu::operation_t` convention
/// (0 = N, 1 = T). `compute_mode` is the integer cast of the
/// `mtk::ozimmu::compute_mode_t` enum so callers can pass
/// `fp64_int8_3` .. `fp64_int8_18` or `fp64_int8_auto` directly.
int baracuda_ozimmu_dgemm(mtk::ozimmu::handle_t handle,
                          int op_a, int op_b,
                          std::size_t m, std::size_t n, std::size_t k,
                          const double *alpha,
                          const double *a_ptr, std::size_t lda,
                          const double *b_ptr, std::size_t ldb,
                          const double *beta,
                          double *c_ptr, std::size_t ldc,
                          int compute_mode) {
    const auto oa = op_a ? mtk::ozimmu::op_t : mtk::ozimmu::op_n;
    const auto ob = op_b ? mtk::ozimmu::op_t : mtk::ozimmu::op_n;
    const auto cm = static_cast<mtk::ozimmu::compute_mode_t>(compute_mode);
    try {
        return mtk::ozimmu::gemm(handle, oa, ob, m, n, k,
                                 alpha, a_ptr, lda, b_ptr, ldb, beta,
                                 c_ptr, ldc, cm, mtk::ozimmu::real);
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
