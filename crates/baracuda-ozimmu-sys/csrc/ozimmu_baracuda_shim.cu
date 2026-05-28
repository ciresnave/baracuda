// baracuda Phase 44 — ozIMMU integration shim.
//
// ozIMMU upstream is structured as an LD_PRELOAD-time interceptor: a user
// runs their cuBLAS app with `LD_PRELOAD=libozimmu.so` and the interceptor's
// own `cublasGemmEx` / `cublasDgemm_v2` definitions transparently replace
// cuBLAS's symbols, routing DGEMM through the Ozaki scheme. The interceptor
// needs to call the *real* cuBLAS underneath, which it grabs via
// `dlsym(RTLD_NEXT, ...)`.
//
// For the baracuda integration we statically link ozIMMU into our own crate
// and call its `mtk::ozimmu::gemm(...)` entry directly — no LD_PRELOAD, no
// runtime symbol games. Three things have to be supplied to the vendored
// ozIMMU translation units that this shim handles:
//
//   1. `mtk::ozimmu::cublasCreate_org`  — declared in `handle.hpp`, used by
//      `handle.cu`. Upstream's definition lives in `cublas.cu`, which we
//      drop because it redefines cuBLAS symbols (would collide with
//      baracuda-cublas-sys at link time). We supply a one-line direct call.
//
//   2. `mtk::ozimmu::cublasDestroy_org` — sibling of (1).
//
//   3. `ozimmu_baracuda_lookup_cublas_symbol` — called from the patched
//      `utils.hpp` (`ozIMMU_get_function_pointer`) when the upstream code
//      asks for "cublasGemmEx" / "cublasGemmStridedBatchedEx". We return the
//      address of the real cuBLAS symbol, which baracuda-cublas-sys's link
//      stage resolves to the cuBLAS .so/.dll for us.
//
// This is the *only* baracuda-specific TU added to the build; everything else
// is the upstream code with the two small in-place edits documented in
// `vendor/ozimmu/VENDOR.md`.

#include <cublas_v2.h>
#include <cstring>
#include <cstddef>

#include "handle.hpp"

// --- (1) and (2): _org indirection ------------------------------------------

cublasStatus_t mtk::ozimmu::cublasCreate_org(cublasHandle_t *handle_ptr) {
    return ::cublasCreate_v2(handle_ptr);
}

cublasStatus_t mtk::ozimmu::cublasDestroy_org(cublasHandle_t handle_ptr) {
    return ::cublasDestroy_v2(handle_ptr);
}

// --- (3): name → function pointer for the patched utils.hpp -----------------

extern "C" void *ozimmu_baracuda_lookup_cublas_symbol(const char *name) {
    if (name == nullptr) return nullptr;
    if (std::strcmp(name, "cublasGemmEx") == 0) {
        return reinterpret_cast<void *>(&::cublasGemmEx);
    }
    if (std::strcmp(name, "cublasGemmStridedBatchedEx") == 0) {
        return reinterpret_cast<void *>(&::cublasGemmStridedBatchedEx);
    }
    if (std::strcmp(name, "cublasCreate_v2") == 0) {
        return reinterpret_cast<void *>(&::cublasCreate_v2);
    }
    if (std::strcmp(name, "cublasDestroy_v2") == 0) {
        return reinterpret_cast<void *>(&::cublasDestroy_v2);
    }
    // Anything else means the upstream caller path is taking us somewhere
    // we did not anticipate. Return null so the existing error-log path in
    // utils.hpp can surface a clear diagnostic.
    return nullptr;
}

// --- (4): tiny C entry points the Rust safe wrapper calls -------------------
//
// ozIMMU's public C++ API uses `std::vector<std::tuple<...>>` for the
// `reallocate_working_memory` overload and a couple of `std::string`
// returners. bindgen on C++ would need `--allowlist-function` plus a
// large set of opaque types just to get plain-Rust signatures. Wrapping
// only what the safe layer needs in flat C is far cleaner.

#include <ozimmu/ozimmu.hpp>

extern "C" {

int baracuda_ozimmu_create(mtk::ozimmu::handle_t *out_handle,
                           int malloc_mode_async) {
    const auto mm = malloc_mode_async ? mtk::ozimmu::malloc_async
                                       : mtk::ozimmu::malloc_sync;
    return mtk::ozimmu::create(out_handle, mm);
}

int baracuda_ozimmu_destroy(mtk::ozimmu::handle_t handle) {
    return mtk::ozimmu::destroy(handle);
}

void baracuda_ozimmu_set_cuda_stream(mtk::ozimmu::handle_t handle,
                                     cudaStream_t stream) {
    mtk::ozimmu::set_cuda_stream(handle, stream);
}

std::size_t baracuda_ozimmu_reallocate_working_memory_bytes(
    mtk::ozimmu::handle_t handle, std::size_t size_in_bytes) {
    return mtk::ozimmu::reallocate_working_memory(handle, size_in_bytes);
}

// FP64 real GEMM only — the only path baracuda exposes today.
// `op_a` / `op_b` are 0 = N, 1 = T (matching mtk::ozimmu::operation_t).
// `compute_mode` is the integer cast of `mtk::ozimmu::compute_mode_t`
// (so callers can pass `fp64_int8_3` .. `fp64_int8_18` or
// `fp64_int8_auto` directly).
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
    return mtk::ozimmu::gemm(handle, oa, ob, m, n, k,
                             alpha, a_ptr, lda, b_ptr, ldb, beta,
                             c_ptr, ldc, cm, mtk::ozimmu::real);
}

} // extern "C"
