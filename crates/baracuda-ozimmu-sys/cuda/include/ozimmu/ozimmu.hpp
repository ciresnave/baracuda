// ozimmu.hpp — baracuda-internalized Ozaki-scheme FP64 GEMM
//
// Public C++ header for the baracuda-owned ozIMMU implementation. The
// algorithm — synthesize an FP64 GEMM out of S² int8 tensor-core
// matmuls — is from:
//
//   Hiroyuki Ootomo, Katsuhisa Ozaki, Rio Yokota.
//   "DGEMM on Integer Matrix Multiplication Unit."
//   IJHPCA 2024 / arXiv:2306.11975.
//
// The original reference implementation is enp1s0/ozIMMU (MIT). We
// vendored it for Phase 44 (alpha.56) and clean-forked it in Phase 44b
// (alpha.57) — see `ATTRIBUTION.md` at the crate root for the full
// provenance story. The C++ interface below intentionally mirrors the
// upstream `mtk::ozimmu::` API so the algorithmic literature (e.g.
// the IJHPCA paper) can be read against this code directly; the
// `mtk::ozimmu::` namespace is preserved for the same reason.
//
// The Rust-facing surface lives one layer up in
// `baracuda_shim.cu` (the small flat C ABI consumed by
// `baracuda-ozimmu-sys/src/lib.rs`). Downstream callers should use
// the `baracuda-ozimmu` safe wrapper instead of touching this header
// directly.

#pragma once
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace mtk {
namespace ozimmu {

struct handle;

using handle_t = handle *;

/// Operand transpose flag — `op_n` is no-transpose, `op_t` is
/// transpose. The Ozaki real path doesn't support conjugate-transpose
/// (`op_c`) — complex inputs are routed through a separate
/// real-emulating path.
enum operation_t { op_n, op_t };

/// Compute-mode discriminant. `dgemm` is a passthrough to
/// `cublasDgemm` (reference fallback / accuracy baseline). The
/// `fp64_int8_*` modes select the slice count `S` for the Ozaki
/// scheme: more slices = more accurate, more expensive. `S = 8` is
/// the upstream-recommended "comparable-to-DGEMM" sweet spot;
/// `fp64_int8_auto` picks `S` per-call based on the inputs'
/// mantissa-loss histogram.
enum compute_mode_t {
    sgemm,
    dgemm,

    fp64_int8_3,
    fp64_int8_4,
    fp64_int8_5,
    fp64_int8_6,
    fp64_int8_7,
    fp64_int8_8,
    fp64_int8_9,
    fp64_int8_10,
    fp64_int8_11,
    fp64_int8_12,
    fp64_int8_13,
    fp64_int8_14,
    fp64_int8_15,
    fp64_int8_16,
    fp64_int8_17,
    fp64_int8_18,

    fp64_int8_auto,
};

/// Data-type discriminant used by the working-memory sizer and the
/// internal split tables. `original` means "the same type the caller
/// passed in"; `int8` is the staging dtype for the Ozaki splitter.
enum data_t { fp64, fp32, fp16, int8, original, none };

/// Working-memory allocator mode. `malloc_sync` uses `cudaMalloc`
/// (default; safe everywhere); `malloc_async` uses `cudaMallocAsync`
/// (only meaningful when the bound stream's memory pool is configured
/// for it).
enum malloc_mode_t { malloc_sync, malloc_async };

/// Operand element category — real (FP64) or complex (cuDoubleComplex).
enum element_kind_t {
    real,
    complx,
};

int create(mtk::ozimmu::handle_t *handle,
           const malloc_mode_t mm = malloc_sync);
int destroy(mtk::ozimmu::handle_t handle);
void set_cuda_stream(mtk::ozimmu::handle_t handle,
                     const cudaStream_t cuda_stream);

void set_auto_mantissa_loss_threashold(mtk::ozimmu::handle_t handle,
                                       const double threshold);
double get_auto_mantissa_loss_threashold(mtk::ozimmu::handle_t handle);

using gemm_params_t =
    std::tuple<mtk::ozimmu::operation_t, mtk::ozimmu::operation_t,
               std::size_t, std::size_t, std::size_t,
               mtk::ozimmu::element_kind_t,
               mtk::ozimmu::compute_mode_t>;
using gemm_list_t = std::vector<gemm_params_t>;

/// Grow the handle's working-memory scratch to the maximum size
/// required by any of the planned GEMM shapes in `gemm_list`. Returns
/// the new total size in bytes, or 0 if no reallocation happened.
std::size_t reallocate_working_memory(mtk::ozimmu::handle_t handle,
                                      const gemm_list_t gemm_list);

/// Grow the handle's working-memory scratch to at least `size_in_byte`.
/// Returns the new size, or 0 if no reallocation happened.
std::size_t reallocate_working_memory(mtk::ozimmu::handle_t handle,
                                      const std::size_t size_in_byte);

/// FP64 GEMM via the Ozaki scheme (or the chosen `compute_mode_t`).
/// `alpha` / `beta` are host-side FP64 (real) or cuDoubleComplex
/// (complex) scalars. `a_ptr` / `b_ptr` / `c_ptr` are device-resident
/// FP64 buffers in cuBLAS column-major layout.
int gemm(mtk::ozimmu::handle_t handle,
         const mtk::ozimmu::operation_t op_A,
         const mtk::ozimmu::operation_t op_B,
         const std::size_t m,
         const std::size_t n,
         const std::size_t k,
         const void *alpha,
         const void *const a_ptr, const std::size_t lda,
         const void *const b_ptr, const std::size_t ldb,
         const void *beta,
         void *const c_ptr, std::size_t ldc,
         const mtk::ozimmu::compute_mode_t compute_mode,
         const mtk::ozimmu::element_kind_t element_kind);

/// Pick the smallest slice-count Ozaki mode whose average mantissa
/// loss is within `mantissa_loss_threshold` of the input. Used
/// internally by `compute_mode_t::fp64_int8_auto`; exposed publicly
/// so callers can pre-select once and re-use the chosen mode across
/// many GEMMs at the same shape.
compute_mode_t auto_mode_select(mtk::ozimmu::handle_t handle,
                                const mtk::ozimmu::operation_t op_A,
                                const mtk::ozimmu::operation_t op_B,
                                const std::size_t m,
                                const std::size_t n,
                                const std::size_t k,
                                const void *const a_ptr,
                                const std::size_t lda,
                                const void *const b_ptr,
                                const std::size_t ldb,
                                const mtk::ozimmu::element_kind_t element_kind,
                                const double mantissa_loss_threshold);

std::string get_compute_mode_name_str(const mtk::ozimmu::compute_mode_t mode);

mtk::ozimmu::data_t get_output_type(const mtk::ozimmu::compute_mode_t mode);

std::size_t get_data_size_in_byte(const mtk::ozimmu::data_t d);

std::uint32_t get_bits_per_int8(const std::uint32_t k);

}  // namespace ozimmu
}  // namespace mtk
