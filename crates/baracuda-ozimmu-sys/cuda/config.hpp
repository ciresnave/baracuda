// config.hpp — Ozaki-scheme split-config tables.
//
// For a given `compute_mode_t` (i.e. slice count `S`), the
// `get_split_config` function returns the layout-of-int8-slices and
// the schedule of int8 tensor-core GEMM pairs the algorithm needs to
// reconstruct the FP64 product. The schedule is built once per
// (compute_mode, problem-shape) and is purely host-side metadata; no
// device launches happen here.

#pragma once
#include <ozimmu/ozimmu.hpp>
#include <string>
#include <vector>

namespace mtk {
namespace ozimmu {
namespace detail {

/// Which operand a split-config entry refers to. C is unused in the
/// real path but is kept for symmetry with the upstream API.
enum matrix_t { matrix_A, matrix_B, matrix_C };

/// GEMM backend variant used by a given pair of int8 slices.
/// `int8tc` is the dominant case (and the only one wired today); the
/// other variants are kept for the upstream paper's experimental
/// tables.
enum gemm_t {
    cublas_dgemm,
    cublas_sgemm,
    cublas_tf32,
    cublas_fp16,
    cublas_bf16,
    int8tc
};

template <class T> inline data_t get_data_t();
template <> inline data_t get_data_t<float>() { return data_t::fp32; }

/// One entry in the schedule: which int8 slice of A times which int8
/// slice of B, computed by which backend variant. `A_id == 0` is the
/// original (non-split) operand reference.
struct gemm_pair_config_t {
    int A_id;     ///< Slice index into A (`-1` would mean "original"; 0 is the actual original here).
    int B_id;     ///< Slice index into B.
    gemm_t gemm_mode;
};

/// Split-config for one `compute_mode_t`. The two `_split_types`
/// vectors describe how each matrix is split (element type per slice,
/// with index 0 == original FP64 input). The pair list is the
/// `S²`-ish schedule of int8 GEMMs whose accumulation reconstructs
/// the FP64 product.
struct split_config_t {
    std::vector<data_t> matrix_A_split_types;
    std::vector<data_t> matrix_B_split_types;
    std::vector<gemm_pair_config_t> gemm_pair_config_list;
};

/// Build a `split_config_t` for the given compute mode. Pure host
/// function; the returned vectors are small enough that copy-by-value
/// is cheaper than caching.
split_config_t get_split_config(const mtk::ozimmu::compute_mode_t compute_mode);

/// Human-readable name for `gemm_t` — used by debug/log paths.
std::string gemm_mode_str(const gemm_t gemm_mode);

/// Compute the total working-memory size (in bytes) needed for one
/// operand of one GEMM under the chosen mode. Sums across all
/// slices; `element_kind` doubles the result for complex inputs (two
/// real-equivalent matrices held back-to-back).
std::size_t
calculate_working_memory_size(const mtk::ozimmu::operation_t op,
                              const std::size_t m, const std::size_t n,
                              const mtk::ozimmu::compute_mode_t compute_mode,
                              const mtk::ozimmu::detail::matrix_t matrix,
                              const mtk::ozimmu::element_kind_t element_kind);

}  // namespace detail
}  // namespace ozimmu
}  // namespace mtk
