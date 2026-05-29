// split.hpp — kernel-launch entry points for the Ozaki int8 splitter
// and the mantissa-loss histogram used by `fp64_int8_auto`.
//
// The actual device kernels live in `split.cu`; this header just
// declares the host-side launch functions and the FP-utility helpers
// (`get_two_to_alpha`) that the splitter shares with the GEMM driver.

#pragma once

#include "config.hpp"
#include "handle.hpp"
#include "utils.hpp"
#include <cmath>
#include <ozimmu/ozimmu.hpp>
#include <unordered_map>

// Pulled in from the baracuda-native FP-bittwiddle header (replaces
// cutf/experimental/fp.hpp).
#include "baracuda_fp_bits.cuh"

namespace mtk {
namespace ozimmu {

/// Split an FP64 (or cuDoubleComplex) matrix into `num_split`
/// int8 slices, recording the per-row max-exponent so the
/// reconstruction stage can re-scale. Asynchronous on `cuda_stream`.
template <class T>
void split_int8(
    std::int8_t *const out_ptr, std::uint32_t ldo,
    typename mtk::ozimmu::detail::real_type<T>::type *const max_exp_ptr,
    const std::size_t m, const std::size_t n, const T *const in_ptr,
    const std::size_t ld, const mtk::ozimmu::operation_t op,
    const mtk::ozimmu::detail::matrix_t matrix, const unsigned num_split,
    const unsigned bits_per_int8, const cudaStream_t cuda_stream);

/// Compute the per-slice-count mantissa-loss totals for a single
/// matrix. The histogram drives `auto_mode_select` — for each
/// candidate slice count `S ∈ 6..13` it returns the total number of
/// lost mantissa bits across the matrix.
template <class T>
std::unordered_map<mtk::ozimmu::compute_mode_t, std::uint64_t>
get_mantissa_loss_total(mtk::ozimmu::handle &handle, const std::size_t m,
                        const std::size_t n, const T *const in_ptr,
                        const std::size_t ld, const mtk::ozimmu::operation_t op,
                        const unsigned bits_per_int8,
                        const cudaStream_t cuda_stream, const bool download);

/// Reset the handle's mantissa-loss histogram to all-zero.
void init_mantissa_loss_counter(mtk::ozimmu::handle &handle);

/// `2^alpha` — the per-matrix scale factor used to re-center the
/// FP64 range into int8 territory. `alpha` is chosen so the mantissa
/// loss across the matrix is balanced; see the upstream paper §3.
template <class InputT>
InputT get_two_to_alpha(const std::size_t k) {
    return 1lu << static_cast<unsigned>(
        std::ceil((baracuda::fp::get_mantissa_size<InputT>() + 1 + std::log2(k)) / 2));
}

}  // namespace ozimmu
}  // namespace mtk
