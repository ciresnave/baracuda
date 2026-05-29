// config.cu — split-config table generator + working-memory sizer.
//
// Pure host-side metadata for the Ozaki splitter. No device code.

#include "config.hpp"
#include "utils.hpp"

mtk::ozimmu::detail::split_config_t
mtk::ozimmu::detail::get_split_config(const mtk::ozimmu::compute_mode_t compute_mode) {
    switch (compute_mode) {
    case mtk::ozimmu::sgemm:
        return split_config_t{ {original}, {original}, {{0, 0, detail::cublas_sgemm}} };
    case mtk::ozimmu::dgemm:
        return split_config_t{ {original}, {original}, {{0, 0, detail::cublas_dgemm}} };

    case mtk::ozimmu::fp64_int8_3:
    case mtk::ozimmu::fp64_int8_4:
    case mtk::ozimmu::fp64_int8_5:
    case mtk::ozimmu::fp64_int8_6:
    case mtk::ozimmu::fp64_int8_7:
    case mtk::ozimmu::fp64_int8_8:
    case mtk::ozimmu::fp64_int8_9:
    case mtk::ozimmu::fp64_int8_10:
    case mtk::ozimmu::fp64_int8_11:
    case mtk::ozimmu::fp64_int8_12:
    case mtk::ozimmu::fp64_int8_13:
    case mtk::ozimmu::fp64_int8_14:
    case mtk::ozimmu::fp64_int8_15:
    case mtk::ozimmu::fp64_int8_16:
    case mtk::ozimmu::fp64_int8_17:
    case mtk::ozimmu::fp64_int8_18: {
        // Map mode → slice count `S`. The compute_mode enum is laid
        // out consecutively from `fp64_int8_3` so a single subtract
        // would also work; the explicit switch matches upstream and
        // is one-time host-side cost.
        const unsigned num_split = static_cast<unsigned>(compute_mode)
                                 - static_cast<unsigned>(mtk::ozimmu::fp64_int8_3)
                                 + 3;

        // Data: index 0 is the original FP64 operand; indices 1..num_split
        // are the int8 slices.
        std::vector<mtk::ozimmu::data_t> split_types(num_split + 1, mtk::ozimmu::int8);
        split_types[0] = mtk::ozimmu::original;

        // Computation schedule: `S²`-ish list of (a_slice, b_slice) pairs.
        // The pair selection is anti-diagonals of the (1..S) × (1..S) grid,
        // ensuring every reachable (i, j) with i + j ≤ S + 1 is covered
        // exactly once. The upstream paper documents this as
        // "increasing-sum" ordering — the pairs with the largest
        // significance bits are computed first so the FP64 accumulator
        // sees them while the round-off margin is still large.
        std::vector<mtk::ozimmu::detail::gemm_pair_config_t> gemm_pair_list;
        for (int sum = 2; sum <= static_cast<int>(num_split) + 1; sum++) {
            for (int j = 1; j < sum; j++) {
                if (j > static_cast<int>(num_split) || sum - j > static_cast<int>(num_split)) {
                    continue;
                }
                gemm_pair_list.push_back({j, sum - j, mtk::ozimmu::detail::int8tc});
            }
        }

        return split_config_t{split_types, split_types, gemm_pair_list};
    }
    default:
        break;
    }
    return split_config_t{{}, {}};
}

std::string mtk::ozimmu::detail::gemm_mode_str(const mtk::ozimmu::detail::gemm_t gemm_mode) {
#define BARACUDA_GEMM_MODE_STR_CASE(mode)                                     \
    case mode:                                                                \
        return #mode
    switch (gemm_mode) {
        BARACUDA_GEMM_MODE_STR_CASE(cublas_sgemm);
        BARACUDA_GEMM_MODE_STR_CASE(cublas_dgemm);
        BARACUDA_GEMM_MODE_STR_CASE(cublas_tf32);
        BARACUDA_GEMM_MODE_STR_CASE(cublas_fp16);
        BARACUDA_GEMM_MODE_STR_CASE(cublas_bf16);
        BARACUDA_GEMM_MODE_STR_CASE(int8tc);
    default:
        break;
    }
#undef BARACUDA_GEMM_MODE_STR_CASE
    return "Unknown";
}

std::size_t mtk::ozimmu::detail::calculate_working_memory_size(
    const mtk::ozimmu::operation_t op, const std::size_t m, const std::size_t n,
    const mtk::ozimmu::compute_mode_t compute_mode,
    const mtk::ozimmu::detail::matrix_t matrix,
    const mtk::ozimmu::element_kind_t element_kind) {
    const auto split_config = mtk::ozimmu::detail::get_split_config(compute_mode);

    decltype(split_config.matrix_A_split_types) split_data_types;
    if (matrix == mtk::ozimmu::detail::matrix_A) {
        split_data_types = split_config.matrix_A_split_types;
    } else {
        split_data_types = split_config.matrix_B_split_types;
    }

    std::size_t size_sum = 0;
    for (const auto d : split_data_types) {
        const auto num_slice_elements = mtk::ozimmu::get_slice_num_elements<void>(
            m, n,
            matrix == mtk::ozimmu::detail::matrix_A ? mtk::ozimmu::op_t
                                                    : mtk::ozimmu::op_n,
            d);
        size_sum += mtk::ozimmu::get_data_size_in_byte(d) * num_slice_elements;
    }

    return size_sum * (element_kind == mtk::ozimmu::real ? 1 : 2);
}
