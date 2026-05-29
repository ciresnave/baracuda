// split.cu — Ozaki int8 splitter + mantissa-loss histogram.
//
// Per-row, the splitter:
//   1. finds the matrix's per-row absolute max (via a hierarchical
//      warp/block reduction);
//   2. computes a per-row scale exponent (`max_exp`) shared across
//      every cell of that row;
//   3. cuts each FP64 cell's mantissa into `num_split` 8-bit slices,
//      stored in the int8 staging buffer.
//
// The reconstruction stage in `gemm.cu` accumulates the int8 tensor-
// core products and re-scales by `max_exp` to recover the FP64
// product.
//
// Phase 44b clean-fork notes:
//
//   - All `cutf::` references replaced with their baracuda equivalents
//     (`baracuda::fp::` / `baracuda::Uint128`).
//   - Inlined `warp_size_const` constant (literal 32) and replaced
//     `cutf::math::max` with `::max` (CUDA already provides
//     overloaded `max(double, double)` / `max(float, float)`).
//   - Replaced `__uint128_t` with `baracuda::Uint128` so the kernel
//     compiles under MSVC. The Linux path typedefs `Uint128` back to
//     `__uint128_t` (bit-for-bit unchanged).
//   - Removed the upstream `CUTF_CHECK_ERROR(cudaDeviceSynchronize())`
//     after the kernel launch — the host driver in `gemm.cu` already
//     synchronizes the stream before reading results, and the extra
//     device-wide sync hurts overlap.

#include "config.hpp"
#include "split.hpp"
#include "utils.hpp"
#include <ozimmu/ozimmu.hpp>

#include "baracuda_fp_bits.cuh"

// Inlined from cutf::thread (the only constant we used). Matches the
// literal 32 that CUDA's `warpSize` builtin reports across every arch
// baracuda supports.
namespace {
constexpr unsigned BARACUDA_WARP_SIZE_CONST = 32;
}

namespace {

template <class T>
__device__ T get_exp_max_element(
    // [length * inc]
    const T *const ptr, const unsigned length, const unsigned inc,
    // [blockDim.x / warp_size]
    typename mtk::ozimmu::detail::real_type<T>::type *const working_smem_ptr) {

    T local_abs_max = 0;

    unsigned i = threadIdx.x;
    const T *local_ptr = ptr + i * inc;
    for (; i < length; i += blockDim.x) {
        const auto v = baracuda::fp::reinterpret_as_fp(
            baracuda::fp::mask_exponent(*local_ptr));

        local_abs_max = ::max(local_abs_max, v);

        local_ptr += inc * blockDim.x;
    }

    // Inner-warp reduction
    for (std::uint32_t offset = BARACUDA_WARP_SIZE_CONST >> 1; offset >= 1; offset >>= 1) {
        local_abs_max = ::max(__shfl_xor_sync(~0u, local_abs_max, offset),
                              local_abs_max);
    }

    // Inner-threadblock reduction
    if ((threadIdx.x & 0x1f) == 0) {
        working_smem_ptr[threadIdx.x >> 5] = local_abs_max;
    }

    __syncthreads();
    local_abs_max = 0;
    if (threadIdx.x < BARACUDA_WARP_SIZE_CONST) {
        if (threadIdx.x < (blockDim.x / BARACUDA_WARP_SIZE_CONST)) {
            local_abs_max = working_smem_ptr[threadIdx.x];
        }

        for (std::uint32_t offset = BARACUDA_WARP_SIZE_CONST >> 1; offset >= 1; offset >>= 1) {
            local_abs_max = ::max(__shfl_xor_sync(~0u, local_abs_max, offset),
                                  local_abs_max);
        }

        if (threadIdx.x == 0) {
            working_smem_ptr[0] = local_abs_max;
        }
    }

    __syncthreads();

    return working_smem_ptr[0];
}

template <>
__device__ cuDoubleComplex get_exp_max_element<cuDoubleComplex>(
    const cuDoubleComplex *const ptr, const unsigned length, const unsigned inc,
    double *const working_smem_ptr) {
    using real_t = typename mtk::ozimmu::detail::real_type<cuDoubleComplex>::type;

    real_t local_real_abs_max = 0;
    real_t local_imag_abs_max = 0;

    unsigned i = threadIdx.x;
    const cuDoubleComplex *local_ptr = ptr + i * inc;
    for (; i < length; i += blockDim.x) {
        const auto v = *local_ptr;
        const auto real_exp = baracuda::fp::reinterpret_as_fp(
            baracuda::fp::mask_exponent(v.x));
        const auto imag_exp = baracuda::fp::reinterpret_as_fp(
            baracuda::fp::mask_exponent(v.y));

        local_real_abs_max = ::max(local_real_abs_max, real_exp);
        local_imag_abs_max = ::max(local_imag_abs_max, imag_exp);

        local_ptr += inc * blockDim.x;
    }

    // Inner-warp reduction (real + imag)
    for (std::uint32_t offset = BARACUDA_WARP_SIZE_CONST >> 1; offset >= 1; offset >>= 1) {
        local_real_abs_max = ::max(__shfl_xor_sync(~0u, local_real_abs_max, offset),
                                   local_real_abs_max);
        local_imag_abs_max = ::max(__shfl_xor_sync(~0u, local_imag_abs_max, offset),
                                   local_imag_abs_max);
    }

    // Real-block reduce
    if ((threadIdx.x & 0x1f) == 0) {
        working_smem_ptr[threadIdx.x >> 5] = local_real_abs_max;
    }
    __syncthreads();
    local_real_abs_max = 0;
    if (threadIdx.x < BARACUDA_WARP_SIZE_CONST) {
        if (threadIdx.x < (blockDim.x / BARACUDA_WARP_SIZE_CONST)) {
            local_real_abs_max = working_smem_ptr[threadIdx.x];
        }
        for (std::uint32_t offset = BARACUDA_WARP_SIZE_CONST >> 1; offset >= 1; offset >>= 1) {
            local_real_abs_max = ::max(__shfl_xor_sync(~0u, local_real_abs_max, offset),
                                       local_real_abs_max);
        }
    }
    __syncthreads();

    // Imag-block reduce
    if ((threadIdx.x & 0x1f) == 0) {
        working_smem_ptr[threadIdx.x >> 5] = local_imag_abs_max;
    }
    __syncthreads();
    local_imag_abs_max = 0;
    if (threadIdx.x < BARACUDA_WARP_SIZE_CONST) {
        if (threadIdx.x < (blockDim.x / BARACUDA_WARP_SIZE_CONST)) {
            local_imag_abs_max = working_smem_ptr[threadIdx.x];
        }
        for (std::uint32_t offset = BARACUDA_WARP_SIZE_CONST >> 1; offset >= 1; offset >>= 1) {
            local_imag_abs_max = ::max(__shfl_xor_sync(~0u, local_imag_abs_max, offset),
                                       local_imag_abs_max);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        working_smem_ptr[0] = local_real_abs_max;
        working_smem_ptr[1] = local_imag_abs_max;
    }

    __syncthreads();

    return make_cuDoubleComplex(working_smem_ptr[0], working_smem_ptr[1]);
}

/// Core int8-extraction step. Builds a `MantissaT` (128-bit) holding
/// register from the FP cell's mantissa + implicit-1 bit, shifts to
/// align with the row's max-exponent, and emits `num_split`
/// `mantissa_length`-bit slices as signed int8s.
template <class InputT, class MantissaT>
__device__ void cut_int8_core(std::int8_t *const out_ptr, const std::size_t inc,
                              const InputT a, const InputT max_exp,
                              const unsigned num_split,
                              const unsigned mantissa_length) {
    const std::uint8_t sign_flag = a > 0;
    // Non-normalized input cell — leave the implicit-1 bit clear.
    const std::uint64_t implict_one_bit =
        baracuda::fp::mask_exponent(a) ? 1lu : 0lu;
    const auto mantissa =
        static_cast<MantissaT>(
            baracuda::fp::mask_mantissa(a) |
            (implict_one_bit << baracuda::fp::get_mantissa_size<InputT>()))
        << ((sizeof(MantissaT) - sizeof(InputT)) * 8
            + baracuda::fp::get_exponent_size<InputT>());
    const auto mantissa_shift_offset =
        (baracuda::fp::reinterpret_as_uint(max_exp)
         - baracuda::fp::mask_exponent(a))
        >> baracuda::fp::get_mantissa_size<InputT>();

    auto shifted_mantissa = mantissa >> mantissa_shift_offset;
    for (unsigned s = 0; s < num_split; s++) {
        const std::int8_t int8 =
            static_cast<std::int8_t>(
                shifted_mantissa
                >> (sizeof(MantissaT) * 8 - mantissa_length))
            * (sign_flag ? 1 : -1);
        shifted_mantissa <<= mantissa_length;

        out_ptr[s * inc] = int8;
    }
}

__device__ cuDoubleComplex x2(const cuDoubleComplex a) {
    return make_cuDoubleComplex(a.x * 2, a.y * 2);
}
__device__ double x2(const double a) { return a * 2; }

template <class InputT, class MantissaT>
__global__ void split_int8_kernel(
    std::int8_t *const out_ptr, const std::uint32_t ldo,
    typename mtk::ozimmu::detail::real_type<InputT>::type *const max_exp_ptr,
    const std::size_t m, const std::size_t n, const InputT *const in_ptr,
    const std::size_t ld, const unsigned num_split,
    const unsigned mantissa_length, const bool col_major) {
    __shared__ typename mtk::ozimmu::detail::real_type<InputT>::type smem[32];
    const auto row_index = blockIdx.x;
    const auto max_exp = x2(
        get_exp_max_element(in_ptr + (col_major ? row_index : (row_index * ld)),
                            n, (col_major ? ld : 1), smem));

    const auto N = m * ldo;
    unsigned i;
    for (i = threadIdx.x; i < n; i += blockDim.x) {
        const auto a =
            in_ptr[(col_major ? (i * ld + row_index) : (i + row_index * ld))];
        if constexpr (std::is_same<cuDoubleComplex, InputT>::value) {
            cut_int8_core<double, MantissaT>(out_ptr + row_index * ldo + i, N, a.x,
                                             max_exp.x, num_split, mantissa_length);
            cut_int8_core<double, MantissaT>(
                out_ptr + row_index * ldo + i + N * num_split, N, a.y, max_exp.y,
                num_split, mantissa_length);
        } else {
            cut_int8_core<double, MantissaT>(out_ptr + row_index * ldo + i, N, a,
                                             max_exp, num_split, mantissa_length);
        }
    }
    // Padding fill — zero the tail of the slice buffer.
    for (; i < ldo; i += blockDim.x) {
        for (std::uint32_t j = 0; j < num_split; j++) {
            if constexpr (std::is_same<cuDoubleComplex, InputT>::value) {
                *(out_ptr + row_index * ldo + i + j * N) = 0;
                *(out_ptr + row_index * ldo + i + N * num_split + j * N) = 0;
            } else {
                *(out_ptr + row_index * ldo + i + j * N) = 0;
            }
        }
    }

    if (threadIdx.x == 0) {
        if constexpr (std::is_same<cuDoubleComplex, InputT>::value) {
            max_exp_ptr[blockIdx.x] = max_exp.x;
            max_exp_ptr[blockIdx.x + m] = max_exp.y;
        } else {
            max_exp_ptr[blockIdx.x] = max_exp;
        }
    }
}

template <class InputT>
void split_int8_A(
    std::int8_t *const out_ptr, const std::uint32_t ldo,
    typename mtk::ozimmu::detail::real_type<InputT>::type *const max_exp_ptr,
    const mtk::ozimmu::operation_t op, const std::size_t m, const std::size_t n,
    const InputT *const in_ptr, const std::size_t ld, const unsigned num_split,
    const unsigned mantissa_length, cudaStream_t cuda_stream) {
    const dim3 block_size = 256;
    const dim3 grid_size = m;

    const bool is_col_major = op == mtk::ozimmu::op_n;

    // Phase 44b — `baracuda::Uint128` is a typedef alias for
    // `__uint128_t` on GCC/Clang (bit-identical to the prior path)
    // and a portable struct emulator on MSVC. See
    // `baracuda_fp_bits.cuh` for the definition.
    using MantissaT = baracuda::Uint128;
    split_int8_kernel<InputT, MantissaT>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            out_ptr, ldo, max_exp_ptr, m, n, in_ptr, ld, num_split,
            mantissa_length, is_col_major);
}

}  // namespace

template <class T>
void mtk::ozimmu::split_int8(
    std::int8_t *const out_ptr, const std::uint32_t ldo,
    typename mtk::ozimmu::detail::real_type<T>::type *const max_exp_ptr,
    const std::size_t m, const std::size_t n, const T *const in_ptr,
    const std::size_t ld, const mtk::ozimmu::operation_t op,
    const mtk::ozimmu::detail::matrix_t matrix, const unsigned num_split,
    const unsigned bits_per_int8, const cudaStream_t cuda_stream) {
    if (matrix == mtk::ozimmu::detail::matrix_A) {
        split_int8_A(out_ptr, ldo, max_exp_ptr, op, m, n, in_ptr, ld, num_split,
                     bits_per_int8, cuda_stream);
    } else {
        split_int8_A(out_ptr, ldo, max_exp_ptr,
                     op == mtk::ozimmu::op_n ? mtk::ozimmu::op_t : mtk::ozimmu::op_n,
                     n, m, in_ptr, ld, num_split, bits_per_int8, cuda_stream);
    }
}

template void mtk::ozimmu::split_int8<double>(
    std::int8_t *const out_ptr, const std::uint32_t ldo,
    double *const max_exp_ptr, const std::size_t m, const std::size_t n,
    const double *const in_ptr, const std::size_t ld,
    const mtk::ozimmu::operation_t op,
    const mtk::ozimmu::detail::matrix_t matrix, const unsigned num_split,
    const unsigned bits_per_int8, const cudaStream_t cuda_stream);

template void mtk::ozimmu::split_int8<cuDoubleComplex>(
    std::int8_t *const out_ptr, const std::uint32_t ldo,
    double *const max_exp_ptr, const std::size_t m, const std::size_t n,
    const cuDoubleComplex *const in_ptr, const std::size_t ld,
    const mtk::ozimmu::operation_t op,
    const mtk::ozimmu::detail::matrix_t matrix, const unsigned num_split,
    const unsigned bits_per_int8, const cudaStream_t cuda_stream);

// ===========================================================================
// Mantissa-loss histogram for `auto_mode_select`
// ===========================================================================

namespace {

__global__ void
init_mantissa_loss_counter_kernel(unsigned long long int *const counter_ptr,
                                  const std::uint64_t counter_length) {
    if (threadIdx.x < counter_length) {
        counter_ptr[threadIdx.x] = 0;
    }
}

void init_mantissa_loss_counter(unsigned long long int *const counter_ptr,
                                const std::uint64_t counter_length,
                                cudaStream_t cuda_stream) {
    init_mantissa_loss_counter_kernel<<<1, counter_length, 0, cuda_stream>>>(
        counter_ptr, counter_length);
}

template <class InputT>
__device__ void calculate_mantissa_loss_core(
    unsigned long long *mantissa_loss_length_buffer, const InputT in,
    const InputT max_exp, const unsigned min_num_split,
    const unsigned max_num_split, const unsigned mantissa_length) {
    if (in == 0 || max_exp == 0) {
        return;
    }
    const auto required_mantissa_space_length =
        ((baracuda::fp::mask_exponent(max_exp)
          - baracuda::fp::mask_exponent(in)) >> 52)
        + 53;
    for (unsigned num_split = min_num_split; num_split <= max_num_split;
         num_split++) {
        unsigned mantissa_loss_length = 0;
        const auto mantissa_space_length = num_split * mantissa_length;
        if (mantissa_space_length < required_mantissa_space_length) {
            mantissa_loss_length = required_mantissa_space_length - mantissa_space_length;
        }

        for (std::uint32_t offset = BARACUDA_WARP_SIZE_CONST >> 1; offset >= 1; offset >>= 1) {
            mantissa_loss_length += __shfl_xor_sync(~0u, mantissa_loss_length, offset);
        }

        if ((threadIdx.x & 0x1f) == 0) {
            atomicAdd(mantissa_loss_length_buffer + (num_split - min_num_split),
                      mantissa_loss_length);
        }
    }
}

template <class InputT>
__global__ void calculate_mantissa_loss_kernel(
    unsigned long long *mantissa_loss_length_buffer, const std::size_t m,
    const std::size_t n, const InputT *const in_ptr, const std::size_t ld,
    const unsigned min_num_split, const unsigned max_num_split,
    const unsigned mantissa_length, const bool col_major) {
    __shared__ typename mtk::ozimmu::detail::real_type<InputT>::type smem[32];
    const auto row_index = blockIdx.x;
    const auto max_exp = x2(
        get_exp_max_element(in_ptr + (col_major ? row_index : (row_index * ld)),
                            n, (col_major ? ld : 1), smem));

    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        const auto a =
            in_ptr[(col_major ? (i * ld + row_index) : (i + row_index * ld))];
        if constexpr (std::is_same<cuDoubleComplex, InputT>::value) {
            calculate_mantissa_loss_core<double>(mantissa_loss_length_buffer, a.x,
                                                 max_exp.x, min_num_split,
                                                 max_num_split, mantissa_length);
            calculate_mantissa_loss_core<double>(mantissa_loss_length_buffer, a.y,
                                                 max_exp.y, min_num_split,
                                                 max_num_split, mantissa_length);
        } else {
            calculate_mantissa_loss_core<double>(mantissa_loss_length_buffer, a,
                                                 max_exp, min_num_split,
                                                 max_num_split, mantissa_length);
        }
    }
}

}  // namespace

template <class T>
std::unordered_map<mtk::ozimmu::compute_mode_t, std::uint64_t>
mtk::ozimmu::get_mantissa_loss_total(
    mtk::ozimmu::handle &handle, const std::size_t m, const std::size_t n,
    const T *const in_ptr, const std::size_t ld,
    const mtk::ozimmu::operation_t op, const unsigned bits_per_int8,
    const cudaStream_t /*cuda_stream*/, const bool download) {
    const dim3 block_size = 256;
    const dim3 grid_size = m;

    const bool is_col_major = op == mtk::ozimmu::op_n;

    calculate_mantissa_loss_kernel<T>
        <<<grid_size, block_size, 0, handle.cuda_stream>>>(
            handle.d_mantissa_loss_counter_ptr, m, n, in_ptr, ld, 3, 18,
            bits_per_int8, is_col_major);

    std::unordered_map<mtk::ozimmu::compute_mode_t, std::uint64_t> result;
    if (download) {
        unsigned long long int
            host_buffer[mtk::ozimmu::handle::mantissa_loss_counter_length];
        const auto status =
            cudaMemcpy(host_buffer, handle.d_mantissa_loss_counter_ptr,
                       sizeof(unsigned long long int)
                           * mtk::ozimmu::handle::mantissa_loss_counter_length,
                       cudaMemcpyDefault);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("ozIMMU mantissa-loss download failed: ")
                + cudaGetErrorString(status));
        }

        // Indices in the host_buffer correspond to slice counts
        // 3..18 — see the upstream paper §3 for the layout.
        for (unsigned i = 0; i < mtk::ozimmu::handle::mantissa_loss_counter_length; i++) {
            const auto mode = static_cast<mtk::ozimmu::compute_mode_t>(
                static_cast<unsigned>(mtk::ozimmu::fp64_int8_3) + i);
            result.emplace(mode, host_buffer[i]);
        }
    }

    return result;
}

void mtk::ozimmu::init_mantissa_loss_counter(mtk::ozimmu::handle &handle) {
    ::init_mantissa_loss_counter(handle.d_mantissa_loss_counter_ptr,
                                 handle.mantissa_loss_counter_length,
                                 handle.cuda_stream);
}

namespace {

template <class T>
mtk::ozimmu::compute_mode_t auto_mode_select_core(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const T *const a_ptr,
    const std::size_t lda, const T *const b_ptr, const std::size_t ldb,
    const double mantissa_loss_threshold) {
    const auto bits_per_int8 = mtk::ozimmu::get_bits_per_int8(k);
    mtk::ozimmu::init_mantissa_loss_counter(*handle);

    mtk::ozimmu::get_mantissa_loss_total(*handle, m, k, a_ptr, lda, op_A,
                                         bits_per_int8, handle->cuda_stream,
                                         false);

    const auto dist = mtk::ozimmu::get_mantissa_loss_total(
        *handle, n, k, b_ptr, ldb,
        op_B == mtk::ozimmu::op_n ? mtk::ozimmu::op_t : mtk::ozimmu::op_n,
        bits_per_int8, handle->cuda_stream, true);

    const std::vector<mtk::ozimmu::compute_mode_t> mode_candidate_order = {
        mtk::ozimmu::fp64_int8_3,  mtk::ozimmu::fp64_int8_4,
        mtk::ozimmu::fp64_int8_5,  mtk::ozimmu::fp64_int8_6,
        mtk::ozimmu::fp64_int8_7,  mtk::ozimmu::fp64_int8_8,
        mtk::ozimmu::fp64_int8_9,  mtk::ozimmu::fp64_int8_10,
        mtk::ozimmu::fp64_int8_11, mtk::ozimmu::fp64_int8_12,
        mtk::ozimmu::fp64_int8_13, mtk::ozimmu::fp64_int8_14,
        mtk::ozimmu::fp64_int8_15, mtk::ozimmu::fp64_int8_16,
        mtk::ozimmu::fp64_int8_17, mtk::ozimmu::fp64_int8_18,
    };

    for (const auto mode : mode_candidate_order) {
        if (dist.count(mode) != 0) {
            if (dist.at(mode) / static_cast<double>(m * k + k * n)
                <= mantissa_loss_threshold) {
                return mode;
            }
        }
    }

    return mtk::ozimmu::dgemm;
}

}  // namespace

mtk::ozimmu::compute_mode_t mtk::ozimmu::auto_mode_select(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const void *const a_ptr,
    const std::size_t lda, const void *const b_ptr, const std::size_t ldb,
    const mtk::ozimmu::element_kind_t element_kind,
    const double mantissa_loss_threshold) {
    mtk::ozimmu::compute_mode_t result;
    if (element_kind == mtk::ozimmu::real) {
        result = auto_mode_select_core(handle, op_A, op_B, m, n, k,
                                       reinterpret_cast<const double *>(a_ptr), lda,
                                       reinterpret_cast<const double *>(b_ptr), ldb,
                                       mantissa_loss_threshold);
    } else {
        result = auto_mode_select_core(
            handle, op_A, op_B, m, n, k,
            reinterpret_cast<const cuDoubleComplex *>(a_ptr), lda,
            reinterpret_cast<const cuDoubleComplex *>(b_ptr), ldb,
            mantissa_loss_threshold);
    }
    return result;
}

std::uint32_t mtk::ozimmu::get_bits_per_int8(const std::uint32_t k) {
    if (k == 0) {
        return 0;
    }
    // ceil(log2(k))
    std::uint32_t log2_k = 0;
    while ((1u << (log2_k + 1)) <= k) {
        log2_k++;
    }
    if ((1u << log2_k) != k) {
        log2_k++;
    }
    return std::min<std::uint32_t>(7, (31 - log2_k) / 2);
}
