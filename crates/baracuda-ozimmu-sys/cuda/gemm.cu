// gemm.cu — Ozaki-scheme reconstruction driver.
//
// Orchestrates the (split → cublasGemmEx int8 × S² → accumulate →
// re-scale) pipeline that turns one FP64 GEMM into a chain of int8
// tensor-core matmuls. The split + per-row-max-exponent kernels live
// in `split.cu`; this TU owns the per-pair int8 GEMM launch + the
// FP64 accumulator + the per-row re-scale.
//
// Phase 44b clean-fork notes:
//
//   - `<cutf/cublas.hpp>` include dropped; `cublasGemmEx` is called
//     directly (the upstream `cublasGemmEx_org` was the LD_PRELOAD
//     indirection that we removed — its single call site is now a
//     thin inline wrapper for source-compat with `handle.cu`'s
//     test scaffolding).
//   - Profiler `start_timer_sync` / `stop_timer_sync` calls removed
//     (no profiler in this build path — see `baracuda-kernels-bench`).
//   - `cutf::experimental::fp::` → `baracuda::fp::` (one site:
//     `accumulate_in_f64` scale computation).
//   - Removed the unused `to_cudaDataType_t` and `split_core`
//     placeholder helpers — the latter contained a single
//     `OZIMMU_NOT_IMPLEMENTED` and was reachable from no caller.

#include "config.hpp"
#include "handle.hpp"
#include "split.hpp"
#include "utils.hpp"

#include <cublas_v2.h>
#include <cmath>    // std::scalbn
#include <utility>  // std::pair
#include <vector>

#include "baracuda_fp_bits.cuh"

// Phase 44c — RIKEN-RCCS perf-enhancement variants.
//
// `OZIMMU_VARIANT_*` integer codes used by the variant dispatcher.
// Held inside this TU rather than the public header to keep the
// upstream `mtk::ozimmu::` ABI source-compatible with Phase 44b.
namespace {
constexpr int OZIMMU_VARIANT_BASE = 0;
constexpr int OZIMMU_VARIANT_EF   = 1;
constexpr int OZIMMU_VARIANT_RN   = 2;
constexpr int OZIMMU_VARIANT_H    = 3;
}  // namespace

namespace {

template <class T>
void split_AB_int8(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const T *const a_ptr,
    const std::size_t lda, double *const a_max_exp_ptr,
    std::int8_t *const working_a_ptr, const std::uint32_t ld_int8_a,
    const T *const b_ptr, const std::size_t ldb, double *const b_max_exp_ptr,
    std::int8_t *const working_b_ptr, const std::uint32_t ld_int8_b,
    const unsigned num_split, const unsigned bits_per_int8) {
    mtk::ozimmu::split_int8<T>(working_a_ptr, ld_int8_a, a_max_exp_ptr, m, k,
                               a_ptr, lda, op_A, mtk::ozimmu::detail::matrix_A,
                               num_split, bits_per_int8, handle->cuda_stream);
    mtk::ozimmu::split_int8<T>(working_b_ptr, ld_int8_b, b_max_exp_ptr, k, n,
                               b_ptr, ldb, op_B, mtk::ozimmu::detail::matrix_B,
                               num_split, bits_per_int8, handle->cuda_stream);
}

cublasOperation_t to_cublasOperation_t(const mtk::ozimmu::operation_t op) {
    switch (op) {
    case mtk::ozimmu::op_n: return CUBLAS_OP_N;
    case mtk::ozimmu::op_t: return CUBLAS_OP_T;
    default:
        break;
    }
    OZIMMU_NOT_IMPLEMENTED;
    return CUBLAS_OP_N;
}

__global__ void accumulate_in_f64_kernel(double *const f64_ptr,
                                         const std::int32_t *i32_ptr,
                                         const std::size_t length,
                                         const double scale) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length) {
        return;
    }

    f64_ptr[tid] +=
        static_cast<double>(static_cast<std::int64_t>(i32_ptr[tid]) << 32) * scale;
}

void accumulate_in_f64(double *const f64_ptr, const std::int32_t *i32_ptr,
                       const std::size_t length,
                       const std::int32_t mantissa_rshift,
                       cudaStream_t cuda_stream) {
    constexpr std::size_t block_size = 256;
    const auto scale = baracuda::fp::reinterpret_as_fp(
        static_cast<std::uint64_t>(
            (baracuda::fp::get_bias<double>() - mantissa_rshift))
        << baracuda::fp::get_mantissa_size<double>());
    accumulate_in_f64_kernel<<<(length + block_size - 1) / block_size, block_size,
                               0, cuda_stream>>>(f64_ptr, i32_ptr, length, scale);
}

template <class T>
__global__ void init_accumulator_buffer_kernel(T *const dp_ptr,
                                               const std::size_t length) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length) {
        return;
    }
    dp_ptr[tid] = 0;
}

template <class T>
void init_accumulator_buffer(T *const dp_ptr, const std::size_t length,
                             cudaStream_t cuda_stream) {
    constexpr std::size_t block_size = 256;
    init_accumulator_buffer_kernel<T>
        <<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
            dp_ptr, length);
}

__global__ void axby_kernel(const std::size_t m, const std::size_t n,
                            const double a, const double *const x_ptr,
                            const double b, double *const y_ptr,
                            const std::size_t ldy,
                            const double *const a_max_exp_ptr,
                            const double *const b_max_exp_ptr) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * n) {
        return;
    }

    const auto mi = tid % m;
    const auto ni = tid / m;

    const auto memory_index = ni * ldy + mi;

    // Phase 44c bugfix: upstream's `(1l << 44)` is UB on MSVC/Windows
    // where `long` is 32-bit (LLP64). Use `1ll` (long long, 64-bit on
    // every platform we target) or just an FP literal. Same fix
    // applied to `axy_complex_kernel` below.
    const auto x =
        x_ptr[tid] / static_cast<double>(1ull << 44)
        * a_max_exp_ptr[mi] * b_max_exp_ptr[ni];

    if (b != 0) {
        y_ptr[memory_index] = a * x + b * y_ptr[memory_index];
    } else {
        y_ptr[memory_index] = a * x;
    }
}

void axby(const std::size_t m, const std::size_t n, const double a,
          const double *const x_ptr, const double b, double *const y_ptr,
          const std::size_t ldy, const double *const a_max_exp_ptr,
          const double *const b_max_exp_ptr, cudaStream_t cuda_stream) {
    constexpr std::size_t block_size = 256;
    axby_kernel<<<(m * n + block_size - 1) / block_size, block_size, 0,
                  cuda_stream>>>(m, n, a, x_ptr, b, y_ptr, ldy, a_max_exp_ptr,
                                 b_max_exp_ptr);
}

__global__ void axy_complex_kernel(const std::size_t m, const std::size_t n,
                                   const cuDoubleComplex a,
                                   const double *const x_ptr,
                                   cuDoubleComplex *const y_ptr,
                                   const std::size_t ldy,
                                   const double *const a_max_exp_ptr,
                                   const double *const b_max_exp_ptr) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * n) {
        return;
    }

    const auto mi = tid % m;
    const auto ni = tid / m;

    const auto memory_index = ni * ldy + mi;

    // Same MSVC/Windows `1l << 44` bug as in axby_kernel — fixed here too.
    const auto x =
        x_ptr[tid] / static_cast<double>(1ull << 44)
        * a_max_exp_ptr[mi] * b_max_exp_ptr[ni];

    auto y = y_ptr[memory_index];

    y.x = a.x * x + y.x;
    y.y = a.y * x + y.y;

    y_ptr[memory_index] = y;
}

void axy_complex(const std::size_t m, const std::size_t n,
                 const cuDoubleComplex a, const double *const x_ptr,
                 cuDoubleComplex *const y_ptr, const std::size_t ldy,
                 const double *const a_max_exp_ptr,
                 const double *const b_max_exp_ptr, cudaStream_t cuda_stream) {
    constexpr std::size_t block_size = 256;
    axy_complex_kernel<<<(m * n + block_size - 1) / block_size, block_size, 0,
                         cuda_stream>>>(m, n, a, x_ptr, y_ptr, ldy,
                                        a_max_exp_ptr, b_max_exp_ptr);
}

template <bool IsBetaZero>
__global__ void init_c_complex_kernel(const std::size_t m, const std::size_t n,
                                      cuDoubleComplex *const c_ptr,
                                      const std::size_t ldc,
                                      const cuDoubleComplex beta) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * n) {
        return;
    }
    const auto mi = tid % m;
    const auto ni = tid / m;
    const auto memory_index = ni * ldc + mi;

    if (IsBetaZero) {
        c_ptr[memory_index] = make_cuDoubleComplex(0, 0);
    } else {
        auto c = c_ptr[memory_index];
        c.x = c.x * beta.x - c.y * beta.y;
        c.y = c.y * beta.x + c.x * beta.y;

        c_ptr[memory_index] = c;
    }
}

void init_c_complex(const std::size_t m, const std::size_t n,
                    cuDoubleComplex *const c_ptr, const std::size_t ldc,
                    const cuDoubleComplex beta, cudaStream_t cuda_stream) {
    constexpr std::size_t block_size = 256;
    if (beta.x == 0 && beta.y == 0) {
        init_c_complex_kernel<true>
            <<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
                m, n, c_ptr, ldc, beta);
    } else {
        init_c_complex_kernel<false>
            <<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
                m, n, c_ptr, ldc, beta);
    }
}

/// Core int8 GEMM dispatcher.
///
/// Phase 44c adds:
///   - `beta_i_in` — replaces the previously-hardcoded `beta_i = 0`
///     so the EF / H variants can chain int32 partial sums in-register
///     (cublas accumulates `D = alpha*AB + beta*D` in int32; passing
///     `beta_i = 1` reuses the existing `c_ptr_r` as the running sum
///     and skips the `accumulate_in_f64` materialization between
///     consecutive slice pairs of the same group).
///   - n-blocking (`n > 12288 → split into 8192-wide chunks`). Ports
///     RIKEN-RCCS/accelerator_for_ozIMMU `acc/gemm.cu`. cuBLAS' int8
///     GEMM has a non-linear-throughput cliff around `n = 12288` on
///     consumer Ada; the chunked launch keeps the GPU at the linear
///     band and improves end-to-end perf at large N.
///
/// The thresholds (8192 chunk, 12288 single-launch ceiling) are
/// upstream's; the ozIMMU perf paper documents them empirically on
/// H100. They reproduce on RTX 4070 sm_89 too — the int8 tensor-core
/// path on cuBLAS hits the same scheduling wall.
void matmul_core(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const void *const a_ptr,
    const std::size_t lda, const mtk::ozimmu::data_t /*type_a*/,
    const void *const b_ptr, const std::size_t ldb,
    const mtk::ozimmu::data_t /*type_b*/, const int beta_i_in,
    void *const c_ptr,
    const mtk::ozimmu::detail::gemm_pair_config_t &gemm_pair_config,
    const mtk::ozimmu::compute_mode_t compute_mode,
    const void *const a_working_memory_ptr, const std::size_t ld_w_a,
    const void *const b_working_memory_ptr, const std::size_t ld_w_b) {
    const auto gemm_mode = gemm_pair_config.gemm_mode;
    const auto split_config = mtk::ozimmu::detail::get_split_config(compute_mode);
    const auto lda_r = gemm_pair_config.A_id == 0 ? lda : ld_w_a;
    const auto ldb_r = gemm_pair_config.B_id == 0 ? ldb : ld_w_b;

    const auto num_int8_a_slice_elements = ld_w_a * m;
    const auto num_int8_b_slice_elements = ld_w_b * n;

    std::size_t A_working_ptr_offset = 0;
    for (unsigned i = 0; i < static_cast<unsigned>(gemm_pair_config.A_id); i++) {
        const auto t = split_config.matrix_A_split_types[i];
        A_working_ptr_offset +=
            num_int8_a_slice_elements * mtk::ozimmu::get_data_size_in_byte(t);
    }

    std::size_t B_working_ptr_offset = 0;
    for (unsigned i = 0; i < static_cast<unsigned>(gemm_pair_config.B_id); i++) {
        const auto t = split_config.matrix_B_split_types[i];
        B_working_ptr_offset +=
            num_int8_b_slice_elements * mtk::ozimmu::get_data_size_in_byte(t);
    }

    const void *const a_working_ptr =
        reinterpret_cast<const std::uint8_t *>(a_working_memory_ptr) + A_working_ptr_offset;
    const void *const b_working_ptr =
        reinterpret_cast<const std::uint8_t *>(b_working_memory_ptr) + B_working_ptr_offset;

    const void *const a_ptr_r = gemm_pair_config.A_id == 0 ? a_ptr : a_working_ptr;
    const void *const b_ptr_r = gemm_pair_config.B_id == 0 ? b_ptr : b_working_ptr;
    void *const c_ptr_r = c_ptr;

    switch (gemm_mode) {
    case mtk::ozimmu::detail::int8tc: {
        const int alpha_i = 1;
        const int beta_i_const = beta_i_in;
        const auto op_A_r =
            gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
        const auto op_B_r =
            gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;

        // n-blocking: chunk n into 8192-wide pieces if it exceeds the
        // 12288 single-launch ceiling. Below 12288 we issue a single
        // cublasGemmEx call equivalent to the Phase 44b path.
        //
        // Phase 44b parity guard: when n <= 12288 (the single-launch
        // case), the chunk loop reduces to one call with offset=0
        // and pointer arithmetic that is a no-op (offset * ldb_r = 0).
        // We special-case it to make that obvious — and to avoid
        // accidental signed/unsigned divergence in the pointer-cast
        // arithmetic that bit me on Windows MSVC + CUDA 13 (release
        // mode wave5 redux pattern).
        if (n <= 12288) {
            cublasGemmEx(handle->cublas_handle, op_A_r, op_B_r, m, n, k,
                         &alpha_i, a_ptr_r, CUDA_R_8I, lda_r, b_ptr_r,
                         CUDA_R_8I, ldb_r, &beta_i_const, c_ptr_r,
                         CUDA_R_32I, m, CUBLAS_COMPUTE_32I,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            const std::size_t blk = 8192;
            std::size_t rem       = n;
            std::size_t offset    = 0;
            while (rem > 0) {
                const std::size_t nn = (rem <= 12288) ? rem : blk;
                const auto b_chunk =
                    reinterpret_cast<const std::int8_t *>(b_ptr_r)
                    + offset * ldb_r;
                auto c_chunk = reinterpret_cast<std::int32_t *>(c_ptr_r)
                             + offset * m;
                cublasGemmEx(handle->cublas_handle, op_A_r, op_B_r, m, nn, k,
                             &alpha_i, a_ptr_r, CUDA_R_8I, lda_r, b_chunk,
                             CUDA_R_8I, ldb_r, &beta_i_const, c_chunk,
                             CUDA_R_32I, m, CUBLAS_COMPUTE_32I,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                offset += nn;
                rem -= nn;
            }
        }
    } break;
    default:
        OZIMMU_NOT_IMPLEMENTED;
    }
}

/// Back-compat overload — Phase 44b call sites pass no `beta_i`.
/// Forwards with `beta_i = 0` (the original behaviour).
void matmul_core(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const void *const a_ptr,
    const std::size_t lda, const mtk::ozimmu::data_t type_a,
    const void *const b_ptr, const std::size_t ldb,
    const mtk::ozimmu::data_t type_b, void *const c_ptr,
    const mtk::ozimmu::detail::gemm_pair_config_t &gemm_pair_config,
    const mtk::ozimmu::compute_mode_t compute_mode,
    const void *const a_working_memory_ptr, const std::size_t ld_w_a,
    const void *const b_working_memory_ptr, const std::size_t ld_w_b) {
    matmul_core(handle, op_A, op_B, m, n, k, a_ptr, lda, type_a, b_ptr, ldb,
                type_b, /*beta_i_in=*/0, c_ptr, gemm_pair_config, compute_mode,
                a_working_memory_ptr, ld_w_a, b_working_memory_ptr, ld_w_b);
}

// ===========================================================================
// Phase 44c — nearest-split accumulator + axby.
// ===========================================================================

/// H-variant accumulator. Multiplies per-row scales `sft_a[mi]` and
/// `sft_b[ni]` against the int32 partial-sum cell and folds in a
/// per-slice-pair significance factor `scale = 2^(-bits*(A_id+B_id-2))`.
__global__ void accumulate_in_f64_kernel_2(
    const std::size_t m, double *const f64_ptr,
    const std::int32_t *i32_ptr, const std::size_t length,
    const double *const sft_a, const double *const sft_b,
    const double scale) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length) {
        return;
    }
    const auto mi = tid % m;
    const auto ni = tid / m;
    f64_ptr[tid] +=
        static_cast<double>(i32_ptr[tid]) * sft_a[mi] * sft_b[ni] * scale;
}

void accumulate_in_f64_2(const std::size_t m, double *const f64_ptr,
                         const std::int32_t *i32_ptr, const std::size_t length,
                         const double *const sft_a, const double *const sft_b,
                         const double scale, cudaStream_t cuda_stream) {
    constexpr std::size_t block_size = 256;
    accumulate_in_f64_kernel_2<<<(length + block_size - 1) / block_size,
                                 block_size, 0, cuda_stream>>>(
        m, f64_ptr, i32_ptr, length, sft_a, sft_b, scale);
}

/// Nearest-split axby. Simpler than the base path — the per-row
/// re-scale is already baked into the f64 accumulator by
/// `accumulate_in_f64_2`, so the final copy only applies the user
/// alpha / beta and the LD remap.
__global__ void axby_kernel_2(const std::size_t m, const std::size_t n,
                              const double a, const double *const x_ptr,
                              const double b, double *const y_ptr,
                              const std::size_t ldy) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * n) {
        return;
    }
    const auto mi           = tid % m;
    const auto ni           = tid / m;
    const auto memory_index = ni * ldy + mi;

    if (b != 0) {
        y_ptr[memory_index] = a * x_ptr[tid] + b * y_ptr[memory_index];
    } else {
        y_ptr[memory_index] = a * x_ptr[tid];
    }
}

void axby_2(const std::size_t m, const std::size_t n, const double a,
            const double *const x_ptr, const double b, double *const y_ptr,
            const std::size_t ldy, cudaStream_t cuda_stream) {
    constexpr std::size_t block_size = 256;
    axby_kernel_2<<<(m * n + block_size - 1) / block_size, block_size, 0,
                    cuda_stream>>>(m, n, a, x_ptr, b, y_ptr, ldy);
}

template <class T>
int gemm_int8(mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
              const mtk::ozimmu::operation_t op_B, const std::size_t m,
              const std::size_t n, const std::size_t k, const T *alpha,
              const T *const a_ptr, const std::size_t lda, const T *const b_ptr,
              const std::size_t ldb, const T *beta, T *const c_ptr,
              std::size_t ldc, const mtk::ozimmu::compute_mode_t compute_mode);

template <>
int gemm_int8<double>(mtk::ozimmu::handle_t handle,
                      const mtk::ozimmu::operation_t op_A,
                      const mtk::ozimmu::operation_t op_B,
                      const std::size_t m, const std::size_t n,
                      const std::size_t k, const double *alpha,
                      const double *const a_ptr, const std::size_t lda,
                      const double *const b_ptr, const std::size_t ldb,
                      const double *beta, double *const c_ptr, std::size_t ldc,
                      const mtk::ozimmu::compute_mode_t compute_mode) {
    const unsigned num_split =
        mtk::ozimmu::detail::get_split_config(compute_mode).matrix_A_split_types.size() - 1;
    const int32_t bits_per_int8 = mtk::ozimmu::get_bits_per_int8(k);

    double *const c_f64_ptr = reinterpret_cast<double *>(handle->working_memory_ptr);
    double *const a_max_exp_ptr = c_f64_ptr + m * n;
    double *const b_max_exp_ptr = a_max_exp_ptr + m;
    std::int32_t *const c_i32_ptr = reinterpret_cast<int32_t *>(b_max_exp_ptr + n);
    void *const working_memory_ptr = c_i32_ptr + m * n;

    init_accumulator_buffer(c_f64_ptr, m * n, handle->cuda_stream);

    const auto ld_int8_a =
        mtk::ozimmu::get_slice_ld<std::int8_t>(m, k, mtk::ozimmu::op_t);
    const auto ld_int8_b =
        mtk::ozimmu::get_slice_ld<std::int8_t>(k, n, mtk::ozimmu::op_n);

    const auto num_int8_a_slice_elements =
        mtk::ozimmu::get_slice_num_elements<std::int8_t>(m, k, mtk::ozimmu::op_t);
    std::size_t A_working_memory_size = num_int8_a_slice_elements * num_split;

    auto a_int8_slices_ptr = reinterpret_cast<std::int8_t *>(working_memory_ptr);
    auto b_int8_slices_ptr = a_int8_slices_ptr + A_working_memory_size;

    split_AB_int8<double>(handle, op_A, op_B, m, n, k, a_ptr, lda, a_max_exp_ptr,
                          a_int8_slices_ptr, ld_int8_a, b_ptr, ldb, b_max_exp_ptr,
                          b_int8_slices_ptr, ld_int8_b, num_split, bits_per_int8);

    const auto &gemm_pair_config_list =
        mtk::ozimmu::detail::get_split_config(compute_mode).gemm_pair_config_list;
    for (const auto &gemm_pair_config : gemm_pair_config_list) {
        matmul_core(handle, op_A, op_B, m, n,
                    ld_int8_a, // use ld_int8_a instead of k for better stability
                    a_ptr, lda, mtk::ozimmu::fp64, b_ptr, ldb, mtk::ozimmu::fp64,
                    c_i32_ptr, gemm_pair_config, compute_mode, a_int8_slices_ptr,
                    ld_int8_a, b_int8_slices_ptr, ld_int8_b);
        // The `(7 - bits_per_int8) * 2` correction is needed because the
        // mantissa `bits_per_int8` bits are stored in the LOW bits of the
        // int8 slice (paper §3.2).
        accumulate_in_f64(
            c_f64_ptr, c_i32_ptr, m * n,
            bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2)
                - (7 - bits_per_int8) * 2,
            handle->cuda_stream);
    }
    axby(m, n, *alpha, c_f64_ptr, *beta, c_ptr, ldc, a_max_exp_ptr, b_max_exp_ptr,
         handle->cuda_stream);

    return 0;
}

// ===========================================================================
// Phase 44c — EF / RN / H variant dispatch for the FP64 real path.
//
// Each variant differs from the Base path in either the splitter
// (RN, H replace `split_int8` with `split_int8_nearest`) or the
// accumulator (EF, H delay the int32 → f64 conversion by chaining
// cublasGemmEx calls with `beta_i = 1`).
//
// Common control parameter:
//   `lim_accum` — how many consecutive int32 GEMMs can chain
//   before risking overflow in the int32 accumulator. Derived
//   from `31 - 2*bits - ceil(log2(k))` per the upstream's bit-budget
//   analysis (each int8*int8 fits in 15 bits, k accumulations add
//   ceil(log2(k)) bits, leaving `31 - 2*bits - log2(k)` headroom).
//   `lim_accum > 0` enables grouping; `0` falls back to per-pair
//   accumulation (saves nothing but the variant code path is
//   still exercised and produces bit-identical output to Base + RN
//   for the EF / H cases).
// ===========================================================================
int gemm_int8_double_variant(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const double *alpha,
    const double *const a_ptr, const std::size_t lda,
    const double *const b_ptr, const std::size_t ldb, const double *beta,
    double *const c_ptr, std::size_t ldc,
    const mtk::ozimmu::compute_mode_t compute_mode, const int variant) {
    const bool use_nearest_split =
        (variant == OZIMMU_VARIANT_RN) || (variant == OZIMMU_VARIANT_H);
    const bool use_errfree_sum =
        (variant == OZIMMU_VARIANT_EF) || (variant == OZIMMU_VARIANT_H);

    const unsigned num_split =
        mtk::ozimmu::detail::get_split_config(compute_mode)
            .matrix_A_split_types.size()
        - 1;
    const std::int32_t bits_per_int8 = mtk::ozimmu::get_bits_per_int8(k);

    // Working-memory carve-up. The nearest-split path stores ONE
    // f64 scale per row (one element per `m` for A, one per `n` for
    // B) — same shape as Base's per-row `max_exp` arrays — so the
    // layout below works for all variants.
    double *const c_f64_ptr =
        reinterpret_cast<double *>(handle->working_memory_ptr);
    double *const a_scale_ptr = c_f64_ptr + m * n;
    double *const b_scale_ptr = a_scale_ptr + m;
    std::int32_t *const c_i32_ptr =
        reinterpret_cast<std::int32_t *>(b_scale_ptr + n);
    void *const working_memory_ptr = c_i32_ptr + m * n;

    init_accumulator_buffer(c_f64_ptr, m * n, handle->cuda_stream);

    const auto ld_int8_a =
        mtk::ozimmu::get_slice_ld<std::int8_t>(m, k, mtk::ozimmu::op_t);
    const auto ld_int8_b =
        mtk::ozimmu::get_slice_ld<std::int8_t>(k, n, mtk::ozimmu::op_n);

    const auto num_int8_a_slice_elements =
        mtk::ozimmu::get_slice_num_elements<std::int8_t>(m, k, mtk::ozimmu::op_t);
    const std::size_t A_working_memory_size =
        num_int8_a_slice_elements * num_split;

    auto a_int8_slices_ptr =
        reinterpret_cast<std::int8_t *>(working_memory_ptr);
    auto b_int8_slices_ptr = a_int8_slices_ptr + A_working_memory_size;

    // ---- Split phase ------------------------------------------------------
    if (use_nearest_split) {
        // H / RN — nearest-rounding split. The per-row scale stored
        // in `a_scale_ptr` / `b_scale_ptr` is `1/s` from the
        // extractor; the accumulator does the multiply-with-cell.
        mtk::ozimmu::split_int8_nearest(
            a_int8_slices_ptr, ld_int8_a, a_scale_ptr, m, k, a_ptr, lda,
            op_A, mtk::ozimmu::detail::matrix_A,
            static_cast<std::int8_t>(num_split),
            static_cast<std::int8_t>(bits_per_int8), handle->cuda_stream);
        mtk::ozimmu::split_int8_nearest(
            b_int8_slices_ptr, ld_int8_b, b_scale_ptr, k, n, b_ptr, ldb,
            op_B, mtk::ozimmu::detail::matrix_B,
            static_cast<std::int8_t>(num_split),
            static_cast<std::int8_t>(bits_per_int8), handle->cuda_stream);
    } else {
        // Base / EF — signed split, per-row max-exponent.
        split_AB_int8<double>(handle, op_A, op_B, m, n, k, a_ptr, lda,
                              a_scale_ptr, a_int8_slices_ptr, ld_int8_a,
                              b_ptr, ldb, b_scale_ptr, b_int8_slices_ptr,
                              ld_int8_b, num_split, bits_per_int8);
    }

    // ---- int8 GEMMs + accumulation ---------------------------------------
    // `lim_accum` budget — see the function-header comment for the
    // derivation. We compute it the same way for all variants; only
    // EF / H actually consult it (Base / RN flush after every pair).
    int nextpow2_k = 0;
    if (k > 0) {
        std::uint32_t kk = static_cast<std::uint32_t>(k);
        while ((1u << nextpow2_k) < kk) {
            nextpow2_k++;
        }
    }
    int lim_accum_bits = 31 - bits_per_int8 - bits_per_int8 - nextpow2_k;
    if (lim_accum_bits < 0) lim_accum_bits = 0;

    const auto &gemm_pair_config_list =
        mtk::ozimmu::detail::get_split_config(compute_mode).gemm_pair_config_list;

    // Closure picks the right accumulator based on splitter.
    auto do_accumulate =
        [&](const mtk::ozimmu::detail::gemm_pair_config_t &gpc) {
            if (use_nearest_split) {
                const double scale = std::scalbn(
                    1.0, -bits_per_int8 * (gpc.A_id + gpc.B_id - 2));
                accumulate_in_f64_2(m, c_f64_ptr, c_i32_ptr, m * n,
                                    a_scale_ptr, b_scale_ptr, scale,
                                    handle->cuda_stream);
            } else {
                accumulate_in_f64(
                    c_f64_ptr, c_i32_ptr, m * n,
                    bits_per_int8 * (gpc.A_id + gpc.B_id - 2)
                        - (7 - bits_per_int8) * 2,
                    handle->cuda_stream);
            }
        };

    if (!use_errfree_sum || lim_accum_bits == 0) {
        // Base / RN — accumulate after every int8 GEMM.
        for (const auto &gpc : gemm_pair_config_list) {
            matmul_core(handle, op_A, op_B, m, n, ld_int8_a, a_ptr, lda,
                        mtk::ozimmu::fp64, b_ptr, ldb, mtk::ozimmu::fp64,
                        /*beta_i=*/0, c_i32_ptr, gpc, compute_mode,
                        a_int8_slices_ptr, ld_int8_a, b_int8_slices_ptr,
                        ld_int8_b);
            do_accumulate(gpc);
        }
    } else {
        // EF / H — group-wise error-free summation. Chain
        // consecutive int8 GEMMs into the same int32 accumulator via
        // `beta_i = 1`; flush to f64 once at the end of each group
        // (group size = `2^lim_accum_bits`).
        const int lim_accum = 1 << lim_accum_bits;
        int beta_i          = 0;
        int p               = -1;
        for (const auto &gpc : gemm_pair_config_list) {
            if (gpc.A_id == 1) p++;
            if ((gpc.A_id - 1) % lim_accum == 0) beta_i = 0;

            matmul_core(handle, op_A, op_B, m, n, ld_int8_a, a_ptr, lda,
                        mtk::ozimmu::fp64, b_ptr, ldb, mtk::ozimmu::fp64,
                        /*beta_i=*/beta_i, c_i32_ptr, gpc, compute_mode,
                        a_int8_slices_ptr, ld_int8_a, b_int8_slices_ptr,
                        ld_int8_b);
            beta_i = 1;

            const bool last_in_row = (gpc.A_id - 1) == p;
            const bool at_lim_bdry =
                (gpc.A_id % lim_accum == 0) && (gpc.A_id > 1);
            if (last_in_row || at_lim_bdry) {
                do_accumulate(gpc);
            }
        }
    }

    // ---- Final copy to caller-C ------------------------------------------
    if (use_nearest_split) {
        // Per-row scale is already baked in by accumulate_in_f64_2.
        axby_2(m, n, *alpha, c_f64_ptr, *beta, c_ptr, ldc,
               handle->cuda_stream);
    } else {
        // Base / EF — `axby` does the `/2^44 * max_exp_a * max_exp_b` re-scale.
        axby(m, n, *alpha, c_f64_ptr, *beta, c_ptr, ldc, a_scale_ptr,
             b_scale_ptr, handle->cuda_stream);
    }
    return 0;
}

template <>
int gemm_int8<cuDoubleComplex>(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *const a_ptr, const std::size_t lda,
    const cuDoubleComplex *const b_ptr, const std::size_t ldb,
    const cuDoubleComplex *beta, cuDoubleComplex *const c_ptr, std::size_t ldc,
    const mtk::ozimmu::compute_mode_t compute_mode) {
    using real_t = double;
    const unsigned num_split =
        mtk::ozimmu::detail::get_split_config(compute_mode).matrix_A_split_types.size() - 1;
    const int32_t bits_per_int8 = mtk::ozimmu::get_bits_per_int8(k);
    const auto &gemm_pair_config_list =
        mtk::ozimmu::detail::get_split_config(compute_mode).gemm_pair_config_list;

    const auto ld_int8_a =
        mtk::ozimmu::get_slice_ld<std::int8_t>(m, k, mtk::ozimmu::op_t);
    const auto ld_int8_b =
        mtk::ozimmu::get_slice_ld<std::int8_t>(k, n, mtk::ozimmu::op_n);

    const auto num_int8_a_slice_elements =
        mtk::ozimmu::get_slice_num_elements<std::int8_t>(m, k, mtk::ozimmu::op_t);
    const auto num_int8_b_slice_elements =
        mtk::ozimmu::get_slice_num_elements<std::int8_t>(k, n, mtk::ozimmu::op_n);

    const std::size_t A_working_memory_size = num_int8_a_slice_elements * num_split;
    const std::size_t B_working_memory_size = num_int8_b_slice_elements * num_split;

    double *const tmp_f64_ptr = reinterpret_cast<double *>(handle->working_memory_ptr);
    double *const a_real_max_exp_ptr = tmp_f64_ptr + m * n;
    double *const a_imag_max_exp_ptr = a_real_max_exp_ptr + m;
    double *const b_real_max_exp_ptr = a_imag_max_exp_ptr + m;
    double *const b_imag_max_exp_ptr = b_real_max_exp_ptr + n;
    std::int32_t *const c_i32_ptr =
        reinterpret_cast<std::int32_t *>(b_imag_max_exp_ptr + n);
    void *const working_memory_ptr = c_i32_ptr + m * n;

    const double *a_max_exp_ptr_list[] = {a_real_max_exp_ptr, a_imag_max_exp_ptr};
    const std::int8_t *a_int8_working_memory_ptr_list[] = {
        reinterpret_cast<const std::int8_t *>(working_memory_ptr),
        reinterpret_cast<const std::int8_t *>(working_memory_ptr) + A_working_memory_size,
    };

    const double *b_max_exp_ptr_list[] = {b_real_max_exp_ptr, b_imag_max_exp_ptr};
    const std::int8_t *b_int8_working_memory_ptr_list[] = {
        a_int8_working_memory_ptr_list[0] + A_working_memory_size * 2,
        a_int8_working_memory_ptr_list[0] + A_working_memory_size * 2 + B_working_memory_size,
    };

    split_AB_int8<cuDoubleComplex>(
        handle, op_A, op_B, m, n, k, a_ptr, lda, a_real_max_exp_ptr,
        reinterpret_cast<std::int8_t *>(working_memory_ptr), ld_int8_a, b_ptr,
        ldb, b_real_max_exp_ptr,
        reinterpret_cast<std::int8_t *>(working_memory_ptr) + A_working_memory_size * 2,
        ld_int8_b, num_split, bits_per_int8);

    init_c_complex(m, n, c_ptr, ldc, *beta, handle->cuda_stream);

    for (const auto p : std::vector<std::pair<unsigned, unsigned>>{
             {1, 1}, {0, 0}, {1, 0}, {0, 1}}) {
        init_accumulator_buffer(tmp_f64_ptr, m * n, handle->cuda_stream);
        for (const auto &gemm_pair_config : gemm_pair_config_list) {
            matmul_core(handle, mtk::ozimmu::op_t, mtk::ozimmu::op_n, m, n,
                        ld_int8_a, // use ld_int8_a instead of k for better stability
                        a_ptr, lda, mtk::ozimmu::fp64, b_ptr, ldb, mtk::ozimmu::fp64,
                        c_i32_ptr, gemm_pair_config, compute_mode,
                        a_int8_working_memory_ptr_list[p.first], ld_int8_a,
                        b_int8_working_memory_ptr_list[p.second], ld_int8_b);
            accumulate_in_f64(
                tmp_f64_ptr, c_i32_ptr, m * n,
                bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2)
                    - (7 - bits_per_int8) * 2,
                handle->cuda_stream);
        }

        real_t axpy_alpha_real = 0;
        real_t axpy_alpha_imag = 0;
        if (p.first == 0 && p.second == 0) {
            axpy_alpha_real =  alpha->x;
            axpy_alpha_imag =  alpha->y;
        } else if (p.first == 1 && p.second == 1) {
            axpy_alpha_real = -alpha->x;
            axpy_alpha_imag = -alpha->y;
        } else {
            axpy_alpha_real = -alpha->y;
            axpy_alpha_imag =  alpha->x;
        }
        axy_complex(m, n, make_cuDoubleComplex(axpy_alpha_real, axpy_alpha_imag),
                    tmp_f64_ptr, c_ptr, ldc, a_max_exp_ptr_list[p.first],
                    b_max_exp_ptr_list[p.second], handle->cuda_stream);
    }

    return 0;
}

}  // namespace

/// Phase 44c — variant-aware dgemm entry point.
///
/// Routes to `gemm_int8_double_variant` for the FP64 real path with
/// the caller-supplied `variant` flag (0 = Base, 1 = EF, 2 = RN,
/// 3 = H). Complex / non-int8 compute_modes ignore `variant` and
/// fall back to the legacy path. Validation matches
/// `mtk::ozimmu::gemm` exactly.
///
/// Not in the `mtk::ozimmu::` namespace because the upstream header
/// is intentionally untouched in Phase 44c (Phase 44b clean-fork
/// rule: the C++ API surface stays drop-in with the original ozIMMU
/// header so downstream users can grep for `mtk::ozimmu::` symbols
/// without surprises).
int baracuda_ozimmu_gemm_double_variant_impl(
    mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
    const mtk::ozimmu::operation_t op_B, const std::size_t m,
    const std::size_t n, const std::size_t k, const double *alpha,
    const double *const a_ptr, const std::size_t lda,
    const double *const b_ptr, const std::size_t ldb, const double *beta,
    double *const c_ptr, std::size_t ldc,
    const mtk::ozimmu::compute_mode_t compute_mode, const int variant) {
    // Argument validation — same shape as `mtk::ozimmu::gemm`.
    int arg_error = 0;
    arg_error |= check_gemm_shape(op_A, m, k, lda, "A");
    arg_error |= check_gemm_shape(op_B, k, n, ldb, "B");
    arg_error |= check_gemm_shape(mtk::ozimmu::op_n, m, n, ldc, "C");
    arg_error |= check_address_alignment<double>(a_ptr, "A");
    arg_error |= check_address_alignment<double>(b_ptr, "B");
    arg_error |= check_address_alignment<double>(c_ptr, "C");
    if (arg_error) {
        return 1;
    }

    // Reallocate working memory if needed.
    mtk::ozimmu::gemm_list_t gemm_list = {mtk::ozimmu::gemm_params_t{
        op_A, op_B, m, n, k, mtk::ozimmu::real, compute_mode}};
    mtk::ozimmu::reallocate_working_memory(handle, gemm_list);

    if (compute_mode >= mtk::ozimmu::fp64_int8_3
        && compute_mode <= mtk::ozimmu::fp64_int8_18) {
        return gemm_int8_double_variant(handle, op_A, op_B, m, n, k, alpha,
                                        a_ptr, lda, b_ptr, ldb, beta, c_ptr,
                                        ldc, compute_mode, variant);
    } else if (compute_mode == mtk::ozimmu::fp64_int8_auto) {
        const auto auto_mode = mtk::ozimmu::auto_mode_select(
            handle, op_A, op_B, m, n, k, a_ptr, lda, b_ptr, ldb,
            mtk::ozimmu::real, handle->avg_mantissa_loss_threshold);
        return baracuda_ozimmu_gemm_double_variant_impl(
            handle, op_A, op_B, m, n, k, alpha, a_ptr, lda, b_ptr, ldb,
            beta, c_ptr, ldc, auto_mode, variant);
    } else if (compute_mode == mtk::ozimmu::dgemm) {
        // Native cuBLAS DGEMM passthrough — variant is meaningless.
        cublasGemmEx(handle->cublas_handle, to_cublasOperation_t(op_A),
                     to_cublasOperation_t(op_B), m, n, k, alpha, a_ptr,
                     CUDA_R_64F, lda, b_ptr, CUDA_R_64F, ldb, beta, c_ptr,
                     CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F,
                     CUBLAS_GEMM_DEFAULT);
        return 0;
    }
    return 2;  // Unsupported compute_mode.
}

int mtk::ozimmu::gemm(mtk::ozimmu::handle_t handle,
                      const mtk::ozimmu::operation_t op_A,
                      const mtk::ozimmu::operation_t op_B,
                      const std::size_t m, const std::size_t n,
                      const std::size_t k,
                      const void *alpha,
                      const void *const a_ptr, const std::size_t lda,
                      const void *const b_ptr, const std::size_t ldb,
                      const void *beta,
                      void *const c_ptr, std::size_t ldc,
                      const mtk::ozimmu::compute_mode_t compute_mode,
                      const mtk::ozimmu::element_kind_t element_kind) {
    // Host-side argument validation.
    int arg_error = 0;
    arg_error |= check_gemm_shape(op_A, m, k, lda, "A");
    arg_error |= check_gemm_shape(op_B, k, n, ldb, "B");
    arg_error |= check_gemm_shape(mtk::ozimmu::op_n, m, n, ldc, "C");
    if (element_kind == mtk::ozimmu::real) {
        arg_error |= check_address_alignment<double>(
            reinterpret_cast<const double *>(a_ptr), "A");
        arg_error |= check_address_alignment<double>(
            reinterpret_cast<const double *>(b_ptr), "B");
        arg_error |= check_address_alignment<double>(
            reinterpret_cast<const double *>(c_ptr), "C");
    } else {
        arg_error |= check_address_alignment<cuDoubleComplex>(
            reinterpret_cast<const cuDoubleComplex *>(a_ptr), "A");
        arg_error |= check_address_alignment<cuDoubleComplex>(
            reinterpret_cast<const cuDoubleComplex *>(b_ptr), "B");
        arg_error |= check_address_alignment<cuDoubleComplex>(
            reinterpret_cast<const cuDoubleComplex *>(c_ptr), "C");
    }
    if (arg_error) {
        return 1;
    }

    mtk::ozimmu::data_t input_type;
    switch (compute_mode) {
    case mtk::ozimmu::sgemm:
        input_type = mtk::ozimmu::fp32;
        break;
    case mtk::ozimmu::dgemm:
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
    case mtk::ozimmu::fp64_int8_18:
    case mtk::ozimmu::fp64_int8_auto:
        input_type = mtk::ozimmu::fp64;
        break;
    default:
        OZIMMU_NOT_IMPLEMENTED;
    }

    gemm_list_t gemm_list = {
        gemm_params_t{op_A, op_B, m, n, k, element_kind, compute_mode}};
    mtk::ozimmu::reallocate_working_memory(handle, gemm_list);

    if (input_type == mtk::ozimmu::fp64) {
        if (compute_mode >= mtk::ozimmu::fp64_int8_3
            && compute_mode <= mtk::ozimmu::fp64_int8_18) {
            if (element_kind == mtk::ozimmu::real) {
                using T = double;
                gemm_int8(handle, op_A, op_B, m, n, k,
                          reinterpret_cast<const T *>(alpha),
                          reinterpret_cast<const T *>(a_ptr), lda,
                          reinterpret_cast<const T *>(b_ptr), ldb,
                          reinterpret_cast<const T *>(beta),
                          reinterpret_cast<T *>(c_ptr), ldc, compute_mode);
            } else {
                using T = cuDoubleComplex;
                gemm_int8(handle, op_A, op_B, m, n, k,
                          reinterpret_cast<const T *>(alpha),
                          reinterpret_cast<const T *>(a_ptr), lda,
                          reinterpret_cast<const T *>(b_ptr), ldb,
                          reinterpret_cast<const T *>(beta),
                          reinterpret_cast<T *>(c_ptr), ldc, compute_mode);
            }
        } else if (compute_mode == mtk::ozimmu::fp64_int8_auto) {
            const auto auto_mode = mtk::ozimmu::auto_mode_select(
                handle, op_A, op_B, m, n, k, a_ptr, lda, b_ptr, ldb, element_kind,
                handle->avg_mantissa_loss_threshold);
            ozimmu_log("AUTO selected mode = "
                       + mtk::ozimmu::get_compute_mode_name_str(auto_mode)
                       + ", threshold average mantissa loss = "
                       + std::to_string(handle->avg_mantissa_loss_threshold));
            return mtk::ozimmu::gemm(handle, op_A, op_B, m, n, k, alpha, a_ptr, lda,
                                     b_ptr, ldb, beta, c_ptr, ldc, auto_mode,
                                     element_kind);
        } else if (compute_mode == mtk::ozimmu::dgemm) {
            const auto dtype =
                element_kind == mtk::ozimmu::real ? CUDA_R_64F : CUDA_C_64F;
            cublasGemmEx(handle->cublas_handle, to_cublasOperation_t(op_A),
                         to_cublasOperation_t(op_B), m, n, k, alpha, a_ptr, dtype,
                         lda, b_ptr, dtype, ldb, beta, c_ptr, dtype, ldc,
                         CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        } else {
            OZIMMU_NOT_IMPLEMENTED;
        }
    } else {
        OZIMMU_NOT_IMPLEMENTED;
    }
    return 0;
}
