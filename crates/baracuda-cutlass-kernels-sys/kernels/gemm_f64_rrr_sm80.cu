// f64 (DGEMM via Ampere FP64 tensor cores), RRR layout, sm_80
// instantiation.
//
// Mirror of gemm_f64_rcr_sm80.cu with B in row-major rather than
// column-major. Full IEEE 754 binary64 throughout.
//
// Layout RRR:
//   A: row-major [M, K]
//   B: row-major [K, N]
//   C: row-major [M, N] (optional)
//   D: row-major [M, N]
//
// Tile shape matches the RCR variant: 128x128 threadblock, 32x64 warp,
// 8x8x4 instruction (DGEMM mma).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace baracuda_cutlass {
namespace f64_rrr {

using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = double;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

using IdentityEpilogue = cutlass::epilogue::thread::LinearCombination<
    double, 1, ElementAcc, ElementAcc>;

using GemmF64RrrSm80 = cutlass::gemm::device::Gemm<
    double, RowMajor,
    double, RowMajor,   // <- only difference from the RCR variant
    double, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    IdentityEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

static int run_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    double alpha, double beta,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream)
{
    using Gemm = GemmF64RrrSm80;

    auto const* c_eff   = c ? static_cast<const double*>(c)
                            : static_cast<const double*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const double*>(a), static_cast<int>(lda)},
        {static_cast<const double*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<double*>(d), static_cast<int>(ldd)},
        {alpha, beta}
    };

    Gemm op;

    auto status = op.can_implement(args);
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    if (status != cutlass::Status::kSuccess)                return 5;

    std::size_t needed = Gemm::get_workspace_size(args);
    if (workspace_bytes < needed) return 4;

    status = op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return 5;

    status = op(stream);
    if (status != cutlass::Status::kSuccess) return 5;

    return 0;
}

static std::size_t workspace_impl(int m, int n, int k) {
    using Gemm = GemmF64RrrSm80;
    typename Gemm::Arguments args{
        {m, n, k},
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
        {1.0, 0.0}
    };
    return Gemm::get_workspace_size(args);
}

static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    using Gemm = GemmF64RrrSm80;

    auto const* c_eff   = c ? static_cast<const double*>(c)
                            : static_cast<const double*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const double*>(a), static_cast<int>(lda)},
        {static_cast<const double*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<double*>(d), static_cast<int>(ldd)},
        {1.0, 0.0}
    };

    Gemm op;
    auto status = op.can_implement(args);
    if (status == cutlass::Status::kSuccess)              return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

}  // namespace f64_rrr
}  // namespace baracuda_cutlass

extern "C" {

int baracuda_cutlass_gemm_f64_rrr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    double alpha, double beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::f64_rrr::run_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta, workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_f64_rrr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::f64_rrr::workspace_impl(m, n, k);
}

int baracuda_cutlass_gemm_f64_rrr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::f64_rrr::can_implement_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

}  // extern "C"
