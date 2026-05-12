// f32 GEMM via TF32 tensor cores, RRR layout, sm_80 instantiation.
//
// Mirror of gemm_tf32_rcr_sm80.cu with B in row-major rather than
// column-major. The natural shape for f32 activations × f32 weights when
// both tensors are stored row-major — no transpose pass needed before
// launch.
//
// Inputs are stored as IEEE 754 binary32 but the multiply-add is reduced
// through Ampere TF32 tensor cores (10-bit mantissa, 8-bit exponent —
// same range as F32, narrower precision than F32 SIMT). Accumulator is
// FP32.
//
// Layout RRR:
//   A: row-major [M, K]
//   B: row-major [K, N]
//   C: row-major [M, N] (optional)
//   D: row-major [M, N]
//
// Status code mapping matches the other kernels (see gemm_rcr_sm80.cu).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace baracuda_cutlass {
namespace tf32_rrr {

using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

// Tile shape matches the RCR TF32 kernel — keeps the launch profile
// predictable across layouts so the selector can swap RCR↔RRR without
// rewinding the rest of its plan.
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

// 4 elements per access matches F32 alignment (16-byte vector load).
using IdentityEpilogue = cutlass::epilogue::thread::LinearCombination<
    float, 4, ElementAcc, ElementAcc>;

using GemmTf32RrrSm80 = cutlass::gemm::device::Gemm<
    float, RowMajor,
    float, RowMajor,   // <- only difference from the RCR variant
    float, RowMajor,
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
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream)
{
    using Gemm = GemmTf32RrrSm80;

    auto const* c_eff   = c ? static_cast<const float*>(c)
                            : static_cast<const float*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const float*>(a), static_cast<int>(lda)},
        {static_cast<const float*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<float*>(d), static_cast<int>(ldd)},
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
    using Gemm = GemmTf32RrrSm80;
    typename Gemm::Arguments args{
        {m, n, k},
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
        {1.0f, 0.0f}
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
    using Gemm = GemmTf32RrrSm80;

    auto const* c_eff   = c ? static_cast<const float*>(c)
                            : static_cast<const float*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const float*>(a), static_cast<int>(lda)},
        {static_cast<const float*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<float*>(d), static_cast<int>(ldd)},
        {1.0f, 0.0f}
    };

    Gemm op;
    auto status = op.can_implement(args);
    if (status == cutlass::Status::kSuccess)              return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

}  // namespace tf32_rrr
}  // namespace baracuda_cutlass

extern "C" {

int baracuda_cutlass_gemm_tf32_rrr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::tf32_rrr::run_impl(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_tf32_rrr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::tf32_rrr::workspace_impl(m, n, k);
}

int baracuda_cutlass_gemm_tf32_rrr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::tf32_rrr::can_implement_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

}  // extern "C"
