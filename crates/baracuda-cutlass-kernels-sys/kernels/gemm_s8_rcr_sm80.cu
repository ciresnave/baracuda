// int8 (signed) GEMM, RCR layout, sm_80 instantiation.
//
// Computes: D = saturating_cast<int8>(alpha * (A * B) + beta * C)
// with int8 inputs, int32 accumulator, float alpha/beta. The Ampere
// int8 tensor core (`mma.sync m16n8k32` integer variant) drives the
// multiply-add. CUTLASS's `LinearCombinationClamp` does the float
// alpha/beta math and saturating round-to-nearest cast back to int8.
//
// This is the dequant-style integer GEMM: the float alpha/beta let
// callers fold per-tensor / per-channel scaling factors into the
// epilogue, while the accumulator stays in int32 to retain full
// precision through the reduction.
//
// Layout RCR:
//   A: row-major    [M, K]  (int8)
//   B: column-major [K, N]  (int8)
//   C: row-major    [M, N]  (int8, optional)
//   D: row-major    [M, N]  (int8)
//
// Tile shape from CUTLASS's `DefaultGemmConfiguration<OpClassTensorOp,
// Sm80, int8_t, int8_t, int8_t, int32_t>`: 128x256x64 threadblock,
// 64x64x64 warp, 16x8x32 instruction (integer mma). EPA = 16
// (128-bit access at int8). Operator = OpMultiplyAddSaturate (clamps
// the int32 accumulator on overflow instead of wrapping — a soft
// guard for extreme s8 reductions over large K).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>

namespace baracuda_cutlass {
namespace s8_rcr {

using RowMajor    = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;
using ElementInput  = int8_t;
using ElementOutput = int8_t;
using ElementAcc    = int32_t;
using ElementCompute = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kStages = 3;
constexpr int kEPA = 128 / cutlass::sizeof_bits<ElementOutput>::value;  // 16

using IdentityEpilogue = cutlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput, kEPA, ElementAcc, ElementCompute>;

using GemmS8RcrSm80 = cutlass::gemm::device::Gemm<
    ElementInput,  RowMajor,
    ElementInput,  ColumnMajor,
    ElementOutput, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    IdentityEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    /* kAlignmentA  = */ 16,
    /* kAlignmentB  = */ 16,
    /* SplitKSerial = */ false,
    cutlass::arch::OpMultiplyAddSaturate>;

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
    using Gemm = GemmS8RcrSm80;

    auto const* c_eff   = c ? static_cast<const ElementOutput*>(c)
                            : static_cast<const ElementOutput*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const ElementInput*>(a), static_cast<int>(lda)},
        {static_cast<const ElementInput*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<ElementOutput*>(d), static_cast<int>(ldd)},
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
    using Gemm = GemmS8RcrSm80;
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
    using Gemm = GemmS8RcrSm80;

    auto const* c_eff   = c ? static_cast<const ElementOutput*>(c)
                            : static_cast<const ElementOutput*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const ElementInput*>(a), static_cast<int>(lda)},
        {static_cast<const ElementInput*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<ElementOutput*>(d), static_cast<int>(ldd)},
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

}  // namespace s8_rcr
}  // namespace baracuda_cutlass

extern "C" {

int baracuda_cutlass_gemm_s8_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::s8_rcr::run_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta, workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_s8_rcr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::s8_rcr::workspace_impl(m, n, k);
}

int baracuda_cutlass_gemm_s8_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::s8_rcr::can_implement_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

}  // extern "C"
