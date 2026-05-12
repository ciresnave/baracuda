// int8 (signed) GEMM, RRR layout (RowMajor A × RowMajor B), sm_80
// instantiation.
//
// Computes: D = saturating_cast<int8>(alpha * (A * B) + beta * C)
// with int8 inputs, int32 accumulator, float alpha/beta. Same kernel
// family as gemm_s8_rcr_sm80.cu — only B's layout differs (RowMajor
// here, ColumnMajor in the RCR sibling).
//
// Layout RRR:
//   A: row-major    [M, K]  (int8)
//   B: row-major    [K, N]  (int8)
//   C: row-major    [M, N]  (int8, optional)
//   D: row-major    [M, N]  (int8)
//
// Tile shape matches the RCR sibling: 128x256x64 threadblock,
// 64x64x64 warp, 16x8x32 instruction. EPA = 16. Operator =
// `OpMultiplyAddSaturate`.
//
// CUTLASS upstream caveat
// -----------------------
// CUTLASS 4.2.0's `DefaultMmaCore` selects
// `RowMajorTensorOpMultiplicandCongruous<8, _>` for B in this layout —
// a smem arrangement that doesn't compose with what
// `mma.sync.m16n8k32.s8` expects (b16 chunks pack N-adjacent bytes,
// the mma needs K-adjacent). This kernel therefore depends on
// `baracuda_int8_rr_default_mma_core.h`, which vendors a more-specific
// `DefaultMmaCore` partial spec routing B through K-contig Crosswise
// smem. The header must be included BEFORE
// `<cutlass/gemm/device/gemm.h>` so the specialisation is in scope at
// GEMM-instantiation time.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include "baracuda_int8_rr_default_mma_core.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>

namespace baracuda_cutlass {
namespace s8_rrr {

using RowMajor    = cutlass::layout::RowMajor;
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

using GemmS8RrrSm80 = cutlass::gemm::device::Gemm<
    ElementInput,  RowMajor,
    ElementInput,  RowMajor,
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
    using Gemm = GemmS8RrrSm80;

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
    using Gemm = GemmS8RrrSm80;
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
    using Gemm = GemmS8RrrSm80;

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
    if (status == cutlass::Status::kSuccess)                return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

}  // namespace s8_rrr
}  // namespace baracuda_cutlass

extern "C" {

int baracuda_cutlass_gemm_s8_rrr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::s8_rrr::run_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta, workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_s8_rrr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::s8_rrr::workspace_impl(m, n, k);
}

int baracuda_cutlass_gemm_s8_rrr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::s8_rrr::can_implement_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

}  // extern "C"
