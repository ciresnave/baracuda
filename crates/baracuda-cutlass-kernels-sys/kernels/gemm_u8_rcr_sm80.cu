// uint8 (unsigned) GEMM, RCR layout, sm_80 instantiation.
//
// Computes: D = saturating_cast<uint8>(alpha * (A * B) + beta * C)
// with uint8 inputs, int32 accumulator, float alpha/beta. Same family
// as gemm_s8_rcr_sm80.cu — operands are unsigned 8-bit and the final
// cast clamps to `[0, 255]` via the `cvt.rni.sat.u8.f32` PTX
// instruction (round-to-nearest, saturate to unsigned 8-bit range).
//
// Layout RCR:
//   A: row-major    [M, K]  (uint8)
//   B: column-major [K, N]  (uint8)
//   C: row-major    [M, N]  (uint8, optional)
//   D: row-major    [M, N]  (uint8)
//
// See gemm_s8_rcr_sm80.cu for the tile-shape and epilogue rationale.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>

namespace baracuda_cutlass {
namespace u8_rcr {

using RowMajor    = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;
using ElementInput  = uint8_t;
using ElementOutput = uint8_t;
using ElementAcc    = int32_t;
using ElementCompute = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kStages = 3;
constexpr int kEPA = 128 / cutlass::sizeof_bits<ElementOutput>::value;  // 16

using IdentityEpilogue = cutlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput, kEPA, ElementAcc, ElementCompute>;

using GemmU8RcrSm80 = cutlass::gemm::device::Gemm<
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
    16,
    16,
    false,
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
    using Gemm = GemmU8RcrSm80;

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
    using Gemm = GemmU8RcrSm80;
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
    using Gemm = GemmU8RcrSm80;

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

}  // namespace u8_rcr
}  // namespace baracuda_cutlass

extern "C" {

int baracuda_cutlass_gemm_u8_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::u8_rcr::run_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta, workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_u8_rcr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::u8_rcr::workspace_impl(m, n, k);
}

int baracuda_cutlass_gemm_u8_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::u8_rcr::can_implement_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

}  // extern "C"
