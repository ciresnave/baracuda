// f16 + bf16 GEMM, RRR layout, sm_80 instantiation.
//
// Layout RRR:
//   A: row-major [M, K], lda elements per row
//   B: row-major [K, N], ldb elements per row
//   C: row-major [M, N], ldc elements per row (optional; pass nullptr
//                                              + beta = 0 to skip)
//   D: row-major [M, N], ldd elements per row
//
// This is the natural shape for Fuel's `Op::MatMul`: activations and
// weights both stored row-major, so neither operand needs a transpose
// pass before the kernel call.
//
// Accumulator and alpha/beta are FP32. Epilogue: Identity only
// (D = alpha * AB + beta * C). Same null-C / status-code conventions
// as gemm_rcr_sm80.cu — see that file for rationale.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace baracuda_cutlass {
namespace rrr {

using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

template <typename Element>
using IdentityEpilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 8, ElementAcc, ElementAcc>;

// All three operands row-major. Tile shape and pipeline depth match
// the RCR kernel — keeps the launch profile predictable across the two
// SKUs so a downstream selector can pick layout without rewinding the
// rest of its plan.
template <typename Element>
using GemmRrrSm80 = cutlass::gemm::device::Gemm<
    Element, RowMajor,
    Element, RowMajor,
    Element, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    IdentityEpilogue<Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

template <typename Element>
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
    using Gemm = GemmRrrSm80<Element>;
    using ElementC = Element;

    auto const* c_eff   = c ? static_cast<const ElementC*>(c)
                            : static_cast<const ElementC*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const Element*>(a), static_cast<int>(lda)},
        {static_cast<const Element*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<Element*>(d), static_cast<int>(ldd)},
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

template <typename Element>
static std::size_t workspace_impl(int m, int n, int k) {
    using Gemm = GemmRrrSm80<Element>;
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

template <typename Element>
static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    using Gemm = GemmRrrSm80<Element>;
    using ElementC = Element;

    auto const* c_eff   = c ? static_cast<const ElementC*>(c)
                            : static_cast<const ElementC*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const Element*>(a), static_cast<int>(lda)},
        {static_cast<const Element*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<Element*>(d), static_cast<int>(ldd)},
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

}  // namespace rrr
}  // namespace baracuda_cutlass

// ============================================================================
// C ABI surface — symbol names match the extern decls in src/lib.rs
// ============================================================================

extern "C" {

int baracuda_cutlass_gemm_f16_rrr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::rrr::run_impl<cutlass::half_t>(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_f16_rrr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::rrr::workspace_impl<cutlass::half_t>(m, n, k);
}

int baracuda_cutlass_gemm_f16_rrr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::rrr::can_implement_impl<cutlass::half_t>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

int baracuda_cutlass_gemm_bf16_rrr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::rrr::run_impl<cutlass::bfloat16_t>(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_bf16_rrr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::rrr::workspace_impl<cutlass::bfloat16_t>(m, n, k);
}

int baracuda_cutlass_gemm_bf16_rrr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::rrr::can_implement_impl<cutlass::bfloat16_t>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

}  // extern "C"
