// f16 + bf16 GEMM, RCR layout, sm_80 instantiation.
//
// Layout RCR:
//   A: row-major    [M, K], lda elements per row
//   B: column-major [K, N], ldb elements per column
//   C: row-major    [M, N], ldc elements per row (optional; pass nullptr
//                                                 + beta = 0 to skip)
//   D: row-major    [M, N], ldd elements per row
//
// Accumulator and alpha/beta are FP32. Epilogue: Identity only
// (D = alpha * AB + beta * C). The Bias variant has been removed from
// the safe API until a `LinearCombinationBias` instantiation lands; this
// file therefore takes no `bias` argument.
//
// Null-C contract: when `c == nullptr`, the safe layer guarantees
// `beta == 0`, but CUTLASS's host adapter still wants a valid pointer
// for the C operand, so we substitute `d` for `c` (the read is
// multiplied by 0 and discarded).
//
// Status code mapping (matches src/lib.rs Rust extern decls):
//   0 = success, 1 = misaligned, 2 = invalid, 3 = unsupported,
//   4 = workspace null/too small, 5 = internal.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace baracuda_cutlass {

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

// Standard Ampere tensor-op tile for f16/bf16: 128x128 threadblock,
// 64x64 warp, 16x8x16 mma. Fits well across batch sizes from tiny
// (M,N >= 128) to large; for very small problems CUTLASS will route to
// SIMT internally.
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Identity epilogue: D = alpha * AB + beta * C.
// 8 elements per access matches f16/bf16 alignment for the standard tile.
template <typename Element>
using IdentityEpilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 8, ElementAcc, ElementAcc>;

// Full GEMM type for one element:
//   A: row-major, B: column-major, C/D: row-major
//   tensor-op math, sm_80, with the standard Ampere f16/bf16 tile,
//   identity threadblock swizzle, 3 software pipeline stages.
template <typename Element>
using GemmRcrSm80 = cutlass::gemm::device::Gemm<
    Element, RowMajor,
    Element, ColumnMajor,
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
    using Gemm = GemmRcrSm80<Element>;
    using ElementC = Element;

    // CUTLASS's host adapter wants a valid pointer for C even when the
    // user passed `c == nullptr`. The safe Rust layer guarantees `beta == 0`
    // in that case (see `GemmPlan::run`), so substituting `d` here is
    // numerically inert — the read is multiplied by zero and discarded.
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
    using Gemm = GemmRcrSm80<Element>;
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
    using Gemm = GemmRcrSm80<Element>;
    using ElementC = Element;

    // Same null-C substitution as run_impl: CUTLASS's host adapter wants a
    // valid pointer for the C operand. We're only checking shape and
    // alignment here, so pointing at D is safe (no read happens).
    auto const* c_eff   = c ? static_cast<const ElementC*>(c)
                            : static_cast<const ElementC*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const Element*>(a), static_cast<int>(lda)},
        {static_cast<const Element*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<Element*>(d), static_cast<int>(ldd)},
        {1.0f, 0.0f}  // alpha/beta don't influence can_implement
    };

    Gemm op;
    auto status = op.can_implement(args);
    if (status == cutlass::Status::kSuccess)              return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

}  // namespace baracuda_cutlass

// ============================================================================
// C ABI surface — symbol names match the extern decls in src/lib.rs
// ============================================================================

extern "C" {

int baracuda_cutlass_gemm_f16_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::run_impl<cutlass::half_t>(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_f16_rcr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::workspace_impl<cutlass::half_t>(m, n, k);
}

int baracuda_cutlass_gemm_f16_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::can_implement_impl<cutlass::half_t>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

int baracuda_cutlass_gemm_bf16_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::run_impl<cutlass::bfloat16_t>(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_bf16_rcr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::workspace_impl<cutlass::bfloat16_t>(m, n, k);
}

int baracuda_cutlass_gemm_bf16_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::can_implement_impl<cutlass::bfloat16_t>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

}  // extern "C"
