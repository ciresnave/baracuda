// f16 + bf16 batched GEMM, RCR layout, sm_80 instantiation.
//
// All batches share the same (M, N, K). Each batch's operands are
// identified by a per-tensor stride (in elements) added to the base
// pointer:
//
//   A[batch_i] = a + i * stride_a
//   B[batch_i] = b + i * stride_b
//   C[batch_i] = c + i * stride_c   (optional; null + beta = 0 to skip)
//   D[batch_i] = d + i * stride_d
//
// Strides are in elements, NOT bytes — matches CUTLASS's GemmBatched
// API. Layout matches gemm_rcr_sm80.cu (A row-major, B column-major,
// C/D row-major). Identity epilogue only.
//
// Status code mapping matches the other kernels.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace baracuda_cutlass {
namespace batched_rcr {

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

template <typename Element>
using IdentityEpilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 8, ElementAcc, ElementAcc>;

template <typename Element>
using GemmBatchedRcrSm80 = cutlass::gemm::device::GemmBatched<
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
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>;

template <typename Element>
static int run_impl(
    int m, int n, int k,
    const void* a, int64_t lda, int64_t stride_a,
    const void* b, int64_t ldb, int64_t stride_b,
    const void* c, int64_t ldc, int64_t stride_c,
    void*       d, int64_t ldd, int64_t stride_d,
    float alpha, float beta,
    int batch_count,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream)
{
    using Gemm = GemmBatchedRcrSm80<Element>;
    using ElementC = Element;

    // Null-C substitution mirrors the single-GEMM case: caller
    // guarantees beta == 0 when C is null, so pointing at D is
    // numerically inert.
    auto const* c_eff   = c ? static_cast<const ElementC*>(c)
                            : static_cast<const ElementC*>(d);
    int64_t      ldc_eff      = c ? ldc      : ldd;
    int64_t      stride_c_eff = c ? stride_c : stride_d;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const Element*>(a), static_cast<int>(lda)}, stride_a,
        {static_cast<const Element*>(b), static_cast<int>(ldb)}, stride_b,
        {c_eff, static_cast<int>(ldc_eff)}, stride_c_eff,
        {static_cast<Element*>(d), static_cast<int>(ldd)}, stride_d,
        {alpha, beta},
        batch_count
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
static std::size_t workspace_impl(int m, int n, int k, int batch_count) {
    using Gemm = GemmBatchedRcrSm80<Element>;
    typename Gemm::Arguments args{
        {m, n, k},
        {nullptr, 0}, 0,
        {nullptr, 0}, 0,
        {nullptr, 0}, 0,
        {nullptr, 0}, 0,
        {1.0f, 0.0f},
        batch_count
    };
    return Gemm::get_workspace_size(args);
}

template <typename Element>
static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda, int64_t stride_a,
    const void* b, int64_t ldb, int64_t stride_b,
    const void* c, int64_t ldc, int64_t stride_c,
    void*       d, int64_t ldd, int64_t stride_d,
    int batch_count)
{
    using Gemm = GemmBatchedRcrSm80<Element>;
    using ElementC = Element;

    auto const* c_eff   = c ? static_cast<const ElementC*>(c)
                            : static_cast<const ElementC*>(d);
    int64_t      ldc_eff      = c ? ldc      : ldd;
    int64_t      stride_c_eff = c ? stride_c : stride_d;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const Element*>(a), static_cast<int>(lda)}, stride_a,
        {static_cast<const Element*>(b), static_cast<int>(ldb)}, stride_b,
        {c_eff, static_cast<int>(ldc_eff)}, stride_c_eff,
        {static_cast<Element*>(d), static_cast<int>(ldd)}, stride_d,
        {1.0f, 0.0f},
        batch_count
    };

    Gemm op;
    auto status = op.can_implement(args);
    if (status == cutlass::Status::kSuccess)              return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

}  // namespace batched_rcr
}  // namespace baracuda_cutlass

extern "C" {

int baracuda_cutlass_gemm_batched_f16_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda, int64_t stride_a,
    const void* b, int64_t ldb, int64_t stride_b,
    const void* c, int64_t ldc, int64_t stride_c,
    void*       d, int64_t ldd, int64_t stride_d,
    float alpha, float beta,
    int batch_count,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::batched_rcr::run_impl<cutlass::half_t>(
        m, n, k,
        a, lda, stride_a,
        b, ldb, stride_b,
        c, ldc, stride_c,
        d, ldd, stride_d,
        alpha, beta,
        batch_count,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_batched_f16_rcr_sm80_workspace_size(
    int m, int n, int k, int batch_count)
{
    return baracuda_cutlass::batched_rcr::workspace_impl<cutlass::half_t>(
        m, n, k, batch_count);
}

int baracuda_cutlass_gemm_batched_f16_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda, int64_t stride_a,
    const void* b, int64_t ldb, int64_t stride_b,
    const void* c, int64_t ldc, int64_t stride_c,
    void*       d, int64_t ldd, int64_t stride_d,
    int batch_count)
{
    return baracuda_cutlass::batched_rcr::can_implement_impl<cutlass::half_t>(
        m, n, k,
        a, lda, stride_a,
        b, ldb, stride_b,
        c, ldc, stride_c,
        d, ldd, stride_d,
        batch_count);
}

int baracuda_cutlass_gemm_batched_bf16_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda, int64_t stride_a,
    const void* b, int64_t ldb, int64_t stride_b,
    const void* c, int64_t ldc, int64_t stride_c,
    void*       d, int64_t ldd, int64_t stride_d,
    float alpha, float beta,
    int batch_count,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::batched_rcr::run_impl<cutlass::bfloat16_t>(
        m, n, k,
        a, lda, stride_a,
        b, ldb, stride_b,
        c, ldc, stride_c,
        d, ldd, stride_d,
        alpha, beta,
        batch_count,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_batched_bf16_rcr_sm80_workspace_size(
    int m, int n, int k, int batch_count)
{
    return baracuda_cutlass::batched_rcr::workspace_impl<cutlass::bfloat16_t>(
        m, n, k, batch_count);
}

int baracuda_cutlass_gemm_batched_bf16_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda, int64_t stride_a,
    const void* b, int64_t ldb, int64_t stride_b,
    const void* c, int64_t ldc, int64_t stride_c,
    void*       d, int64_t ldd, int64_t stride_d,
    int batch_count)
{
    return baracuda_cutlass::batched_rcr::can_implement_impl<cutlass::bfloat16_t>(
        m, n, k,
        a, lda, stride_a,
        b, ldb, stride_b,
        c, ldc, stride_c,
        d, ldd, stride_d,
        batch_count);
}

}  // extern "C"
