// f16 + bf16 bias-fused GEMM, RCR layout, sm_80 instantiation.
//
// Computes: D = alpha * (A * B) + beta * C + bias_broadcast(N)
//
// where `bias` is a length-N vector broadcast across rows. The bias add
// is fused into the epilogue — single memory pass, same launch profile
// as the Identity-epilogue Rcr kernel except for the extra vector load.
//
// Layout RCR (matches gemm_rcr_sm80.cu):
//   A: row-major    [M, K]
//   B: column-major [K, N]
//   C: row-major    [M, N] (optional; pass nullptr + beta = 0 to skip)
//   D: row-major    [M, N]
// Bias: contiguous [N] vector (one element per output column)
//
// Epilogue uses cutlass::epilogue::thread::LinearCombinationBiasElementwise
// (default math: z = alpha*AB + beta*C + V), driven by
// cutlass::gemm::device::GemmUniversalWithBroadcast.
//
// Status code mapping matches the other kernels (see gemm_rcr_sm80.cu).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>

namespace baracuda_cutlass {
namespace bias_rcr {

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// LinearCombinationBiasElementwise computes:
//   z = binary_op(alpha * AB + beta * C, V)   [V = bias]
//   T_out = z                                  [auxiliary; we set kStoreT=false]
//   Z_out = elementwise_op(z)                  [Identity here, no activation]
// With BinaryOp = plus and ElementwiseOp = Identity, this gives us
// exactly D = alpha*AB + beta*C + bias_broadcast(N).
template <typename Element>
using BiasEpilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    Element,                                    // ElementC (and D output)
    ElementAcc,                                 // ElementAccumulator
    ElementAcc,                                 // ElementCompute
    Element,                                    // ElementZ (= D)
    Element,                                    // ElementT (auxiliary; unused, kStoreT=false)
    8,                                          // ElementsPerAccess
    cutlass::epilogue::thread::Identity<ElementAcc>,  // ElementwiseOp (no activation)
    cutlass::plus<ElementAcc>,                  // BinaryOp (z = AB + V)
    /* StoreT = */ false,                       // skip auxiliary tensor write
    Element                                     // ElementVector (bias dtype)
>;

template <typename Element>
using GemmBiasRcrSm80 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    Element, RowMajor,
    Element, ColumnMajor,
    Element, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    BiasEpilogue<Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

template <typename Element>
static int run_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream)
{
    using Gemm = GemmBiasRcrSm80<Element>;

    // Null-C handling: caller guarantees beta == 0 when c is null. Some
    // CUTLASS broadcast kernels still touch C1's iterator; pointing it
    // at D with beta=0 is the same numerically-inert pattern used in
    // the Identity-epilogue kernel.
    auto const* c1_eff      = c    ? c    : d;
    int64_t      ldc1_eff   = c    ? ldc  : ldd;

    // GemmUniversalWithBroadcast with `LinearCombinationBiasElementwise`
    // is single-source (kIsSingleSource = true) → the Arguments
    // constructor takes one C operand, not C1 + C2.
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        /* batch_count = */ 1,
        typename Gemm::EpilogueOutputOp::Params{alpha, beta},
        a,                  // ptr_A
        b,                  // ptr_B
        c1_eff,             // ptr_C (the source operand)
        d,                  // ptr_D (= Z output)
        const_cast<void*>(bias), // ptr_Vector (per-N broadcast)
        nullptr,            // ptr_Tensor (T auxiliary; kStoreT=false → unused)
        /* batch_stride_A      */ 0,
        /* batch_stride_B      */ 0,
        /* batch_stride_C      */ 0,
        /* batch_stride_D      */ 0,
        /* batch_stride_Vector */ 0,
        /* batch_stride_Tensor */ 0,
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc1_eff),
        static_cast<int>(ldd),
        /* ldr  */ 0,       // bias vector: stride 0 → broadcast across rows
        /* ldt  */ 0
    );

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
    using Gemm = GemmBiasRcrSm80<Element>;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename Gemm::EpilogueOutputOp::Params{1.0f, 0.0f},
        /* ptr_A      */ static_cast<void const*>(nullptr),
        /* ptr_B      */ static_cast<void const*>(nullptr),
        /* ptr_C      */ static_cast<void const*>(nullptr),
        /* ptr_D      */ static_cast<void*>(nullptr),
        /* ptr_Vector */ static_cast<void*>(nullptr),
        /* ptr_Tensor */ static_cast<void*>(nullptr),
        /* batch_stride_A      */ int64_t(0),
        /* batch_stride_B      */ int64_t(0),
        /* batch_stride_C      */ int64_t(0),
        /* batch_stride_D      */ int64_t(0),
        /* batch_stride_Vector */ int64_t(0),
        /* batch_stride_Tensor */ int64_t(0),
        /* lda */ 0, /* ldb */ 0, /* ldc */ 0, /* ldd */ 0, /* ldr */ 0, /* ldt */ 0
    );
    return Gemm::get_workspace_size(args);
}

template <typename Element>
static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias)
{
    using Gemm = GemmBiasRcrSm80<Element>;

    auto const* c1_eff      = c    ? c    : d;
    int64_t      ldc1_eff   = c    ? ldc  : ldd;

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename Gemm::EpilogueOutputOp::Params{1.0f, 0.0f},
        a, b, c1_eff, d,
        const_cast<void*>(bias),
        /* ptr_Tensor */ static_cast<void*>(nullptr),
        /* batch_stride_A      */ int64_t(0),
        /* batch_stride_B      */ int64_t(0),
        /* batch_stride_C      */ int64_t(0),
        /* batch_stride_D      */ int64_t(0),
        /* batch_stride_Vector */ int64_t(0),
        /* batch_stride_Tensor */ int64_t(0),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc1_eff),
        static_cast<int>(ldd),
        /* ldr */ 0, /* ldt */ 0
    );

    Gemm op;
    auto status = op.can_implement(args);
    if (status == cutlass::Status::kSuccess)              return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

}  // namespace bias_rcr
}  // namespace baracuda_cutlass

// ============================================================================
// C ABI surface — symbol names match the extern decls in src/lib.rs
// ============================================================================

extern "C" {

int baracuda_cutlass_gemm_bias_f16_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::bias_rcr::run_impl<cutlass::half_t>(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        bias,
        alpha, beta,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_bias_f16_rcr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::bias_rcr::workspace_impl<cutlass::half_t>(m, n, k);
}

int baracuda_cutlass_gemm_bias_f16_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias)
{
    return baracuda_cutlass::bias_rcr::can_implement_impl<cutlass::half_t>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);
}

int baracuda_cutlass_gemm_bias_bf16_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::bias_rcr::run_impl<cutlass::bfloat16_t>(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        bias,
        alpha, beta,
        workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_bias_bf16_rcr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::bias_rcr::workspace_impl<cutlass::bfloat16_t>(m, n, k);
}

int baracuda_cutlass_gemm_bias_bf16_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias)
{
    return baracuda_cutlass::bias_rcr::can_implement_impl<cutlass::bfloat16_t>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);
}

}  // extern "C"
