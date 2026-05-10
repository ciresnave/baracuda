// f16 + bf16 bias-fused GEMM with optional activation, RRR layout, sm_80.
//
// Mirrors gemm_rcr_sm80_bias.cu except that B is row-major rather than
// column-major. Computes:
//
//   D = activation(alpha * (A * B) + beta * C + bias_broadcast(N))
//
// where `bias` is a length-N vector broadcast across rows. Activations:
// {Identity, ReLu, GELU, SiLu}. The bias and activation are both fused
// into the epilogue — single memory pass for the full row-major
// Linear-plus-activation pipeline.
//
// Layout RRR (matches gemm_rrr_sm80.cu):
//   A: row-major [M, K], lda elements per row
//   B: row-major [K, N], ldb elements per row
//   C: row-major [M, N], ldc elements per row (optional; null + beta=0)
//   D: row-major [M, N], ldd elements per row
// Bias: contiguous [N] vector (one element per output column)
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
namespace bias_rrr {

using RowMajor   = cutlass::layout::RowMajor;
using ElementAcc = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

template <typename Element, typename ActivationOp>
using BiasActEpilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    Element,                                    // ElementC (and D output)
    ElementAcc,                                 // ElementAccumulator
    ElementAcc,                                 // ElementCompute
    Element,                                    // ElementZ (= D)
    Element,                                    // ElementT (auxiliary; unused, kStoreT=false)
    8,                                          // ElementsPerAccess
    ActivationOp,                               // pluggable elementwise
    cutlass::plus<ElementAcc>,                  // BinaryOp (z = AB + V)
    /* StoreT = */ false,                       // skip auxiliary tensor write
    Element                                     // ElementVector (bias dtype)
>;

template <typename Element, typename ActivationOp>
using GemmBiasActRrrSm80 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    Element, RowMajor,
    Element, RowMajor,   // <- only difference from the RCR variant
    Element, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    BiasActEpilogue<Element, ActivationOp>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

template <typename Element, typename ActivationOp>
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
    using Gemm = GemmBiasActRrrSm80<Element, ActivationOp>;

    auto const* c_eff      = c    ? c    : d;
    int64_t      ldc_eff   = c    ? ldc  : ldd;

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        /* batch_count = */ 1,
        typename Gemm::EpilogueOutputOp::Params{alpha, beta},
        a, b, c_eff, d,
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
        static_cast<int>(ldc_eff),
        static_cast<int>(ldd),
        /* ldr  */ 0,   // bias vector broadcast across rows
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

template <typename Element, typename ActivationOp>
static std::size_t workspace_impl(int m, int n, int k) {
    using Gemm = GemmBiasActRrrSm80<Element, ActivationOp>;
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

template <typename Element, typename ActivationOp>
static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias)
{
    using Gemm = GemmBiasActRrrSm80<Element, ActivationOp>;

    auto const* c_eff      = c    ? c    : d;
    int64_t      ldc_eff   = c    ? ldc  : ldd;

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename Gemm::EpilogueOutputOp::Params{1.0f, 0.0f},
        a, b, c_eff, d,
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
        static_cast<int>(ldc_eff),
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

using ActIdentity = cutlass::epilogue::thread::Identity<ElementAcc>;
using ActReLu     = cutlass::epilogue::thread::ReLu<ElementAcc>;
using ActGELU     = cutlass::epilogue::thread::GELU<ElementAcc>;
using ActSiLu     = cutlass::epilogue::thread::SiLu<ElementAcc>;

}  // namespace bias_rrr
}  // namespace baracuda_cutlass

// ============================================================================
// C ABI surface — symbol names match the extern decls in src/lib.rs
//
// Naming convention: baracuda_cutlass_gemm_<bias-flavor>_<dtype>_rrr_sm80_<op>
// = 4 * 2 * 3 = 24 entry points (same layout as the RCR bias kernel).
// ============================================================================

namespace br = baracuda_cutlass::bias_rrr;

#define BARACUDA_BIAS_RRR_GEMM_ENTRIES(SUFFIX, ACTIVATION_OP)                              \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_f16_rrr_sm80_run(                      \
        int m, int n, int k,                                                               \
        const void* a, int64_t lda,                                                        \
        const void* b, int64_t ldb,                                                        \
        const void* c, int64_t ldc,                                                        \
        void*       d, int64_t ldd,                                                        \
        const void* bias,                                                                  \
        float alpha, float beta,                                                           \
        void* workspace, std::size_t workspace_bytes,                                      \
        void* stream)                                                                      \
    {                                                                                      \
        return br::run_impl<cutlass::half_t, ACTIVATION_OP>(                               \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,                                 \
            alpha, beta, workspace, workspace_bytes,                                       \
            static_cast<cudaStream_t>(stream));                                            \
    }                                                                                      \
                                                                                           \
    extern "C" std::size_t baracuda_cutlass_gemm_##SUFFIX##_f16_rrr_sm80_workspace_size(   \
        int m, int n, int k)                                                               \
    {                                                                                      \
        return br::workspace_impl<cutlass::half_t, ACTIVATION_OP>(m, n, k);                \
    }                                                                                      \
                                                                                           \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_f16_rrr_sm80_can_implement(            \
        int m, int n, int k,                                                               \
        const void* a, int64_t lda,                                                        \
        const void* b, int64_t ldb,                                                        \
        const void* c, int64_t ldc,                                                        \
        void*       d, int64_t ldd,                                                        \
        const void* bias)                                                                  \
    {                                                                                      \
        return br::can_implement_impl<cutlass::half_t, ACTIVATION_OP>(                     \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);                                \
    }                                                                                      \
                                                                                           \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_bf16_rrr_sm80_run(                     \
        int m, int n, int k,                                                               \
        const void* a, int64_t lda,                                                        \
        const void* b, int64_t ldb,                                                        \
        const void* c, int64_t ldc,                                                        \
        void*       d, int64_t ldd,                                                        \
        const void* bias,                                                                  \
        float alpha, float beta,                                                           \
        void* workspace, std::size_t workspace_bytes,                                      \
        void* stream)                                                                      \
    {                                                                                      \
        return br::run_impl<cutlass::bfloat16_t, ACTIVATION_OP>(                           \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,                                 \
            alpha, beta, workspace, workspace_bytes,                                       \
            static_cast<cudaStream_t>(stream));                                            \
    }                                                                                      \
                                                                                           \
    extern "C" std::size_t baracuda_cutlass_gemm_##SUFFIX##_bf16_rrr_sm80_workspace_size(  \
        int m, int n, int k)                                                               \
    {                                                                                      \
        return br::workspace_impl<cutlass::bfloat16_t, ACTIVATION_OP>(m, n, k);            \
    }                                                                                      \
                                                                                           \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_bf16_rrr_sm80_can_implement(           \
        int m, int n, int k,                                                               \
        const void* a, int64_t lda,                                                        \
        const void* b, int64_t ldb,                                                        \
        const void* c, int64_t ldc,                                                        \
        void*       d, int64_t ldd,                                                        \
        const void* bias)                                                                  \
    {                                                                                      \
        return br::can_implement_impl<cutlass::bfloat16_t, ACTIVATION_OP>(                 \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);                                \
    }

BARACUDA_BIAS_RRR_GEMM_ENTRIES(bias,      br::ActIdentity)
BARACUDA_BIAS_RRR_GEMM_ENTRIES(bias_relu, br::ActReLu)
BARACUDA_BIAS_RRR_GEMM_ENTRIES(bias_gelu, br::ActGELU)
BARACUDA_BIAS_RRR_GEMM_ENTRIES(bias_silu, br::ActSiLu)

#undef BARACUDA_BIAS_RRR_GEMM_ENTRIES
