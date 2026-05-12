// f32 (TF32 tensor core) bias-fused GEMM with optional activation,
// RRR layout, sm_80 instantiation.
//
// Computes: D = activation(alpha * (A * B) + beta * C + bias_broadcast(N))
//
// Mirror of gemm_tf32_rcr_sm80_bias.cu with B in row-major rather than
// column-major. All operands and the bias vector are IEEE 754 binary32 in
// storage; multiply-add reduces through TF32 tensor cores (10-bit mantissa,
// 8-bit exponent); accumulator is FP32. Activations: {Identity, ReLu,
// GELU, SiLu}.
//
// Layout RRR (matches gemm_tf32_rrr_sm80.cu and gemm_rrr_sm80_bias.cu):
//   A: row-major [M, K]
//   B: row-major [K, N]
//   C: row-major [M, N] (optional; pass nullptr + beta = 0 to skip)
//   D: row-major [M, N]
// Bias: contiguous [N] vector (one element per output column)
//
// Tile shape matches the non-bias TF32 kernel for predictable launch
// profile across layouts. Status code mapping matches the other kernels.

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
namespace bias_tf32_rrr {

using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

// TF32 tile shape, same as the non-bias TF32 kernel.
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

// 4 elements per access matches f32 alignment (16-byte vector load,
// 4 floats). Same as the RCR TF32 bias kernel.
template <typename ActivationOp>
using BiasActEpilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    float,                                      // ElementC (and D output)
    ElementAcc,                                 // ElementAccumulator
    ElementAcc,                                 // ElementCompute
    float,                                      // ElementZ (= D)
    float,                                      // ElementT (auxiliary; unused)
    4,                                          // ElementsPerAccess
    ActivationOp,                               // pluggable elementwise
    cutlass::plus<ElementAcc>,                  // BinaryOp (z = AB + V)
    /* StoreT = */ false,                       // skip auxiliary tensor write
    float                                       // ElementVector (bias dtype)
>;

template <typename ActivationOp>
using GemmTf32BiasRrrSm80 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    float, RowMajor,
    float, RowMajor,   // <- only difference from the RCR variant
    float, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    BiasActEpilogue<ActivationOp>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

template <typename ActivationOp>
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
    using Gemm = GemmTf32BiasRrrSm80<ActivationOp>;

    auto const* c_eff   = c    ? c    : d;
    int64_t      ldc_eff = c   ? ldc  : ldd;

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
        /* ldr */ 0,
        /* ldt */ 0
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

template <typename ActivationOp>
static std::size_t workspace_impl(int m, int n, int k) {
    using Gemm = GemmTf32BiasRrrSm80<ActivationOp>;
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

template <typename ActivationOp>
static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias)
{
    using Gemm = GemmTf32BiasRrrSm80<ActivationOp>;

    auto const* c_eff   = c    ? c    : d;
    int64_t      ldc_eff = c   ? ldc  : ldd;

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

}  // namespace bias_tf32_rrr
}  // namespace baracuda_cutlass

// ============================================================================
// C ABI surface — naming convention:
//   baracuda_cutlass_gemm_<bias-flavor>_tf32_rrr_sm80_<op>
//   bias-flavor ∈ {bias, bias_relu, bias_gelu, bias_silu}
//   op          ∈ {run, workspace_size, can_implement}
// = 4 * 3 = 12 entry points (single element type since TF32 implies f32).
// ============================================================================

namespace br = baracuda_cutlass::bias_tf32_rrr;

#define BARACUDA_TF32_BIAS_RRR_GEMM_ENTRIES(SUFFIX, ACTIVATION_OP)                              \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_tf32_rrr_sm80_run(                          \
        int m, int n, int k,                                                                    \
        const void* a, int64_t lda,                                                             \
        const void* b, int64_t ldb,                                                             \
        const void* c, int64_t ldc,                                                             \
        void*       d, int64_t ldd,                                                             \
        const void* bias,                                                                       \
        float alpha, float beta,                                                                \
        void* workspace, std::size_t workspace_bytes,                                           \
        void* stream)                                                                           \
    {                                                                                           \
        return br::run_impl<ACTIVATION_OP>(                                                     \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,                                      \
            alpha, beta, workspace, workspace_bytes,                                            \
            static_cast<cudaStream_t>(stream));                                                 \
    }                                                                                           \
                                                                                                \
    extern "C" std::size_t baracuda_cutlass_gemm_##SUFFIX##_tf32_rrr_sm80_workspace_size(       \
        int m, int n, int k)                                                                    \
    {                                                                                           \
        return br::workspace_impl<ACTIVATION_OP>(m, n, k);                                      \
    }                                                                                           \
                                                                                                \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_tf32_rrr_sm80_can_implement(                \
        int m, int n, int k,                                                                    \
        const void* a, int64_t lda,                                                             \
        const void* b, int64_t ldb,                                                             \
        const void* c, int64_t ldc,                                                             \
        void*       d, int64_t ldd,                                                             \
        const void* bias)                                                                       \
    {                                                                                           \
        return br::can_implement_impl<ACTIVATION_OP>(                                           \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);                                     \
    }

BARACUDA_TF32_BIAS_RRR_GEMM_ENTRIES(bias,      br::ActIdentity)
BARACUDA_TF32_BIAS_RRR_GEMM_ENTRIES(bias_relu, br::ActReLu)
BARACUDA_TF32_BIAS_RRR_GEMM_ENTRIES(bias_gelu, br::ActGELU)
BARACUDA_TF32_BIAS_RRR_GEMM_ENTRIES(bias_silu, br::ActSiLu)

#undef BARACUDA_TF32_BIAS_RRR_GEMM_ENTRIES
