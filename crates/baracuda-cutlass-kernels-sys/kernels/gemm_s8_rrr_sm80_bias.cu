// int8 (signed) bias-fused GEMM with optional activation, RRR layout,
// sm_80. Sibling of gemm_s8_rcr_sm80_bias.cu — only B's layout differs
// (RowMajor here, ColumnMajor in the RCR sibling).
//
// Computes: D = saturating_cast<int8>(
//   activation( alpha * (A * B) + beta * C + bias_broadcast(N) ))
//
// See gemm_s8_rcr_sm80_bias.cu for the dequant-in-epilogue rationale.
//
// Layout RRR:
//   A: row-major [M, K]  (int8)
//   B: row-major [K, N]  (int8)
//   C: row-major [M, N]  (int8, optional)
//   D: row-major [M, N]  (int8)
//   bias: contiguous [N] (float OR int32; pick at the symbol level)
//
// Tile shape: 128x256x64 threadblock, 64x64x64 warp, 16x8x32
// instruction. EPA = 16. Operator = `OpMultiplyAddSaturate`.
//
// Depends on the vendored Crosswise-B `DefaultMmaCore` (see
// `baracuda_int8_rr_default_mma_core.h`) — included before
// `<cutlass/gemm/device/gemm_universal_with_broadcast.h>` so the
// partial specialisation is in scope at instantiation time.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include "baracuda_int8_rr_default_mma_core.h"
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>

namespace baracuda_cutlass {
namespace bias_s8_rrr {

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

template <typename ActivationOp, typename ElementBias>
using BiasActEpilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    ElementOutput,                              // ElementC (and D output)
    ElementAcc,                                 // ElementAccumulator (int32)
    ElementCompute,                             // ElementCompute (float)
    ElementOutput,                              // ElementZ (= D)
    ElementOutput,                              // ElementT (auxiliary; unused, kStoreT=false)
    kEPA,                                       // ElementsPerAccess
    ActivationOp,                               // pluggable elementwise (in float)
    cutlass::plus<ElementCompute>,              // BinaryOp (z = AB + V), in float
    /* StoreT = */ false,
    ElementBias                                 // ElementVector (bias dtype)
>;

template <typename ActivationOp, typename ElementBias>
using GemmS8BiasRrrSm80 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    ElementInput,  RowMajor,
    ElementInput,  RowMajor,
    ElementOutput, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    BiasActEpilogue<ActivationOp, ElementBias>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    16,                                          // kAlignmentA
    16,                                          // kAlignmentB
    cutlass::arch::OpMultiplyAddSaturate>;

template <typename ActivationOp, typename ElementBias>
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
    using Gemm = GemmS8BiasRrrSm80<ActivationOp, ElementBias>;

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

template <typename ActivationOp, typename ElementBias>
static std::size_t workspace_impl(int m, int n, int k) {
    using Gemm = GemmS8BiasRrrSm80<ActivationOp, ElementBias>;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename Gemm::EpilogueOutputOp::Params{1.0f, 0.0f},
        static_cast<void const*>(nullptr),
        static_cast<void const*>(nullptr),
        static_cast<void const*>(nullptr),
        static_cast<void*>(nullptr),
        static_cast<void*>(nullptr),
        static_cast<void*>(nullptr),
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        0, 0, 0, 0, 0, 0
    );
    return Gemm::get_workspace_size(args);
}

template <typename ActivationOp, typename ElementBias>
static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias)
{
    using Gemm = GemmS8BiasRrrSm80<ActivationOp, ElementBias>;

    auto const* c_eff   = c    ? c    : d;
    int64_t      ldc_eff = c   ? ldc  : ldd;

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename Gemm::EpilogueOutputOp::Params{1.0f, 0.0f},
        a, b, c_eff, d,
        const_cast<void*>(bias),
        static_cast<void*>(nullptr),
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc_eff),
        static_cast<int>(ldd),
        0, 0
    );

    Gemm op;
    auto status = op.can_implement(args);
    if (status == cutlass::Status::kSuccess)                return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

using ActIdentity = cutlass::epilogue::thread::Identity<ElementCompute>;
using ActReLu     = cutlass::epilogue::thread::ReLu<ElementCompute>;
using ActGELU     = cutlass::epilogue::thread::GELU<ElementCompute>;
using ActSiLu     = cutlass::epilogue::thread::SiLu<ElementCompute>;

}  // namespace bias_s8_rrr
}  // namespace baracuda_cutlass

// ============================================================================
// C ABI surface — naming convention mirrors the RCR sibling:
//   baracuda_cutlass_gemm_<bias-flavor>_<bias-elem>_s8_rrr_sm80_<op>
//   bias-flavor ∈ {bias, bias_relu, bias_gelu, bias_silu}
//   bias-elem   ∈ {f32bias, i32bias}
//   op          ∈ {run, workspace_size, can_implement}
// = 4 * 2 * 3 = 24 entry points.
// ============================================================================

namespace bs8r = baracuda_cutlass::bias_s8_rrr;

#define BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(EPI_SUFFIX, ACTIVATION_OP, BIAS_SUFFIX, BIAS_ELEM)            \
    extern "C" int baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_s8_rrr_sm80_run(                \
        int m, int n, int k,                                                                            \
        const void* a, int64_t lda,                                                                     \
        const void* b, int64_t ldb,                                                                     \
        const void* c, int64_t ldc,                                                                     \
        void*       d, int64_t ldd,                                                                     \
        const void* bias,                                                                               \
        float alpha, float beta,                                                                        \
        void* workspace, std::size_t workspace_bytes,                                                   \
        void* stream)                                                                                   \
    {                                                                                                   \
        return bs8r::run_impl<ACTIVATION_OP, BIAS_ELEM>(                                                \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,                                              \
            alpha, beta, workspace, workspace_bytes,                                                    \
            static_cast<cudaStream_t>(stream));                                                         \
    }                                                                                                   \
                                                                                                        \
    extern "C" std::size_t baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_s8_rrr_sm80_workspace_size( \
        int m, int n, int k)                                                                            \
    {                                                                                                   \
        return bs8r::workspace_impl<ACTIVATION_OP, BIAS_ELEM>(m, n, k);                                 \
    }                                                                                                   \
                                                                                                        \
    extern "C" int baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_s8_rrr_sm80_can_implement(      \
        int m, int n, int k,                                                                            \
        const void* a, int64_t lda,                                                                     \
        const void* b, int64_t ldb,                                                                     \
        const void* c, int64_t ldc,                                                                     \
        void*       d, int64_t ldd,                                                                     \
        const void* bias)                                                                               \
    {                                                                                                   \
        return bs8r::can_implement_impl<ACTIVATION_OP, BIAS_ELEM>(                                      \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);                                             \
    }

BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(bias,      bs8r::ActIdentity, f32bias, float)
BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(bias_relu, bs8r::ActReLu,     f32bias, float)
BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(bias_gelu, bs8r::ActGELU,     f32bias, float)
BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(bias_silu, bs8r::ActSiLu,     f32bias, float)
BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(bias,      bs8r::ActIdentity, i32bias, int32_t)
BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(bias_relu, bs8r::ActReLu,     i32bias, int32_t)
BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(bias_gelu, bs8r::ActGELU,     i32bias, int32_t)
BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES(bias_silu, bs8r::ActSiLu,     i32bias, int32_t)

#undef BARACUDA_S8_BIAS_RRR_GEMM_ENTRIES
