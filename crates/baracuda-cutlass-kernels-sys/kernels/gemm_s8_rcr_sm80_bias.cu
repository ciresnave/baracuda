// int8 (signed) bias-fused GEMM with optional activation, RCR layout,
// sm_80.
//
// Computes: D = saturating_cast<int8>(
//   activation( alpha * (A * B) + beta * C + bias_broadcast(N) ))
//
// with int8 inputs, int32 accumulator, FLOAT alpha/beta and FLOAT or
// INT32 bias broadcast — caller chooses the bias element type via the
// `_f32bias` vs `_i32bias` symbol suffix. The activation runs in FLOAT
// space (after the int32→float dequant), so GELU and SiLU are
// numerically meaningful; the final saturating cast clamps back to
// int8 range via `cvt.rni.sat.s8.f32`.
//
// This is the dequant-in-epilogue path: the float compute element type
// of `LinearCombinationBiasElementwise` already implements the
// quantize/dequant sandwich for us — no custom epilogue functor
// needed. See `cutlass/epilogue/thread/linear_combination_bias_elementwise.h`
// for the conversion sequence (int32→float→activate→saturating-cast-to-int8).
//
// Layout RCR (matches gemm_s8_rcr_sm80.cu):
//   A: row-major    [M, K]  (int8)
//   B: column-major [K, N]  (int8)
//   C: row-major    [M, N]  (int8, optional)
//   D: row-major    [M, N]  (int8)
//   bias: contiguous [N]    (float OR int32; pick at the symbol level)
//
// Tile shape matches the identity int8 kernel: 128x256x64 threadblock,
// 64x64x64 warp, 16x8x32 instruction. EPA = 16. Operator =
// OpMultiplyAddSaturate.

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
namespace bias_s8_rcr {

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

// `ElementCompute = float` is the dequant-in-epilogue knob: int32 accum
// converts to float for the alpha/beta/bias/activation chain, then
// converts back to int8 via the saturating PTX cvt. `ElementVector`
// (last template arg) picks the bias element type — float or int32.
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
    /* StoreT = */ false,                       // skip auxiliary tensor write
    ElementBias                                 // ElementVector (bias dtype)
>;

template <typename ActivationOp, typename ElementBias>
using GemmS8BiasRcrSm80 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    ElementInput,  RowMajor,
    ElementInput,  ColumnMajor,
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
    using Gemm = GemmS8BiasRcrSm80<ActivationOp, ElementBias>;

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
    using Gemm = GemmS8BiasRcrSm80<ActivationOp, ElementBias>;
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
    using Gemm = GemmS8BiasRcrSm80<ActivationOp, ElementBias>;

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
    if (status == cutlass::Status::kSuccess)              return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

// Activation functors operate on `ElementCompute` (= float). They run
// after dequant and before saturating cast back to int8, so the standard
// CUTLASS float activations apply unchanged.
using ActIdentity = cutlass::epilogue::thread::Identity<ElementCompute>;
using ActReLu     = cutlass::epilogue::thread::ReLu<ElementCompute>;
using ActGELU     = cutlass::epilogue::thread::GELU<ElementCompute>;
using ActSiLu     = cutlass::epilogue::thread::SiLu<ElementCompute>;

}  // namespace bias_s8_rcr
}  // namespace baracuda_cutlass

// ============================================================================
// C ABI surface — naming convention:
//   baracuda_cutlass_gemm_<bias-flavor>_<bias-elem>_s8_rcr_sm80_<op>
//   bias-flavor ∈ {bias, bias_relu, bias_gelu, bias_silu}
//   bias-elem   ∈ {f32bias, i32bias}
//   op          ∈ {run, workspace_size, can_implement}
// = 4 * 2 * 3 = 24 entry points.
// ============================================================================

namespace bs8 = baracuda_cutlass::bias_s8_rcr;

#define BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(EPI_SUFFIX, ACTIVATION_OP, BIAS_SUFFIX, BIAS_ELEM)            \
    extern "C" int baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_s8_rcr_sm80_run(                \
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
        return bs8::run_impl<ACTIVATION_OP, BIAS_ELEM>(                                                 \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,                                              \
            alpha, beta, workspace, workspace_bytes,                                                    \
            static_cast<cudaStream_t>(stream));                                                         \
    }                                                                                                   \
                                                                                                        \
    extern "C" std::size_t baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_s8_rcr_sm80_workspace_size( \
        int m, int n, int k)                                                                            \
    {                                                                                                   \
        return bs8::workspace_impl<ACTIVATION_OP, BIAS_ELEM>(m, n, k);                                  \
    }                                                                                                   \
                                                                                                        \
    extern "C" int baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_s8_rcr_sm80_can_implement(      \
        int m, int n, int k,                                                                            \
        const void* a, int64_t lda,                                                                     \
        const void* b, int64_t ldb,                                                                     \
        const void* c, int64_t ldc,                                                                     \
        void*       d, int64_t ldd,                                                                     \
        const void* bias)                                                                               \
    {                                                                                                   \
        return bs8::can_implement_impl<ACTIVATION_OP, BIAS_ELEM>(                                       \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);                                             \
    }

BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(bias,      bs8::ActIdentity, f32bias, float)
BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(bias_relu, bs8::ActReLu,     f32bias, float)
BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(bias_gelu, bs8::ActGELU,     f32bias, float)
BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(bias_silu, bs8::ActSiLu,     f32bias, float)
BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(bias,      bs8::ActIdentity, i32bias, int32_t)
BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(bias_relu, bs8::ActReLu,     i32bias, int32_t)
BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(bias_gelu, bs8::ActGELU,     i32bias, int32_t)
BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES(bias_silu, bs8::ActSiLu,     i32bias, int32_t)

#undef BARACUDA_S8_BIAS_RCR_GEMM_ENTRIES
