// uint8 (unsigned) bias-fused GEMM with optional activation, RCR layout,
// sm_80.
//
// Same template family as gemm_s8_rcr_sm80_bias.cu — operands are
// unsigned 8-bit, the final saturating cast clamps to `[0, 255]` via
// `cvt.rni.sat.u8.f32`, and the dequant-in-epilogue path is identical
// (int32 accum → float compute → activation → saturating cast to u8).
//
// Layout RCR (matches gemm_u8_rcr_sm80.cu):
//   A: row-major    [M, K]  (uint8)
//   B: column-major [K, N]  (uint8)
//   C: row-major    [M, N]  (uint8, optional)
//   D: row-major    [M, N]  (uint8)
//   bias: contiguous [N]    (float OR int32)
//
// See gemm_s8_rcr_sm80_bias.cu for the full rationale on epilogue
// composition and bias-element generalization.

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
namespace bias_u8_rcr {

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
using GemmU8BiasRcrSm80 = cutlass::gemm::device::GemmUniversalWithBroadcast<
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
    16,
    16,
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
    using Gemm = GemmU8BiasRcrSm80<ActivationOp, ElementBias>;

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
    using Gemm = GemmU8BiasRcrSm80<ActivationOp, ElementBias>;
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
    using Gemm = GemmU8BiasRcrSm80<ActivationOp, ElementBias>;

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

using ActIdentity = cutlass::epilogue::thread::Identity<ElementCompute>;
using ActReLu     = cutlass::epilogue::thread::ReLu<ElementCompute>;
using ActGELU     = cutlass::epilogue::thread::GELU<ElementCompute>;
using ActSiLu     = cutlass::epilogue::thread::SiLu<ElementCompute>;

}  // namespace bias_u8_rcr
}  // namespace baracuda_cutlass

namespace bu8 = baracuda_cutlass::bias_u8_rcr;

#define BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(EPI_SUFFIX, ACTIVATION_OP, BIAS_SUFFIX, BIAS_ELEM)            \
    extern "C" int baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_u8_rcr_sm80_run(                \
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
        return bu8::run_impl<ACTIVATION_OP, BIAS_ELEM>(                                                 \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,                                              \
            alpha, beta, workspace, workspace_bytes,                                                    \
            static_cast<cudaStream_t>(stream));                                                         \
    }                                                                                                   \
                                                                                                        \
    extern "C" std::size_t baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_u8_rcr_sm80_workspace_size( \
        int m, int n, int k)                                                                            \
    {                                                                                                   \
        return bu8::workspace_impl<ACTIVATION_OP, BIAS_ELEM>(m, n, k);                                  \
    }                                                                                                   \
                                                                                                        \
    extern "C" int baracuda_cutlass_gemm_##EPI_SUFFIX##_##BIAS_SUFFIX##_u8_rcr_sm80_can_implement(      \
        int m, int n, int k,                                                                            \
        const void* a, int64_t lda,                                                                     \
        const void* b, int64_t ldb,                                                                     \
        const void* c, int64_t ldc,                                                                     \
        void*       d, int64_t ldd,                                                                     \
        const void* bias)                                                                               \
    {                                                                                                   \
        return bu8::can_implement_impl<ACTIVATION_OP, BIAS_ELEM>(                                       \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);                                             \
    }

BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(bias,      bu8::ActIdentity, f32bias, float)
BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(bias_relu, bu8::ActReLu,     f32bias, float)
BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(bias_gelu, bu8::ActGELU,     f32bias, float)
BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(bias_silu, bu8::ActSiLu,     f32bias, float)
BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(bias,      bu8::ActIdentity, i32bias, int32_t)
BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(bias_relu, bu8::ActReLu,     i32bias, int32_t)
BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(bias_gelu, bu8::ActGELU,     i32bias, int32_t)
BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES(bias_silu, bu8::ActSiLu,     i32bias, int32_t)

#undef BARACUDA_U8_BIAS_RCR_GEMM_ENTRIES
