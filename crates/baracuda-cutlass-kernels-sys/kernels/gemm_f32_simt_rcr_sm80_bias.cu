// f32 SIMT (CUDA cores) bias-fused GEMM with optional activation,
// RCR layout, sm_80 instantiation.
//
// The strict-precision counterpart to gemm_tf32_rcr_sm80_bias.cu.
// Computes: D = activation(alpha * (A * B) + beta * C + bias_broadcast(N))
// with full IEEE 754 binary32 throughout (inputs, accumulator, output,
// and the bias vector). Activations: {Identity, ReLu, GELU, SiLu}.
//
// Layout RCR:
//   A: row-major    [M, K]
//   B: column-major [K, N]
//   C: row-major    [M, N] (optional)
//   D: row-major    [M, N]
// Bias: contiguous [N] vector (one element per output column)
//
// Tile shape matches the non-bias SIMT kernel — 128×128 threadblock,
// 32×64 warp, 1×1×1 instruction shape (no MMA), 1 element per access.
// Status code mapping matches the other kernels.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
// Vendored partial specialization of `DefaultGemmWithBroadcast` for
// `OpClassSimt`. MUST be included BEFORE
// `gemm_universal_with_broadcast.h` so the SIMT routing is visible at
// template-instantiation time. See the header for full rationale.
#include "baracuda_simt_broadcast_epilogue.h"
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>

namespace baracuda_cutlass {
namespace bias_f32_simt_rcr {

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

// 1 element per access — SIMT doesn't vectorize epilogue stores the
// same way tensor cores do.
template <typename ActivationOp>
using BiasActEpilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    float,                                      // ElementC (and D output)
    ElementAcc,                                 // ElementAccumulator
    ElementAcc,                                 // ElementCompute
    float,                                      // ElementZ (= D)
    float,                                      // ElementT (auxiliary; unused)
    1,                                          // ElementsPerAccess
    ActivationOp,                               // pluggable elementwise
    cutlass::plus<ElementAcc>,                  // BinaryOp (z = AB + V)
    /* StoreT = */ false,
    float                                       // ElementVector (bias dtype)
>;

template <typename ActivationOp>
using GemmF32SimtBiasRcrSm80 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    float, RowMajor,
    float, ColumnMajor,
    float, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    BiasActEpilogue<ActivationOp>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

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
    using Gemm = GemmF32SimtBiasRcrSm80<ActivationOp>;

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
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc_eff),
        static_cast<int>(ldd),
        /* ldr */ 0, /* ldt */ 0
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
    using Gemm = GemmF32SimtBiasRcrSm80<ActivationOp>;
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

template <typename ActivationOp>
static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    const void* bias)
{
    using Gemm = GemmF32SimtBiasRcrSm80<ActivationOp>;

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

using ActIdentity = cutlass::epilogue::thread::Identity<ElementAcc>;
using ActReLu     = cutlass::epilogue::thread::ReLu<ElementAcc>;
using ActGELU     = cutlass::epilogue::thread::GELU<ElementAcc>;
using ActSiLu     = cutlass::epilogue::thread::SiLu<ElementAcc>;

}  // namespace bias_f32_simt_rcr
}  // namespace baracuda_cutlass

namespace bs = baracuda_cutlass::bias_f32_simt_rcr;

#define BARACUDA_F32SIMT_BIAS_RCR_GEMM_ENTRIES(SUFFIX, ACTIVATION_OP)                              \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_f32_simt_rcr_sm80_run(                         \
        int m, int n, int k,                                                                       \
        const void* a, int64_t lda,                                                                \
        const void* b, int64_t ldb,                                                                \
        const void* c, int64_t ldc,                                                                \
        void*       d, int64_t ldd,                                                                \
        const void* bias,                                                                          \
        float alpha, float beta,                                                                   \
        void* workspace, std::size_t workspace_bytes,                                              \
        void* stream)                                                                              \
    {                                                                                              \
        return bs::run_impl<ACTIVATION_OP>(                                                        \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,                                         \
            alpha, beta, workspace, workspace_bytes,                                               \
            static_cast<cudaStream_t>(stream));                                                    \
    }                                                                                              \
                                                                                                   \
    extern "C" std::size_t baracuda_cutlass_gemm_##SUFFIX##_f32_simt_rcr_sm80_workspace_size(      \
        int m, int n, int k)                                                                       \
    {                                                                                              \
        return bs::workspace_impl<ACTIVATION_OP>(m, n, k);                                         \
    }                                                                                              \
                                                                                                   \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_f32_simt_rcr_sm80_can_implement(               \
        int m, int n, int k,                                                                       \
        const void* a, int64_t lda,                                                                \
        const void* b, int64_t ldb,                                                                \
        const void* c, int64_t ldc,                                                                \
        void*       d, int64_t ldd,                                                                \
        const void* bias)                                                                          \
    {                                                                                              \
        return bs::can_implement_impl<ACTIVATION_OP>(                                              \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);                                        \
    }

BARACUDA_F32SIMT_BIAS_RCR_GEMM_ENTRIES(bias,      bs::ActIdentity)
BARACUDA_F32SIMT_BIAS_RCR_GEMM_ENTRIES(bias_relu, bs::ActReLu)
BARACUDA_F32SIMT_BIAS_RCR_GEMM_ENTRIES(bias_gelu, bs::ActGELU)
BARACUDA_F32SIMT_BIAS_RCR_GEMM_ENTRIES(bias_silu, bs::ActSiLu)

#undef BARACUDA_F32SIMT_BIAS_RCR_GEMM_ENTRIES
