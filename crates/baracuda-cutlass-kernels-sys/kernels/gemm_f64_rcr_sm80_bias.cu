// f64 (DGEMM) bias-fused GEMM with optional activation, RCR layout, sm_80.
//
// Computes: D = activation(alpha * (A * B) + beta * C + bias_broadcast(N))
// with full IEEE 754 binary64 throughout (inputs, accumulator, scalars,
// output, bias). Activations: {Identity, ReLu, GELU, SiLu}.
//
// Layout RCR (matches gemm_f64_rcr_sm80.cu):
//   A: row-major    [M, K]
//   B: column-major [K, N]
//   C: row-major    [M, N] (optional)
//   D: row-major    [M, N]
// Bias: contiguous [N] f64 vector.
//
// Tile shape matches the non-bias f64 kernel: 128x128 threadblock,
// 32x64 warp, 8x8x4 instruction (DGEMM mma). 1 element per access.

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
namespace bias_f64_rcr {

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = double;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

template <typename ActivationOp>
using BiasActEpilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    double,
    ElementAcc,
    ElementAcc,
    double,
    double,
    1,                                          // ElementsPerAccess (DGEMM)
    ActivationOp,
    cutlass::plus<ElementAcc>,
    /* StoreT = */ false,
    double                                      // ElementVector (bias dtype)
>;

template <typename ActivationOp>
using GemmF64BiasRcrSm80 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    double, RowMajor,
    double, ColumnMajor,
    double, RowMajor,
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
    double alpha, double beta,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream)
{
    using Gemm = GemmF64BiasRcrSm80<ActivationOp>;

    auto const* c_eff   = c    ? c    : d;
    int64_t      ldc_eff = c   ? ldc  : ldd;

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename Gemm::EpilogueOutputOp::Params{alpha, beta},
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
    using Gemm = GemmF64BiasRcrSm80<ActivationOp>;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename Gemm::EpilogueOutputOp::Params{1.0, 0.0},
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
    using Gemm = GemmF64BiasRcrSm80<ActivationOp>;

    auto const* c_eff   = c    ? c    : d;
    int64_t      ldc_eff = c   ? ldc  : ldd;

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename Gemm::EpilogueOutputOp::Params{1.0, 0.0},
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

}  // namespace bias_f64_rcr
}  // namespace baracuda_cutlass

namespace bf64 = baracuda_cutlass::bias_f64_rcr;

#define BARACUDA_F64_BIAS_RCR_GEMM_ENTRIES(SUFFIX, ACTIVATION_OP)                                  \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_f64_rcr_sm80_run(                              \
        int m, int n, int k,                                                                       \
        const void* a, int64_t lda,                                                                \
        const void* b, int64_t ldb,                                                                \
        const void* c, int64_t ldc,                                                                \
        void*       d, int64_t ldd,                                                                \
        const void* bias,                                                                          \
        double alpha, double beta,                                                                 \
        void* workspace, std::size_t workspace_bytes,                                              \
        void* stream)                                                                              \
    {                                                                                              \
        return bf64::run_impl<ACTIVATION_OP>(                                                      \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,                                         \
            alpha, beta, workspace, workspace_bytes,                                               \
            static_cast<cudaStream_t>(stream));                                                    \
    }                                                                                              \
                                                                                                   \
    extern "C" std::size_t baracuda_cutlass_gemm_##SUFFIX##_f64_rcr_sm80_workspace_size(           \
        int m, int n, int k)                                                                       \
    {                                                                                              \
        return bf64::workspace_impl<ACTIVATION_OP>(m, n, k);                                       \
    }                                                                                              \
                                                                                                   \
    extern "C" int baracuda_cutlass_gemm_##SUFFIX##_f64_rcr_sm80_can_implement(                    \
        int m, int n, int k,                                                                       \
        const void* a, int64_t lda,                                                                \
        const void* b, int64_t ldb,                                                                \
        const void* c, int64_t ldc,                                                                \
        void*       d, int64_t ldd,                                                                \
        const void* bias)                                                                          \
    {                                                                                              \
        return bf64::can_implement_impl<ACTIVATION_OP>(                                            \
            m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias);                                        \
    }

BARACUDA_F64_BIAS_RCR_GEMM_ENTRIES(bias,      bf64::ActIdentity)
BARACUDA_F64_BIAS_RCR_GEMM_ENTRIES(bias_relu, bf64::ActReLu)
BARACUDA_F64_BIAS_RCR_GEMM_ENTRIES(bias_gelu, bf64::ActGELU)
BARACUDA_F64_BIAS_RCR_GEMM_ENTRIES(bias_silu, bf64::ActSiLu)

#undef BARACUDA_F64_BIAS_RCR_GEMM_ENTRIES
