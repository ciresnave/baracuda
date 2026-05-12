// f32 GEMM via SIMT (CUDA cores, no tensor cores), RCR layout, sm_80
// instantiation.
//
// The strict-precision counterpart to gemm_tf32_rcr_sm80.cu. Inputs,
// accumulator, and output are all IEEE 754 binary32 throughout. Slower
// than the TF32 path but produces bit-identical results across runs on
// the same hardware (no tensor-core warp-reduction nondeterminism in
// the last bit). The natural choice for consumers who pick precision
// over throughput — analogous to cuBLAS's full-precision F32 GEMM.
//
// Layout RCR (matches the rest of the RCR family):
//   A: row-major    [M, K]
//   B: column-major [K, N]
//   C: row-major    [M, N] (optional)
//   D: row-major    [M, N]
//
// Tile shape is CUTLASS's standard SIMT GEMM tile: 128×128 threadblock,
// 32×64 warp, 1×1×1 instruction (no MMA), 1 element per access (SIMT
// doesn't have the vectorized memory access pattern that tensor cores
// have). 2 pipeline stages (vs 3 on the tensor-core kernels) matches
// CUTLASS's reference SIMT instantiation.
//
// Status code mapping matches the other kernels (see gemm_rcr_sm80.cu).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace baracuda_cutlass {
namespace f32_simt_rcr {

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using IdentityEpilogue = cutlass::epilogue::thread::LinearCombination<
    float, 1, ElementAcc, ElementAcc>;

using GemmF32SimtRcrSm80 = cutlass::gemm::device::Gemm<
    float, RowMajor,
    float, ColumnMajor,
    float, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    IdentityEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

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
    using Gemm = GemmF32SimtRcrSm80;

    auto const* c_eff   = c ? static_cast<const float*>(c)
                            : static_cast<const float*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const float*>(a), static_cast<int>(lda)},
        {static_cast<const float*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<float*>(d), static_cast<int>(ldd)},
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

static std::size_t workspace_impl(int m, int n, int k) {
    using Gemm = GemmF32SimtRcrSm80;
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

static int can_implement_impl(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    using Gemm = GemmF32SimtRcrSm80;

    auto const* c_eff   = c ? static_cast<const float*>(c)
                            : static_cast<const float*>(d);
    int64_t      ldc_eff = c ? ldc : ldd;

    typename Gemm::Arguments args{
        {m, n, k},
        {static_cast<const float*>(a), static_cast<int>(lda)},
        {static_cast<const float*>(b), static_cast<int>(ldb)},
        {c_eff, static_cast<int>(ldc_eff)},
        {static_cast<float*>(d), static_cast<int>(ldd)},
        {1.0f, 0.0f}
    };

    Gemm op;
    auto status = op.can_implement(args);
    if (status == cutlass::Status::kSuccess)              return 0;
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    return 5;
}

}  // namespace f32_simt_rcr
}  // namespace baracuda_cutlass

extern "C" {

int baracuda_cutlass_gemm_f32_simt_rcr_sm80_run(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd,
    float alpha, float beta,
    void* workspace, std::size_t workspace_bytes,
    void* stream)
{
    return baracuda_cutlass::f32_simt_rcr::run_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
        alpha, beta, workspace, workspace_bytes,
        static_cast<cudaStream_t>(stream));
}

std::size_t baracuda_cutlass_gemm_f32_simt_rcr_sm80_workspace_size(int m, int n, int k) {
    return baracuda_cutlass::f32_simt_rcr::workspace_impl(m, n, k);
}

int baracuda_cutlass_gemm_f32_simt_rcr_sm80_can_implement(
    int m, int n, int k,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    const void* c, int64_t ldc,
    void*       d, int64_t ldd)
{
    return baracuda_cutlass::f32_simt_rcr::can_implement_impl(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd);
}

}  // extern "C"
