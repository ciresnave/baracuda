// f16 + bf16 grouped GEMM, RCR layout, sm_80 instantiation.
//
// Per the Fuel team's design review, baracuda-cutlass treats grouped GEMM
// as a first-class op family for MoE workloads (variable M per expert,
// shared N/K). This file ships the v0 sm_80 instantiations.
//
// Layout RCR (per-group, identical to single-GEMM):
//   A: row-major    [M, K], lda elements per row
//   B: column-major [K, N], ldb elements per column
//   C: row-major    [M, N], ldc elements per row (optional; pass nullptr
//                                                 + beta = 0 to skip)
//   D: row-major    [M, N], ldd elements per row
//
// Accumulator and alpha/beta are FP32. Epilogue: Identity only
// (D = alpha * AB + beta * C). v0 uses one shared (alpha, beta) pair
// across all groups; per-group epilogues require LinearCombinationGeneric
// and would land in a follow-up.
//
// Workspace contract (caller-supplied; see baracuda-cutlass plan.rs for the
// metadata layout): the safe Rust layer packs problem_sizes, ptr arrays,
// and ld arrays into the front of the workspace, then hands us pointers
// at fixed offsets. The CUTLASS internal scratch (size from
// `*_scratch_bytes`) lives at the tail.
//
// Status code mapping (matches src/lib.rs Rust extern decls):
//   0 = success, 1 = misaligned, 2 = invalid, 3 = unsupported,
//   4 = workspace null/too small, 5 = internal.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace baracuda_cutlass {

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;
using ElementAcc  = float;

// Same Ampere standard tile as the single-GEMM RCR sm_80 kernel —
// keeps code shape consistent and matches well across MoE expert sizes.
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

template <typename Element>
using IdentityEpilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 8, ElementAcc, ElementAcc>;

template <typename Element>
using GroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, RowMajor,
    cutlass::ComplexTransform::kNone,
    8,
    Element, ColumnMajor,
    cutlass::ComplexTransform::kNone,
    8,
    Element, RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    IdentityEpilogue<Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
>::GemmKernel;

template <typename Element>
using GroupedGemm = cutlass::gemm::device::GemmGrouped<GroupedKernel<Element>>;

template <typename Element>
static int sufficient_impl(
    const int* h_m, const int* h_n, const int* h_k, int group_count)
{
    // Pack host problem_sizes for CUTLASS's static `sufficient` query.
    // It only reads the array; we drop it on return.
    cutlass::gemm::GemmCoord stack_buf[64];
    cutlass::gemm::GemmCoord* problems = nullptr;
    bool heap = group_count > 64;
    if (heap) {
        problems = new cutlass::gemm::GemmCoord[group_count];
    } else {
        problems = stack_buf;
    }
    for (int i = 0; i < group_count; ++i) {
        problems[i] = cutlass::gemm::GemmCoord{h_m[i], h_n[i], h_k[i]};
    }
    int tb = GroupedGemm<Element>::sufficient(problems, group_count);
    if (heap) delete[] problems;
    return tb;
}

template <typename Element>
static std::size_t scratch_bytes_impl(
    const int* h_m, const int* h_n, const int* h_k,
    int group_count, int threadblock_count)
{
    using Gemm = GroupedGemm<Element>;

    // Build a minimal Arguments struct with only the fields
    // get_workspace_size touches (problem_sizes_host + count).
    cutlass::gemm::GemmCoord stack_buf[64];
    cutlass::gemm::GemmCoord* problems = nullptr;
    bool heap = group_count > 64;
    if (heap) {
        problems = new cutlass::gemm::GemmCoord[group_count];
    } else {
        problems = stack_buf;
    }
    for (int i = 0; i < group_count; ++i) {
        problems[i] = cutlass::gemm::GemmCoord{h_m[i], h_n[i], h_k[i]};
    }

    typename Gemm::Arguments args(
        /*problem_sizes_device*/ nullptr,
        group_count,
        threadblock_count,
        /*epilogue_op*/ {1.0f, 0.0f},
        /*ptr_A*/ nullptr, /*ptr_B*/ nullptr, /*ptr_C*/ nullptr, /*ptr_D*/ nullptr,
        /*lda*/ nullptr, /*ldb*/ nullptr, /*ldc*/ nullptr, /*ldd*/ nullptr,
        /*host_problem_sizes*/ problems
    );

    std::size_t bytes = Gemm::get_workspace_size(args);
    if (heap) delete[] problems;
    return bytes;
}

template <typename Element>
static int run_impl(
    int group_count,
    int threadblock_count,
    const void* d_problem_sizes,
    const void* d_ptr_a,
    const void* d_ptr_b,
    const void* d_ptr_c,
    void*       d_ptr_d,
    const void* d_lda,
    const void* d_ldb,
    const void* d_ldc,
    const void* d_ldd,
    const void* h_problem_sizes,
    float alpha, float beta,
    void* scratch, std::size_t scratch_bytes,
    cudaStream_t stream)
{
    using Gemm = GroupedGemm<Element>;

    // Cast through reinterpret_cast — the layout of GemmCoord (Coord<3, int>)
    // is just int[3], which the Rust side packs as i32x3.
    auto problems_d = const_cast<cutlass::gemm::GemmCoord*>(
        reinterpret_cast<const cutlass::gemm::GemmCoord*>(d_problem_sizes));
    auto problems_h = const_cast<cutlass::gemm::GemmCoord*>(
        reinterpret_cast<const cutlass::gemm::GemmCoord*>(h_problem_sizes));

    typename Gemm::Arguments args(
        problems_d,
        group_count,
        threadblock_count,
        {alpha, beta},
        const_cast<Element**>(reinterpret_cast<const Element* const*>(d_ptr_a)),
        const_cast<Element**>(reinterpret_cast<const Element* const*>(d_ptr_b)),
        const_cast<Element**>(reinterpret_cast<const Element* const*>(d_ptr_c)),
        reinterpret_cast<Element**>(d_ptr_d),
        const_cast<int64_t*>(reinterpret_cast<const int64_t*>(d_lda)),
        const_cast<int64_t*>(reinterpret_cast<const int64_t*>(d_ldb)),
        const_cast<int64_t*>(reinterpret_cast<const int64_t*>(d_ldc)),
        const_cast<int64_t*>(reinterpret_cast<const int64_t*>(d_ldd)),
        problems_h
    );

    Gemm op;

    auto status = op.can_implement(args);
    if (status == cutlass::Status::kErrorMisalignedOperand) return 1;
    if (status == cutlass::Status::kErrorInvalidProblem)    return 2;
    if (status == cutlass::Status::kErrorNotSupported)      return 3;
    if (status != cutlass::Status::kSuccess)                return 5;

    std::size_t needed = Gemm::get_workspace_size(args);
    if (scratch_bytes < needed) return 4;

    status = op.initialize(args, scratch, stream);
    if (status != cutlass::Status::kSuccess) return 5;

    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) return 5;

    return 0;
}

template <typename Element>
static int can_implement_impl(
    const int* h_m, const int* h_n, const int* h_k, int group_count)
{
    // Per-problem alignment / shape sanity. CUTLASS doesn't have a
    // standalone grouped can_implement, so we do the same checks the
    // single-problem can_implement_impl does, applied per group.
    using Gemm = GroupedGemm<Element>;
    static constexpr int kAlign = 8;  // 8-element-per-access for f16/bf16

    for (int i = 0; i < group_count; ++i) {
        if (h_m[i] <= 0 || h_n[i] <= 0 || h_k[i] <= 0) return 2;
        // Inner reduction stride (k) must be aligned for the f16/bf16 tile.
        if (h_k[i] % kAlign != 0) return 1;
        if (h_n[i] % kAlign != 0) return 1;
    }
    (void)Gemm{};  // keep the symbol referenced
    return 0;
}

}  // namespace baracuda_cutlass

// ============================================================================
// C ABI surface — symbol names match the extern decls in src/lib.rs
// ============================================================================

extern "C" {

// ---------- f16 ----------

int baracuda_cutlass_grouped_gemm_f16_rcr_sm80_sufficient(
    const int* h_m, const int* h_n, const int* h_k, int group_count)
{
    return baracuda_cutlass::sufficient_impl<cutlass::half_t>(
        h_m, h_n, h_k, group_count);
}

std::size_t baracuda_cutlass_grouped_gemm_f16_rcr_sm80_scratch_bytes(
    const int* h_m, const int* h_n, const int* h_k,
    int group_count, int threadblock_count)
{
    return baracuda_cutlass::scratch_bytes_impl<cutlass::half_t>(
        h_m, h_n, h_k, group_count, threadblock_count);
}

int baracuda_cutlass_grouped_gemm_f16_rcr_sm80_can_implement(
    const int* h_m, const int* h_n, const int* h_k, int group_count)
{
    return baracuda_cutlass::can_implement_impl<cutlass::half_t>(
        h_m, h_n, h_k, group_count);
}

int baracuda_cutlass_grouped_gemm_f16_rcr_sm80_run(
    int group_count,
    int threadblock_count,
    const void* d_problem_sizes,
    const void* d_ptr_a,
    const void* d_ptr_b,
    const void* d_ptr_c,
    void*       d_ptr_d,
    const void* d_lda,
    const void* d_ldb,
    const void* d_ldc,
    const void* d_ldd,
    const void* h_problem_sizes,
    float alpha, float beta,
    void* scratch, std::size_t scratch_bytes,
    void* stream)
{
    return baracuda_cutlass::run_impl<cutlass::half_t>(
        group_count, threadblock_count,
        d_problem_sizes,
        d_ptr_a, d_ptr_b, d_ptr_c, d_ptr_d,
        d_lda, d_ldb, d_ldc, d_ldd,
        h_problem_sizes,
        alpha, beta,
        scratch, scratch_bytes,
        static_cast<cudaStream_t>(stream));
}

// ---------- bf16 ----------

int baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_sufficient(
    const int* h_m, const int* h_n, const int* h_k, int group_count)
{
    return baracuda_cutlass::sufficient_impl<cutlass::bfloat16_t>(
        h_m, h_n, h_k, group_count);
}

std::size_t baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_scratch_bytes(
    const int* h_m, const int* h_n, const int* h_k,
    int group_count, int threadblock_count)
{
    return baracuda_cutlass::scratch_bytes_impl<cutlass::bfloat16_t>(
        h_m, h_n, h_k, group_count, threadblock_count);
}

int baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_can_implement(
    const int* h_m, const int* h_n, const int* h_k, int group_count)
{
    return baracuda_cutlass::can_implement_impl<cutlass::bfloat16_t>(
        h_m, h_n, h_k, group_count);
}

int baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_run(
    int group_count,
    int threadblock_count,
    const void* d_problem_sizes,
    const void* d_ptr_a,
    const void* d_ptr_b,
    const void* d_ptr_c,
    void*       d_ptr_d,
    const void* d_lda,
    const void* d_ldb,
    const void* d_ldc,
    const void* d_ldd,
    const void* h_problem_sizes,
    float alpha, float beta,
    void* scratch, std::size_t scratch_bytes,
    void* stream)
{
    return baracuda_cutlass::run_impl<cutlass::bfloat16_t>(
        group_count, threadblock_count,
        d_problem_sizes,
        d_ptr_a, d_ptr_b, d_ptr_c, d_ptr_d,
        d_lda, d_ldb, d_ldc, d_ldd,
        h_problem_sizes,
        alpha, beta,
        scratch, scratch_bytes,
        static_cast<cudaStream_t>(stream));
}

}  // extern "C"
