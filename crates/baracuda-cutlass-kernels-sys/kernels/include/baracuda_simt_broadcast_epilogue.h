// Vendored partial specialization of `cutlass::gemm::kernel::DefaultGemmWithBroadcast`
// for `OperatorClass = cutlass::arch::OpClassSimt`.
//
// Why this exists
// ---------------
// CUTLASS 4.x's `DefaultGemmWithBroadcast` (primary template in
// `cutlass/gemm/kernel/default_gemm_with_broadcast.h`) unconditionally
// routes its epilogue through `DefaultEpilogueWithBroadcastTensorOp`,
// regardless of the `OperatorClass` template parameter. The TensorOp
// epilogue path includes `default_epilogue_tensor_op.h`, which fails to
// compile when the underlying mainloop is `OpClassSimt` (it tries to
// derive an MMA-tensor-op iterator that doesn't exist for SIMT).
//
// `cutlass::epilogue::threadblock::DefaultEpilogueWithBroadcastSimt`
// already exists in CUTLASS (see
// `cutlass/epilogue/threadblock/default_epilogue_with_broadcast.h`) —
// it just isn't reachable through the device-level
// `GemmUniversalWithBroadcast` template because nothing in the kernel-
// level selector routes SIMT to it.
//
// What this header does
// ---------------------
// Adds a partial specialization on `OperatorClass = OpClassSimt` that
// reroutes the epilogue to `DefaultEpilogueWithBroadcastSimt`. This is
// the smallest possible vendor patch — no CUTLASS internals are
// duplicated; we just wire the existing SIMT broadcast epilogue into
// the kernel-level chain that `GemmUniversalWithBroadcast` reaches via
// `DefaultGemmWithBroadcast`.
//
// Include this header BEFORE any `#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>`
// in a translation unit that needs SIMT bias + broadcast GEMM. The
// primary `DefaultGemmWithBroadcast` template still applies for all
// other OperatorClass values (TensorOp, etc), so this is a strictly
// additive change at the template-resolution level.
//
// Attribution
// -----------
// Structure mirrors NVIDIA CUTLASS's existing partial specializations
// in `default_gemm_with_broadcast.h` (BSD-3-Clause). The SIMT routing
// follows the same shape as the Sm70 partial specialization in that
// file. See `NOTICE` for CUTLASS attribution.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_with_fused_epilogue.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm_with_broadcast.h"
#include "cutlass/epilogue/threadblock/default_epilogue_with_broadcast.h"

namespace cutlass {
namespace gemm {
namespace kernel {

/// Partial specialization for `OperatorClass = OpClassSimt`.
///
/// Reaches the same `GemmBase::Mma` mainloop as the primary template
/// (via `DefaultGemmUniversal` with `OpClassSimt`), but composes the
/// epilogue through `DefaultEpilogueWithBroadcastSimt` instead of
/// `DefaultEpilogueWithBroadcastTensorOp`. The SIMT broadcast epilogue
/// takes a different template-argument list (no `PartitionsK`), which
/// is why this specialization is structurally distinct from the
/// primary.
template <
  typename ElementA_,
  typename LayoutA_,
  cutlass::ComplexTransform TransformA,
  int kAlignmentA,
  typename ElementB_,
  typename LayoutB_,
  cutlass::ComplexTransform TransformB,
  int kAlignmentB,
  typename ElementC_,
  typename LayoutC_,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename Operator
>
struct DefaultGemmWithBroadcast<
  ElementA_, LayoutA_, TransformA, kAlignmentA,
  ElementB_, LayoutB_, TransformB, kAlignmentB,
  ElementC_, LayoutC_,
  ElementAccumulator,
  cutlass::arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  Operator,
  void
> {

  using GemmBase = typename DefaultGemmUniversal<
    ElementA_, LayoutA_, TransformA, kAlignmentA,
    ElementB_, LayoutB_, TransformB, kAlignmentB,
    ElementC_, LayoutC_,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    Operator
  >::GemmKernel;

  // SIMT broadcast epilogue — note the template argument list differs
  // from the TensorOp variant (no PartitionsK).
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithBroadcastSimt<
    typename GemmBase::Epilogue::Shape,
    typename GemmBase::Epilogue::WarpMmaOperator,
    ElementC_,
    typename EpilogueOutputOp::ElementT,
    typename EpilogueOutputOp::ElementVector,
    EpilogueOutputOp,
    GemmBase::Epilogue::kElementsPerAccess
  >::Epilogue;

  using GemmKernel = GemmWithFusedEpilogue<
    typename GemmBase::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
