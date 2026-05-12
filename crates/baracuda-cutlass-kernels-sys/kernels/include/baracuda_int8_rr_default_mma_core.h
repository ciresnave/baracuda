// Vendored partial specialization of
// `cutlass::gemm::threadblock::DefaultMmaCore` for
// `int8 × RowMajor × RowMajor × OpClassTensorOp × Sm80`, routing B
// through `ColumnMajorTensorOpMultiplicandCrosswise` shared-memory
// layout (instead of upstream's `RowMajor...Congruous`).
//
// Why this exists
// ---------------
// CUTLASS 4.2.0's `default_mma_core_sm80.h` (line ~1669) picks, for the
// generic `RowMajor × RowMajor × OpClassTensorOp` chain:
//
//   SmemLayoutA = RowMajorTensorOpMultiplicandCrosswise<sizeof_bits<ElementA>, Shape::kK>
//   SmemLayoutB = RowMajorTensorOpMultiplicandCongruous<sizeof_bits<ElementB>, Crosswise_B>
//
// At 16-bit (f16, bf16) this works. At 8-bit (int8, uint8) the
// `Congruous<8, _>` smem layout for B doesn't compose with what
// `mma.sync.m16n8k32.s8` expects in its register fragment: the b16
// chunks emitted by `ldmatrix.x4.trans` over Congruous-8 smem pack
// N-adjacent bytes, while the mma needs K-adjacent bytes within each
// 32-bit register. The vendored 8-bit Congruous warp iterator in
// `baracuda_int8_congruous_warp_iterator.h` is a necessary precondition
// for that path to even *compile*, but GPU validation showed it still
// produces incorrect results (Phase 2b post-mortem captured the s8 /
// u8 numerical asymmetry).
//
// What this header does
// ---------------------
// Adds a more-specific partial specialisation (fixed `ElementA ==
// ElementB == 8-bit integer`) of `DefaultMmaCore` that:
//
//   1. Keeps A unchanged — `RowMajorTensorOpMultiplicandCrosswise<8, Shape::kK>`
//      already composes correctly at 8-bit (A's gmem is K-contig
//      row-major, smem is K-contig Crosswise, natural fit).
//   2. Switches B's smem layout to
//      `ColumnMajorTensorOpMultiplicandCrosswise<8, Shape::kK>` so each
//      b16 chunk read by `ldmatrix.x4` packs K-adjacent bytes — what
//      `mma.sync.m16n8k32.s8` expects.
//
//      Naming detail: `ColumnMajorTensorOpMultiplicandCrosswise` maps
//      matrix `(row, col)` directly to PitchLinearCoord `(row, col)`,
//      so for B[K, N] (row=K, col=N) it puts K-contig in the
//      pitch-linear view — exactly what we want regardless of the fact
//      that the gmem layout is RowMajor.
//   3. Reuses CUTLASS's existing warp-level iterator at
//      `mma_tensor_op_tile_iterator.h:~2149` (generic on
//      `TensorOpMultiplicandCrosswise<sizeof_bits, Crosswise>`), so no
//      vendored warp iterator is needed for the working path.
//   4. Bridges the gmem → smem layout mismatch with
//      `TransposePitchLinearThreadMap`: gmem stays N-contig (good
//      coalescing on row-major B[K,N]), smem stores land at K-contig
//      positions inside the Crosswise smem via the transposed thread
//      arrangement. The pattern mirrors CUTLASS's column-major-
//      interleaved spec at `default_mma_core_sm80.h:~1821` — the only
//      existing upstream spec that bridges gmem-contig ≠ smem-contig.
//
// WarpThreadArrangement choice
// ----------------------------
// `TransposePitchLinearThreadMap` requires the underlying ThreadMap's
// `Iterations::kStrided == 1` (after transpose, that becomes the new
// `Iterations::kContiguous == 1`, which the primitive asserts).
//
// For the default 128×256×64 threadblock with 64×64×64 warp shape
// (`kThreads = 8 × 32 = 256`, `kEPA = 16`), set
// `WarpThreadArrangement_B = <4, 8>` (4 contig × 8 strided lanes per
// warp):
//
//   ShapeInAccesses = <Shape::kN / kEPA, Shape::kK> = <16, 64>
//   WarpAccessIter  = <16/4, 64/8>                 = <4, 8>
//   kWarpsStrided   = min(8, kWarpCount=8)         = 8
//   kWarpsContig    = 8 / 8                        = 1
//   Iterations      = <4/1, 8/8>                   = <4, 1>  ✓
//
// Coalesced gmem load: 4 lanes × 16 bytes = 64-byte transactions on B.
// Smem stores after transpose land K-contiguously into ColumnMajor
// Crosswise, served by the same swizzle that the existing A side uses.
//
// If a future tile shape doesn't satisfy `Iterations::kStrided == 1`
// with this arrangement, the `TransposePitchLinearThreadMap` static_assert
// will fire at instantiation. The static_assert below pre-empts that
// with a clearer message for the default tile.
//
// Why a specialisation on fixed `int8_t` / `uint8_t` (not SFINAE)
// --------------------------------------------------------------
// CUTLASS's primary `DefaultMmaCore` has no SFINAE escape parameter.
// To win partial-order selection against the generic RR spec at line
// ~1669 (parameterised on `ElementA_`, `ElementB_`), the vendor spec
// must fix at least one type. Fixing both `ElementA == ElementB ==
// int8_t` (and a sibling for `uint8_t`) gives a strictly-more-specific
// match without touching the unsigned, mixed-precision, or 16-bit paths.
//
// Attribution
// -----------
// Structurally derived from CUTLASS 4.2.0
// `include/cutlass/gemm/threadblock/default_mma_core_sm80.h`, lines
// ~1640–1774 (the `A: row-major, B: row-major, OpClassTensorOp` partial
// spec) and ~1821–1965 (the column-major-interleaved spec's
// `TransposePitchLinearThreadMap` pattern). Both are
// BSD-3-Clause-licensed; see CUTLASS `LICENSE.txt`.

#pragma once

#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/cache_operation.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor_op_multiplicand_sm75.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/platform/platform.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/regular_tile_access_iterator.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

namespace baracuda_int8_rr_detail {

// Shared body for the int8_t / uint8_t × RowMajor × RowMajor partial
// specialisations.
template <
    typename Shape_,
    typename WarpShape_,
    typename InstructionShape_,
    typename Element_,
    typename ElementC_,
    typename LayoutC_,
    int Stages,
    typename Operator_,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCoreInt8RrBody {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = Element_;
  using LayoutA = layout::RowMajor;
  using ElementB = Element_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN,
                              Shape::kK / WarpShape::kK>;

  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access (128 bits)
  static int const kAccessSizeInBits = 128;
  static int const kElementsPerAccess =
      kAccessSizeInBits / sizeof_bits<Element_>::value;

  /// Default Operator (e.g. OpMultiplyAddSaturate)
  using Operator = Operator_;

  //
  // A side: identical to upstream RR spec — K is gmem-contig (row-major
  // A) and smem-contig (RowMajorTensorOpMultiplicandCrosswise<8, K>).
  // Natural fit, no transposing iterator.
  //
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / kElementsPerAccess;
  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK>;

  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kElementsPerAccess>;

  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  //
  // B side: gmem is N-contig (row-major B[K,N]); smem is K-contig via
  // ColumnMajorTensorOpMultiplicandCrosswise<8, K> (so ldmatrix.x4
  // packs K-adjacent bytes for mma.sync.m16n8k32.s8). The gmem→smem
  // contig mismatch is bridged with TransposePitchLinearThreadMap.
  //
  // WarpThreadArrangement = <4, 8> chosen so the gmem ThreadMap has
  // Iterations::kStrided == 1 (required for the post-transpose
  // Iterations::kContiguous == 1 assert inside
  // TransposePitchLinearThreadMap). See the file header for the
  // arithmetic.
  //
  static int const kWarpThreadArrangementContiguousB = 4;
  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;  // 8

  static_assert(kWarpThreadArrangementContiguousB *
                        kWarpThreadArrangementStridedB ==
                    kWarpSize,
                "WarpThreadArrangement product must equal warp size.");

  static_assert(
      Shape::kK == kWarpThreadArrangementStridedB * 8,
      "baracuda int8 RR DefaultMmaCore: this specialisation is tuned "
      "for Shape::kK == 64 (matches WarpThreadArrangement<4, 8> with "
      "8-iteration warp-strided coverage). Re-derive the arrangement "
      "for a different K.");

  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, Shape::kK>;

  /// Gmem-source ThreadMap: PitchLinearShape<N, K> = N-contig, K-stride
  /// (natural row-major B view, coalesced gmem loads).
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kElementsPerAccess>;

  /// Smem-store ThreadMap: same threads, transposed so the smem store
  /// follows the K-contig pitch-linear view of
  /// ColumnMajorTensorOpMultiplicandCrosswise.
  using SmemThreadMapB = transform::TransposePitchLinearThreadMap<
      IteratorThreadMapB,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>>;

  /// AdvanceRank = 1: after transpose, the strided dim is N — advance
  /// along N across iterations. Same as CUTLASS's NN-spec and
  /// interleaved-spec B iterators.
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      SmemThreadMapB>;

  //
  // Warp-level tensor op: routes to CUTLASS's generic warp iterator on
  // `TensorOpMultiplicandCrosswise<sizeof_bits, Crosswise>` at
  // `mma_tensor_op_tile_iterator.h:~2149`.
  //
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                              MatrixShape<0, 0>, WarpCount::kK>;
};

}  // namespace baracuda_int8_rr_detail

// Partial specialisation: int8_t × RowMajor × int8_t × RowMajor ×
// OpClassTensorOp. Strictly more specific than the generic RR spec
// (which is parameterised on `ElementA_`, `ElementB_`).
template <
    typename Shape_,
    typename WarpShape_,
    typename InstructionShape_,
    typename ElementC_,
    typename LayoutC_,
    int Stages,
    typename Operator_,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, int8_t,
                      layout::RowMajor, int8_t, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB>
    : baracuda_int8_rr_detail::DefaultMmaCoreInt8RrBody<
          Shape_, WarpShape_, InstructionShape_, int8_t, ElementC_, LayoutC_,
          Stages, Operator_, CacheOpA, CacheOpB> {};

// Sibling for uint8_t × uint8_t.
template <
    typename Shape_,
    typename WarpShape_,
    typename InstructionShape_,
    typename ElementC_,
    typename LayoutC_,
    int Stages,
    typename Operator_,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, uint8_t,
                      layout::RowMajor, uint8_t, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB>
    : baracuda_int8_rr_detail::DefaultMmaCoreInt8RrBody<
          Shape_, WarpShape_, InstructionShape_, uint8_t, ElementC_, LayoutC_,
          Stages, Operator_, CacheOpA, CacheOpB> {};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
