// Vendored partial specialization of
// `cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator` for the
// 8-bit `TensorOpMultiplicandCongruous<8, Crosswise>` shared-memory
// layout that the `OpClassTensorOp / RowMajor × RowMajor / int8`
// threadblock chain selects on sm_80.
//
// Why this exists
// ---------------
// CUTLASS 4.2.0's `mma_tensor_op_tile_iterator.h` ships partial
// specializations of `MmaTensorOpMultiplicandTileIterator` for
// `TensorOpMultiplicandCongruous` at:
//
//   - generic `<sizeof_bits<Element>, 64>` (line ~105)
//   - `<32, 32>` (line ~506, 32-bit operand)
//   - `<16, 32>` (line ~871, 16-bit operand)
//   - `<16, 16>` (line ~1275, 16-bit operand, smaller crosswise)
//
// For `RowMajor × RowMajor × int8 × sm_80 × OpClassTensorOp`,
// `DefaultMmaCore` selects:
//
//   SmemLayoutB = RowMajorTensorOpMultiplicandCongruous<8, Crosswise_B>
//   Crosswise_B = min(128 / sizeof(int8_t), Shape::kN)
//                = min(128, 256)
//                = 128    (for the 128x256x64 threadblock)
//
// The `RowMajor` smem-layout wrapper at line ~1914 of the upstream file
// delegates to a base iterator on
// `TensorOpMultiplicandCongruous<sizeof_bits, Crosswise>`. For 8-bit
// operands, that base iterator is NOT specialized in CUTLASS 4.2.0 —
// the only generic 8-bit-compatible spec hardcodes `Crosswise = 64`,
// and neither `<8, 128>` nor `<8, 64>` after partial-spec deduction
// matches a dense warp iterator. The kernel therefore fails to
// instantiate ("incomplete type" at the
// `MmaTensorOpMultiplicandTileIterator` reference).
//
// What this header does
// ---------------------
// Adds a partial specialization on `Element_` with
// `sizeof_bits<Element_>::value == 8` and a free `Crosswise` template
// parameter, so any 8-bit Congruous configuration (Crosswise = 64 or
// 128) becomes instantiable. The body mirrors the existing
// `<16, 32>` spec (line ~871 upstream) — the math is parameterized
// over `Layout::kFactor`, `Layout::kElementsPerAccess`,
// `Layout::TileShape`, and `Layout::PartitionShape` (all derived
// generically by `TensorOpMultiplicandCongruous<ElementSize,
// Crosswise>` for any pairing), and `Layout::kFactor == 1` for
// `<8, 128>` cleanly degenerates the divisions to no-ops.
//
// LDSM dispatch
// -------------
// For `m16n8k32` integer mma (`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
// and its `.u8.u8.s32` sibling), the policy computes:
//
//   kLdsmOpInner = 8
//   LdsmShapeStrided = InstructionShape::kStrided / kLdsmOpInner = 32 / 8 = 4
//   LdsmShapeContiguous = 4 / LdsmShapeStrided = 1
//
// which routes into the `LdsmShape::kContiguous == 1` branch in the
// constructor. The branch's lane-to-pointer math (originally written
// for sparse 16-bit m16n8k32 in the upstream `<16, 32>` spec) maps
// correctly for dense 8-bit when `Layout::kFactor == 1`. The XOR
// pattern is the same bank-conflict-avoiding swizzle the layout class
// applies to smem, just re-derived per-lane.
//
// Note that this iterator runs the dense int8 m16n8k32 mma, not the
// `m16n8k32.sp` (sparse) variant — the kernel's `MmaTensorOp` template
// at the `DefaultMmaTensorOp` selector level picks
// `arch::Mma<m16n8k32, ..., OpMultiplyAddSaturate>` (dense). The LDSM
// arrangement is shared between the dense and sparse paths; only the
// mma instruction differs.
//
// Include this header BEFORE any
// `<cutlass/gemm/device/gemm.h>` /
// `<cutlass/gemm/device/gemm_universal_with_broadcast.h>` in a
// translation unit that instantiates an int8 RowMajor×RowMajor
// tensor-op GEMM, so the partial specialization is in scope at
// template-resolution time.
//
// Attribution
// -----------
// Structure mirrors NVIDIA CUTLASS's existing partial specializations
// in `cutlass/gemm/warp/mma_tensor_op_tile_iterator.h` (BSD-3-Clause).
// Specifically, the body is derived from the `<16, 32>` specialization
// at line ~871, generalized over `Crosswise` and re-specialized on
// 8-bit elements. See `NOTICE` for CUTLASS attribution.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////
//
// Partial specialization on `TensorOpMultiplicandCongruous<8, Crosswise>`.
// SFINAE-restricted to 8-bit elements via the explicit `Element_` size
// and the `sizeof_bits<Element_>::value == 8` static assertion below.
//
////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements (must be 8-bit: int8_t / uint8_t / cutlass::int8_t-like)
    typename Element_,
    /// Crosswise extent of the smem layout (free parameter)
    int Crosswise,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<8, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  static_assert(cutlass::sizeof_bits<Element_>::value == 8,
    "baracuda 8-bit Congruous warp iterator: Element must be 8-bit.");

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = cutlass::TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Stride-index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        cutlass::layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = cutlass::layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

 private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous / Layout::kFactor;

  /// Pointer type used for accesses
  using AccessType = cutlass::Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
      cutlass::Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

 private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

 public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
    byte_offset_(0),
    k_group_idx_(0) {

    int quad_pair = (lane_id >> 3);
    int quad_quad = (lane_id >> 4);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      if (Policy::LdsmShape::kContiguous == 4) {
        // Matrix multiply 1688 A/B
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = quad_pair ^ (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = lane_in_quad_pair / Layout::kFactor;
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (((quad_pair & 1) + i * 2) ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = (lane_in_quad_pair + (lane_id >> 4 << 3)) / 2;
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (quad_quad + i * 2) ^ (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = (lane_in_quad_quad / Layout::kFactor);
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832 (dense int8 B path).
        //
        // For our kFactor = 1 case (Crosswise = 128 with 8-bit
        // elements), the two upstream formulations are algebraically
        // equivalent — both produce `access_contiguous` =
        // `lane_in_quad_pair ^ i`. Keep the kFactor-parameterized
        // form since it generalizes if a future caller picks a
        // different Crosswise.
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor) ^ i;
        access_strided_idx = lane_id / Layout::kFactor;
      }

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * cutlass::sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
      if (tile_offset.contiguous() % 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) *
                     stride_ * Layout::kElementsPerAccess / Layout::kFactor +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided *
                    (cutlass::sizeof_bits<Element>::value / 8) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    cutlass::Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
      reinterpret_cast<cutlass::Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_[c % kPointerCount] +
            Layout::TileShape::kContiguous * (c / kPointerCount) +
            Policy::kLdsmOpInner * Policy::LdsmShape::kStrided * s * stride_ / Layout::kFactor;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        cutlass::arch::ldsm<cutlass::layout::ColumnMajor, Policy::LdsmShape::kCount>(
          fetch_ptr[access_idx],
          source_byte_ptr
        );
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * cutlass::sizeof_bits<Element>::value / 8);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * cutlass::sizeof_bits<Element>::value / 8);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess +
      tile_offset.strided() * InstructionShape::kStrided * stride_ / Layout::kFactor;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  CUTLASS_DEVICE
  void set_kgroup_index(int /*k_group*/) {
    // no op
  }
};

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass
