//! The op **algorithm** IR — a small, backend-agnostic tensor expression.
//!
//! An op is the *pure function* computed at each output coordinate ([`OpDef`]),
//! described as a scalar-op DAG ([`ScalarExpr`]) over its input operands plus an
//! access pattern ([`Access`]). The emitter lowers this to a concrete backend
//! and *schedule* (chosen per [`baracuda_kernels_types::StructureKey`] cell).
//! Describing the math here — rather than as opaque CUDA — is what lets the
//! emitter vectorize, hoist, and fuse, because it can see the dataflow.

use baracuda_kernels_types::{AxisMask, ElementKind};
use std::collections::HashMap;

/// A scalar compute expression — the per-output-coordinate math, as a typed DAG.
///
/// Backend-agnostic: the emitter lowers it to CUDA today (and other backends
/// later) by walking the tree with a per-backend accessor for the leaves.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarExpr {
    /// The value of input operand `i` at the current coordinate.
    Input(u8),
    /// A compile-time scalar constant — the same value at every coordinate.
    Const(f64),
    /// A runtime scalar parameter — the op's `p{i}` launch argument. Distinct
    /// from [`ScalarExpr::Const`]: a `Const` is folded into the kernel, a
    /// `Param` is passed at launch (and, in a fused graph, comes from an
    /// `AddScalar`/`MulScalar` attribute via the pattern's `extract:`).
    Param(u8),
    /// The per-row reduced scalar produced by [`Access::RowReduce`] stage `i`,
    /// broadcast across every element of the row. A leaf exactly like
    /// [`ScalarExpr::Input`]/`Param` — to the per-element math a reduction result
    /// is just another scalar source. Legal **only** inside a `RowReduce`: in a
    /// stage `pre` referencing an earlier stage (`Reduced(j)`, `j < i`) or in the
    /// `epilogue` (any `Reduced(0..n_stages)`). Never an `Input` — it carries no
    /// bind index and must not be folded across rows by the optimizer.
    Reduced(u8),
    /// The **output element's coordinate** along axis `axis` (increment 0d) —
    /// the row-major unravel of the output index over the cell's iteration
    /// shape, converted to the compute dtype. `Coord(1)` at output element
    /// `[i, j]` of a rank-2 op is the value `j` as a `float`/`double`.
    ///
    /// A leaf exactly like [`ScalarExpr::Reduced`]: opaque to the optimizer
    /// (hash/intern by `axis`, never an `Input`, never const-folded — its
    /// value varies per coordinate), lowered by the backend's coordinate
    /// accessor. Legality (enforced at the top of `plan::build_plan`, every
    /// `Access` arm, with independent emitter backstops in `cuda`):
    ///
    /// - **dtype**: `F32`/`F32Strict`/`F64` ONLY. f16/bf16 reject — their max
    ///   exactly-representable integer is 2048 (bf16: 256), which real axis
    ///   extents exceed, so a half coordinate would silently round. Int
    ///   dtypes reject — the coordinate is spelled as a float cast
    ///   (`(float)c{d}`), the same double-math hazard as `Const`/`Param` at
    ///   int dtypes; an int-literal coordinate spelling is the queued
    ///   follow-up.
    /// - **exactness bound (documented honestly)**: an f32 coordinate is
    ///   exact only while the axis extent ≤ 2²⁴ (f64: 2⁵³). The per-axis
    ///   "extent fits the compute dtype's exact-integer range" check is a
    ///   **caller precondition**: the structure key deliberately abstracts
    ///   numeric extents away — the same trust level as the established
    ///   RowReduce column-weight extent precondition (see
    ///   `plan::validate_row_reduce`'s caller-pre-condition note).
    /// - **access**: [`Access::Elementwise`] bodies ONLY (v1). Reduction-class
    ///   bodies reject: a coordinate along a reduced/folded axis is ambiguous
    ///   (which fold iteration?), and RowReduce/Contraction epilogues iterate
    ///   their own coordinate spaces (row/column, m/n) — lifting Coord into
    ///   them needs explicit per-arm semantics, deferred.
    /// - **axis**: must be `< key.rank` (validated at plan time).
    /// - **schedule**: a Coord body always lowers via `Schedule::Strided` —
    ///   the one emitter that materializes the per-axis coordinates `c{d}`;
    ///   never `Vectorized`/`Scalar` (a linear-index kernel has no per-axis
    ///   coordinates to read).
    Coord(u8),
    /// Sum of two sub-expressions.
    Add(Box<ScalarExpr>, Box<ScalarExpr>),
    /// Difference of two sub-expressions.
    Sub(Box<ScalarExpr>, Box<ScalarExpr>),
    /// Product of two sub-expressions.
    Mul(Box<ScalarExpr>, Box<ScalarExpr>),
    /// Quotient of two sub-expressions.
    Div(Box<ScalarExpr>, Box<ScalarExpr>),
    /// A unary math / activation op applied to a sub-expression.
    Unary(UnaryOp, Box<ScalarExpr>),
    /// A non-infix binary op (`max`/`min`/`pow`/`rem`/`atan2`/…) — a backend
    /// function call.
    Binary(BinaryOp, Box<ScalarExpr>, Box<ScalarExpr>),
}

/// A unary math / activation op. Variant names line up with the FKC §4.1
/// graph-`Op` vocabulary, so [`crate::derive_pattern`] maps them by name —
/// **except** the increment-0a scalar-fn extension (`Erfc` through `Lgamma`),
/// which Fuel's `OpTag`/§4.1 vocabulary does not name yet: those lower and
/// validate like any other op but are rejected by pattern derivation (an
/// honest miss — no invented tags) until Fuel adds the vocabulary.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Negation `-x`.
    Neg,
    /// Absolute value `|x|`.
    Abs,
    /// Square `x²`.
    Sqr,
    /// Square root `√x`.
    Sqrt,
    /// Reciprocal square root `1/√x`.
    Rsqrt,
    /// Reciprocal `1/x`.
    Recip,
    /// Natural exponential `eˣ`.
    Exp,
    /// Natural logarithm `ln x`.
    Log,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid `1/(1+e⁻ˣ)`.
    Sigmoid,
    /// Rectified linear unit `max(x, 0)`.
    Relu,
    /// Gauss error function.
    Erf,
    /// Exact (erf-based) GELU — emits the FKC §4.1 `GeluErf` op (bare `Gelu` is
    /// the tanh approximation, per §4.1's B6/E2 resolution).
    Gelu,
    /// SiLU / swish `x·sigmoid(x)`.
    Silu,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Floor — round toward −∞.
    Floor,
    /// Ceil — round toward +∞.
    Ceil,
    /// Round to nearest (ties to even).
    Round,
    /// Sign `−1 / 0 / +1`.
    Sign,
    /// Heaviside step `x > 0 ? 1 : 0` (`heaviside(x, values=0)`; `step(0) = 0`).
    Step,
    // --- increment-0a scalar-fn extension (no FKC §4.1 name yet; see enum docs) ---
    /// Complementary error function `erfc x = 1 − erf x`.
    Erfc,
    /// Truncate — round toward zero (exact on finite values).
    Trunc,
    /// Base-2 exponential `2ˣ`.
    Exp2,
    /// `eˣ − 1`, accurate near 0.
    Expm1,
    /// Base-2 logarithm.
    Log2,
    /// Base-10 logarithm.
    Log10,
    /// `ln(1 + x)`, accurate near 0.
    Log1p,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Tangent.
    Tan,
    /// Arcsine (domain `[-1, 1]`).
    Asin,
    /// Arccosine (domain `[-1, 1]`).
    Acos,
    /// Arctangent.
    Atan,
    /// Inverse hyperbolic sine.
    Asinh,
    /// Inverse hyperbolic cosine (domain `[1, ∞)`).
    Acosh,
    /// Inverse hyperbolic tangent (domain `(-1, 1)`).
    Atanh,
    /// Cube root (defined for negative inputs, unlike `pow(x, 1/3)`).
    Cbrt,
    /// Log-gamma `ln|Γ(x)|`.
    Lgamma,
}

/// A non-infix binary op — lowered as a backend **function call** (`fmaxf`,
/// `powf`) or C operator (the `Cmp*` predicates and the increment-0c integer
/// ops), unlike the infix arithmetic [`ScalarExpr::Add`]/`Sub`/`Mul`/`Div`.
/// Variant names line up with the FKC §4.1 graph-`Op` vocabulary — except the
/// increment-0a extension (`Atan2` through `RemTrunc`), which §4.1 does not
/// name yet (see the [`UnaryOp`] docs; same honest-miss rule), the
/// increment-0b comparisons, whose §4.1 names drop the `Cmp` prefix
/// (`CmpEq` → `Equal`, `CmpNe` → `Ne`, `CmpLt` → `Lt`, `CmpLe` → `Le`,
/// `CmpGt` → `Gt`, `CmpGe` → `Ge` — the mapping lives in `pattern::binary_name`),
/// and the increment-0c bitwise/shift/logical ops, which neither `OpTag`
/// (fuel-kernel-seam-types 0.10.2) nor Fuel's `lower_op_kind` dispatch table
/// names — those ops lower and validate but emit NO contract (honest miss).
///
/// # Op × dtype admissibility (increment 0c — audited against the bespoke surface)
///
/// The compute-dtype legality table, enforced at the TOP of `plan::build_plan`
/// (`assert_int_op_admissibility`, every `Access` arm) with independent
/// emitter backstops in `cuda::binary_int` / `cuda::binary_f32` / `binary_f64`
/// and the JIT's `dtype_compatible`. "int" = `I32`/`I64`/`S8`(FKC `I8`)/`U8`.
///
/// | op set                                          | f16/bf16/f32/f32s/f64 | I32/I64 | S8/U8 | evidence / semantics |
/// |-------------------------------------------------|-----------------------|---------|-------|----------------------|
/// | infix `Add`/`Sub`/`Mul`                         | legal                 | legal   | legal | wrapping two's-complement (see below); bespoke `reduce_sum_int`/`reduce_prod_int` carry int +/× |
/// | infix `Div`                                     | legal                 | REJECT  | REJECT| bespoke elementwise div is `_fp`-only (`binary_div_fp.cu`); C `/` div-by-zero is device-UB |
/// | every [`UnaryOp`]                               | legal (0a/0b gates)   | REJECT  | REJECT| bespoke unary elementwise surface is `_fp`-only |
/// | `Max`/`Min`/`Pow`/`Rem` + 0a fns                | legal (Nextafter f32/f64) | REJECT | REJECT | float device fns only; no bespoke int instantiation |
/// | `CmpEq`…`CmpGe`                                 | legal                 | REJECT  | REJECT| bespoke cmp is `_fp`-only (`binary_cmp_*_fp.cu`) |
/// | `BitAnd`/`BitOr`/`BitXor`/`Shl`/`Shr`           | REJECT                | legal   | legal, LEAF operands only | bespoke `binary_bitwise_*_int.cu` instantiates i32/i64; 8-bit legal per the 0c charter with the promote-then-truncate semantics documented per variant — and at `S8`/`U8` every operand must be a leaf `Input` (a composed operand observes the un-truncated promoted value when inlined but the truncated 8-bit tmp when hoisted, so its result would depend on DAG sharing; see `plan::assert_int_op_admissibility` rule 3) |
/// | `LogicalAnd`/`LogicalOr`/`LogicalXor`           | REJECT                | REJECT  | U8 only, LEAF operands only | bespoke `binary_logical_*_bool.cu` instantiates ONLY `uint8_t` (Bool); the `!= 0` tests observe un-truncated composed values, so the same 8-bit leaf-operand pin applies |
/// | `Const`/`Param` leaves in the body              | legal (Param f32-only)| REJECT  | REJECT| a `Const` is spelled as an f64 C literal — at an int dtype it would silently run double math (and f64 cannot represent all i64); an int-literal speller is a follow-up |
///
/// **Integer wrapping semantics.** For `I32`/`I64`, infix `+ - *` lower to the
/// native C operators. Signed arithmetic overflow remains formally UB in ALL
/// current ISO C++ standards — C++20 (P0907) standardized the two's-complement
/// REPRESENTATION, shifts, and narrowing conversions, NOT arithmetic overflow,
/// which stays UB in C++20 and C++23. The wrapping contract therefore rests on
/// the NVCC/PTX lowering: every CUDA target compiles `+ - *` to the wrapping
/// two's-complement PTX forms (`add.s32`/`mul.lo.s64`) — observed-defined,
/// matched by the bespoke `reduce_sum_int.cu`/`reduce_prod_int.cu` kernels'
/// native-operator int accumulation, and the same architecture-inherited
/// reliance the bespoke right-shift kernel documents for arithmetic `>>`.
/// For `S8`/`U8`, C integer promotion widens both operands to
/// `int`, the arithmetic is exact in `int` (no 8-bit pair can overflow it),
/// and the store truncates back to 8 bits — mod-2⁸ wrapping by construction
/// (well-defined for `unsigned char`; implementation-defined-but-two's-
/// complement for `signed char` on every CUDA compiler, standardized by
/// C++20). "Exact"/`correctly_rounded` for these ops means exact WRAPPING
/// semantics, not infinite-precision arithmetic.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Elementwise maximum — **NaN-propagating** (`torch.maximum`; commutative).
    /// Deliberately distinct from [`BinaryOp::FmaxIeee`] (NaN-suppressing).
    Max,
    /// Elementwise minimum — **NaN-propagating** (`torch.minimum`; commutative).
    /// Deliberately distinct from [`BinaryOp::FminIeee`] (NaN-suppressing).
    Min,
    /// Power `aᵇ` (not commutative).
    Pow,
    /// Floored remainder — `a - floor(a/b)·b`, sign-of-divisor (`torch.remainder`,
    /// Fuel's `Op::Rem`; not commutative). Distinct from C `fmod` (sign-of-dividend
    /// — that is [`BinaryOp::RemTrunc`]); never merge the two.
    Rem,
    // --- increment-0a scalar-fn extension (no FKC §4.1 name yet; see UnaryOp docs) ---
    /// Four-quadrant arctangent `atan2(a, b)` — `a` is `y`, `b` is `x` (not
    /// commutative; the ±0 quadrant conventions are IEEE's).
    Atan2,
    /// `copysign(a, b)` — bit-level sign transfer: magnitude of `a`, sign bit of
    /// `b`, including signed zero and the sign of a NaN payload (not commutative).
    Copysign,
    /// `nextafter(a, b)` — the next representable value after `a` toward `b`, in
    /// the **kernel dtype's own lattice** (bit-level). f16/bf16 are rejected as an
    /// honest miss: the half path computes promoted-to-f32, which would step the
    /// f32 lattice and round back to `a` — the wrong neighbor, silently.
    Nextafter,
    /// IEEE-754 `maxNum` (CUDA `fmaxf`) — **NaN-suppressing**: if exactly one
    /// operand is NaN, returns the other. DISTINCT from the house NaN-propagating
    /// [`BinaryOp::Max`]; never alias or merge them (the reference kernel
    /// `binary_maximum_fp.cu` reserves `fmaxf` for exactly this separate op).
    FmaxIeee,
    /// IEEE-754 `minNum` (CUDA `fminf`) — NaN-suppressing dual of
    /// [`BinaryOp::FmaxIeee`]; distinct from the NaN-propagating [`BinaryOp::Min`].
    FminIeee,
    /// Truncated remainder — C `fmod`, sign-of-**dividend** (`torch.fmod`; not
    /// commutative). Distinct from the floored [`BinaryOp::Rem`] (sign-of-divisor).
    RemTrunc,
    // --- increment-0b comparison predicates (FKC §4.1 names them: `Equal`/`Ne`/
    // `Lt`/`Le`/`Gt`/`Ge`, "→ U8 mask") ---
    //
    // Semantics (all six): the IEEE-754 ordered/unordered comparison of the two
    // operands in the COMPUTE dtype, producing EXACTLY 1.0 or 0.0 in that dtype
    // (the C ternary `a < b ? 1.0f : 0.0f`). NaN semantics are the C operators':
    // any comparison with a NaN operand is FALSE — except `CmpNe`, which is TRUE.
    // f16/bf16 compare via promote-to-f32, which is EXACT: half→f32 is a lossless,
    // order-preserving embedding, so the f32 compare decides identically to a
    // native half compare (and demoting the exact 1.0/0.0 back is exact) — unlike
    // Nextafter, no lattice is stepped. Nested inside a float body a Cmp* is just
    // an inline 0.0/1.0 float (the mask-multiply `dy * (x > 0)` shape); as a
    // TOP-LEVEL body it may pair with `OpDef::elementwise_pred`'s `out_dtype =
    // Some(U8)` to store the predicate as a `u8` 1/0 mask (exact — see `plan`'s
    // `assert_valid_out_dtype`). NEVER rewrite `CmpEq(x, x)` to 1.0: it is FALSE
    // for NaN `x` (and dually `CmpNe(x, x)` is TRUE for NaN).
    /// Equality `a == b ? 1 : 0` (NaN == anything is false, including NaN == NaN).
    CmpEq,
    /// Inequality `a != b ? 1 : 0` — the one comparison that is TRUE on NaN
    /// operands (`NaN != x` for every `x`, including NaN).
    CmpNe,
    /// Less-than `a < b ? 1 : 0` (false on any NaN operand).
    CmpLt,
    /// Less-or-equal `a <= b ? 1 : 0` (false on any NaN operand; `-0 <= +0` true).
    CmpLe,
    /// Greater-than `a > b ? 1 : 0` (false on any NaN operand).
    CmpGt,
    /// Greater-or-equal `a >= b ? 1 : 0` (false on any NaN operand).
    CmpGe,
    // --- increment-0c integer bitwise / shift / logical ops (INT-ONLY — see the
    // admissibility table in the enum docs; no FKC §4.1/OpTag name, no
    // lower_op_kind dispatch spelling ⇒ no contract, honest miss) ---
    /// Bitwise AND `a & b` — int-only (`I32`/`I64`/`S8`/`U8`). Matches
    /// `binary_bitwise_and_int.cu`'s `BitwiseAndFunctor` (`return a & b;`)
    /// exactly: no rounding, no overflow. On `S8`/`U8` the C integer promotion
    /// (sign-/zero-extend to `int`) followed by the 8-bit store truncation is
    /// bit-identical to a native 8-bit AND.
    BitAnd,
    /// Bitwise OR `a | b` — int-only; `binary_bitwise_or_int.cu` semantics.
    /// Same promotion/truncation reasoning as [`BinaryOp::BitAnd`].
    BitOr,
    /// Bitwise XOR `a ^ b` — int-only; `binary_bitwise_xor_int.cu` semantics.
    /// Same promotion/truncation reasoning as [`BinaryOp::BitAnd`].
    BitXor,
    /// Bitwise left shift `a << b` — int-only. Matches
    /// `binary_bitwise_left_shift_int.cu` exactly: the RAW C `<<`, no masking
    /// or clamping. Caveat carried verbatim from the bespoke kernel:
    /// out-of-range shift amounts (`b < 0` or `b >= 8 * sizeof(promoted T)`)
    /// are undefined behavior in C/C++ on signed types — PyTorch documents the
    /// result as undefined/hardware-dependent there too, so we inherit the
    /// architecture's behavior rather than masking; callers who need defined
    /// behavior clamp `b` before launch. Left-shifting a negative value is
    /// likewise formally UB pre-C++20 (wrapping two's-complement in practice
    /// on every CUDA target; C++20 standardizes it). **8-bit note:** `S8`/`U8`
    /// operands promote to `int` BEFORE the shift, so the effective shift
    /// width is 32 (amounts 8..31 are in-promoted-range, where a native 8-bit
    /// shift would have no defined range at all) and the store truncates the
    /// 32-bit result mod 2⁸ — for in-range amounts this equals the native
    /// 8-bit wrapping shift. Formal-UB caveat, carried honestly: a promoted
    /// result that overflows `int` (e.g. `200 << 24`) is still formally UB
    /// pre-C++20 (C++20 defines it as mod-2³²), and while nvcc via forge
    /// compiles C++20, the headerless nvrtc path passes no `-std` flag — on
    /// NVCC/PTX both observably wrap, the same architecture-inherited
    /// contract as the bespoke shift kernels. Bespoke has no 8-bit
    /// instantiation to defer to; these promotion-composition semantics are
    /// the documented 0c contract. At `S8`/`U8` both operands must be leaf
    /// `Input`s (the plan gate's 8-bit composition pin — the shift AMOUNT and,
    /// for `Shr`, the shifted value observe the un-truncated promoted value,
    /// so a composed operand's result would depend on DAG sharing).
    Shl,
    /// Bitwise right shift `a >> b` — int-only. Matches
    /// `binary_bitwise_right_shift_int.cu` exactly: **arithmetic** shift on
    /// signed types (`I32`/`I64`/`S8` — the sign bit replicates; formally
    /// implementation-defined pre-C++20, but NVCC/MSVC/GCC/Clang all lower
    /// signed `>>` to the arithmetic PTX `shr.s32`/`shr.s64`, the reliance the
    /// bespoke kernel pins) and **logical** shift on `U8` (zero-extension —
    /// the natural unsigned semantics). Out-of-range amounts (`b < 0` or
    /// `b >= 8 * sizeof(promoted T)`) inherit the architecture's behavior,
    /// same caller contract as [`BinaryOp::Shl`]. `S8`/`U8` promote to `int`
    /// first (sign-/zero-extended), so the 8-bit result always fits and the
    /// store truncation is exact. Same 8-bit leaf-operand pin as
    /// [`BinaryOp::Shl`]: `Shr` observes the un-truncated promoted value of a
    /// composed operand in BOTH positions (the shifted value's high bits and
    /// the amount), so at `S8`/`U8` both operands must be leaf `Input`s.
    Shr,
    /// Logical AND — **U8 (Bool) only**. Matches `binary_logical_and_bool.cu`'s
    /// `LogicalAndFunctor` exactly: `(a != 0 && b != 0) ? 1 : 0` — each input
    /// is NORMALIZED to 0/1 before the op, so the output is strictly 0 or 1
    /// even for unnormalized byte inputs (e.g. `2 && 4 == 1`, never `2 & 4 == 0`).
    /// Operands must be leaf `Input`s (the plan gate's 8-bit composition pin:
    /// the `!= 0` test observes the un-truncated promoted value of a composed
    /// operand — `255+1` is 0 truncated but 256 promoted — so its result would
    /// depend on DAG sharing); U8 is this op's only dtype, so the pin always
    /// applies. Same contract for [`BinaryOp::LogicalOr`]/[`BinaryOp::LogicalXor`].
    LogicalAnd,
    /// Logical OR — U8 only; `binary_logical_or_bool.cu`:
    /// `(a != 0 || b != 0) ? 1 : 0`, same normalization contract as
    /// [`BinaryOp::LogicalAnd`].
    LogicalOr,
    /// Logical XOR (boolean inequality) — U8 only; `binary_logical_xor_bool.cu`:
    /// `((a != 0) != (b != 0)) ? 1 : 0`, same normalization contract as
    /// [`BinaryOp::LogicalAnd`].
    LogicalXor,
}

impl BinaryOp {
    /// `true` for the comparison predicates (`CmpEq`…`CmpGe`) — the ops whose
    /// value is exactly 1.0/0.0 and which may drive a `u8`-mask output
    /// ([`OpDef::elementwise_pred`]).
    #[must_use]
    pub fn is_cmp(self) -> bool {
        matches!(
            self,
            BinaryOp::CmpEq
                | BinaryOp::CmpNe
                | BinaryOp::CmpLt
                | BinaryOp::CmpLe
                | BinaryOp::CmpGt
                | BinaryOp::CmpGe
        )
    }

    /// `true` for the increment-0c INT-ONLY ops (bitwise/shift/logical) — the
    /// ops that lower via `cuda::binary_int` and are legal ONLY at the integer
    /// compute dtypes (`I32`/`I64`/`S8`/`U8`; logical narrows further to `U8`
    /// — see [`BinaryOp::is_logical`]). Float dtypes validate-reject at the
    /// plan gate; the float spellers carry an independent panic backstop.
    #[must_use]
    pub fn is_int_only(self) -> bool {
        matches!(
            self,
            BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::BitXor
                | BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::LogicalAnd
                | BinaryOp::LogicalOr
                | BinaryOp::LogicalXor
        )
    }

    /// `true` for the logical (0/1-normalizing) ops — legal at `U8` (the FKC
    /// Bool spelling) ONLY, per the bespoke surface: `binary_logical_*_bool.cu`
    /// instantiates exactly `uint8_t`, no wider int.
    #[must_use]
    pub fn is_logical(self) -> bool {
        matches!(
            self,
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr | BinaryOp::LogicalXor
        )
    }
}

// ===========================================================================
// Value-numbered DAG (derived from the authored `ScalarExpr` tree)
// ===========================================================================

/// Dense-arena node index into an [`ExprDag`].
pub type NodeId = u32;

/// A node of a value-numbered op-DAG: the [`ScalarExpr`] op shape, but with
/// children referenced by [`NodeId`] instead of `Box`, so a value reachable by
/// two paths is stored — and emitted — once. Built by [`ExprDag::from_expr`].
#[derive(Clone, Debug, PartialEq)]
pub enum DagNode {
    /// Input operand `i` at the current coordinate. (Leaf.)
    Input(u8),
    /// Compile-time scalar constant. (Leaf.)
    Const(f64),
    /// Runtime scalar parameter `p{i}`. (Leaf.)
    Param(u8),
    /// Per-row reduced scalar from [`Access::RowReduce`] stage `i`. (Leaf.)
    Reduced(u8),
    /// Output coordinate along axis `axis` ([`ScalarExpr::Coord`]). (Leaf.)
    Coord(u8),
    /// Sum of two nodes.
    Add(NodeId, NodeId),
    /// Difference of two nodes.
    Sub(NodeId, NodeId),
    /// Product of two nodes.
    Mul(NodeId, NodeId),
    /// Quotient of two nodes.
    Div(NodeId, NodeId),
    /// A unary op over one node.
    Unary(UnaryOp, NodeId),
    /// A non-infix binary op over two nodes.
    Binary(BinaryOp, NodeId, NodeId),
}

impl DagNode {
    /// `true` for a source leaf (`Input`/`Const`/`Param`/`Reduced`/`Coord`) — a
    /// value with no children. Leaves are never hoisted to a `tmp` (a leaf
    /// reference is free); only shared *interior* nodes are.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            DagNode::Input(_)
                | DagNode::Const(_)
                | DagNode::Param(_)
                | DagNode::Reduced(_)
                | DagNode::Coord(_)
        )
    }
}

/// A value-numbered op-DAG: nodes stored once (index == [`NodeId`]), children by
/// id, with a per-node **consumer count** (how many edges reference the node).
///
/// Built from a [`ScalarExpr`] tree by [`ExprDag::from_expr`] via hash-consing:
/// structurally-equal subtrees collapse to one node, so the diamond
/// `Add(Mul(x,y), Mul(x,y))` stores one `Mul` with `consumers == 2` instead of
/// two. Two consumer notions must be kept distinct (design doc §5.3):
///
/// 1. **Intra-body sharing** — [`ExprDag::consumers`], the edge count *inside this
///    op body*. `> 1` on a non-leaf ⇒ the emitter hoists it to a named `tmp` so a
///    shared interior is computed once (killing the tree emitter's `O(2^depth)`
///    blow-up). Always Baracuda-internal and safe.
/// 2. **FKC cross-region `consumers:`** — a *different*, fusion-safety notion (does
///    the value escape the fused region?) that only the seam sets; an AOT body is
///    the whole region, so a non-root interior stays externally sole-consumer.
///    This type carries only notion (1).
///
/// `Const` is interned by `f64::to_bits()` (NaN-safe by bits), mirroring the
/// e-graph in [`crate::optimize`]. A `Reduced`/`Param` leaf interns once but is
/// never merged with a structurally different node, so the RowReduce per-row-leaf
/// invariant holds for free (a leaf has no children to fold across rows).
#[derive(Clone, Debug)]
pub struct ExprDag {
    nodes: Vec<DagNode>,
    consumers: Vec<u32>,
    root: NodeId,
}

impl ExprDag {
    /// Hash-cons a [`ScalarExpr`] tree into a value-numbered DAG.
    #[must_use]
    pub fn from_expr(e: &ScalarExpr) -> ExprDag {
        let mut b = DagBuilder {
            nodes: Vec::new(),
            consumers: Vec::new(),
            memo: HashMap::new(),
        };
        let root = b.intern(e);
        ExprDag {
            nodes: b.nodes,
            consumers: b.consumers,
            root,
        }
    }

    /// The output node (the value the op computes).
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// The node at `id`.
    #[must_use]
    pub fn node(&self, id: NodeId) -> &DagNode {
        &self.nodes[id as usize]
    }

    /// How many edges reference `id` inside this body (intra-body sharing, notion
    /// (1) in the type docs). `> 1` on a non-leaf ⇒ the emitter hoists it.
    #[must_use]
    pub fn consumers(&self, id: NodeId) -> u32 {
        self.consumers[id as usize]
    }

    /// Number of distinct nodes (a shared subtree counts once).
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` if the DAG has no nodes (never, for a well-formed expr).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Reconstruct a [`ScalarExpr`] tree by inlining every node (a shared node
    /// re-expands to duplicated subtrees). Semantics-preserving — used to test
    /// that interning is a value-identity, not for emission.
    #[must_use]
    pub fn to_expr(&self) -> ScalarExpr {
        self.rebuild(self.root)
    }

    fn rebuild(&self, id: NodeId) -> ScalarExpr {
        match self.nodes[id as usize] {
            DagNode::Input(i) => ScalarExpr::Input(i),
            DagNode::Const(v) => ScalarExpr::Const(v),
            DagNode::Param(i) => ScalarExpr::Param(i),
            DagNode::Reduced(i) => ScalarExpr::Reduced(i),
            DagNode::Coord(d) => ScalarExpr::Coord(d),
            DagNode::Add(a, b) => {
                ScalarExpr::Add(Box::new(self.rebuild(a)), Box::new(self.rebuild(b)))
            }
            DagNode::Sub(a, b) => {
                ScalarExpr::Sub(Box::new(self.rebuild(a)), Box::new(self.rebuild(b)))
            }
            DagNode::Mul(a, b) => {
                ScalarExpr::Mul(Box::new(self.rebuild(a)), Box::new(self.rebuild(b)))
            }
            DagNode::Div(a, b) => {
                ScalarExpr::Div(Box::new(self.rebuild(a)), Box::new(self.rebuild(b)))
            }
            DagNode::Unary(op, x) => ScalarExpr::Unary(op, Box::new(self.rebuild(x))),
            DagNode::Binary(op, a, b) => {
                ScalarExpr::Binary(op, Box::new(self.rebuild(a)), Box::new(self.rebuild(b)))
            }
        }
    }
}

/// Hashable interning key — `Const` folded to bits so NaN / ±0 intern by identity
/// (a bare `f64` is neither `Eq` nor `Hash`).
#[derive(Clone, PartialEq, Eq, Hash)]
enum DagKey {
    Input(u8),
    ConstBits(u64),
    Param(u8),
    Reduced(u8),
    Coord(u8),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Unary(UnaryOp, NodeId),
    Binary(BinaryOp, NodeId, NodeId),
}

impl DagKey {
    fn of(n: &DagNode) -> DagKey {
        match *n {
            DagNode::Input(i) => DagKey::Input(i),
            DagNode::Const(v) => DagKey::ConstBits(v.to_bits()),
            DagNode::Param(i) => DagKey::Param(i),
            DagNode::Reduced(i) => DagKey::Reduced(i),
            DagNode::Coord(d) => DagKey::Coord(d),
            DagNode::Add(a, b) => DagKey::Add(a, b),
            DagNode::Sub(a, b) => DagKey::Sub(a, b),
            DagNode::Mul(a, b) => DagKey::Mul(a, b),
            DagNode::Div(a, b) => DagKey::Div(a, b),
            DagNode::Unary(op, x) => DagKey::Unary(op, x),
            DagNode::Binary(op, a, b) => DagKey::Binary(op, a, b),
        }
    }
}

struct DagBuilder {
    nodes: Vec<DagNode>,
    consumers: Vec<u32>,
    memo: HashMap<DagKey, NodeId>,
}

impl DagBuilder {
    fn intern(&mut self, e: &ScalarExpr) -> NodeId {
        let node = match e {
            ScalarExpr::Input(i) => DagNode::Input(*i),
            ScalarExpr::Const(v) => DagNode::Const(*v),
            ScalarExpr::Param(i) => DagNode::Param(*i),
            ScalarExpr::Reduced(i) => DagNode::Reduced(*i),
            ScalarExpr::Coord(d) => DagNode::Coord(*d),
            ScalarExpr::Add(a, b) => {
                let (a, b) = (self.intern(a), self.intern(b));
                DagNode::Add(a, b)
            }
            ScalarExpr::Sub(a, b) => {
                let (a, b) = (self.intern(a), self.intern(b));
                DagNode::Sub(a, b)
            }
            ScalarExpr::Mul(a, b) => {
                let (a, b) = (self.intern(a), self.intern(b));
                DagNode::Mul(a, b)
            }
            ScalarExpr::Div(a, b) => {
                let (a, b) = (self.intern(a), self.intern(b));
                DagNode::Div(a, b)
            }
            ScalarExpr::Unary(op, x) => {
                let x = self.intern(x);
                DagNode::Unary(*op, x)
            }
            ScalarExpr::Binary(op, a, b) => {
                let (a, b) = (self.intern(a), self.intern(b));
                DagNode::Binary(*op, a, b)
            }
        };
        self.hashcons(node)
    }

    /// Return the id for `node`, creating it if new. On creation, register each
    /// outgoing edge by bumping the referenced child's consumer count once — so a
    /// re-interned (memoized) parent never double-counts edges that already exist,
    /// and `Mul(a, a)` correctly counts `a` twice (same-parent-twice is a shared
    /// value).
    fn hashcons(&mut self, node: DagNode) -> NodeId {
        let key = DagKey::of(&node);
        if let Some(&id) = self.memo.get(&key) {
            return id;
        }
        for child in node_children(&node) {
            self.consumers[child as usize] += 1;
        }
        let id = u32::try_from(self.nodes.len()).expect("DAG node count exceeds u32");
        self.nodes.push(node);
        self.consumers.push(0);
        self.memo.insert(key, id);
        id
    }
}

/// The child ids a node references, with multiplicity (`Mul(a, a)` → `[a, a]`).
fn node_children(n: &DagNode) -> Vec<NodeId> {
    match *n {
        DagNode::Input(_)
        | DagNode::Const(_)
        | DagNode::Param(_)
        | DagNode::Reduced(_)
        | DagNode::Coord(_) => Vec::new(),
        DagNode::Unary(_, x) => vec![x],
        DagNode::Add(a, b)
        | DagNode::Sub(a, b)
        | DagNode::Mul(a, b)
        | DagNode::Div(a, b)
        | DagNode::Binary(_, a, b) => vec![a, b],
    }
}

/// The associative combine of an [`Access::Reduction`]. The identity is implied
/// (`Sum`/`Mean` → 0; `Prod` → 1; `Max`/`Min` peel the first element, so no ±∞
/// literal — that keeps the emitted source header-light under nvrtc).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Sum over the reduced axis (`SumDim`).
    Sum,
    /// Arithmetic mean — `sum / extent` (`MeanDim`).
    Mean,
    /// Maximum — NaN-propagating (`torch.amax`).
    Max,
    /// Minimum — NaN-propagating (`torch.amin`).
    Min,
    /// Product over the reduced axis (increment 0e — `torch.prod`). Identity 1
    /// (`acc = 1; acc *= elem`), pass-through finalize (no Mean-style divisor).
    /// Semantics match the bespoke `reduce_prod_fp.cu` / `reduce_prod_int.cu`:
    /// f16/bf16 multiply through an f32 accumulator; f32/f16/bf16 fold in
    /// `float`, f64/f32-strict in `double`; `I32`/`I64` accumulate in the
    /// widened `long long` (the bespoke i64/u64 accumulator) and the store
    /// truncates back to the input width with wrap-on-overflow. Fuel has no
    /// `ProdReduce` `OpKind` yet (`fuel-cuda-backend/src/baracuda/reduce.rs`:
    /// "Prod … ship in baracuda but don't have matching Fuel `OpKind`s yet"),
    /// so a Prod reduction — like every reduction — emits no contract (honest
    /// miss, and reductions are not in the elementwise pattern vocabulary).
    Prod,
}

/// Ergonomic builder handle wrapping a [`ScalarExpr`]. Overloads arithmetic so
/// op bodies read like math: `input(0) + input(1) * input(2)`.
#[derive(Clone, Debug)]
pub struct Expr(pub ScalarExpr);

/// The value of input operand `i` — the leaf of an op body expression.
#[must_use]
pub fn input(i: u8) -> Expr {
    Expr(ScalarExpr::Input(i))
}

/// The per-row reduced scalar from [`Access::RowReduce`] stage `i` (broadcast
/// across the row) — a leaf for fused-reduction epilogues (e.g.
/// `input(0) * (reduced(0) + konst(eps)).unary(UnaryOp::Rsqrt)` for RmsNorm).
#[must_use]
pub fn reduced(i: u8) -> Expr {
    Expr(ScalarExpr::Reduced(i))
}

/// The output element's coordinate along `axis` ([`ScalarExpr::Coord`]) as a
/// value in the compute dtype — the iota/coordinate leaf (increment 0d). E.g.
/// the main-diagonal triu mask is
/// `input(0) * coord(1).binary(BinaryOp::CmpGe, coord(0))`. f32/f64
/// Elementwise bodies only; see the [`ScalarExpr::Coord`] legality table.
#[must_use]
pub fn coord(axis: u8) -> Expr {
    Expr(ScalarExpr::Coord(axis))
}

/// A compile-time scalar constant leaf (e.g. `input(0) * konst(0.5)`).
#[must_use]
pub fn konst(v: f64) -> Expr {
    Expr(ScalarExpr::Const(v))
}

/// A runtime scalar-parameter leaf — the op's `p{i}` launch argument
/// (e.g. `input(0) * param(0) + param(1)`).
#[must_use]
pub fn param(i: u8) -> Expr {
    Expr(ScalarExpr::Param(i))
}

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr(ScalarExpr::Add(Box::new(self.0), Box::new(rhs.0)))
    }
}
impl std::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr(ScalarExpr::Sub(Box::new(self.0), Box::new(rhs.0)))
    }
}
impl std::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr(ScalarExpr::Mul(Box::new(self.0), Box::new(rhs.0)))
    }
}
impl std::ops::Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr(ScalarExpr::Div(Box::new(self.0), Box::new(rhs.0)))
    }
}

impl Expr {
    /// Apply a unary op to this expression (`expr.unary(UnaryOp::Relu)`).
    #[must_use]
    pub fn unary(self, op: UnaryOp) -> Expr {
        Expr(ScalarExpr::Unary(op, Box::new(self.0)))
    }
    /// ReLU `max(x, 0)`.
    #[must_use]
    pub fn relu(self) -> Expr {
        self.unary(UnaryOp::Relu)
    }
    /// SiLU / swish `x·sigmoid(x)`.
    #[must_use]
    pub fn silu(self) -> Expr {
        self.unary(UnaryOp::Silu)
    }
    /// Exact (erf-based) GELU.
    #[must_use]
    pub fn gelu(self) -> Expr {
        self.unary(UnaryOp::Gelu)
    }
    /// Logistic sigmoid.
    #[must_use]
    pub fn sigmoid(self) -> Expr {
        self.unary(UnaryOp::Sigmoid)
    }
    /// Hyperbolic tangent.
    #[must_use]
    pub fn tanh(self) -> Expr {
        self.unary(UnaryOp::Tanh)
    }
    /// Natural exponential.
    #[must_use]
    pub fn exp(self) -> Expr {
        self.unary(UnaryOp::Exp)
    }
    /// Square root.
    #[must_use]
    pub fn sqrt(self) -> Expr {
        self.unary(UnaryOp::Sqrt)
    }
    /// Sine.
    #[must_use]
    pub fn sin(self) -> Expr {
        self.unary(UnaryOp::Sin)
    }
    /// Floor.
    #[must_use]
    pub fn floor(self) -> Expr {
        self.unary(UnaryOp::Floor)
    }

    /// Apply a non-infix binary op (`expr.binary(BinaryOp::Max, rhs)`).
    #[must_use]
    pub fn binary(self, op: BinaryOp, rhs: Expr) -> Expr {
        Expr(ScalarExpr::Binary(op, Box::new(self.0), Box::new(rhs.0)))
    }
    /// Elementwise maximum.
    #[must_use]
    pub fn max(self, rhs: Expr) -> Expr {
        self.binary(BinaryOp::Max, rhs)
    }
    /// Elementwise minimum.
    #[must_use]
    pub fn min(self, rhs: Expr) -> Expr {
        self.binary(BinaryOp::Min, rhs)
    }
    /// Power `aᵇ`.
    #[must_use]
    pub fn pow(self, rhs: Expr) -> Expr {
        self.binary(BinaryOp::Pow, rhs)
    }
}

/// One reduction stage of an [`Access::RowReduce`]: fold `pre` (the per-element
/// pre-reduction expression) over the last axis with `op`. Stage `i` produces the
/// scalar [`ScalarExpr::Reduced`]`(i)`; its `pre` may reference `Reduced(j)` for
/// `j < i` (e.g. Softmax's exp-sum stage reads the row max from stage 0).
#[derive(Clone, Debug, PartialEq)]
pub struct ReduceStage {
    /// Per-element expression reduced along the last axis (`Input`/`Const`/`Param`
    /// and earlier-stage `Reduced(j)`).
    pub pre: ScalarExpr,
    /// The associative combine.
    pub op: ReduceOp,
}

/// Iteration / access pattern of an op — tells the emitter the loop-nest shape
/// and which schedules are legal.
///
/// `#[non_exhaustive]`: windowed/stencil and gather patterns are still the growth
/// path; arbitrary/multiple reduction axes, strided-input reductions, and keepdim
/// layout extend [`Access::Reduction`] later.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Access {
    /// Output coordinate equals input coordinate (a per-element map).
    Elementwise,
    /// Reduce the axes in `axes` with `op`: each output element is `op` folded
    /// over the reduced axes' run of `body` values. `axes == AxisMask::EMPTY` is
    /// the legacy sentinel for the **last (contiguous, trailing) axis** — the
    /// `MeanDim`/`SumDim` core of RmsNorm/Softmax that `OpDef::reduction` builds.
    /// A non-empty mask names arbitrary outer/middle/multiple reduced axes, and
    /// `keepdim` selects whether the reduced axes collapse (rank drops) or stay
    /// size-1 (broadcast-back). The IR *represents* all of these; the emitter
    /// generalizes past the contiguous-last-axis fast path in a follow-up (item
    /// 03 step 3), and integer accumulation is item 04.
    Reduction {
        /// The associative combine (+ implied identity).
        op: ReduceOp,
        /// Canonical reduced-axis set (bit `i` ⇒ axis `i`). `AxisMask::EMPTY` ⇒
        /// the legacy last-axis default (`OpDef::reduction` preserves this).
        axes: AxisMask,
        /// Keep reduced axes as size-1 (broadcast-back) vs. collapse them.
        keepdim: bool,
        /// **Fused post-expression / epilogue** (increment 0e), applied to the
        /// finalized fold result before the store. Mirrors the `Contraction`
        /// epilogue convention: it references the reduced result as
        /// [`ScalarExpr::Reduced`]`(0)` and MAY reference `Const`/`Param`, but
        /// NOT `Input` (the reduced axis is gone — an `Input` at the output
        /// coordinate is a different, ambiguous thing; rejected at
        /// `plan::assert_valid_reduction_post`) nor `Coord` (reduction-class,
        /// Elementwise-only). The default is the identity `Reduced(0)`
        /// (`OpDef::reduction`/`reduction_axes` set this), so every existing
        /// reduction emits byte-identically. **Ordering vs. Mean (pinned):** the
        /// post sees the POST-Mean value — `Mean` divides first, then the post
        /// applies to the mean (the natural "finalize" semantics). Unlocks
        /// `norm2 = Sqrt(Sum(Sqr(x)))`, logsumexp-finalize, dot-with-scale, and
        /// the boolean/count hetero-out reductions (via a `Cmp*` post + a `U8`
        /// `out_dtype`). The post lowers through the SAME accumulator-width
        /// scalar-expr emitter as the fold body, so all 0a–0d vocabulary
        /// composes in it.
        post: ScalarExpr,
    },
    /// Fused **reduce → broadcast → elementwise** over the contiguous last axis:
    /// the `stages` fold per-row reduced scalars (`Reduced(0..n)`), then `epilogue`
    /// (which may read those scalars and the `Input`s) is the per-element,
    /// full-width output. RmsNorm (1 stage) and Softmax (2 stages) are instances —
    /// one block per row, no hand-written per-op CUDA. v1: single input,
    /// float-dtype, contiguous; per-column weight/bias (LayerNorm) is a follow-up.
    RowReduce {
        /// Ordered reduction stages; stage `i` produces `Reduced(i)`.
        stages: Vec<ReduceStage>,
        /// Per-element output expression (references `Input`s + `Reduced(0..n)`).
        epilogue: ScalarExpr,
    },
    /// Batched **contraction** (the terminal ORDER-3 node; item-10 spike):
    /// `out[m,n] = epilogue( Σ_k lhs[m,k] · rhs[k,n] )`. One contracted axis
    /// group (K), free axes M (input 0) and N (input 1) — the K-fold is fused
    /// with TWO free axes, which neither [`Access::Reduction`] (one free axis)
    /// nor [`Access::RowReduce`] (row-broadcast, no second free axis) can
    /// express. The K-accumulator reaches the epilogue as the `Reduced(0)` leaf
    /// (the same bridge `RowReduce` uses — the item-02 "contraction producer"
    /// hook in its v1 form). v1: rank-2 single-K dense row-major, epilogue over
    /// `Reduced(0)` only; batch axes, transposes, and fused bias inputs are the
    /// node's growth axes.
    Contraction {
        /// Per-operand axis roles (the AxisRole vocabulary, wired here per the
        /// item-10 spike / `docs/design/axis-role-vocabulary.md`).
        axes: ContractionAxes,
        /// K-accumulation policy.
        accum: AccumSpec,
        /// Per-output-element epilogue over the K-sum (`Reduced(0)`).
        epilogue: ScalarExpr,
    },
}

/// Per-axis role in a contraction — the `{Batch, FreeM, FreeN, ContractedK}`
/// projection of the unified AxisRole vocabulary (`axis-role-vocabulary.md`;
/// reductions carry the `{Reduced}` projection as `StructureKey::reduce_axes`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AxisRole {
    /// Shared, iterated, not summed (lhs & rhs & out). v1: unused (rank-2).
    Batch,
    /// Free on the lhs → a row of the output.
    FreeM,
    /// Free on the rhs → a column of the output.
    FreeN,
    /// Shared and summed; absent from the output.
    ContractedK,
}

/// Which axis of each input plays which role. v1 pins the canonical dense
/// matmul assignment; the constructor exists so the vocabulary (not a bare
/// convention) is what the emitter and key read — general einsum role vectors
/// are the growth path without reshaping this type's consumers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractionAxes {
    /// Roles of input 0's axes, in axis order.
    pub lhs: Vec<AxisRole>,
    /// Roles of input 1's axes, in axis order.
    pub rhs: Vec<AxisRole>,
}

impl ContractionAxes {
    /// The canonical rank-2 matmul: `lhs [M,K]`, `rhs [K,N]`.
    #[must_use]
    pub fn matmul() -> Self {
        Self {
            lhs: vec![AxisRole::FreeM, AxisRole::ContractedK],
            rhs: vec![AxisRole::ContractedK, AxisRole::FreeN],
        }
    }
}

/// K-accumulation policy for a contraction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AccumSpec {
    /// Accumulate in `float` (`double` for f64/f32-strict inputs) — the SIMT
    /// path, deterministic for a fixed schedule; the same widening discipline
    /// as [`Access::Reduction`]. Tensor-core/TF32 policies join as variants
    /// with honest contract flips (see the item-10 spike §5.3).
    WideFloat,
}

/// How input operand `i` is read relative to the op's iteration space — a
/// structural (compile-time) layout fact the emitter folds into address math.
/// It is deliberately **not** part of [`ScalarExpr`] (per-coordinate *value*
/// math) and **not** an [`Access`] variant (a whole-op loop-nest change): a view
/// is a *per-operand read-through*, so it lives orthogonally on [`OpDef::views`].
/// That keeps the value-math walkers (optimizer/e-graph, `contract`, `pattern`)
/// untouched. `Identity` reads at the iteration coordinate (today's behavior);
/// the other variants let a fused op read an input *through* a layout change in
/// one pass, skipping a materialized `contiguize`/transpose copy (the §1 win).
///
/// v1 emits `Transpose` (= rank-2 `Permute`) / `Permute` / `Broadcast`;
/// `Reshape` is carried for recognition + keying only (a reshape of a contiguous
/// producer is the identity linear-index map — genuine rank-change emit belongs
/// to items 03/10).
#[derive(Clone, Debug, PartialEq, Default)]
pub enum View {
    /// Read operand `i` at the iteration coordinate — no layout change (default).
    #[default]
    Identity,
    /// Read a permutation of the producer: iteration axis `d` indexes producer
    /// axis `perm[d]`. `perm` is a permutation of `0..rank` (the rank-2 case is a
    /// transpose); validate with [`View::is_valid`].
    Permute {
        /// Permutation of `0..rank`: iteration axis `d` → producer axis `perm[d]`.
        perm: Vec<u8>,
    },
    /// Broadcast a lower-rank / size-1 producer up to the iteration shape: `bcast`
    /// marks the iteration axes the producer does **not** vary along (stride 0).
    /// The named IR form of what [`baracuda_kernels_types::OperandKey`]'s
    /// broadcast mask already encodes on the schedule side.
    Broadcast {
        /// Iteration axes along which the producer is broadcast (stride 0).
        bcast: AxisMask,
    },
    /// The producer is contiguous with a different logical rank but the **same**
    /// linear element order, so reading is a pure linear-index pass-through.
    /// Carries the producer rank for contract/keying only (no address math).
    Reshape {
        /// Logical rank of the pre-reshape producer.
        producer_rank: u8,
    },
}

impl View {
    /// `true` iff structurally well-formed for an op iterating over `rank` axes: a
    /// `Permute` must carry a true permutation of `0..rank`; the other variants are
    /// always well-formed. Extent agreement between the declared view and the
    /// runtime `shape[]`/stride arrays is a *caller* precondition (the same trust
    /// level as the RowReduce `n_out`/`k` contract), because
    /// [`baracuda_kernels_types::StructureKey`] deliberately abstracts numeric
    /// extents away.
    #[must_use]
    pub fn is_valid(&self, rank: u8) -> bool {
        match self {
            View::Identity | View::Broadcast { .. } | View::Reshape { .. } => true,
            View::Permute { perm } => is_permutation(perm, rank),
        }
    }

    /// `true` for [`View::Identity`] — the back-compat default that leaves address
    /// math unchanged.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        matches!(self, View::Identity)
    }
}

/// `true` iff `perm` is a permutation of `0..rank` (each axis in range, no dup).
fn is_permutation(perm: &[u8], rank: u8) -> bool {
    if perm.len() != rank as usize {
        return false;
    }
    let mut seen = 0u64;
    for &a in perm {
        // `a >= 64` guard keeps the shift in range regardless of a bogus `rank`;
        // any valid axis is `< rank <= MAX_RANK (8)`.
        if a as usize >= rank as usize || a >= 64 {
            return false;
        }
        let bit = 1u64 << a;
        if seen & bit != 0 {
            return false; // duplicate axis
        }
        seen |= bit;
    }
    true
}

/// An op definition — the **algorithm** half of the algorithm/schedule split.
///
/// Names the op, its input-operand count, the output expression, the accepted
/// dtypes, and the access pattern. The generator fans one `OpDef` out across
/// many [`baracuda_kernels_types::StructureKey`] cells (the schedule half).
#[derive(Clone, Debug)]
pub struct OpDef {
    /// Stable op name — used in generated symbol names and the FKC contract.
    pub name: String,
    /// Number of input operands the body references.
    pub n_inputs: u8,
    /// Output `= body` evaluated at each coordinate.
    pub body: ScalarExpr,
    /// Dtypes this op accepts.
    pub dtypes: Vec<ElementKind>,
    /// Iteration pattern.
    pub access: Access,
    /// Per-input layout view (index `i` ↔ `Input(i)`). Empty ⇒ every input is
    /// [`View::Identity`] (back-compat: every existing `OpDef` is view-free). When
    /// non-empty, length **must** equal `n_inputs`. Set via [`OpDef::with_views`].
    pub views: Vec<View>,
    /// Output element dtype when it differs from the key (input) dtype. `None`
    /// (every uniform-dtype constructor) ⇒ output dtype == key dtype — the
    /// behavior every pre-0b op has, unchanged. The legal `Some(_)` set is
    /// validated AOT by `plan::assert_valid_out_dtype`:
    ///
    /// - **`Some(U8)`** — a `u8` 0/1 mask. On an [`Access::Elementwise`] body
    ///   whose ROOT is a `Cmp*` predicate (the `elementwise_pred` case, the FKC
    ///   "comparison → U8 mask" convention), or on an [`Access::Reduction`]
    ///   whose `post` root is a `Cmp*` (a boolean `any`/`all` reduce). The
    ///   store converts the exact 0.0/1.0 to `u8`.
    /// - **`Some(I64)`** — a widened integer output (increment 0e), only on an
    ///   [`Access::Reduction`] with `Sum` + an identity `Reduced(0)` post
    ///   (a `count`/sum-widening reduce). Exact for an integer accumulator;
    ///   exact for a float accumulator only while the count ≤ 2²⁴ — a caller
    ///   precondition (the key abstracts extents away), at the `Coord` trust
    ///   level.
    ///
    /// Keying is untouched: `StructureKey.dtype` stays operand-0 (input) dtype
    /// per the schema ("v1 assumes a uniform operand dtype; mixed-dtype folds
    /// in a follow-up"); the caller's output `OperandDesc` carries the hetero
    /// dtype, which only shapes that operand's own layout facts.
    pub out_dtype: Option<ElementKind>,
}

impl OpDef {
    /// Build an elementwise op from a name, input count, accepted dtypes, and a
    /// body expression.
    #[must_use]
    pub fn elementwise(name: &str, n_inputs: u8, dtypes: &[ElementKind], body: Expr) -> Self {
        Self {
            name: name.to_string(),
            n_inputs,
            body: body.0,
            dtypes: dtypes.to_vec(),
            access: Access::Elementwise,
            views: Vec::new(),
            out_dtype: None,
        }
    }

    /// Build an elementwise **predicate** op: `body`'s root must be a `Cmp*`
    /// comparison, and the output is stored as a **u8 `1`/`0` mask**
    /// (`out_dtype = Some(U8)`) — the FKC §4.1 "comparison → U8 mask" shape.
    /// `dtypes` are the accepted *input* dtypes (the key dtype); the store's
    /// float→u8 conversion is exact because a `Cmp*` root yields exactly 0.0 or
    /// 1.0. The root-is-cmp rule (and the Elementwise-only rule) is enforced at
    /// plan time by `assert_valid_out_dtype` — a non-predicate body with a u8
    /// output would truncate silently and is an authoring error, not a miss.
    #[must_use]
    pub fn elementwise_pred(name: &str, n_inputs: u8, dtypes: &[ElementKind], body: Expr) -> Self {
        Self {
            name: name.to_string(),
            n_inputs,
            body: body.0,
            dtypes: dtypes.to_vec(),
            access: Access::Elementwise,
            views: Vec::new(),
            out_dtype: Some(ElementKind::U8),
        }
    }

    /// Build a **last-axis reduction** op: `body` is the per-element pre-reduction
    /// expression (e.g. `input(0).unary(Sqr)` for a mean-of-squares), folded over
    /// the contiguous trailing axis by `op`. The output holds one element per
    /// outer coordinate. This is the legacy default — `axes = EMPTY`, no keepdim —
    /// and is byte-identical to before item 03. See [`Access::Reduction`].
    #[must_use]
    pub fn reduction(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        body: Expr,
        op: ReduceOp,
    ) -> Self {
        Self::reduction_axes(name, n_inputs, dtypes, body, op, AxisMask::EMPTY, false)
    }

    /// Build a reduction over an explicit `axes` set (bit `i` ⇒ axis `i`), with
    /// `keepdim` selecting broadcast-back (size-1 reduced axes) vs. collapse.
    /// `axes == AxisMask::EMPTY` is the last-axis legacy default and reproduces
    /// [`OpDef::reduction`] exactly. The emitter's generalized outer/middle/multi
    /// axis + keepdim handling lands in item 03 step 3; until then a non-empty
    /// mask is *representable* here but only lowered by that follow-up. The
    /// post-expression is the identity `Reduced(0)` (see [`OpDef::reduction_post`]
    /// for the fused-epilogue form), so emission is byte-identical to pre-0e.
    #[must_use]
    pub fn reduction_axes(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        body: Expr,
        op: ReduceOp,
        axes: AxisMask,
        keepdim: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            n_inputs,
            body: body.0,
            dtypes: dtypes.to_vec(),
            access: Access::Reduction {
                op,
                axes,
                keepdim,
                post: ScalarExpr::Reduced(0),
            },
            views: Vec::new(),
            out_dtype: None,
        }
    }

    /// Build a **last-axis reduction with a fused post-expression** (increment
    /// 0e). `body` is the per-element pre-reduction expression (as
    /// [`OpDef::reduction`]); `op` is the combine; `post` is the epilogue applied
    /// to the finalized fold result, which it references as
    /// [`reduced`]`(0)` — e.g. `norm2 = Sqrt(Sum(Sqr(x)))` is
    /// `reduction_post("norm2", 1, dt, input(0).unary(Sqr), Sum, reduced(0).sqrt())`.
    /// `post` MAY read `Const`/`Param` but NOT `Input` (validated at
    /// `plan::assert_valid_reduction_post`). The post sees the POST-Mean value.
    /// This is the last-axis (`AxisMask::EMPTY`, no keepdim) form; a fused post
    /// over an explicit axis set is a follow-up (construct the `Access` directly).
    #[must_use]
    pub fn reduction_post(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        body: Expr,
        op: ReduceOp,
        post: Expr,
    ) -> Self {
        Self {
            name: name.to_string(),
            n_inputs,
            body: body.0,
            dtypes: dtypes.to_vec(),
            access: Access::Reduction {
                op,
                axes: AxisMask::EMPTY,
                keepdim: false,
                post: post.0,
            },
            views: Vec::new(),
            out_dtype: None,
        }
    }

    /// Build a **fused row-reduction** op (reduce → broadcast → elementwise over
    /// the last axis). `stages` are the ordered reductions (stage `i` →
    /// `Reduced(i)`); `epilogue` is the per-element output (references `Input`s and
    /// `Reduced(0..stages.len())`). `body` is set to the epilogue so the existing
    /// body-walkers (`params_used`/`count_flops`/dtype plumbing) operate on the
    /// row-output expression unchanged. See [`Access::RowReduce`] for the v1 scope.
    #[must_use]
    pub fn row_reduce(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        stages: Vec<ReduceStage>,
        epilogue: Expr,
    ) -> Self {
        Self {
            name: name.to_string(),
            n_inputs,
            body: epilogue.0.clone(),
            dtypes: dtypes.to_vec(),
            access: Access::RowReduce {
                stages,
                epilogue: epilogue.0,
            },
            views: Vec::new(),
            out_dtype: None,
        }
    }

    /// Build a **contraction** op (`out[m,n] = epilogue(Σ_k lhs[m,k]·rhs[k,n])`).
    /// `epilogue` references the K-sum as `Reduced(0)` (identity: `reduced(0)`);
    /// `body == epilogue`, mirroring [`OpDef::row_reduce`], so every body-walker
    /// (params/flops/ulp/DAG) operates unchanged. v1: exactly 2 inputs, epilogue
    /// over `Reduced(0)` only (fused bias inputs are a follow-up).
    #[must_use]
    pub fn contraction(
        name: &str,
        dtypes: &[ElementKind],
        axes: ContractionAxes,
        epilogue: Expr,
    ) -> Self {
        Self {
            name: name.to_string(),
            n_inputs: 2,
            body: epilogue.0.clone(),
            dtypes: dtypes.to_vec(),
            access: Access::Contraction {
                axes,
                accum: AccumSpec::WideFloat,
                epilogue: epilogue.0,
            },
            views: Vec::new(),
            out_dtype: None,
        }
    }

    /// Attach per-input layout [`View`]s (item 01). `views[i]` applies to
    /// `Input(i)`; `views.len()` must equal `n_inputs`. A view-free op (the common
    /// case) never calls this and keeps `views` empty. The debug assert catches a
    /// generator bug at catalog-build time; per-`Permute` validity is checked later
    /// (in `plan`/`cuda`) once the iteration rank is known.
    #[must_use]
    pub fn with_views(mut self, views: Vec<View>) -> Self {
        debug_assert_eq!(
            views.len(),
            self.n_inputs as usize,
            "OpDef::with_views: views.len() must equal n_inputs"
        );
        self.views = views;
        self
    }
}

#[cfg(test)]
mod view_tests {
    use super::*;

    #[test]
    fn view_default_is_identity() {
        assert_eq!(View::default(), View::Identity);
        assert!(View::Identity.is_identity());
        assert!(!View::Permute { perm: vec![1, 0] }.is_identity());
    }

    #[test]
    fn existing_constructors_are_view_free() {
        // Back-compat: every current OpDef builds with empty `views`.
        let ew = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu());
        assert!(ew.views.is_empty());
        let red = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        assert!(red.views.is_empty());
    }

    #[test]
    fn existing_constructors_have_uniform_out_dtype() {
        // Back-compat (increment 0b): every pre-existing constructor sets
        // out_dtype = None (output dtype == key dtype) — zero behavior change.
        let ew = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu());
        assert_eq!(ew.out_dtype, None);
        let red = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        assert_eq!(red.out_dtype, None);
        let rr = OpDef::row_reduce(
            "rms",
            1,
            &[ElementKind::F32],
            vec![ReduceStage { pre: ScalarExpr::Input(0), op: ReduceOp::Mean }],
            reduced(0),
        );
        assert_eq!(rr.out_dtype, None);
        let mm = OpDef::contraction(
            "mm",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        assert_eq!(mm.out_dtype, None);
    }

    #[test]
    fn elementwise_pred_sets_u8_out_dtype_and_is_cmp_covers_exactly_six() {
        let p = OpDef::elementwise_pred(
            "cmp_lt",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpLt, input(1)),
        );
        assert_eq!(p.out_dtype, Some(ElementKind::U8));
        assert!(matches!(p.access, Access::Elementwise));
        // is_cmp: exactly the six predicates, nothing else (FmaxIeee/Max etc.
        // must never gain mask-store semantics by drifting into this set).
        for op in [
            BinaryOp::CmpEq,
            BinaryOp::CmpNe,
            BinaryOp::CmpLt,
            BinaryOp::CmpLe,
            BinaryOp::CmpGt,
            BinaryOp::CmpGe,
        ] {
            assert!(op.is_cmp());
        }
        for op in [
            BinaryOp::Max,
            BinaryOp::Min,
            BinaryOp::Pow,
            BinaryOp::Rem,
            BinaryOp::Atan2,
            BinaryOp::Copysign,
            BinaryOp::Nextafter,
            BinaryOp::FmaxIeee,
            BinaryOp::FminIeee,
            BinaryOp::RemTrunc,
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::LogicalAnd,
            BinaryOp::LogicalOr,
            BinaryOp::LogicalXor,
        ] {
            assert!(!op.is_cmp());
        }
    }

    #[test]
    fn is_int_only_and_is_logical_cover_exactly_the_0c_sets() {
        // is_int_only: exactly the 8 increment-0c ops — no float op may drift
        // into the int lowering path, and no int op may reach a float speller.
        let int_only = [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::LogicalAnd,
            BinaryOp::LogicalOr,
            BinaryOp::LogicalXor,
        ];
        for op in int_only {
            assert!(op.is_int_only(), "{op:?}");
        }
        for op in [
            BinaryOp::Max,
            BinaryOp::Min,
            BinaryOp::Pow,
            BinaryOp::Rem,
            BinaryOp::Atan2,
            BinaryOp::Copysign,
            BinaryOp::Nextafter,
            BinaryOp::FmaxIeee,
            BinaryOp::FminIeee,
            BinaryOp::RemTrunc,
            BinaryOp::CmpEq,
            BinaryOp::CmpNe,
            BinaryOp::CmpLt,
            BinaryOp::CmpLe,
            BinaryOp::CmpGt,
            BinaryOp::CmpGe,
        ] {
            assert!(!op.is_int_only(), "{op:?}");
        }
        // is_logical: exactly the three 0/1-normalizing ops (U8-only set) —
        // the bitwise ops must NOT gain the U8-only restriction, and the
        // logical ops must NOT silently widen to I32/I64.
        for op in [BinaryOp::LogicalAnd, BinaryOp::LogicalOr, BinaryOp::LogicalXor] {
            assert!(op.is_logical(), "{op:?}");
        }
        for op in [BinaryOp::BitAnd, BinaryOp::BitOr, BinaryOp::BitXor, BinaryOp::Shl, BinaryOp::Shr] {
            assert!(!op.is_logical(), "{op:?}");
        }
    }

    #[test]
    fn with_views_sets_per_input_views() {
        let op = OpDef::elementwise("add_t", 2, &[ElementKind::F32], input(0) + input(1))
            .with_views(vec![View::Permute { perm: vec![1, 0] }, View::Identity]);
        assert_eq!(op.views.len(), 2);
        assert_eq!(op.views[0], View::Permute { perm: vec![1, 0] });
        assert!(op.views[1].is_identity());
    }

    #[test]
    fn permute_validity() {
        assert!(View::Permute { perm: vec![1, 0] }.is_valid(2));
        assert!(View::Permute { perm: vec![2, 0, 1] }.is_valid(3));
        assert!(!View::Permute { perm: vec![0, 1] }.is_valid(3)); // wrong length
        assert!(!View::Permute { perm: vec![0, 0] }.is_valid(2)); // duplicate axis
        assert!(!View::Permute { perm: vec![0, 5] }.is_valid(2)); // out-of-range axis
        assert!(View::Identity.is_valid(4));
        assert!(View::Broadcast { bcast: AxisMask::EMPTY }.is_valid(4));
        assert!(View::Reshape { producer_rank: 2 }.is_valid(3));
    }
}

#[cfg(test)]
mod reduction_axes_tests {
    use super::*;

    #[test]
    fn reduction_defaults_to_last_axis_empty_mask() {
        // OpDef::reduction stays the legacy last-axis default: empty mask, no
        // keepdim — byte-identical to before item 03.
        match OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum).access {
            Access::Reduction { op, axes, keepdim, post } => {
                assert_eq!(op, ReduceOp::Sum);
                assert!(axes.is_empty());
                assert!(!keepdim);
                // Default post is the identity Reduced(0) — byte-identical emission.
                assert_eq!(post, ScalarExpr::Reduced(0));
            }
            other => panic!("expected Access::Reduction, got {other:?}"),
        }
    }

    #[test]
    fn reduction_axes_carries_axis_set_and_keepdim() {
        // Reduce axis 0, keepdim on.
        match OpDef::reduction_axes(
            "mean0",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Mean,
            AxisMask(0b01),
            true,
        )
        .access
        {
            Access::Reduction { op, axes, keepdim, post } => {
                assert_eq!(op, ReduceOp::Mean);
                assert!(axes.is_set(0));
                assert!(!axes.is_set(1));
                assert!(keepdim);
                assert_eq!(post, ScalarExpr::Reduced(0));
            }
            other => panic!("expected Access::Reduction, got {other:?}"),
        }
    }

    #[test]
    fn reduction_post_carries_the_epilogue_and_defaults_last_axis() {
        // reduction_post: last-axis (empty mask, no keepdim), Prod combiner, and a
        // real Sqrt post over Reduced(0) — the norm2 shape.
        match OpDef::reduction_post(
            "norm2",
            1,
            &[ElementKind::F32],
            input(0).unary(UnaryOp::Sqr),
            ReduceOp::Sum,
            reduced(0).sqrt(),
        )
        .access
        {
            Access::Reduction { op, axes, keepdim, post } => {
                assert_eq!(op, ReduceOp::Sum);
                assert!(axes.is_empty());
                assert!(!keepdim);
                assert_eq!(
                    post,
                    ScalarExpr::Unary(UnaryOp::Sqrt, Box::new(ScalarExpr::Reduced(0)))
                );
            }
            other => panic!("expected Access::Reduction, got {other:?}"),
        }
    }

    #[test]
    fn prod_is_a_distinct_combiner() {
        match OpDef::reduction("p", 1, &[ElementKind::F32], input(0), ReduceOp::Prod).access {
            Access::Reduction { op, .. } => assert_eq!(op, ReduceOp::Prod),
            other => panic!("expected Access::Reduction, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod dag_tests {
    use super::*;

    fn ipt(i: u8) -> ScalarExpr {
        ScalarExpr::Input(i)
    }
    fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Mul(Box::new(a), Box::new(b))
    }
    fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Add(Box::new(a), Box::new(b))
    }

    /// Find the single node matching `pred`, asserting there is exactly one.
    fn only<F: Fn(&DagNode) -> bool>(dag: &ExprDag, pred: F) -> NodeId {
        let hits: Vec<NodeId> = (0..dag.len() as NodeId).filter(|&i| pred(dag.node(i))).collect();
        assert_eq!(hits.len(), 1, "expected exactly one matching node, got {hits:?}");
        hits[0]
    }

    #[test]
    fn diamond_shares_one_interior_with_two_consumers() {
        // g = a*b; out = g + g. The two structurally-identical Mul subtrees must
        // collapse to ONE node with consumers == 2.
        let g = mul(ipt(0), ipt(1));
        let dag = ExprDag::from_expr(&add(g.clone(), g));
        assert_eq!(dag.len(), 4, "Input0, Input1, Mul, Add — Mul stored once");
        let m = only(&dag, |n| matches!(n, DagNode::Mul(..)));
        assert_eq!(dag.consumers(m), 2, "the shared Mul feeds both Add operands");
        // Its two operands are the same interior value; the Add references it twice.
        assert!(matches!(dag.node(dag.root()), DagNode::Add(a, b) if a == b));
    }

    #[test]
    fn same_parent_twice_still_counts_as_shared() {
        // x*x = Mul(Input0, Input0): one Input0 node, consumers == 2, and the Mul's
        // two children are the SAME id (a leaf — hoisting is the emitter's call).
        let dag = ExprDag::from_expr(&mul(ipt(0), ipt(0)));
        assert_eq!(dag.len(), 2, "Input0 stored once + the Mul");
        let i0 = only(&dag, |n| matches!(n, DagNode::Input(0)));
        assert_eq!(dag.consumers(i0), 2);
        assert!(dag.node(i0).is_leaf());
        assert!(matches!(dag.node(dag.root()), DagNode::Mul(a, b) if a == b));
    }

    #[test]
    fn pure_chain_has_all_unit_consumers_and_round_trips() {
        // relu(a + b) * c — no repeats: every node has one consumer, and the DAG
        // reconstructs to the original tree (interning is a value identity).
        let expr = mul(
            ScalarExpr::Unary(UnaryOp::Relu, Box::new(add(ipt(0), ipt(1)))),
            ipt(2),
        );
        let dag = ExprDag::from_expr(&expr);
        for id in 0..dag.len() as NodeId {
            if id != dag.root() {
                assert_eq!(dag.consumers(id), 1, "node {id} is single-use in a chain");
            }
        }
        assert_eq!(dag.to_expr(), expr, "round-trip preserves the expression");
    }

    #[test]
    fn const_interns_by_bits_including_nan() {
        // Two Const(NaN) share one node (NaN-safe by bits), like the e-graph.
        let nan = ScalarExpr::Const(f64::NAN);
        let dag = ExprDag::from_expr(&add(nan.clone(), nan));
        assert_eq!(dag.len(), 2, "one Const(NaN) + the Add");
        let c = only(&dag, |n| matches!(n, DagNode::Const(_)));
        assert_eq!(dag.consumers(c), 2);
        // Distinct constants stay distinct.
        let two = ExprDag::from_expr(&add(ScalarExpr::Const(1.0), ScalarExpr::Const(2.0)));
        assert_eq!(two.len(), 3, "1.0, 2.0, Add — no false merge");
    }

    #[test]
    fn reduced_leaf_shared_but_never_merged_across_indices() {
        // A shared Reduced(0) (the Softmax shape: exp(x - r0) reused) interns once;
        // Reduced(0) and Reduced(1) never merge.
        let r0 = ScalarExpr::Reduced(0);
        let dag = ExprDag::from_expr(&add(r0.clone(), r0));
        let r = only(&dag, |n| matches!(n, DagNode::Reduced(0)));
        assert_eq!(dag.consumers(r), 2);
        assert!(dag.node(r).is_leaf());
        let mixed = ExprDag::from_expr(&add(ScalarExpr::Reduced(0), ScalarExpr::Reduced(1)));
        assert_eq!(mixed.len(), 3, "Reduced(0) and Reduced(1) are distinct leaves");
    }

    #[test]
    fn coord_leaf_shared_but_never_merged_across_axes_or_kinds() {
        // A shared Coord(1) (the triu-mask shape: c1 compared and reused)
        // interns once and is a LEAF (never hoisted); Coord(0) and Coord(1)
        // never merge; Coord(i) never merges with Input(i)/Reduced(i)/Param(i)
        // (distinct DagKey kinds).
        let c1 = ScalarExpr::Coord(1);
        let dag = ExprDag::from_expr(&add(c1.clone(), c1));
        assert_eq!(dag.len(), 2, "one Coord(1) + the Add");
        let c = only(&dag, |n| matches!(n, DagNode::Coord(1)));
        assert_eq!(dag.consumers(c), 2);
        assert!(dag.node(c).is_leaf());
        let mixed = ExprDag::from_expr(&add(ScalarExpr::Coord(0), ScalarExpr::Coord(1)));
        assert_eq!(mixed.len(), 3, "Coord(0) and Coord(1) are distinct leaves");
        let kinds = ExprDag::from_expr(&add(
            add(ScalarExpr::Coord(0), ScalarExpr::Input(0)),
            add(ScalarExpr::Reduced(0), ScalarExpr::Param(0)),
        ));
        assert_eq!(kinds.len(), 7, "same-index leaves of different kinds never merge");
        // Round-trip: interning is a value identity for Coord bodies too.
        let body = mul(ScalarExpr::Coord(1), ipt(0));
        assert_eq!(ExprDag::from_expr(&body).to_expr(), body);
    }

    #[test]
    fn diamond_chain_stays_linear_not_exponential() {
        // Each level squares the *shared* value: v0 = a*b; v1 = v0*v0; v2 = v1*v1; …
        // A tree would be O(2^k) nodes; the DAG is O(k).
        let mut e = mul(ipt(0), ipt(1));
        for _ in 0..8 {
            e = mul(e.clone(), e);
        }
        let dag = ExprDag::from_expr(&e);
        // 2 inputs + 9 distinct Mul levels = 11 nodes (not 2^8-scale).
        assert_eq!(dag.len(), 11, "one node per level, shared — linear in depth");
    }
}
