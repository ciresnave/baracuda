//! The op **algorithm** IR — a small, backend-agnostic tensor expression.
//!
//! An op is the *pure function* computed at each output coordinate ([`OpDef`]),
//! described as a scalar-op DAG ([`ScalarExpr`]) over its input operands plus an
//! access pattern ([`Access`]). The emitter lowers this to a concrete backend
//! and *schedule* (chosen per [`baracuda_kernels_types::StructureKey`] cell).
//! Describing the math here — rather than as opaque CUDA — is what lets the
//! emitter vectorize, hoist, and fuse, because it can see the dataflow.

use baracuda_kernels_types::{AxisMask, ElementKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A scalar compute expression — the per-output-coordinate math, as a typed DAG.
///
/// Backend-agnostic: the emitter lowers it to CUDA today (and other backends
/// later) by walking the tree with a per-backend accessor for the leaves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScalarExpr {
    /// The value of input operand `i` at the current coordinate.
    Input(u8),
    /// A compile-time scalar constant — the same value at every coordinate.
    #[serde(with = "crate::text::f64_repr")]
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
    /// Bitwise ternary select: `out = if cond != 0 { a } else { b }` — the
    /// WHERE/SELECT increment, and the IR's first 3-child node.
    ///
    /// Operand order is `(cond, a, b)` — Fuel's `Where` order. `cond` is any
    /// expression in the compute dtype, tested `!= 0` at lowering
    /// (nonzero-true; `-0.0` is false — both zero signs are `!= 0`-false in
    /// every width; NaN is true — numpy truthiness. torch never sees a float
    /// cond: `torch.where` requires bool; Fuel never does either: its `Where`
    /// cond is a U8 tensor, IEEE-unordered handling lives upstream in the
    /// compare family).
    ///
    /// The chosen arm's bits move **UNTOUCHED**: no arithmetic, no conversion
    /// ever touches an arm — the C ternary with both arms typed in the
    /// compute dtype is data movement only (setp+selp), so the sign of zero
    /// and NaN payloads (quiet AND signaling) survive the pick. This is what
    /// distinguishes `select(cond, x, 0)` from the mask-multiply `x * cond`:
    /// the multiply stores `-0.0` for a masked negative and `NaN` for a
    /// masked NaN, and `x * 1.0` quiets a kept sNaN — the exact bespoke-triu
    /// bit gap the 0d on-device audit measured. The two forms are NEVER
    /// rewritten into each other (see `crate::optimize`).
    ///
    /// NOT commutative in any operand; never folded or rewritten (zero
    /// optimizer rules — `select_is_never_folded_or_rewritten` pins it).
    /// Legality (enforced at the top of `plan::build_plan` with independent
    /// emitter backstops in `cuda`): float compute dtypes only in v1
    /// (f32/f32s/f64/f16/bf16 — int select would raise the 0c U8/I8
    /// cond-observer question, rejected outright with zero bespoke-parity
    /// loss); legal in every `Access` arm at those dtypes (a select inside a
    /// Reduction pre-expr is the masked-sum shape). At f16/bf16 only the
    /// *cond* promotes to f32 (exact); arms are picked as raw half bits.
    Select(Box<ScalarExpr>, Box<ScalarExpr>, Box<ScalarExpr>),
}

/// A unary math / activation op. Variant names line up with the FKC §4.1
/// graph-`Op` vocabulary, so [`crate::derive_pattern`] maps them by name —
/// **except** the increment-0a scalar-fn extension (`Erfc` through `Lgamma`),
/// which Fuel's `OpTag`/§4.1 vocabulary does not name yet: those lower and
/// validate like any other op but are rejected by pattern derivation (an
/// honest miss — no invented tags) until Fuel adds the vocabulary.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// Rectified linear unit `x < 0 ? 0 : x` — NaN-propagating, -0.0-preserving
    /// (torch.relu convention; NOT `max(x, 0)`, which scrubs NaN and normalizes
    /// -0.0). The distinguishing fact the `ReluElementwise` lift + the bespoke
    /// propagating kernel rest on; every executable site (cuda.rs, optimize.rs
    /// const-fold) matches this spelling.
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// Bitwise ternary select over three nodes — `(cond, a, b)`, the first
    /// 3-child node ([`ScalarExpr::Select`]). Non-leaf: a shared select hoists
    /// to a `tmp` like any interior (both arms then evaluate eagerly —
    /// value-identical for the pure IR expressions arms are, no GPU traps).
    Select(NodeId, NodeId, NodeId),
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
    /// One output root per body — `[root]` for the single-body [`Self::from_expr`],
    /// one entry per output body for the multi-output [`Self::from_exprs`]. Because
    /// every body is interned into this SAME arena, a subexpression shared across
    /// bodies collapses to one node (its `consumers` count reflects the cross-body
    /// edges) — the cross-body CSE the multi-output emitter lowers once.
    roots: Vec<NodeId>,
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
            roots: vec![root],
        }
    }

    /// Hash-cons **several** [`ScalarExpr`] bodies into ONE value-numbered DAG,
    /// interning across bodies so a subexpression shared between outputs collapses
    /// to a single node — the cross-body CSE that makes a multi-output kernel load
    /// `dy` once and compute a shared interior once, then store to N outputs
    /// ([`OpDef::elementwise_multi`]). Returns the roots in body order via
    /// [`Self::roots`]; `root()` is `roots[0]` (output 0) for API compatibility.
    ///
    /// Order-of-composition note (pinned): the optimizer ([`crate::optimize`])
    /// simplifies a single `ScalarExpr`, so a multi-output op is optimized
    /// **per body first, THEN interned here** — the interning is what preserves the
    /// cross-body sharing, and running it after per-body optimization keeps both
    /// the correctness (each body simplified independently) and the sharing
    /// (structurally-equal simplified subtrees still hash-cons to one node).
    ///
    /// # Panics
    /// If `exprs` is empty (a DAG needs at least one root).
    #[must_use]
    pub fn from_exprs(exprs: &[&ScalarExpr]) -> ExprDag {
        assert!(
            !exprs.is_empty(),
            "ExprDag::from_exprs: needs at least one body"
        );
        let mut b = DagBuilder {
            nodes: Vec::new(),
            consumers: Vec::new(),
            memo: HashMap::new(),
        };
        let roots: Vec<NodeId> = exprs.iter().map(|e| b.intern(e)).collect();
        ExprDag {
            nodes: b.nodes,
            consumers: b.consumers,
            root: roots[0],
            roots,
        }
    }

    /// The output node of body 0 (the value output 0 computes). For a
    /// single-body DAG this is *the* output; for a multi-output DAG see
    /// [`Self::roots`].
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// The output root of every body, in body order (`[root()]` for a single-body
    /// DAG). The multi-output emitter lowers each, sharing one hoisted prelude.
    #[must_use]
    pub fn roots(&self) -> &[NodeId] {
        &self.roots
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
            DagNode::Select(c, a, b) => ScalarExpr::Select(
                Box::new(self.rebuild(c)),
                Box::new(self.rebuild(a)),
                Box::new(self.rebuild(b)),
            ),
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
    /// 3-child select key — hash-conses like any node (strictly positional:
    /// `(cond, a, b)` is never reordered, so no two orderings ever merge).
    Select(NodeId, NodeId, NodeId),
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
            DagNode::Select(c, a, b) => DagKey::Select(c, a, b),
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
            ScalarExpr::Select(c, a, b) => {
                let (c, a, b) = (self.intern(c), self.intern(a), self.intern(b));
                DagNode::Select(c, a, b)
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
        DagNode::Select(c, a, b) => vec![c, a, b],
    }
}

/// The associative combine of an [`Access::Reduction`]. The identity is implied
/// (`Sum`/`Mean` → 0; `Prod` → 1; `Max`/`Min` peel the first element, so no ±∞
/// literal — that keeps the emitted source header-light under nvrtc).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Sort direction of an [`Access::RowSort`] (increment 8). Ascending places the
/// smallest key first; descending the largest. The NaN convention is PINNED
/// (PyTorch): NaN compares GREATER than every non-NaN, so ascending places the
/// NaN block last and descending places it first. `-0.0`/`+0.0` and NaN-vs-NaN
/// are key-ties, resolved by the (ascending) original index (stability).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortOrder {
    /// Smallest key first (NaN block last).
    Asc,
    /// Largest key first (NaN block first).
    Desc,
}

/// Which buffers a row sort writes (increment 9 FUSED_ARGSORT). All three share
/// the SAME total order on `(key, original-index)` pairs, so the three outputs
/// are mutually consistent by construction; only the STORE differs.
///
/// - `Values` — a dtype-preserving RAW-BIT value permutation (today's
///   [`OpDef::row_sort`]); `out_dtype == None`, one output buffer.
/// - `Indices` — the `I32` sort permutation (today's [`OpDef::row_argsort`]);
///   `out_dtype == Some(I32)`, one output buffer. Caller precondition
///   `k <= 2^31 - 1` (the index cannot represent a position past it).
/// - `Both` — one kernel writes the value permutation to `out_val` AND the `I32`
///   index permutation to `out_idx` in a single launch (the fused
///   `(values, indices)` sort — bespoke's native one-kernel shape). `out_dtype ==
///   None` (output 0 = values, dtype-preserving; the I32 index is emitter-
///   hardwired off the entry-point symbol, not a per-operand dtype channel), the
///   key carries THREE operands `[in0, out_val, out_idx]`. Inherits argsort's
///   `k <= 2^31 - 1` cap (the I32 index output).
///
/// `body` stays `Input(0)` for all three; the index is a STRUCTURAL output (not a
/// [`ScalarExpr`] body), so it never rides `extra_out_bodies`, and `n_outputs()`
/// stays body-derived = 1 even for `Both` (the second buffer is owned locally by
/// this state + the 3-operand key, exactly as the emitter owns "argsort writes
/// index vs value" today).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SortOut {
    /// Values output (raw-bit permutation, dtype-preserving) — today's `row_sort`.
    Values,
    /// `I32` index output (the sort permutation) — today's `row_argsort`.
    Indices,
    /// Fused two-output: values to `out_val` AND `I32` indices to `out_idx`.
    Both,
}

/// Whether a row sort caps its output to a runtime top-`k_out` (increment 10
/// TOPK/BOTTOMK). ORTHOGONAL to [`SortOut`] (which buffers) — a capped sort still
/// independently writes Values / Indices / Both. `order` picks the direction:
/// `Desc` + `TopK` = top-k (largest first, torch.topk `largest=True`); `Asc` +
/// `TopK` = bottom-k (smallest first, `largest=False`). `Full` reproduces today's
/// sort byte-for-byte (single `k`, no store guard).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SortLimit {
    /// Sort the whole row; out extent == in extent (today's
    /// `row_sort`/`row_argsort`/`row_sort_indices`, single `long long k`).
    Full,
    /// Cap the output to a runtime `k_out <= k_in`: write only ranks `< k_out`
    /// (the top-`k_out` under `order`); out extent = `[batch, k_out]`. `k_out`
    /// rides a `long long` launch arg (the Window `(n_out, k_in, k_out)` ABI
    /// precedent); `k_out <= k_in` is a caller / on-device precondition (the
    /// structure key carries no numeric extent), on-device-validated by
    /// `initcheck` — the same trust tier as the bitonic `k <= 1024`. The
    /// index-writing states (`Indices`/`Both`) inherit argsort's `k_in <= 2^31-1`
    /// cap (the `I32` index).
    TopK,
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
    /// ReLU `x < 0 ? 0 : x` — NaN-propagating, -0.0-preserving (torch.relu; NOT
    /// `max(x, 0)`, which scrubs NaN). See [`UnaryOp::Relu`].
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

    /// Bitwise ternary select with `self` as the condition:
    /// `cond.select(a, b)` = `if cond != 0 { a } else { b }` — operand order
    /// `(cond, a, b)`, Fuel's `Where` order ([`ScalarExpr::Select`]). The
    /// chosen arm's bits move untouched (a pick, not arithmetic) — e.g. the
    /// bit-exact triu is
    /// `coord(1).binary(BinaryOp::CmpGe, coord(0) + konst(k)).select(input(0), konst(0.0))`,
    /// NOT the mask-multiply `input(0) * cond` (which stores `-0.0` for a
    /// masked negative).
    #[must_use]
    pub fn select(self, a: Expr, b: Expr) -> Expr {
        Expr(ScalarExpr::Select(
            Box::new(self.0),
            Box::new(a.0),
            Box::new(b.0),
        ))
    }
}

/// One reduction stage of an [`Access::RowReduce`]: fold `pre` (the per-element
/// pre-reduction expression) over the last axis with `op`. Stage `i` produces the
/// scalar [`ScalarExpr::Reduced`]`(i)`; its `pre` may reference `Reduced(j)` for
/// `j < i` (e.g. Softmax's exp-sum stage reads the row max from stage 0).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
        #[serde(with = "crate::text::axis")]
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
    /// **Prefix scan** along a single axis with a monoid combine (increment 6):
    /// `out[.., j] = op_folded( pre(in[.., 0..=j]) )` (inclusive) — a
    /// cumsum/cumprod/cummax/cummin. Unlike [`Access::Reduction`] (one output per
    /// row) the scan produces a **full-width** output: a running prefix that varies
    /// with `j`. v1 monoids are `Sum | Prod | Max | Min` on the innermost
    /// (contiguous) axis; the index-carrying cummax/cummin backward pair and
    /// `LogCumsumExp` (non-monoid) are DEFERRED.
    Scan {
        /// The associative monoid combine (reuses [`ReduceOp`]; `Mean` is rejected
        /// at `plan::validate_scan` — it is not a monoid).
        op: ReduceOp,
        /// The scanned axis. v1 asserts `axis == rank - 1` (innermost, contiguous).
        axis: u8,
        /// Walk the axis descending (`reverse` scan) instead of ascending.
        reverse: bool,
        /// **Exclusive** scan: output position `j` holds the fold of the elements
        /// STRICTLY before `j` (the first visited position holds the monoid
        /// identity). The default (`false`) is the inclusive scan.
        exclusive: bool,
        /// Per-element **pre-map** applied to each input element before it enters
        /// the fold (identity default = `ScalarExpr::Input(0)`).
        pre: ScalarExpr,
        /// Per-element **epilogue** applied to the running prefix after the combine
        /// (identity default = `ScalarExpr::Reduced(0)`, carrying the scan result
        /// unchanged — the running prefix reaches it as the `Reduced(0)` leaf, the
        /// same bridge `RowReduce`/`Contraction` epilogues use).
        post: ScalarExpr,
    },
    /// **Sliding-window reduction** along a single axis (increment 7) — the
    /// POOLING family (`max_pool`/`avg_pool`, plus the free `min_pool`/`sum_pool`
    /// the same monoid machinery yields). Structurally a cousin of
    /// [`Access::Reduction`], but the fold runs over a **sliding local
    /// neighborhood** of `size` taps rather than a whole axis, and the output axis
    /// is **DOWNSAMPLED**: `out_len = floor((in_len + pad_lo + pad_hi -
    /// dilation*(size-1) - 1)/stride) + 1`. Not a [`View`] (a view is a 1:1
    /// coordinate remap — one read per output); a window is a MULTI-TAP access
    /// (`size` reads per output) with a different output extent.
    ///
    /// For output coord `o` the window taps input position
    /// `p = o*stride - pad_lo + k*dilation` for `k in 0..size`. A tap with
    /// `p ∉ [0, in_len)` is **out of bounds**: it is SKIPPED for `Max`/`Min`
    /// (padding never wins), contributes the additive identity `0` for `Sum`, and
    /// for `Mean` (avg_pool) is excluded from the sum and — per
    /// `count_include_pad` — from the divisor (`false`: divide by the valid-tap
    /// count; `true`: divide by `size`). NaN propagates through `Max`/`Min` via the
    /// `v != v` probe (the same rule as [`Access::Scan`]/[`Access::Reduction`]).
    ///
    /// v1 scope (`plan::validate_window`): `axis == rank - 1` (innermost,
    /// contiguous); `op ∈ {Max, Min, Sum, Mean}` (`Prod` rejected — not a pool;
    /// `Mean` requires a float dtype — integer average rounds); `size`/`stride`/
    /// `dilation >= 1`; `2*pad_lo <= span` and `2*pad_hi <= span` where
    /// `span = dilation*(size-1)+1` (each edge window overlaps the input — the
    /// bespoke `pool1d` `pad*2 <= window` constraint, generalized to dilation).
    /// The `in_len → out_len` window arithmetic is a **runtime-launch-arg caller
    /// precondition** (the structure key abstracts numeric extents away, so it
    /// cannot be validated at plan time — the same trust level as RowReduce's
    /// `k`/`n_out`). Deferred: im2col (dimension EXPANSION), causal_conv1d (needs a
    /// weight operand → windowed contraction), interpolate/bilinear (Coord weights,
    /// 2-D), N-D / multi-axis windows, and overlap-backward (rides atomics /
    /// gather-sum).
    Window {
        /// The window combine (reuses [`ReduceOp`]). `Mean` = avg_pool (fold as a
        /// sum, then divide by the count per `count_include_pad`); `Sum` = sum_pool;
        /// `Max`/`Min` = max/min pool. `Prod` is rejected at `plan::validate_window`.
        op: ReduceOp,
        /// The pooled axis. v1 asserts `axis == rank - 1` (innermost, contiguous).
        axis: u8,
        /// Window length in taps (`>= 1`).
        size: u8,
        /// Output downsampling stride (`>= 1`).
        stride: u8,
        /// Inter-tap dilation (`>= 1`; `1` = dense taps).
        dilation: u8,
        /// Zero-/skip padding before the axis (low side).
        pad_lo: u8,
        /// Zero-/skip padding after the axis (high side).
        pad_hi: u8,
        /// **Mean only**: divide by `size` (`true`, count padding in the divisor —
        /// the TensorFlow / `count_include_pad=True` convention) vs. by the
        /// valid-tap count (`false` — the PyTorch avg_pool default). Ignored by
        /// `Max`/`Min`/`Sum` (padding is skipped / contributes the identity there).
        count_include_pad: bool,
        /// Per-tap **pre-map** applied to each in-bounds input tap before it enters
        /// the fold (identity default = `ScalarExpr::Input(0)`). Reads inputs only
        /// (no running result exists yet) — a `Reduced` leaf is rejected.
        pre: ScalarExpr,
        /// Per-output **epilogue** applied to the finalized window result, which it
        /// references as the single `Reduced(0)` leaf (identity default =
        /// `ScalarExpr::Reduced(0)`) — the same bridge `Scan`/`RowReduce`/
        /// `Contraction` epilogues use.
        post: ScalarExpr,
    },
    /// **Row sort / argsort** along the innermost (contiguous) axis (increment
    /// 8) — the `sort`/`argsort`/`msort` family. Each output row is a permutation
    /// of its input row under a total order on `(key, original-index)` pairs
    /// (index tie-break ⇒ every pair is distinct ⇒ a UNIQUE sorted sequence, so
    /// the result is deterministic, algorithm-independent, and stable = ascending
    /// original index within equal keys). NOT a [`View`] (a permutation is
    /// data-dependent, not a fixed coordinate remap) and NOT a [`Access::Scan`]
    /// (no running prefix; the whole row is reordered).
    ///
    /// The [`SortOut`] state selects which buffer(s) this variant writes — all
    /// three share the same total order, so the outputs are mutually consistent by
    /// construction: [`OpDef::row_sort`] (`out = Values`, dtype-preserving values)
    /// and [`OpDef::row_argsort`] (`out = Indices`, `I32` index output via the
    /// single-output `out_dtype` precedent) are the two single-output ops;
    /// [`OpDef::row_sort_indices`] (`out = Both`) is the fused increment-9 kernel
    /// that writes BOTH the value permutation AND the `I32` index permutation in
    /// one launch (bespoke's native one-kernel shape — no double-sort).
    ///
    /// NaN convention (PINNED, PyTorch): NaN compares GREATER than every non-NaN
    /// (asc ⇒ NaN block last, desc ⇒ NaN block first). NaN-vs-NaN and
    /// `-0.0`-vs-`+0.0` are key-ties resolved by index. The values writeback is a
    /// RAW-BIT permutation (it gathers original storage bytes), so NaN payloads
    /// and `-0.0` sign bits are preserved exactly (`memcmp`-checkable).
    ///
    /// v1 scope (`plan::validate_row_sort`): `stable == true` only (the emitter
    /// always pair-sorts, so stability is free — an unstable network would emit
    /// byte-identical code under a different symbol, dead keying); the innermost
    /// axis only; dtypes `F32|F32Strict|F64|F16|Bf16|I32|I64`. `topk`/`sparsemax`,
    /// hetero dual-output, non-inner axis, and `S8`/`U8` are deferred.
    RowSort {
        /// Ascending (smallest first) or descending (largest first). NaN orders
        /// GREATEST in both (asc ⇒ NaN last, desc ⇒ NaN first).
        order: SortOrder,
        /// v1 must be `true` (`plan::validate_row_sort` rejects `false`): the
        /// emitter always pair-sorts `(key, original-index)`, which is stable by
        /// construction. The field exists for a future faster unstable network.
        stable: bool,
        /// Which buffer(s) the sort writes: `Values` (raw-bit value permutation,
        /// `out_dtype = None`), `Indices` (the `I32` sort permutation, `out_dtype
        /// = Some(I32)`), or `Both` (the fused two-output kernel — values to
        /// `out_val`, `I32` indices to `out_idx`, `out_dtype = None`, a 3-operand
        /// key). See [`SortOut`].
        out: SortOut,
        /// Whether the output is capped to a runtime top-`k_out` (increment 10):
        /// `Full` = today's whole-row sort (byte-for-byte); `TopK` = write only the
        /// first `k_out` ranks under `order` (topk = `Desc`, bottomk = `Asc`), out
        /// extent `[batch, k_out]`, `k_out` a `long long` launch arg. ORTHOGONAL to
        /// `out`. See [`SortLimit`].
        limit: SortLimit,
    },
    /// **2-D im2col / unfold (increment 11)** — the conv-lowering workhorse
    /// (`Conv2d` = im2col then GEMM then reshape). A pure EXPANDING structured
    /// gather: each of the `kh*kw` window taps over a rank-4 `[N,C,H_in,W_in]` NCHW
    /// input becomes its OWN output cell, producing the column matrix
    /// `[N, C*kh*kw, oH*oW]` (Layout A — channel-major then tap, spatial row-major).
    /// The extent-INVERSE of [`Access::Window`] (which folds taps into one
    /// downsampled output); this expands them. NOT a [`View`] (1:1 remap) and NOT a
    /// data-dependent [`ReadIndex::Indexed`] gather — the source index is
    /// CLOSED-FORM from the loop coords.
    ///
    /// For output cell `(n, c, ki, kj, oh, ow)` (row = `c*kh*kw + ki*kw + kj`,
    /// col = `oh*oW + ow`) the source coord is
    /// `in_h = oh*stride.0 - pad.0 + ki*dilation.0`,
    /// `in_w = ow*stride.1 - pad.1 + kj*dilation.1`. A tap with
    /// `in_h` outside `[0,H_in)` or `in_w` outside `[0,W_in)` is OUT OF BOUNDS: it
    /// stores the op's typed ZERO (zero-pad convention, matching the bespoke
    /// `zero_of<T>()`). The copied value is RAW-BIT verbatim (no arithmetic) — every
    /// dtype is bit-exact, NaN payloads / -0 signs preserved.
    ///
    /// v1 scope (`plan::validate_im2col`): single input, rank-4 forward-dense
    /// contiguous NCHW; `groups == 1` (dense); output rank-3 `[N, C*kh*kw, oH*oW]`
    /// forward-dense contiguous; `kh,kw,stride.*,dilation.* >= 1`. The
    /// `(H_in,W_in) -> (oH,oW)` conv arithmetic is a **runtime-launch-arg caller
    /// precondition** (the key carries no extents — same tier as Window's
    /// `k_in`->`k_out`). Deferred: col2im/backward, 1-D/3-D, grouped conv, a per-tap
    /// `pre` map, and the im2col->GEMM->reshape `Conv2D` FUSION (the only
    /// advertisable path). See the increment-11 brief §7.
    Im2Col {
        /// `(kh, kw)` — window taps per spatial axis (compile-time literals).
        kernel: (u8, u8),
        /// `(stride_h, stride_w)` — output downsampling stride per axis (`>= 1`).
        stride: (u8, u8),
        /// `(pad_h, pad_w)` — zero-padding per axis (low == high, symmetric v1).
        pad: (u8, u8),
        /// `(dilation_h, dilation_w)` — inter-tap dilation per axis (`>= 1`).
        dilation: (u8, u8),
    },
}

/// Per-axis role in a contraction — the `{Batch, FreeM, FreeN, ContractedK}`
/// projection of the unified AxisRole vocabulary (`axis-role-vocabulary.md`;
/// reductions carry the `{Reduced}` projection as `StructureKey::reduce_axes`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
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
        #[serde(with = "crate::text::axis")]
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

/// Out-of-bounds policy for an [`ReadIndex::Indexed`] read (increment 4, GATHER).
/// The index tensor's value can name a position outside `[0, extent)` along the
/// gathered axis (a stale index, a negative index, an index past the source
/// extent); this picks the semantics — matched EXACTLY to what the bespoke
/// kernels do (`crates/baracuda-kernels-sys/kernels/include/baracuda_indexing.cuh`
/// / `baracuda_embedding.cuh`), the charter being "express the bespoke
/// functionality".
///
/// **Negative indices are always treated as out-of-bounds** (there is NO
/// PyTorch-style from-end wrap): the bespoke gather / index_select / embedding
/// all bounds-check `idx < 0 || idx >= extent` and skip / zero — confirmed per
/// kernel. A from-end-wrap policy is a deliberate non-feature here (bespoke
/// parity).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OobPolicy {
    /// Leave the output element **unwritten** when the index is out of range —
    /// the bespoke `gather` / `index_select` semantics (`continue;` before the
    /// store; the cell keeps its prior content). A store predicate; NO OOB load
    /// occurs (the emitter clamps the load address in-bounds and guards the
    /// store).
    Skip,
    /// Clamp the index to `[0, extent-1]` (`min(max(idx,0),extent-1)`) and always
    /// store — a generator-only policy (no bespoke op clamps; it exists for the
    /// coverage surface + the on-device probe). Assumes a **non-empty** gathered
    /// axis (`extent >= 1`); a 0-extent source axis has no valid target and is a
    /// degenerate shape.
    Clamp,
    /// Store the op's zero fill (`0`) when the index is out of range — the
    /// bespoke `embedding` semantics (OOB / negative / padding row → a zeroed
    /// output row). A store SELECT; no OOB load (address clamped in-bounds).
    ZeroFill,
}

/// How input operand `i` gets the address for ONE iteration axis — the
/// **data-dependent** read role (increment 4, GATHER; the first access pattern
/// whose address is a runtime tensor value, not a compile-time layout fact).
/// Lives per-input on [`OpDef::read_index`], parallel to [`OpDef::views`] and for
/// the identical reason: it is a per-operand *read-through* the value-math
/// walkers (optimizer/e-graph, `contract`, `pattern`) must not see, so it is NOT
/// a [`ScalarExpr`] node and NOT an [`Access`] variant. `Direct` (the default)
/// reads at the iteration coordinate (today's behavior); `Indexed` replaces the
/// coordinate along one axis with a value loaded from an integer index operand.
///
/// This is orthogonal to [`View`] (a compile-time layout remap): a view reorders
/// which *coordinate* pairs with which stride; an `Indexed` read substitutes a
/// runtime *value* for one coordinate. v1 keeps them mutually exclusive on the
/// same input (a gathered-and-permuted operand is deferred — see the plan gate).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum ReadIndex {
    /// Read operand `i` at the iteration coordinate — no indexing (default; every
    /// pre-increment-4 op). Byte-identical emission.
    #[default]
    Direct,
    /// Read operand `i` with iteration axis `axis` replaced by the value loaded
    /// from `index_operand` (an integer tensor) at the iteration coordinate:
    /// `operand_i[..., index_operand[coord], ...]` (the substituted term is
    /// `idx·stride[axis]` instead of `c{axis}·stride[axis]`). The index operand is
    /// itself a keyed input read through its own strides — a **full-shape** index
    /// (its key varies on every axis) is a torch-`gather`; a **1-D** index (its
    /// key broadcasts on every axis except `axis`) is an `index_select`/embedding.
    /// One mechanism, distinguished purely by the index operand's broadcast mask.
    ///
    /// `index_dtype` (`I32`/`I64`) rides HERE, on the op — NOT in the structure
    /// key (which carries a single operand-0 dtype and cannot express a
    /// per-operand index dtype). It selects the emitted index-load type and the
    /// `entry_point` symbol infix (`gather_f32_i32` vs `gather_f32_i64`), the same
    /// way increment 0b's non-primary u8 output dtype rides `OpDef::out_dtype` +
    /// the symbol rather than the token. See the plan gate for the full rule set.
    Indexed {
        /// Which input operand supplies the index values (an integer tensor).
        index_operand: u8,
        /// The iteration axis whose coordinate the index value replaces.
        axis: u8,
        /// Out-of-range behavior (bespoke-matched).
        oob: OobPolicy,
        /// Index element dtype — `I32` or `I64` (rides the op, not the key).
        #[serde(with = "crate::text::ek")]
        index_dtype: ElementKind,
    },
}

impl ReadIndex {
    /// `true` for [`ReadIndex::Direct`] — the back-compat default that reads at
    /// the iteration coordinate (byte-identical address math).
    #[must_use]
    pub fn is_direct(&self) -> bool {
        matches!(self, ReadIndex::Direct)
    }
}

/// Runtime base element offset for one operand (BASE_OFFSET SLICE — the post-ramp
/// increment that closes rope's pair-partner cross-read). [`Self::Zero`] (the
/// default) emits nothing and is byte-identical to today's kernels;
/// [`Self::Runtime`] adds a `long long off{i}` launch argument to the operand's
/// base pointer at kernel ENTRY — a per-launch value in **element units**, applied
/// BEFORE all per-element address math (so on a strided pair-split view it is the
/// view ORIGIN, never multiplied by any stride). This is a runtime launch-arg
/// slice: NOT a compile-time [`View`] layout remap (`View::is_valid` has no slot
/// for a runtime value) and NOT a [`ReadIndex`] data-dependent gather (there is no
/// index tensor) — it rides its own parallel [`OpDef::base_offsets`] Vec for the
/// identical reason those do (a per-operand *address-through* the value-math
/// walkers — optimizer/e-graph, `contract`, `pattern` — must not see, so it is
/// NOT a [`ScalarExpr`] node and NOT an [`Access`] variant, forcing zero new
/// exhaustive-match arms). Only the PRESENCE mask (which operands carry a
/// `Runtime` offset) is compile-time; the offset VALUE never enters the IR or the
/// structure key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaseOffset {
    /// No offset — the operand's base pointer is unchanged (byte-identical
    /// emission; every pre-increment operand).
    #[default]
    Zero,
    /// A runtime `long long off` launch argument added to the operand base at
    /// kernel entry (element units, applied before all address math).
    Runtime,
}

impl BaseOffset {
    /// `true` for [`BaseOffset::Zero`] — the byte-identical default (no launch
    /// arg, no pointer bump, no symbol suffix).
    #[must_use]
    pub fn is_zero(self) -> bool {
        matches!(self, BaseOffset::Zero)
    }
}

/// How a [`WriteIndex::ScatterIndexed`] output combines the scattered value into
/// its (data-dependent) destination cell (increment 5, SCATTER). Duplicate
/// indices name the same output cell from multiple threads, so the combine op's
/// algebra decides the op's **determinism class** — the core discipline of this
/// increment:
///
/// - [`Self::Assign`] — a plain store (last-writer-wins on a duplicate-target
///   race). Deterministic **iff the caller guarantees unique target indices**
///   (the documented `scatter`/`scatter_nd` precondition — bespoke
///   `indexing/scatter.cu` documents the same "last writer wins" race). No
///   accumulation, no atomic.
/// - [`Self::AtomicAdd`] — `atomicAdd`. Its determinism SPLITS on dtype:
///   **integer** add is exact + associative, so the accumulated RESULT is
///   order-independent ⇒ deterministic (bincount / integer scatter_add ship
///   unconditionally); **floating-point** add is non-associative and
///   `atomicAdd`'s completion order varies run-to-run ⇒ genuinely
///   [`crate::backend::VariantFidelity::Nondeterministic`] (ships only as a
///   gated variant; the deterministic default is the gather-sum reformulation).
/// - [`Self::AtomicMax`] / [`Self::AtomicMin`] — `atomicMax`/`atomicMin`. A
///   selection (returns one input verbatim, no rounding), so max/min over a set
///   is associative + commutative + order-independent for BOTH integer and
///   floating cells ⇒ the accumulated result is deterministic. v1 supports the
///   native-atomic INTEGER cells only (float has no native `atomicMax`; a CAS
///   emulation is a follow-up) — the plan gate enforces this.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WriteCombine {
    /// Plain store. Deterministic only with unique target indices.
    Assign,
    /// `atomicAdd`. Deterministic for integer cells; non-deterministic for FP.
    AtomicAdd,
    /// `atomicMax`. Order-independent (selection); integer cells in v1.
    AtomicMax,
    /// `atomicMin`. Order-independent (selection); integer cells in v1.
    AtomicMin,
}

impl WriteCombine {
    /// `true` if this combine accumulates through a **floating-point atomic add**
    /// for `out_dtype` — the one order-nondeterministic case (FP `atomicAdd`).
    /// Integer `AtomicAdd`, `Assign`, and `AtomicMax`/`AtomicMin` all produce an
    /// order-independent result ⇒ `false`. Drives the base-vs-variant split in
    /// [`crate::cuda`] (an FP-atomic scatter lowers its deterministic gather-sum
    /// as the base and the atomic as the `Nondeterministic` variant).
    #[must_use]
    pub fn is_fp_atomic_add(self, out_dtype: ElementKind) -> bool {
        matches!(self, WriteCombine::AtomicAdd) && !is_integer_kind(out_dtype)
    }
}

/// `true` for an integer [`ElementKind`] (the exact-associative-add dtypes). A
/// local mirror of `plan::is_int_dtype` so [`WriteCombine`] can classify without
/// a plan dependency; the `_` arm keeps a future dtype conservative (treated as
/// non-integer ⇒ the safe nondeterministic-variant route for an atomic add).
#[must_use]
fn is_integer_kind(dt: ElementKind) -> bool {
    matches!(
        dt,
        ElementKind::I32 | ElementKind::I64 | ElementKind::U8 | ElementKind::S8
    )
}

/// How the **single output** gets the address for ONE iteration axis — the
/// **data-dependent WRITE** role (increment 5, SCATTER; the write-side mirror of
/// increment 4's [`ReadIndex`]). Lives on [`OpDef::write_index`] as ONE role (v1
/// scatters into one output), and — exactly like [`ReadIndex`] — is a per-operand
/// *address-through* the value-math walkers (optimizer/e-graph, `contract`,
/// `pattern`) must not see, so it is NOT a [`ScalarExpr`] node and NOT an
/// [`Access`] variant. `Direct` (the default) writes at the iteration coordinate
/// (today's behavior, byte-identical); `ScatterIndexed` replaces the output
/// coordinate along one axis with a value loaded from an integer index operand,
/// and the store becomes the [`WriteCombine`] op.
///
/// The iteration domain of a scatter is the **updates/source** domain (one thread
/// per update element), NOT the destination — the destination extent along the
/// scattered axis differs, so it rides a dedicated launch scalar (`sext`), the
/// write-side mirror of gather's `gext`. Out-of-range indices are **skipped**
/// (bespoke `scatter`/`scatter_add`/`index_add`/`bincount` all `continue;` — no
/// negative-index wrap).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum WriteIndex {
    /// Write the output at the iteration coordinate — no scatter (default; every
    /// pre-increment-5 op). Byte-identical emission.
    #[default]
    Direct,
    /// Write the output with iteration axis `axis` replaced by the value loaded
    /// from `index_operand` (an integer input tensor) at the iteration
    /// coordinate: `out[..., index_operand[coord], ...] (combine)= value` (the
    /// substituted term is `idx·stride_out[axis]` instead of
    /// `c{axis}·stride_out[axis]`). A **full-shape** index (its key varies on
    /// every axis) is a torch-`scatter`/`scatter_add`; a **1-D** index (its key
    /// broadcasts on every axis except `axis`) is an `index_add`. One mechanism,
    /// distinguished purely by the index operand's broadcast mask — the write-side
    /// mirror of [`ReadIndex::Indexed`].
    ///
    /// `index_dtype` (`I32`/`I64`) rides HERE, on the op — NOT the structure key
    /// (single operand-0 dtype), exactly like [`ReadIndex::Indexed`]; it selects
    /// the emitted index-load type + the `entry_point` symbol infix.
    ScatterIndexed {
        /// Which input operand supplies the index values (an integer tensor). May
        /// equal the value operand (bincount indexes `Input(0)` = the data itself
        /// and writes a `Const(1)`), unlike a gather where an input can't index
        /// itself — a scatter's index selects the DESTINATION, not a source read.
        index_operand: u8,
        /// The iteration axis whose coordinate the index value replaces in the
        /// OUTPUT address.
        axis: u8,
        /// How the value combines into the destination cell (store / atomic).
        combine: WriteCombine,
        /// Out-of-range behavior. v1 supports [`OobPolicy::Skip`] only — every
        /// bespoke scatter/scatter_add/index_add/bincount skips an OOB target.
        oob: OobPolicy,
        /// Index element dtype — `I32` or `I64` (rides the op, not the key).
        #[serde(with = "crate::text::ek")]
        index_dtype: ElementKind,
    },
}

impl WriteIndex {
    /// `true` for [`WriteIndex::Direct`] — the back-compat default that writes at
    /// the iteration coordinate (byte-identical address math).
    #[must_use]
    pub fn is_direct(&self) -> bool {
        matches!(self, WriteIndex::Direct)
    }

    /// The [`WriteIndex::ScatterIndexed`] fields `(index_operand, axis, combine,
    /// oob, index_dtype)`, or `None` for [`WriteIndex::Direct`]. The single
    /// accessor the emitter + its backstop read so they stay in lockstep.
    #[must_use]
    pub fn scatter(&self) -> Option<(u8, u8, WriteCombine, OobPolicy, ElementKind)> {
        match self {
            WriteIndex::ScatterIndexed {
                index_operand,
                axis,
                combine,
                oob,
                index_dtype,
            } => Some((*index_operand, *axis, *combine, *oob, *index_dtype)),
            WriteIndex::Direct => None,
        }
    }
}

/// An op definition — the **algorithm** half of the algorithm/schedule split.
///
/// Names the op, its input-operand count, the output expression, the accepted
/// dtypes, and the access pattern. The generator fans one `OpDef` out across
/// many [`baracuda_kernels_types::StructureKey`] cells (the schedule half).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpDef {
    /// Stable op name — used in generated symbol names and the FKC contract.
    pub name: String,
    /// Number of input operands the body references.
    pub n_inputs: u8,
    /// Output `= body` evaluated at each coordinate.
    pub body: ScalarExpr,
    /// Dtypes this op accepts.
    #[serde(with = "crate::text::ek_vec")]
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
    #[serde(with = "crate::text::ek_opt")]
    pub out_dtype: Option<ElementKind>,
    /// Additional output bodies for a **multi-output** elementwise op (increment
    /// 1). Output 0 is [`OpDef::body`]; each entry here is one further output,
    /// evaluated at the same coordinate over the same inputs. **Empty for every
    /// single-output op** (`n_outputs() == 1`), which is every op built by every
    /// pre-increment-1 constructor — so `body` stays output 0 and every existing
    /// body-walker (`params_used`/`count_flops`/dtype plumbing/`derive_pattern`)
    /// operates on it unchanged, and emission is **byte-identical**. Non-empty
    /// only via [`OpDef::elementwise_multi`]; `Access::Elementwise` only in v1
    /// (validated at `plan::build_plan`, with an emitter backstop in `cuda`).
    ///
    /// The value proposition is **cross-body CSE**: all output bodies are
    /// interned into ONE [`ExprDag`] ([`ExprDag::from_exprs`]) so a subexpression
    /// shared between outputs (the `dy` load, an interior product) becomes one
    /// hoisted `tmp` referenced by multiple stores — strictly fewer global loads
    /// than decomposing into N single-output kernels.
    pub extra_out_bodies: Vec<ScalarExpr>,
    /// Per-EXTRA-output element dtype for a **hetero multi-output** op (the
    /// dropout-class increment). `extra_out_dtypes[j]` is output `(j+1)`'s dtype;
    /// `None` ⇒ that output is uniform (`== key dtype`). Output 0's hetero dtype
    /// stays on [`Self::out_dtype`] (unchanged). **EMPTY for every single-output
    /// op and every UNIFORM multi-output op** — so emission is byte-identical
    /// (`extra_out_dtypes.is_empty()` ⇒ `out_dtype_of(j)` resolves to the key
    /// dtype for every extra output, exactly the pre-increment behaviour). Length,
    /// when non-empty, **MUST equal `extra_out_bodies.len()`**. The legal `Some(_)`
    /// set is exactly the single-output [`Self::out_dtype`] set applied per-output:
    /// `Some(U8)` with a `Cmp*`-root body (the FKC "comparison → U8 mask"
    /// convention). This authored legality is validated at
    /// `plan::assert_valid_multi_output` (G1) and emitter-backstopped in
    /// `cuda::assert_multi_output_lowerable` (G5). Note the store is driven by this
    /// AUTHORED dtype (`out_dtype_of`), so the gate and the emitter read one source
    /// — no key-side cross-check is possible or needed: `OperandKey` carries no
    /// per-operand dtype, so author↔caller output-dtype agreement is an honest
    /// CALLER PRECONDITION (like buffer aliasing / exact extents), documented at
    /// `plan::assert_valid_multi_output`. Set only via
    /// [`OpDef::elementwise_multi_hetero`].
    #[serde(with = "crate::text::ek_opt_vec")]
    pub extra_out_dtypes: Vec<Option<ElementKind>>,
    /// Per-input **data-dependent read role** (index `i` ↔ `Input(i)`; increment
    /// 4, GATHER). Empty ⇒ every input is [`ReadIndex::Direct`] (back-compat:
    /// every pre-increment-4 op is index-free, so address math + the whole
    /// `views`-orthogonal read stays byte-identical). When non-empty, length
    /// **must** equal `n_inputs`. Set via [`OpDef::with_indexed`] (or the
    /// [`OpDef::gather`]/[`OpDef::index_select`]/[`OpDef::embedding`] convenience
    /// constructors). Only a [`ReadIndex::Indexed`] entry changes emission (the
    /// axis-substitution in the strided offset); validated at the TOP of
    /// `plan::build_plan` (`assert_valid_gather`) with an independent emitter
    /// backstop in [`crate::cuda::Cuda::lower`].
    pub read_index: Vec<ReadIndex>,
    /// The output's **data-dependent write role** (increment 5, SCATTER) — the
    /// write-side mirror of [`Self::read_index`]. [`WriteIndex::Direct`] (the
    /// default for every pre-increment-5 constructor) ⇒ the output is written at
    /// the iteration coordinate, byte-identical. A [`WriteIndex::ScatterIndexed`]
    /// role substitutes a runtime index value for one OUTPUT-axis coordinate and
    /// turns the store into a [`WriteCombine`] op; set via [`OpDef::with_scatter`]
    /// (or the [`OpDef::scatter`]/[`OpDef::scatter_add`]/[`OpDef::index_add`]/
    /// [`OpDef::bincount`] convenience constructors). Validated at the TOP of
    /// `plan::build_plan` (`assert_valid_scatter`) with an independent emitter
    /// backstop in [`crate::cuda::Cuda::lower`].
    pub write_index: WriteIndex,
    /// Per-input **runtime base element offset** (BASE_OFFSET SLICE) — index `i` ↔
    /// `Input(i)`, the parallel-Vec mirror of [`Self::views`]/[`Self::read_index`].
    /// **Empty ⇒ every input is [`BaseOffset::Zero`]** (byte-identical back-compat:
    /// every pre-increment op is offset-free, so address math stays byte-identical).
    /// When non-empty, length **must** equal `n_inputs` (same rule as `views`/
    /// `read_index`). A [`BaseOffset::Runtime`] entry adds a `long long off{i}`
    /// launch arg bumped onto the operand's base pointer at kernel entry (element
    /// units, applied before all per-element address math). Set via
    /// [`OpDef::with_base_offsets`]. A non-empty all-`Zero` vec is semantically
    /// identical to empty — presence is `any(Runtime)` (`plan::op_has_offset`).
    /// Validated at the TOP of `plan::build_plan` (`assert_valid_offsets`) with an
    /// independent emitter backstop in [`crate::cuda::assert_offsets_lowerable`].
    pub base_offsets: Vec<BaseOffset>,
    /// The **single output's** runtime base element offset — the output-side mirror
    /// of [`Self::write_index`] (default [`BaseOffset::Zero`], byte-identical). A
    /// [`BaseOffset::Runtime`] output adds a `long long offo` launch arg bumped onto
    /// the output base pointer at kernel entry (element units).
    pub out_base_offset: BaseOffset,
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
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
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
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: Some(ElementKind::U8),
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// Build a **multi-output** elementwise op (increment 1): one kernel that
    /// writes `bodies.len()` outputs from a shared body-DAG. `bodies[0]` is
    /// output 0 ([`OpDef::body`]); `bodies[1..]` become [`OpDef::extra_out_bodies`].
    /// Every body is evaluated at the same coordinate over the same `n_inputs`
    /// inputs, and all outputs share the iteration shape (elementwise-map).
    ///
    /// This clears the elementwise BACKWARD surface: `mul_backward` computes
    /// `da = dy·b` AND `db = dy·a` in one pass, and the shared subexpressions (the
    /// `dy` load, an interior product) CSE into one hoisted `tmp` referenced by
    /// both stores (via [`ExprDag::from_exprs`]) — the whole value proposition
    /// over decomposing into N single-output kernels.
    ///
    /// v1 scope (validated at `plan::build_plan`, emitter-backstopped in `cuda`):
    /// `Access::Elementwise` only; **uniform dtype across all outputs**
    /// (`out_dtype`/`extra_out_dtypes` all `None` — for a per-output hetero dtype,
    /// e.g. a U8 keep-mask beside an F32 value, use
    /// [`OpDef::elementwise_multi_hetero`]); each body may read
    /// `Input`/`Const`/`Param` but NOT `Reduced`/`Coord` (those are other access
    /// spaces); outputs must not be broadcast/flipped and must not alias inputs
    /// (in-place is deferred). `1 ≤ n_outputs` and `n_inputs + n_outputs ≤
    /// MAX_OPERANDS`.
    ///
    /// # Panics
    /// If `bodies` is empty (a multi-output op needs at least one output).
    #[must_use]
    pub fn elementwise_multi(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        bodies: Vec<Expr>,
    ) -> Self {
        // Delegate to the hetero constructor with an all-`None` (uniform) dtype
        // per output — byte-identical to the pre-hetero behaviour (`out_dtype` and
        // `extra_out_dtypes` both stay empty/`None`, so `out_dtype_of(j)` resolves
        // to the key dtype for every output and every store is unchanged).
        Self::elementwise_multi_hetero(
            name,
            n_inputs,
            dtypes,
            bodies.into_iter().map(|b| (b, None)).collect(),
        )
    }

    /// Build a **hetero multi-output** elementwise op (the dropout-class
    /// increment): one kernel that writes `bodies.len()` outputs, each through its
    /// **own** element dtype, from a shared body-DAG. `bodies[j]` is `(body_j,
    /// dtype_j)`: `bodies[0]`'s body is output 0 ([`OpDef::body`]) and its dtype
    /// (when `Some`) is [`OpDef::out_dtype`]; `bodies[1..]`'s bodies become
    /// [`OpDef::extra_out_bodies`] and their dtypes [`OpDef::extra_out_dtypes`]. A
    /// `None` dtype means that output is uniform (`== key dtype`) — passing all
    /// `None` reproduces [`OpDef::elementwise_multi`] exactly (`out_dtype` `None`,
    /// `extra_out_dtypes` empty), which is the byte-identical uniform path.
    ///
    /// The v1 vehicle is `dropout_fw`: output 0 = the F32 value
    /// `Input(0) * Select(Input(1) < keep_prob, scale, 0.0)` (uniform), output 1 =
    /// the U8 keep-mask `Input(1) < keep_prob` (a shared `Cmp*` node, hoisted once
    /// by cross-body CSE). The per-output store conversion (`Cmp*` `0.0/1.0` → U8,
    /// exact) is applied at the STORE SITE, never baked into the shared DAG node,
    /// so output 0 still consumes the compute-dtype comparison inside its `Select`.
    ///
    /// v1 hetero scope (authored legality validated at
    /// `plan::assert_valid_multi_output`, G1, emitter-backstopped in
    /// `cuda::assert_multi_output_lowerable`, G5 — no key-side cross-check, since
    /// `OperandKey` carries no per-operand dtype; see the caller-precondition note
    /// there): the only legal per-output hetero dtype is `Some(U8)`, and only when
    /// that output's body ROOT is a `Cmp*` predicate (so the U8 store is the exact
    /// `0.0/1.0` mask). Every other per-output legality/shape rule is inherited from
    /// [`OpDef::elementwise_multi`] (`Access::Elementwise`, no `Reduced`/`Coord`,
    /// non-broadcast/-flipped outputs, `n_inputs + n_outputs ≤ MAX_OPERANDS`).
    ///
    /// `extra_out_dtypes` is stored **empty when every extra output is uniform**
    /// (all `bodies[1..]` dtypes are `None`) so the uniform emission stays
    /// byte-identical; it is materialized (length `== extra_out_bodies.len()`) only
    /// when at least one extra output is hetero.
    ///
    /// # Panics
    /// If `bodies` is empty (a multi-output op needs at least one output).
    #[must_use]
    pub fn elementwise_multi_hetero(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        bodies: Vec<(Expr, Option<ElementKind>)>,
    ) -> Self {
        assert!(
            !bodies.is_empty(),
            "OpDef::elementwise_multi_hetero: needs at least one output body"
        );
        let mut it = bodies.into_iter();
        let (body0, out_dtype) = it.next().expect("non-empty checked above");
        let mut extra_out_bodies: Vec<ScalarExpr> = Vec::new();
        let mut extra_out_dtypes: Vec<Option<ElementKind>> = Vec::new();
        for (b, d) in it {
            extra_out_bodies.push(b.0);
            extra_out_dtypes.push(d);
        }
        // Keep `extra_out_dtypes` EMPTY when every extra output is uniform — the
        // byte-identical uniform-multi path (`out_dtype_of` reads the key dtype).
        if extra_out_dtypes.iter().all(Option::is_none) {
            extra_out_dtypes.clear();
        }
        Self {
            name: name.to_string(),
            n_inputs,
            body: body0.0,
            dtypes: dtypes.to_vec(),
            access: Access::Elementwise,
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype,
            extra_out_bodies,
            extra_out_dtypes,
        }
    }

    /// Number of outputs this op writes: `1 + extra_out_bodies.len()`. `1` for
    /// every single-output op (every pre-increment-1 constructor).
    #[must_use]
    pub fn n_outputs(&self) -> u8 {
        // `+ 1` for `body` (output 0). A catalog author who overflows `u8` with
        // output bodies has bigger problems; `MAX_OPERANDS` (8) gates it long first.
        1 + u8::try_from(self.extra_out_bodies.len()).expect("extra_out_bodies exceeds u8")
    }

    /// All output bodies in order: `body` (output 0) then `extra_out_bodies`. The
    /// canonical iterator for the multi-output emitters (interned together into
    /// one [`ExprDag`]) and the plan/backstop walks (gate every output body).
    #[must_use]
    pub fn output_bodies(&self) -> Vec<&ScalarExpr> {
        std::iter::once(&self.body)
            .chain(self.extra_out_bodies.iter())
            .collect()
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
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
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
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
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
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
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
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// Build a **prefix scan** op (increment 6): a cumsum/cumprod/cummax/cummin
    /// along `axis` with monoid `op`. `pre` is the per-element pre-map applied
    /// before the fold (identity: `input(0)`); `post` is the per-element epilogue
    /// over the running prefix, which it references as `reduced(0)` (identity:
    /// `reduced(0)`). `body == post`, mirroring [`OpDef::row_reduce`]/
    /// [`OpDef::contraction`], so every body-walker (params/flops/ulp/DAG/
    /// `derive_pattern`) operates on the primary output expr unchanged. v1 rejects
    /// `Mean` (not a monoid) and asserts `axis == rank - 1` at `plan::validate_scan`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn scan(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        op: ReduceOp,
        axis: u8,
        reverse: bool,
        exclusive: bool,
        pre: Expr,
        post: Expr,
    ) -> Self {
        Self {
            name: name.to_string(),
            n_inputs,
            body: post.0.clone(),
            dtypes: dtypes.to_vec(),
            access: Access::Scan {
                op,
                axis,
                reverse,
                exclusive,
                pre: pre.0,
                post: post.0,
            },
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// The common no-pre/no-post [`OpDef::scan`]: `pre = input(0)`, `post =
    /// reduced(0)` (both identities — the plain cumulative op).
    #[must_use]
    pub fn scan_simple(
        name: &str,
        dtypes: &[ElementKind],
        op: ReduceOp,
        axis: u8,
        reverse: bool,
        exclusive: bool,
    ) -> Self {
        Self::scan(
            name,
            1,
            dtypes,
            op,
            axis,
            reverse,
            exclusive,
            Expr(ScalarExpr::Input(0)),
            Expr(ScalarExpr::Reduced(0)),
        )
    }

    /// Build a **sliding-window reduction** op (increment 7): a max_pool / avg_pool
    /// / sum_pool / min_pool along `axis` with `size`/`stride`/`dilation`/`pad_lo`/
    /// `pad_hi` and a `count_include_pad` divisor policy (Mean only). `pre` is the
    /// per-tap pre-map applied before the fold (identity: `input(0)`); `post` is the
    /// per-output epilogue over the finalized window result, referenced as
    /// `reduced(0)` (identity: `reduced(0)`). `body == post`, mirroring
    /// [`OpDef::scan`]/[`OpDef::row_reduce`], so every body-walker (params/flops/ulp/
    /// DAG/`derive_pattern`) operates on the primary output expr unchanged. v1
    /// asserts `axis == rank - 1` and the window-param legality at
    /// `plan::validate_window`. See [`Access::Window`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn window(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        op: ReduceOp,
        axis: u8,
        size: u8,
        stride: u8,
        dilation: u8,
        pad_lo: u8,
        pad_hi: u8,
        count_include_pad: bool,
        pre: Expr,
        post: Expr,
    ) -> Self {
        Self {
            name: name.to_string(),
            n_inputs,
            body: post.0.clone(),
            dtypes: dtypes.to_vec(),
            access: Access::Window {
                op,
                axis,
                size,
                stride,
                dilation,
                pad_lo,
                pad_hi,
                count_include_pad,
                pre: pre.0,
                post: post.0,
            },
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// The common no-pre/no-post single-input [`OpDef::window`]: `pre = input(0)`,
    /// `post = reduced(0)` (the plain pool). `count_include_pad` picks the avg_pool
    /// divisor policy (ignored for Max/Min/Sum).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn window_simple(
        name: &str,
        dtypes: &[ElementKind],
        op: ReduceOp,
        axis: u8,
        size: u8,
        stride: u8,
        dilation: u8,
        pad_lo: u8,
        pad_hi: u8,
        count_include_pad: bool,
    ) -> Self {
        Self::window(
            name,
            1,
            dtypes,
            op,
            axis,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            count_include_pad,
            Expr(ScalarExpr::Input(0)),
            Expr(ScalarExpr::Reduced(0)),
        )
    }

    /// Build a **row sort** op (increment 8): a values-output sort along the
    /// innermost axis with direction `order`. The output row is a raw-bit
    /// permutation of the input row (dtype-preserving; NaN payloads and `-0.0`
    /// signs preserved). v1 always emits the STABLE pair-sort (`stable: true`
    /// hardwired), so ties keep input order and NaN orders greatest (PyTorch:
    /// asc ⇒ NaN last, desc ⇒ NaN first). `body == Input(0)`, mirroring
    /// [`OpDef::scan`]/[`OpDef::window`], so every body-walker (params/flops/ulp/
    /// DAG/`derive_pattern`) operates on a well-formed expr unchanged. Validated
    /// at `plan::validate_row_sort`. See [`Access::RowSort`].
    #[must_use]
    pub fn row_sort(name: &str, dtype: ElementKind, order: SortOrder) -> Self {
        Self {
            name: name.to_string(),
            n_inputs: 1,
            body: ScalarExpr::Input(0),
            dtypes: vec![dtype],
            access: Access::RowSort {
                order,
                stable: true,
                out: SortOut::Values,
                limit: SortLimit::Full,
            },
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// Build a **row argsort** op (increment 8): a single `I32` index output — the
    /// sort permutation of the input row along the innermost axis. Rides the
    /// single-output hetero `out_dtype` precedent (`out_dtype = Some(I32)`, the
    /// bincount/0b shape); the bespoke layer also pegs indices to `i32`. Otherwise
    /// identical to [`OpDef::row_sort`] (same total order ⇒ mutually consistent
    /// with the values-sort of the same `order`). v1 always emits the STABLE
    /// pair-sort. Validated at `plan::validate_row_sort`. See [`Access::RowSort`].
    ///
    /// **Caller precondition:** `k <= 2^31 - 1` — the `I32` index output cannot
    /// represent a position past it (a runtime launch fact; the structure key
    /// carries no extents). The VALUES-sort has no such cap: its `long long` tie
    /// index (review-hardened) is exact at every addressable `k`.
    #[must_use]
    pub fn row_argsort(name: &str, dtype: ElementKind, order: SortOrder) -> Self {
        Self {
            name: name.to_string(),
            n_inputs: 1,
            body: ScalarExpr::Input(0),
            dtypes: vec![dtype],
            access: Access::RowSort {
                order,
                stable: true,
                out: SortOut::Indices,
                limit: SortLimit::Full,
            },
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: Some(ElementKind::I32),
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// Build a **fused row sort + argsort** op (increment 9 FUSED_ARGSORT): ONE
    /// kernel writes BOTH the values permutation (to `out_val`, dtype-preserving
    /// raw bits) AND the `I32` index permutation (to `out_idx`) in a single launch
    /// — bespoke's native one-kernel shape (`sort_block_kernel` writes `y_vals`
    /// AND `y_idx` together), eliminating the double-sort of launching
    /// [`OpDef::row_sort`] and [`OpDef::row_argsort`] separately. The two projections
    /// share the identical total order, so they are mutually consistent by
    /// construction (`memcmp`-exact vs the two single-output kernels).
    ///
    /// `out_dtype = None` (output 0 = values, dtype-preserving; the I32 index
    /// output is emitter-hardwired off the entry-point symbol, NOT a per-operand
    /// dtype channel — the same `(int)i` / `int* out` precedent as `row_argsort`).
    /// The structure key must carry THREE operands `[in0, out_val, out_idx]`.
    ///
    /// **Caller precondition:** `k <= 2^31 - 1` — like [`OpDef::row_argsort`] (not
    /// like the values-only [`OpDef::row_sort`], which has no cap): the `I32`
    /// `out_idx` cannot represent a position past it. A runtime launch fact (the
    /// structure key carries no extents). Validated at `plan::validate_row_sort`.
    /// See [`Access::RowSort`] and [`SortOut::Both`].
    #[must_use]
    pub fn row_sort_indices(name: &str, dtype: ElementKind, order: SortOrder) -> Self {
        Self {
            name: name.to_string(),
            n_inputs: 1,
            body: ScalarExpr::Input(0),
            dtypes: vec![dtype],
            access: Access::RowSort {
                order,
                stable: true,
                out: SortOut::Both,
                limit: SortLimit::Full,
            },
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// Build a **2-D im2col / unfold** op (increment 11): the pure expanding
    /// structured gather that lowers `Conv2d`. A single rank-4 NCHW input
    /// `[N,C,H_in,W_in]` unfolds into the column matrix `[N, C*kh*kw, oH*oW]`
    /// (Layout A — channel-major then tap, spatial row-major), the exact bespoke
    /// `im2col_2d` + PyTorch `F.unfold` order. `body == Input(0)` (a RAW-BIT copy),
    /// mirroring [`OpDef::window_simple`]/[`OpDef::row_sort`], so every body-walker
    /// (params/flops/ulp/DAG/`derive_pattern`) operates on a well-formed expr
    /// unchanged. The conv geometry `(kernel, stride, pad, dilation)` rides the
    /// [`Access::Im2Col`] node as compile-time literals; the six runtime extents
    /// `(N,C,H_in,W_in,oH,oW)` ride `long long` launch args. `out_dtype = None` (a
    /// pure gather preserves dtype). Validated at `plan::validate_im2col`. See
    /// [`Access::Im2Col`].
    #[must_use]
    pub fn im2col_2d(
        name: &str,
        dtype: ElementKind,
        kernel: (u8, u8),
        stride: (u8, u8),
        pad: (u8, u8),
        dilation: (u8, u8),
    ) -> Self {
        Self {
            name: name.to_string(),
            n_inputs: 1,
            body: ScalarExpr::Input(0),
            dtypes: vec![dtype],
            access: Access::Im2Col {
                kernel,
                stride,
                pad,
                dilation,
            },
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// Build a **row top-k** op (increment 10 TOPK/BOTTOMK): the fused two-output
    /// `(values, indices)` of [`OpDef::row_sort_indices`] with its writeback CAPPED
    /// to a runtime top-`k_out` — the `k_out` largest elements of each innermost
    /// row, descending-sorted (torch.topk `largest=True`, `sorted=True`), NaN-first
    /// (kernelgen's NaN-greatest convention). Output extent `[batch, k_out]`;
    /// `k_out` is the OUT operand's inner extent, resolved at plan/launch as a
    /// `long long` launch arg (NOT a ctor arg — exactly like `k` for `row_sort`).
    ///
    /// TopK is the strict generalization of `row_sort_indices` — the current code
    /// is exactly its `k_out == k_in` special case (`Full`). It reuses verbatim the
    /// same `pair_lt` total order, both validated RowSort emitter paths (rank base +
    /// bitonic), the raw-bit value writeback, and the `I32` index. The single row
    /// length splits into `k_in`/`k_out`; the store is guarded `if (r < k_out)`.
    ///
    /// **Caller / on-device precondition:** `k_out <= k_in` — a `k_out > k_in`
    /// would leave slots `[k_in, k_out)` unwritten (ranks only reach `k_in - 1`).
    /// The structure key carries no numeric extents, so this is a runtime launch
    /// fact (on-device-validated by `initcheck`), NOT a plan assert — the same trust
    /// tier as the bitonic `k <= 1024`. Inherits argsort's `k_in <= 2^31-1` cap (the
    /// `I32` index output). Validated at `plan::validate_row_sort`. See
    /// [`Access::RowSort`], [`SortOut::Both`], and [`SortLimit::TopK`].
    #[must_use]
    pub fn row_topk(name: &str, dtype: ElementKind) -> Self {
        Self {
            name: name.to_string(),
            n_inputs: 1,
            body: ScalarExpr::Input(0),
            dtypes: vec![dtype],
            access: Access::RowSort {
                order: SortOrder::Desc,
                stable: true,
                out: SortOut::Both,
                limit: SortLimit::TopK,
            },
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
        }
    }

    /// Build a **row bottom-k** op (increment 10 TOPK/BOTTOMK): [`OpDef::row_topk`]
    /// with ascending direction — the `k_out` SMALLEST elements of each innermost
    /// row, ascending-sorted (torch.topk `largest=False`, `sorted=True`), NaN-last
    /// (kernelgen's NaN-greatest convention: NaN is excluded until the row holds
    /// fewer than `k_out` non-NaN values). Only `order` differs from `row_topk`
    /// (`Asc` vs `Desc`) — the cap, the fused two-output shape, and every
    /// precondition (`k_out <= k_in`, `k_in <= 2^31-1`) are identical. See
    /// [`OpDef::row_topk`] and [`SortLimit::TopK`].
    #[must_use]
    pub fn row_bottomk(name: &str, dtype: ElementKind) -> Self {
        Self {
            name: name.to_string(),
            n_inputs: 1,
            body: ScalarExpr::Input(0),
            dtypes: vec![dtype],
            access: Access::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Both,
                limit: SortLimit::TopK,
            },
            views: Vec::new(),
            read_index: Vec::new(),
            write_index: WriteIndex::Direct,
            base_offsets: Vec::new(),
            out_base_offset: BaseOffset::Zero,
            out_dtype: None,
            extra_out_bodies: Vec::new(),
            extra_out_dtypes: Vec::new(),
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

    /// Attach per-input **data-dependent read roles** ([`ReadIndex`], increment
    /// 4). `read_index[i]` applies to `Input(i)`; `read_index.len()` must equal
    /// `n_inputs`. An index-free op (the common case) never calls this and keeps
    /// `read_index` empty. The debug assert catches a generator bug at
    /// catalog-build time; the full v1 rule set (index dtype integer, axis in
    /// range, one gathered input, gather ⊥ view, …) is enforced at plan time by
    /// `assert_valid_gather` once the iteration rank + operand keys are known.
    #[must_use]
    pub fn with_indexed(mut self, read_index: Vec<ReadIndex>) -> Self {
        debug_assert_eq!(
            read_index.len(),
            self.n_inputs as usize,
            "OpDef::with_indexed: read_index.len() must equal n_inputs"
        );
        self.read_index = read_index;
        self
    }

    /// Attach per-input **runtime base element offsets** ([`BaseOffset`], the
    /// BASE_OFFSET SLICE increment). `base_offsets[i]` applies to `Input(i)`;
    /// `base_offsets.len()` must equal `n_inputs`. `out_base_offset` is the single
    /// output's offset. An offset-free op (the common case) never calls this and
    /// keeps `base_offsets` empty + `out_base_offset == Zero` (byte-identical). The
    /// debug assert catches a generator bug at catalog-build time; the full v1 rule
    /// set (Elementwise-only, single-output, force-Strided) is enforced at plan
    /// time by `assert_valid_offsets` and backstopped in `cuda`.
    #[must_use]
    pub fn with_base_offsets(
        mut self,
        base_offsets: Vec<BaseOffset>,
        out_base_offset: BaseOffset,
    ) -> Self {
        debug_assert_eq!(
            base_offsets.len(),
            self.n_inputs as usize,
            "OpDef::with_base_offsets: base_offsets.len() must equal n_inputs"
        );
        self.base_offsets = base_offsets;
        self.out_base_offset = out_base_offset;
        self
    }

    /// Build a **gather** op — `out[coord] = data[coord with `axis` replaced by
    /// index[coord]]` (torch `gather` along `axis`). Two inputs: `Input(0)` is the
    /// gathered `data`, `Input(1)` is the integer `index` tensor (dtype
    /// `index_dtype`, `I32`/`I64`); the body is the identity copy `Input(0)`, read
    /// through the [`ReadIndex::Indexed`] role. The output shape equals the index
    /// shape (a **full-shape** index — the caller keys `Input(1)` dense on every
    /// axis). `oob` picks the out-of-range semantics; the bespoke gather is
    /// [`OobPolicy::Skip`] (silently skips OOB / negative indices), so pass `Skip`
    /// to match it exactly.
    #[must_use]
    pub fn gather(
        name: &str,
        dtypes: &[ElementKind],
        axis: u8,
        oob: OobPolicy,
        index_dtype: ElementKind,
    ) -> Self {
        Self::elementwise(name, 2, dtypes, Expr(ScalarExpr::Input(0))).with_indexed(vec![
            ReadIndex::Indexed {
                index_operand: 1,
                axis,
                oob,
                index_dtype,
            },
            ReadIndex::Direct,
        ])
    }

    /// Build an **index_select** op — `out[..., j, ...] = data[..., idx[j], ...]`
    /// along `axis`, where `idx` is a **1-D** index of length
    /// `out.shape[axis]`. Structurally identical to [`OpDef::gather`] (same
    /// `Input(0)` copy through an `Indexed` role); the ONLY difference is that the
    /// caller keys `Input(1)` as a 1-D index that **broadcasts on every axis
    /// except `axis`** — so the emitted index offset degenerates to
    /// `c{axis}·stride`, exactly the bespoke `index_select` 1-D lookup. Bespoke is
    /// [`OobPolicy::Skip`].
    #[must_use]
    pub fn index_select(
        name: &str,
        dtypes: &[ElementKind],
        axis: u8,
        oob: OobPolicy,
        index_dtype: ElementKind,
    ) -> Self {
        Self::gather(name, dtypes, axis, oob, index_dtype)
    }

    /// Build an **embedding** op — `out[n, :] = weight[ids[n], :]` (a row gather
    /// on `axis 0` of the `[V, D]` weight, `ids` broadcast over the feature axis).
    /// Bespoke embedding zeros the output row on an OOB / negative index, so the
    /// OOB policy is [`OobPolicy::ZeroFill`]. (The bespoke `padding_idx` — zero
    /// the row where `ids[n] == padding_idx` — is a per-op runtime scalar
    /// predicate NOT modeled here in v1; pass a disabled padding_idx, i.e. the
    /// `INT32_MIN` sentinel, to the bespoke oracle so only the OOB path is
    /// exercised. See the deliverable note.)
    #[must_use]
    pub fn embedding(name: &str, dtypes: &[ElementKind], index_dtype: ElementKind) -> Self {
        Self::gather(name, dtypes, 0, OobPolicy::ZeroFill, index_dtype)
    }

    /// Attach the output's **data-dependent write role** ([`WriteIndex`],
    /// increment 5). A non-scatter op (the common case) never calls this and keeps
    /// [`WriteIndex::Direct`]. The full v1 rule set (index dtype integer, axis in
    /// range, combine legal for the output dtype, scatter ⊥ view/gather, …) is
    /// enforced at plan time by `assert_valid_scatter` once the iteration rank +
    /// operand keys are known.
    #[must_use]
    pub fn with_scatter(mut self, write_index: WriteIndex) -> Self {
        self.write_index = write_index;
        self
    }

    /// Build a **scatter** op (pure assign) — `out[..., index[..., j, ...], ...] =
    /// updates[..., j, ...]` along `axis` (torch `scatter`). Two inputs: `Input(0)`
    /// is the `updates` value, `Input(1)` is the integer `index`; the body is the
    /// identity copy `Input(0)`, written through the [`WriteIndex::ScatterIndexed`]
    /// role with [`WriteCombine::Assign`]. The iteration domain is the updates
    /// shape (a **full-shape** index — the caller keys `Input(1)` dense on every
    /// axis). **Determinism precondition:** unique target indices — on a duplicate
    /// target the store races (last-writer-wins), exactly as bespoke
    /// `indexing/scatter.cu` documents. Bespoke skips an OOB/negative target
    /// ([`OobPolicy::Skip`]).
    #[must_use]
    pub fn scatter(name: &str, dtypes: &[ElementKind], axis: u8, index_dtype: ElementKind) -> Self {
        Self::elementwise(name, 2, dtypes, Expr(ScalarExpr::Input(0))).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis,
                combine: WriteCombine::Assign,
                oob: OobPolicy::Skip,
                index_dtype,
            },
        )
    }

    /// Build a **scatter_add** op — `out[..., index[..., j, ...], ...] +=
    /// updates[..., j, ...]` along `axis` ([`WriteCombine::AtomicAdd`], dup-safe
    /// accumulation). Structurally identical to [`OpDef::scatter`] but the store is
    /// an `atomicAdd`. **Determinism** depends on the value dtype: an INTEGER
    /// output accumulates order-independently (deterministic, ships
    /// unconditionally); a FLOATING output is run-to-run non-deterministic (ships
    /// as the gated variant, with the gather-sum default as the base — see
    /// [`crate::cuda`]). Bespoke `indexing/index_add.cu`
    /// (`scatter_add`)/[`OobPolicy::Skip`].
    #[must_use]
    pub fn scatter_add(
        name: &str,
        dtypes: &[ElementKind],
        axis: u8,
        index_dtype: ElementKind,
    ) -> Self {
        Self::elementwise(name, 2, dtypes, Expr(ScalarExpr::Input(0))).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis,
                combine: WriteCombine::AtomicAdd,
                oob: OobPolicy::Skip,
                index_dtype,
            },
        )
    }

    /// Build an **index_add** op — `dst[..., idx[j], ...] += src[..., j, ...]`
    /// along `axis`, where `idx` is a **1-D** index of length `src.shape[axis]`.
    /// Structurally identical to [`OpDef::scatter_add`] (same `Input(0)` copy
    /// through an `AtomicAdd` scatter); the ONLY difference is the caller keys
    /// `Input(1)` as a 1-D index that **broadcasts on every axis except `axis`** —
    /// so the emitted index offset degenerates to `c{axis}·stride`, exactly the
    /// bespoke `index_add` 1-D lookup. Same integer-vs-FP determinism split as
    /// `scatter_add`. Bespoke `indexing/index_add.cu`/[`OobPolicy::Skip`].
    #[must_use]
    pub fn index_add(
        name: &str,
        dtypes: &[ElementKind],
        axis: u8,
        index_dtype: ElementKind,
    ) -> Self {
        Self::scatter_add(name, dtypes, axis, index_dtype)
    }

    /// Build a **bincount** op — `out[x[i]] += 1` (the integer-count representative
    /// of the ATOMIC_HISTOGRAM family). ONE input `Input(0)` = the integer data
    /// `x` (dtype `index_dtype`, `I32`/`I64`), which is ALSO the index operand
    /// (a scatter's index selects the destination, so an input indexing itself is
    /// legal here); the body is the constant increment `1`, written through an
    /// [`WriteCombine::AtomicAdd`] scatter into the `I32` counts output (`axis 0`).
    /// INTEGER atomic-add ⇒ order-independent ⇒ **deterministic, ships
    /// unconditionally**. OOB (`x[i] < 0 || x[i] >= num_bins`) is skipped — bespoke
    /// `sort/histogram.cu` `bincount`. (Float `histogram` = an elementwise
    /// bin-index map — `floor((x-lo)·scale)` clamped, expressible today — composed
    /// with this bincount; the computed-bin scatter is a follow-up.)
    #[must_use]
    pub fn bincount(name: &str, index_dtype: ElementKind) -> Self {
        // Input 0 = x (integer); accepted key dtype is the index dtype. Output is
        // an I32 count (hetero out); the stored value is the constant `1`.
        Self {
            out_dtype: Some(ElementKind::I32),
            ..Self::elementwise(name, 1, &[index_dtype], Expr(ScalarExpr::Const(1.0)))
        }
        .with_scatter(WriteIndex::ScatterIndexed {
            index_operand: 0,
            axis: 0,
            combine: WriteCombine::AtomicAdd,
            oob: OobPolicy::Skip,
            index_dtype,
        })
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
    fn base_offset_default_is_zero() {
        assert_eq!(BaseOffset::default(), BaseOffset::Zero);
        assert!(BaseOffset::Zero.is_zero());
        assert!(!BaseOffset::Runtime.is_zero());
    }

    #[test]
    fn existing_constructors_are_offset_free() {
        // Back-compat (BASE_OFFSET SLICE): every current OpDef builds with empty
        // `base_offsets` + `out_base_offset == Zero` — byte-identical emission.
        let ew = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu());
        assert!(ew.base_offsets.is_empty());
        assert_eq!(ew.out_base_offset, BaseOffset::Zero);
        let red = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        assert!(red.base_offsets.is_empty());
        assert_eq!(red.out_base_offset, BaseOffset::Zero);
        let g = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I32,
        );
        assert!(g.base_offsets.is_empty());
        assert_eq!(g.out_base_offset, BaseOffset::Zero);
    }

    #[test]
    fn with_base_offsets_sets_per_operand_offsets() {
        // A two-input op with a Runtime input offset + a Runtime output offset —
        // the rope odd-lane shape.
        let op = OpDef::elementwise("rope_odd", 2, &[ElementKind::F32], input(0))
            .with_base_offsets(
                vec![BaseOffset::Runtime, BaseOffset::Zero],
                BaseOffset::Runtime,
            );
        assert_eq!(op.base_offsets, vec![BaseOffset::Runtime, BaseOffset::Zero]);
        assert_eq!(op.out_base_offset, BaseOffset::Runtime);
    }

    // cfg-gated: the debug_assert compiles OUT under --release — ungated,
    // `cargo test --release` would red on "test did not panic". The
    // release-safe G1 arity cover is `assert_valid_offsets`'s own assert_eq!
    // (plan gate), tested via the struct-literal bypass in cuda.rs.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "base_offsets.len() must equal n_inputs")]
    fn with_base_offsets_arity_mismatch_panics_in_debug() {
        // Debug-assert catches a generator bug at catalog-build time (len != n_inputs).
        let _ = OpDef::elementwise("bad", 2, &[ElementKind::F32], input(0))
            .with_base_offsets(vec![BaseOffset::Runtime], BaseOffset::Zero);
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
            vec![ReduceStage {
                pre: ScalarExpr::Input(0),
                op: ReduceOp::Mean,
            }],
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
        for op in [
            BinaryOp::LogicalAnd,
            BinaryOp::LogicalOr,
            BinaryOp::LogicalXor,
        ] {
            assert!(op.is_logical(), "{op:?}");
        }
        for op in [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
        ] {
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
        assert!(
            View::Permute {
                perm: vec![2, 0, 1]
            }
            .is_valid(3)
        );
        assert!(!View::Permute { perm: vec![0, 1] }.is_valid(3)); // wrong length
        assert!(!View::Permute { perm: vec![0, 0] }.is_valid(2)); // duplicate axis
        assert!(!View::Permute { perm: vec![0, 5] }.is_valid(2)); // out-of-range axis
        assert!(View::Identity.is_valid(4));
        assert!(
            View::Broadcast {
                bcast: AxisMask::EMPTY
            }
            .is_valid(4)
        );
        assert!(View::Reshape { producer_rank: 2 }.is_valid(3));
    }

    // ---- increment 4: gather constructor shapes ----

    #[test]
    fn gather_constructor_builds_the_indexed_read_role() {
        let op = OpDef::gather(
            "gather",
            &[ElementKind::F32],
            1,
            OobPolicy::Skip,
            ElementKind::I64,
        );
        assert_eq!(op.n_inputs, 2);
        assert_eq!(op.n_outputs(), 1);
        // Body is the identity copy of the gathered data operand.
        assert_eq!(op.body, ScalarExpr::Input(0));
        // read_index[0] is the Indexed role; input 1 (the index tensor) is Direct.
        assert_eq!(
            op.read_index[0],
            ReadIndex::Indexed {
                index_operand: 1,
                axis: 1,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I64,
            }
        );
        assert!(op.read_index[1].is_direct());
    }

    #[test]
    fn embedding_constructor_is_axis0_zerofill() {
        // embedding zeros the OOB / negative row (bespoke) — ZeroFill on axis 0.
        let op = OpDef::embedding("emb", &[ElementKind::F32], ElementKind::I32);
        match op.read_index[0] {
            ReadIndex::Indexed {
                axis,
                oob,
                index_dtype,
                ..
            } => {
                assert_eq!(axis, 0);
                assert_eq!(oob, OobPolicy::ZeroFill);
                assert_eq!(index_dtype, ElementKind::I32);
            }
            ReadIndex::Direct => panic!("embedding input 0 must be Indexed"),
        }
    }

    #[test]
    fn index_free_op_has_empty_read_index() {
        // Back-compat: every plain constructor leaves read_index empty.
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        assert!(op.read_index.is_empty());
    }

    // ---- increment 5: scatter constructor shapes ----

    #[test]
    fn scatter_constructor_is_assign() {
        let op = OpDef::scatter("scatter", &[ElementKind::F32], 1, ElementKind::I64);
        assert_eq!(op.n_inputs, 2);
        assert_eq!(op.n_outputs(), 1);
        // Body is the identity copy of the updates operand.
        assert_eq!(op.body, ScalarExpr::Input(0));
        assert_eq!(
            op.write_index,
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 1,
                combine: WriteCombine::Assign,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I64,
            }
        );
        // Scatter is a WRITE role — read_index stays empty (byte-identical reads).
        assert!(op.read_index.is_empty());
    }

    #[test]
    fn scatter_add_and_index_add_are_atomic_add() {
        let sa = OpDef::scatter_add("scatter_add", &[ElementKind::F32], 0, ElementKind::I32);
        let (_, _, sc, _, _) = sa.write_index.scatter().expect("scatter role");
        assert_eq!(sc, WriteCombine::AtomicAdd);
        // FP scatter_add is the non-deterministic case.
        assert!(sc.is_fp_atomic_add(ElementKind::F32));
        // Integer scatter_add accumulates order-independently ⇒ deterministic.
        assert!(!sc.is_fp_atomic_add(ElementKind::I32));
        // index_add is scatter_add with a 1-D index (same role).
        let ia = OpDef::index_add("index_add", &[ElementKind::F32], 0, ElementKind::I32);
        assert_eq!(ia.write_index, sa.write_index);
    }

    #[test]
    fn bincount_is_const1_atomic_add_into_i32() {
        let op = OpDef::bincount("bincount", ElementKind::I64);
        assert_eq!(op.n_inputs, 1);
        assert_eq!(op.body, ScalarExpr::Const(1.0));
        // Hetero out: i64 data key, i32 counts.
        assert_eq!(op.out_dtype, Some(ElementKind::I32));
        match op.write_index {
            WriteIndex::ScatterIndexed {
                index_operand,
                combine,
                index_dtype,
                ..
            } => {
                // The lone input indexes itself (a scatter's index selects the dst).
                assert_eq!(index_operand, 0);
                assert_eq!(combine, WriteCombine::AtomicAdd);
                assert_eq!(index_dtype, ElementKind::I64);
                // Integer counts ⇒ deterministic.
                assert!(!combine.is_fp_atomic_add(ElementKind::I32));
            }
            WriteIndex::Direct => panic!("bincount must be a scatter"),
        }
    }

    #[test]
    fn non_scatter_op_is_write_direct() {
        // Back-compat: every plain constructor leaves write_index Direct.
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        assert!(op.write_index.is_direct());
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
            Access::Reduction {
                op,
                axes,
                keepdim,
                post,
            } => {
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
            Access::Reduction {
                op,
                axes,
                keepdim,
                post,
            } => {
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
            Access::Reduction {
                op,
                axes,
                keepdim,
                post,
            } => {
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
        let hits: Vec<NodeId> = (0..dag.len() as NodeId)
            .filter(|&i| pred(dag.node(i)))
            .collect();
        assert_eq!(
            hits.len(),
            1,
            "expected exactly one matching node, got {hits:?}"
        );
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
        assert_eq!(
            dag.consumers(m),
            2,
            "the shared Mul feeds both Add operands"
        );
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
        assert_eq!(
            mixed.len(),
            3,
            "Reduced(0) and Reduced(1) are distinct leaves"
        );
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
        assert_eq!(
            kinds.len(),
            7,
            "same-index leaves of different kinds never merge"
        );
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
        assert_eq!(
            dag.len(),
            11,
            "one node per level, shared — linear in depth"
        );
    }

    #[test]
    fn from_expr_has_one_root_equal_to_root() {
        // Back-compat: the single-body constructor exposes exactly one root, and
        // it equals `root()`.
        let dag = ExprDag::from_expr(&mul(ipt(0), ipt(1)));
        assert_eq!(dag.roots(), &[dag.root()]);
        assert_eq!(dag.roots().len(), 1);
    }

    #[test]
    fn from_exprs_shares_the_dy_load_across_bodies() {
        // mul_backward: da = dy*b, db = dy*a. dy = Input(0) is loaded by BOTH
        // bodies; interned across bodies it is ONE node with consumers == 2 (the
        // cross-body CSE the emitter hoists to a single `tmp`). The two roots are
        // DISTINCT Mul nodes (dy*b != dy*a).
        let da = mul(ipt(0), ipt(2)); // dy*b
        let db = mul(ipt(0), ipt(1)); // dy*a
        let dag = ExprDag::from_exprs(&[&da, &db]);
        // Input0(dy), Input1(a), Input2(b), Mul(dy,b), Mul(dy,a) = 5 nodes.
        assert_eq!(dag.len(), 5, "dy interned once across both bodies");
        let dy = only(&dag, |n| matches!(n, DagNode::Input(0)));
        assert_eq!(dag.consumers(dy), 2, "dy feeds both output bodies");
        assert_eq!(dag.roots().len(), 2, "two output roots");
        assert_ne!(
            dag.roots()[0],
            dag.roots()[1],
            "da and db are distinct nodes"
        );
    }

    #[test]
    fn from_exprs_shares_an_interior_across_bodies() {
        // div_backward shape: da = dy/b; db = -((dy/b)*a/b). The dy/b interior is
        // the ROOT of body 0 AND an interior of body 1 — interned once. Its
        // consumer count reflects the interior reference (from body 1's Mul); the
        // emitter additionally treats a shared root as a use, so it hoists once.
        let dyb = ScalarExpr::Div(Box::new(ipt(0)), Box::new(ipt(2))); // dy/b
        let da = dyb.clone();
        let db = ScalarExpr::Unary(
            UnaryOp::Neg,
            Box::new(ScalarExpr::Div(
                Box::new(mul(dyb.clone(), ipt(1))),
                Box::new(ipt(2)),
            )),
        );
        let dag = ExprDag::from_exprs(&[&da, &db]);
        // Body 0's root is the dy/b Div; it is referenced by body 1's interior
        // (there are two Div nodes total — dy/b and the outer .../b — so we key
        // off root[0] directly rather than the ambiguous "the Div node").
        let dyb_node = dag.roots()[0];
        assert!(
            matches!(dag.node(dyb_node), DagNode::Div(..)),
            "body 0 root is dy/b"
        );
        assert!(
            dag.consumers(dyb_node) >= 1,
            "the shared dy/b interior is referenced by body 1"
        );
    }

    #[test]
    fn from_exprs_single_body_matches_from_expr() {
        // A one-element slice is identical to `from_expr` (single-output stays
        // byte-identical — the emitter's no-regression guarantee).
        let e = mul(ipt(0), ipt(1));
        let a = ExprDag::from_expr(&e);
        let b = ExprDag::from_exprs(&[&e]);
        assert_eq!(a.len(), b.len());
        assert_eq!(a.roots(), b.roots());
        assert_eq!(a.to_expr(), b.to_expr());
    }
}
