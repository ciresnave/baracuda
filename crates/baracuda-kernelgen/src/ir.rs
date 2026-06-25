//! The op **algorithm** IR — a small, backend-agnostic tensor expression.
//!
//! An op is the *pure function* computed at each output coordinate ([`OpDef`]),
//! described as a scalar-op DAG ([`ScalarExpr`]) over its input operands plus an
//! access pattern ([`Access`]). The emitter lowers this to a concrete backend
//! and *schedule* (chosen per [`baracuda_kernels_types::StructureKey`] cell).
//! Describing the math here — rather than as opaque CUDA — is what lets the
//! emitter vectorize, hoist, and fuse, because it can see the dataflow.

use baracuda_kernels_types::ElementKind;

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
    /// A non-infix binary op (`max`/`min`/`pow`/`rem`) — a backend function call.
    Binary(BinaryOp, Box<ScalarExpr>, Box<ScalarExpr>),
}

/// A unary math / activation op. Variant names line up with the FKC §4.1
/// graph-`Op` vocabulary, so [`crate::derive_pattern`] maps them by name.
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
}

/// A non-infix binary op — lowered as a backend **function call** (`fmaxf`,
/// `powf`), unlike the infix arithmetic [`ScalarExpr::Add`]/`Sub`/`Mul`/`Div`.
/// Variant names line up with the FKC §4.1 graph-`Op` vocabulary.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Elementwise maximum (commutative).
    Max,
    /// Elementwise minimum (commutative).
    Min,
    /// Power `aᵇ` (not commutative).
    Pow,
    /// Floored remainder — `a - floor(a/b)·b`, sign-of-divisor (`torch.remainder`,
    /// Fuel's `Op::Rem`; not commutative). Distinct from C `fmod` (sign-of-dividend).
    Rem,
}

/// The associative combine of an [`Access::Reduction`]. The identity is implied
/// (`Sum`/`Mean` → 0; `Max`/`Min` peel the first element, so no ±∞ literal — that
/// keeps the emitted source header-light under nvrtc).
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
    /// Reduce the **last (contiguous, trailing) axis** with `op`: each output
    /// element is `op` folded over that axis's run of `body` values. v1 covers
    /// the contiguous last-axis float-dtype case — the `MeanDim`/`SumDim` core of
    /// RmsNorm/Softmax. Strided inputs, arbitrary/multiple axes, keepdim layout,
    /// and integer accumulation are follow-ups.
    Reduction {
        /// The associative combine (+ implied identity).
        op: ReduceOp,
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
        }
    }

    /// Build a **last-axis reduction** op: `body` is the per-element pre-reduction
    /// expression (e.g. `input(0).unary(Sqr)` for a mean-of-squares), folded over
    /// the contiguous trailing axis by `op`. The output holds one element per
    /// outer coordinate. See [`Access::Reduction`] for the v1 scope.
    #[must_use]
    pub fn reduction(
        name: &str,
        n_inputs: u8,
        dtypes: &[ElementKind],
        body: Expr,
        op: ReduceOp,
    ) -> Self {
        Self {
            name: name.to_string(),
            n_inputs,
            body: body.0,
            dtypes: dtypes.to_vec(),
            access: Access::Reduction { op },
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
        }
    }
}
