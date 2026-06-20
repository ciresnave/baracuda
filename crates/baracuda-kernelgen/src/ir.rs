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
}

/// A unary math / activation op. Variant names line up with the FKC §4.1
/// graph-`Op` vocabulary, so [`crate::derive_pattern`] maps them by name.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    /// Exact (erf-based) GELU. (FKC §4.1 names `Gelu` vs `GeluErf` — the bare
    /// `Gelu` flavor mapping is pending Fuel review item E2.)
    Gelu,
    /// SiLU / swish `x·sigmoid(x)`.
    Silu,
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
}

/// Iteration / access pattern of an op — tells the emitter the loop-nest shape
/// and which schedules are legal.
///
/// `#[non_exhaustive]`: v1 is elementwise only; `Reduction { axes, combine,
/// identity }`, windowed/stencil, and gather patterns are the growth path.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Access {
    /// Output coordinate equals input coordinate (a per-element map).
    Elementwise,
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
}
