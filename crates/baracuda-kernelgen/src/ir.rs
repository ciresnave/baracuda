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
    /// `true` for a source leaf (`Input`/`Const`/`Param`/`Reduced`) — a value with
    /// no children. Leaves are never hoisted to a `tmp` (a leaf reference is free);
    /// only shared *interior* nodes are.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            DagNode::Input(_) | DagNode::Const(_) | DagNode::Param(_) | DagNode::Reduced(_)
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
        DagNode::Input(_) | DagNode::Const(_) | DagNode::Param(_) | DagNode::Reduced(_) => {
            Vec::new()
        }
        DagNode::Unary(_, x) => vec![x],
        DagNode::Add(a, b)
        | DagNode::Sub(a, b)
        | DagNode::Mul(a, b)
        | DagNode::Div(a, b)
        | DagNode::Binary(_, a, b) => vec![a, b],
    }
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
    /// mask is *representable* here but only lowered by that follow-up.
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
            access: Access::Reduction { op, axes, keepdim },
            views: Vec::new(),
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
            Access::Reduction { op, axes, keepdim } => {
                assert_eq!(op, ReduceOp::Sum);
                assert!(axes.is_empty());
                assert!(!keepdim);
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
            Access::Reduction { op, axes, keepdim } => {
                assert_eq!(op, ReduceOp::Mean);
                assert!(axes.is_set(0));
                assert!(!axes.is_set(1));
                assert!(keepdim);
            }
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
