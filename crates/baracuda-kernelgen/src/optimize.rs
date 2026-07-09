//! Inward algebraic optimizer — an e-graph over the op IR (Kernel-Seam §5.1).
//!
//! §5.1 charges the synthesizer with building the **best** kernel for the
//! Fuel-chosen region, and explicitly permits an **e-graph / equality-saturation**
//! optimizer pointed *only inward* at that region. This is it: intern the op body
//! ([`ScalarExpr`]) into an e-graph, saturate a set of algebraic rewrites that
//! merge equivalent forms into one e-class, then **extract the lowest-cost form**.
//!
//! It is pointed strictly inward — it rewrites the *value* expression of one op,
//! never scanning a graph or choosing regions (that's Fuel's, §5.1). [`optimize`]
//! is a pure `ScalarExpr -> ScalarExpr` simplification used by JIT synthesis for
//! codegen; the recipe (`pattern:`/`decompose:`) stays the original region so
//! Fuel's matcher still recognizes the subgraph.
//!
//! # Scope (first cut)
//!
//! Total, precision-safe rewrites only: the const-`0`/`1` identities, constant
//! folding of the *algebraic* ops (transcendentals are left unfolded to avoid
//! host-f64 vs device-f32 divergence), and the `neg(neg x) -> x` involution.
//! Equality-saturation extraction picks the cheapest equivalent. The rewrite set
//! is the growth surface (factoring, FMA, perspective-diverse identities); the
//! e-graph machinery underneath does not change as rules are added.
//!
//! # Bit-preservation contract
//!
//! Every rewrite preserves the device result **bits** for all inputs, with one
//! documented carve-out: an eliminated arithmetic op no longer *quietens* a
//! signaling-NaN input, so sNaN payloads are out of contract (the platform
//! compilers make the same call). Zero **signs** and quiet-NaN payloads are IN
//! contract — which is why the zero identities are sign-gated by bits
//! (`x + (-0) -> x` is exact for every `x`, `x + (+0)` is NOT: `(-0)+(+0) = +0`;
//! dually `x - (+0) -> x` is exact and `x - (-0)` is not) and why const folds
//! skip NaN operands (a folded host NaN would drop the device's payload
//! propagation and the authored sign).

use crate::ir::{BinaryOp, ScalarExpr, UnaryOp};
use std::collections::HashMap;

type Id = usize;

/// An e-node: an op shape whose children are e-class ids. `Const` stores the
/// `f64` bit pattern so the node is `Hash`/`Eq` (NaN-safe by bits).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ENode {
    Input(u8),
    Const(u64),
    Param(u8),
    /// Opaque per-row reduced scalar ([`ScalarExpr::Reduced`]) — a leaf, never
    /// folded (a row scalar must not be CSE'd/constant-folded across rows).
    Reduced(u8),
    /// Opaque output-coordinate leaf ([`ScalarExpr::Coord`], increment 0d) —
    /// hash/eq by axis, ZERO rewrite/fold rules (its value varies per output
    /// coordinate, so no host fold is even well-typed). `Coord(d) == Coord(d)`
    /// hash-consing into one e-class is fine (same value at every coordinate);
    /// no rule may equate `Coord(i)` with anything else. Pinned by
    /// `coord_is_an_opaque_leaf_with_no_rules`.
    Coord(u8),
    Add(Id, Id),
    Sub(Id, Id),
    Mul(Id, Id),
    Div(Id, Id),
    Binary(BinaryOp, Id, Id),
    Unary(UnaryOp, Id),
}

/// An e-graph: union-find over e-classes + per-class e-node sets + a hashcons.
#[derive(Default)]
struct EGraph {
    parent: Vec<Id>,
    class_nodes: HashMap<Id, Vec<ENode>>,
    memo: HashMap<ENode, Id>,
}

impl EGraph {
    fn find(&mut self, mut x: Id) -> Id {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    /// Read-only find (no compression) — for extraction's shared borrows.
    fn find_imm(&self, mut x: Id) -> Id {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    /// Canonicalize an e-node's child ids through the union-find.
    fn canon(&mut self, n: &ENode) -> ENode {
        match *n {
            ENode::Add(a, b) => ENode::Add(self.find(a), self.find(b)),
            ENode::Sub(a, b) => ENode::Sub(self.find(a), self.find(b)),
            ENode::Mul(a, b) => ENode::Mul(self.find(a), self.find(b)),
            ENode::Div(a, b) => ENode::Div(self.find(a), self.find(b)),
            ENode::Binary(op, a, b) => ENode::Binary(op, self.find(a), self.find(b)),
            ENode::Unary(op, a) => ENode::Unary(op, self.find(a)),
            ref leaf => leaf.clone(),
        }
    }

    /// Intern an e-node (hashcons), returning its e-class id.
    fn add(&mut self, n: ENode) -> Id {
        let c = self.canon(&n);
        if let Some(&id) = self.memo.get(&c) {
            return self.find(id);
        }
        let id = self.parent.len();
        self.parent.push(id);
        self.class_nodes.entry(id).or_default().push(c.clone());
        self.memo.insert(c, id);
        id
    }

    /// Merge two e-classes; returns whether they were distinct.
    fn union(&mut self, a: Id, b: Id) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.parent[rb] = ra;
        if let Some(rb_nodes) = self.class_nodes.remove(&rb) {
            self.class_nodes.entry(ra).or_default().extend(rb_nodes);
        }
        true
    }

    /// The constant value of an e-class, if it contains a `Const` e-node.
    fn class_const(&self, id: Id) -> Option<f64> {
        let rc = self.find_imm(id);
        self.class_nodes.get(&rc)?.iter().find_map(|n| match n {
            ENode::Const(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        })
    }

    /// Re-canonicalize the class/hashcons index after a batch of unions (child
    /// ids through find, dedup). No congruence merging is needed: a single
    /// interned expression shares each subterm's class, so a simplification
    /// propagates to parents through the shared class id at extraction time.
    fn rebuild_index(&mut self) {
        let old: Vec<(Id, Vec<ENode>)> = self.class_nodes.drain().collect();
        self.memo.clear();
        let mut fresh: HashMap<Id, Vec<ENode>> = HashMap::new();
        for (c, nodes) in old {
            let rc = self.find(c);
            for n in nodes {
                let cn = self.canon(&n);
                self.memo.insert(cn.clone(), rc);
                let v = fresh.entry(rc).or_default();
                if !v.contains(&cn) {
                    v.push(cn);
                }
            }
        }
        self.class_nodes = fresh;
    }
}

fn add_expr(eg: &mut EGraph, e: &ScalarExpr) -> Id {
    match e {
        ScalarExpr::Input(i) => eg.add(ENode::Input(*i)),
        ScalarExpr::Const(v) => eg.add(ENode::Const(v.to_bits())),
        ScalarExpr::Param(i) => eg.add(ENode::Param(*i)),
        ScalarExpr::Reduced(i) => eg.add(ENode::Reduced(*i)),
        ScalarExpr::Coord(d) => eg.add(ENode::Coord(*d)),
        ScalarExpr::Add(a, b) => {
            let (a, b) = (add_expr(eg, a), add_expr(eg, b));
            eg.add(ENode::Add(a, b))
        }
        ScalarExpr::Sub(a, b) => {
            let (a, b) = (add_expr(eg, a), add_expr(eg, b));
            eg.add(ENode::Sub(a, b))
        }
        ScalarExpr::Mul(a, b) => {
            let (a, b) = (add_expr(eg, a), add_expr(eg, b));
            eg.add(ENode::Mul(a, b))
        }
        ScalarExpr::Div(a, b) => {
            let (a, b) = (add_expr(eg, a), add_expr(eg, b));
            eg.add(ENode::Div(a, b))
        }
        ScalarExpr::Binary(op, a, b) => {
            let (a, b) = (add_expr(eg, a), add_expr(eg, b));
            eg.add(ENode::Binary(*op, a, b))
        }
        ScalarExpr::Unary(op, x) => {
            let x = add_expr(eg, x);
            eg.add(ENode::Unary(*op, x))
        }
    }
}

/// Fold a unary op on a constant — algebraic ops only; transcendentals return
/// `None` (host-f64 vs device-f32 would diverge).
fn eval_unary(op: UnaryOp, v: f64) -> Option<f64> {
    // Never fold a NaN operand: the emitted `NAN` literal is the positive
    // canonical quiet NaN, which would drop an authored sign/payload the
    // runtime op preserves. Left symbolic, the device computes it faithfully.
    if v.is_nan() {
        return None;
    }
    Some(match op {
        UnaryOp::Neg => -v,
        UnaryOp::Abs => v.abs(),
        UnaryOp::Sqr => v * v,
        UnaryOp::Sqrt => v.sqrt(),
        // Rsqrt is NOT folded: device `rsqrtf` is an approximation (~2 ulp), so a
        // host `1/sqrt(v)` fold would change the bits the kernel emits.
        UnaryOp::Recip => 1.0 / v,
        UnaryOp::Relu => {
            if v < 0.0 {
                0.0
            } else {
                v
            }
        }
        UnaryOp::Floor => v.floor(),
        UnaryOp::Ceil => v.ceil(),
        UnaryOp::Round => v.round_ties_even(),
        // Trunc is exact on FINITE values only by policy: NaN is already guarded
        // above, and ±Inf stays symbolic too (house lesson: fold nothing
        // non-finite — the emitted INFINITY literal round-trip is not worth the
        // risk surface). Every other increment-0a fn is a device approximation
        // and is deliberately NOT folded (the Rsqrt-fold lesson).
        UnaryOp::Trunc if v.is_finite() => v.trunc(),
        UnaryOp::Sign => {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UnaryOp::Step => {
            if v > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        // Sin/Cos/Rsqrt, the activations, and the whole increment-0a fn set
        // (Erfc…Lgamma): device-approximate — never folded.
        _ => return None,
    })
}

/// The exact reciprocal of `c` when `c` is a finite **normal power of two**
/// whose reciprocal is also finite and normal — the precondition under which
/// `x / c == x * (1/c)` bit-exactly (the reciprocal is exact, so the true
/// product equals the true quotient and both round identically). `None`
/// otherwise (non-pow2, zero, subnormal, or a reciprocal that would leave the
/// normal range).
fn exact_pow2_recip(c: f64) -> Option<f64> {
    const MANTISSA_MASK: u64 = (1u64 << 52) - 1;
    const EXP_MASK: u64 = 0x7ff;
    let is_normal_pow2 = |v: f64| {
        let bits = v.to_bits();
        let exp = (bits >> 52) & EXP_MASK;
        bits & MANTISSA_MASK == 0 && exp != 0 && exp != EXP_MASK
    };
    if !is_normal_pow2(c) {
        return None;
    }
    let r = 1.0 / c;
    is_normal_pow2(r).then_some(r)
}

/// Fold a non-infix binary op on two constants — `Max`/`Min` and integer-clean
/// `Rem`; `Pow` is skipped (host-f64 vs device-f32), `Rem` by zero is skipped.
/// The increment-0a binaries are ALL skipped: `Atan2` is approximate, and the
/// exact bit-level ops (`Copysign`/`Nextafter`/`FmaxIeee`/`FminIeee`/`RemTrunc`)
/// stay unfolded under the when-in-doubt-add-no-rule policy (`Nextafter` in
/// particular is dtype-lattice-dependent, so a host-f64 fold would be wrong).
/// The increment-0b `Cmp*` predicates are ALSO all skipped — even const-const:
/// a fold would need the full NaN gate (any cmp with NaN is false except
/// `CmpNe`), the host f64 compare of two demote-destined constants can disagree
/// with the device's dtype-width compare (two f64 constants that are distinct
/// on the host collapse to one f32/f16 value on the device, flipping
/// `==`/`!=`/`<`), and when-in-doubt-add-no-rule holds. Pinned by
/// `cmp_predicates_are_never_folded_or_rewritten`.
fn eval_binary(op: BinaryOp, x: f64, y: f64) -> Option<f64> {
    // Max/Min only fold when neither operand is NaN — the kernel propagates NaN
    // (NaN-select), so folding a NaN operand away (host f64::max suppresses it)
    // would disagree with the device.
    Some(match op {
        BinaryOp::Max if !x.is_nan() && !y.is_nan() => x.max(y),
        BinaryOp::Min if !x.is_nan() && !y.is_nan() => x.min(y),
        // floored remainder (torch.remainder), matching the kernel — not `x % y`.
        // Gated finite-in AND finite-out: NaN operands must stay symbolic (the
        // fold would canonicalize the payload/sign the device would propagate),
        // and an overflowing (x/y).floor()*y can produce ±inf, whose `-INFINITY`
        // literal the headerless-nvrtc discipline forbids (see `ScalarExpr` docs).
        BinaryOp::Rem if x.is_finite() && y.is_finite() && y != 0.0 => {
            let r = x - (x / y).floor() * y;
            if !r.is_finite() {
                return None;
            }
            r
        }
        _ => return None,
    })
}

/// One rewrite pass: recognize equivalent forms and `union` them in. Returns
/// whether anything merged.
fn rules(eg: &mut EGraph) -> bool {
    let snapshot: Vec<ENode> = eg.class_nodes.values().flatten().cloned().collect();
    let mut changed = false;
    for node in snapshot {
        let nid = eg.add(node.clone());
        match node {
            ENode::Add(a, b) => {
                // x + (-0) -> x: bit-exact for EVERY x ((+0)+(-0) = +0,
                // (-0)+(-0) = -0). x + (+0) is NOT an identity: (-0)+(+0) = +0
                // under round-to-nearest, but the passthrough would keep -0.
                // Gate by BITS — `== Some(0.0)` would match -0.0 too.
                let neg_zero = Some((-0.0f64).to_bits());
                if eg.class_const(b).map(f64::to_bits) == neg_zero {
                    changed |= eg.union(nid, a);
                }
                if eg.class_const(a).map(f64::to_bits) == neg_zero {
                    changed |= eg.union(nid, b);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if !x.is_nan() && !y.is_nan() {
                        let c = eg.add(ENode::Const((x + y).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Sub(a, b) => {
                // x - (+0) -> x: bit-exact for EVERY x ((-0)-(+0) = -0,
                // (+0)-(+0) = +0). x - (-0) is NOT: (-0)-(-0) = +0, but the
                // passthrough would keep -0. Gate by bits.
                if eg.class_const(b).map(f64::to_bits) == Some(0.0f64.to_bits()) {
                    changed |= eg.union(nid, a);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if !x.is_nan() && !y.is_nan() {
                        let c = eg.add(ENode::Const((x - y).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Mul(a, b) => {
                if eg.class_const(b) == Some(1.0) {
                    changed |= eg.union(nid, a);
                }
                if eg.class_const(a) == Some(1.0) {
                    changed |= eg.union(nid, b);
                }
                // NOTE: `x * 0 -> 0` is deliberately ABSENT — it is not
                // value-preserving (NaN*0 = NaN, Inf*0 = NaN, and -x*0 = -0);
                // folding it would silently change the bits a kernel computes.
                // Two-const products fold below (0*0 included, exactly).
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if !x.is_nan() && !y.is_nan() {
                        let c = eg.add(ENode::Const((x * y).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Div(a, b) => {
                if eg.class_const(b) == Some(1.0) {
                    changed |= eg.union(nid, a);
                }
                // x / 2^k  ->  x * 2^-k: bit-exact (an exact power-of-two
                // reciprocal makes the true product equal the true quotient, so
                // both round identically — incl. NaN/Inf/±0 propagation), and
                // device FDIV is ~4x an FMUL (weights 8 vs 2 drive extraction).
                if let Some(c) = eg.class_const(b) {
                    if let Some(r) = exact_pow2_recip(c) {
                        let rc = eg.add(ENode::Const(r.to_bits()));
                        let m = eg.add(ENode::Mul(a, rc));
                        changed |= eg.union(nid, m);
                    }
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if y != 0.0 && !x.is_nan() && !y.is_nan() {
                        let c = eg.add(ENode::Const((x / y).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Unary(UnaryOp::Neg, x) => {
                // neg(neg(y)) -> y
                let xc = eg.find(x);
                let inner = eg.class_nodes.get(&xc).and_then(|ns| {
                    ns.iter().find_map(|n| match n {
                        ENode::Unary(UnaryOp::Neg, y) => Some(*y),
                        _ => None,
                    })
                });
                if let Some(y) = inner {
                    changed |= eg.union(nid, y);
                }
                if let Some(v) = eg.class_const(x) {
                    if !v.is_nan() {
                        let c = eg.add(ENode::Const((-v).to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Unary(op @ (UnaryOp::Abs | UnaryOp::Relu), x) => {
                // abs(abs(y)) -> abs(y) ; relu(relu(y)) -> relu(y): idempotent.
                // abs(neg(y)) -> abs(y): |-y| == |y| bit-exactly (abs clears the
                // sign bit either way; NaN payload untouched). relu(neg) is NOT
                // an identity — do not generalize.
                let xc = eg.find(x);
                let inner = eg.class_nodes.get(&xc).and_then(|ns| {
                    ns.iter().find_map(|n| match n {
                        ENode::Unary(i, y) if *i == op => Some((op, *y)),
                        ENode::Unary(UnaryOp::Neg, y) if op == UnaryOp::Abs => {
                            Some((UnaryOp::Abs, *y))
                        }
                        _ => None,
                    })
                });
                if let Some((outer, y)) = inner {
                    let collapsed = eg.add(ENode::Unary(outer, y));
                    changed |= eg.union(nid, collapsed);
                }
                if let Some(v) = eg.class_const(x) {
                    if let Some(r) = eval_unary(op, v) {
                        let c = eg.add(ENode::Const(r.to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Binary(op, a, b) => {
                // max(x, x) = x ; min(x, x) = x. STRICTLY Max/Min — never the
                // Cmp* predicates: CmpEq(x, x) is NOT 1.0 (it is FALSE for NaN
                // x) and CmpNe(x, x) is NOT 0.0 (TRUE for NaN x); the 0a review
                // proved a widened rule arm here can pass the suite, so the
                // non-rewrite is pinned per op in
                // `cmp_predicates_are_never_folded_or_rewritten`.
                if matches!(op, BinaryOp::Max | BinaryOp::Min) && eg.find(a) == eg.find(b) {
                    changed |= eg.union(nid, a);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if let Some(r) = eval_binary(op, x, y) {
                        let c = eg.add(ENode::Const(r.to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            ENode::Unary(op, x) => {
                if let Some(v) = eg.class_const(x) {
                    if let Some(r) = eval_unary(op, v) {
                        let c = eg.add(ENode::Const(r.to_bits()));
                        changed |= eg.union(nid, c);
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

fn saturate(eg: &mut EGraph, max_iters: usize) {
    for _ in 0..max_iters {
        let changed = rules(eg);
        eg.rebuild_index();
        if !changed {
            break;
        }
    }
}

/// Relative op cost for extraction — division and transcendentals dominate.
fn weight(n: &ENode) -> u64 {
    match n {
        // Coord sits in the leaf tier: on the strided schedule the unraveled
        // c{d} already exists for the offset math, so reading it costs a cast.
        ENode::Input(_)
        | ENode::Param(_)
        | ENode::Const(_)
        | ENode::Reduced(_)
        | ENode::Coord(_) => 1,
        ENode::Add(..) | ENode::Sub(..) | ENode::Mul(..) => 2,
        ENode::Div(..) => 8,
        ENode::Binary(op, ..) => match op {
            // Copysign/Nextafter are bit-manipulation ops; FmaxIeee/FminIeee are
            // hardware min/max — all cheap, same tier as the Max/Min selects.
            // The Cmp* predicates (increment 0b) are compare-selects too: one
            // setp + one sel, the same tier as Max/Min. The increment-0c
            // bitwise/shift ops are single ALU instructions and the logical
            // ops a setp pair + sel — all the same compare-select tier.
            BinaryOp::Max
            | BinaryOp::Min
            | BinaryOp::Copysign
            | BinaryOp::Nextafter
            | BinaryOp::FmaxIeee
            | BinaryOp::FminIeee
            | BinaryOp::CmpEq
            | BinaryOp::CmpNe
            | BinaryOp::CmpLt
            | BinaryOp::CmpLe
            | BinaryOp::CmpGt
            | BinaryOp::CmpGe
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::LogicalAnd
            | BinaryOp::LogicalOr
            | BinaryOp::LogicalXor => 2,
            BinaryOp::Rem | BinaryOp::RemTrunc => 8, // division-class
            BinaryOp::Pow | BinaryOp::Atan2 => 16,   // transcendental
        },
        ENode::Unary(op, _) => match op {
            UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Relu => 1,
            UnaryOp::Sqr
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Sign
            | UnaryOp::Step
            | UnaryOp::Trunc => 2,
            UnaryOp::Sqrt | UnaryOp::Rsqrt | UnaryOp::Recip => 8,
            UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Tanh
            | UnaryOp::Sigmoid
            | UnaryOp::Erf
            | UnaryOp::Gelu
            | UnaryOp::Silu
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Erfc
            | UnaryOp::Exp2
            | UnaryOp::Expm1
            | UnaryOp::Log2
            | UnaryOp::Log10
            | UnaryOp::Log1p
            | UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tan
            | UnaryOp::Asin
            | UnaryOp::Acos
            | UnaryOp::Atan
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh
            | UnaryOp::Cbrt
            | UnaryOp::Lgamma => 16,
        },
    }
}

fn children(n: &ENode) -> Vec<Id> {
    match *n {
        ENode::Add(a, b)
        | ENode::Sub(a, b)
        | ENode::Mul(a, b)
        | ENode::Div(a, b)
        | ENode::Binary(_, a, b) => vec![a, b],
        ENode::Unary(_, a) => vec![a],
        _ => vec![],
    }
}

/// Total cost of an e-node given the best costs of its children, or `None` if a
/// child has no cost yet.
fn enode_cost(eg: &EGraph, n: &ENode, best: &HashMap<Id, (u64, ENode)>) -> Option<u64> {
    let mut sum = weight(n);
    for k in children(n) {
        sum = sum.saturating_add(best.get(&eg.find_imm(k))?.0);
    }
    Some(sum)
}

/// Extract the lowest-cost equivalent expression for `root` (equality-saturation
/// extraction: relax per-class min costs to a fixpoint, then reconstruct).
fn extract(eg: &EGraph, root: Id) -> ScalarExpr {
    let mut best: HashMap<Id, (u64, ENode)> = HashMap::new();
    loop {
        let mut changed = false;
        for (&c, nodes) in &eg.class_nodes {
            let rc = eg.find_imm(c);
            for n in nodes {
                if let Some(k) = enode_cost(eg, n, &best) {
                    if best.get(&rc).is_none_or(|(bk, _)| k < *bk) {
                        best.insert(rc, (k, n.clone()));
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    build(eg, eg.find_imm(root), &best)
}

fn build(eg: &EGraph, c: Id, best: &HashMap<Id, (u64, ENode)>) -> ScalarExpr {
    let n = best[&eg.find_imm(c)].1.clone();
    match n {
        ENode::Input(i) => ScalarExpr::Input(i),
        ENode::Const(bits) => ScalarExpr::Const(f64::from_bits(bits)),
        ENode::Param(i) => ScalarExpr::Param(i),
        ENode::Reduced(i) => ScalarExpr::Reduced(i),
        ENode::Coord(d) => ScalarExpr::Coord(d),
        ENode::Add(a, b) => ScalarExpr::Add(bx(build(eg, a, best)), bx(build(eg, b, best))),
        ENode::Sub(a, b) => ScalarExpr::Sub(bx(build(eg, a, best)), bx(build(eg, b, best))),
        ENode::Mul(a, b) => ScalarExpr::Mul(bx(build(eg, a, best)), bx(build(eg, b, best))),
        ENode::Div(a, b) => ScalarExpr::Div(bx(build(eg, a, best)), bx(build(eg, b, best))),
        ENode::Binary(op, a, b) => {
            ScalarExpr::Binary(op, bx(build(eg, a, best)), bx(build(eg, b, best)))
        }
        ENode::Unary(op, x) => ScalarExpr::Unary(op, bx(build(eg, x, best))),
    }
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

/// Algebraically simplify an op body to the lowest-cost equivalent form via
/// equality saturation. Semantics-preserving within the precision-safe rule set
/// (see the module scope note). Pure: `optimize(optimize(e)) == optimize(e)`.
#[must_use]
pub fn optimize(e: &ScalarExpr) -> ScalarExpr {
    let mut eg = EGraph::default();
    let root = add_expr(&mut eg, e);
    saturate(&mut eg, 32);
    extract(&eg, root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{input, konst};

    fn opt(e: crate::ir::Expr) -> ScalarExpr {
        optimize(&e.0)
    }

    fn neg(e: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Unary(UnaryOp::Neg, Box::new(e))
    }

    #[test]
    fn mul_one_is_identity() {
        assert_eq!(opt(input(0) * konst(1.0)), ScalarExpr::Input(0));
    }

    #[test]
    fn zero_identities_are_sign_gated() {
        // The bit-exact identities: x + (-0) and x - (+0) pass through…
        assert_eq!(opt(input(0) + konst(-0.0)), ScalarExpr::Input(0));
        assert_eq!(opt(input(2) - konst(0.0)), ScalarExpr::Input(2));
        // …but the sign-flipping forms must NOT: (-0)+(+0) = +0 and
        // (-0)-(-0) = +0, so eliminating the op would leak a -0 through.
        assert!(matches!(opt(input(0) + konst(0.0)), ScalarExpr::Add(_, _)));
        assert!(matches!(opt(input(2) - konst(-0.0)), ScalarExpr::Sub(_, _)));
    }

    #[test]
    fn nan_constants_are_never_folded() {
        // Folding a host NaN would emit the positive canonical `NAN` literal,
        // dropping the sign/payload the runtime device op preserves.
        let e = optimize(&neg(ScalarExpr::Const(f64::NAN)));
        assert!(
            matches!(e, ScalarExpr::Unary(UnaryOp::Neg, ref x) if matches!(**x, ScalarExpr::Const(v) if v.is_nan())),
            "neg(NaN) stays symbolic, got {e:?}"
        );
        assert!(matches!(
            opt(konst(f64::NAN) + konst(1.0)),
            ScalarExpr::Add(_, _)
        ));
    }

    #[test]
    fn mul_zero_is_not_folded_for_nonconst_operand() {
        // x * 0 must NOT collapse to 0: for x = NaN or ±Inf the kernel computes
        // NaN (and for finite negative x, -0), so the fold would change bits.
        // The rewrite set is precision-safe by contract.
        let e = opt(input(0) * konst(0.0));
        assert!(
            matches!(e, ScalarExpr::Mul(_, _)),
            "x*0 stays symbolic, got {e:?}"
        );
        // Two-const products still fold exactly (0*0 included).
        assert_eq!(opt(konst(0.0) * konst(5.0)), ScalarExpr::Const(0.0));
    }

    #[test]
    fn div_by_pow2_becomes_mul_by_exact_reciprocal() {
        // x / 4 -> x * 0.25 (bit-exact; FDIV ~4x an FMUL, so extraction prefers it).
        assert_eq!(
            opt(input(0) / konst(4.0)),
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Const(0.25))
            )
        );
        // Negative power of two too: x / -2 -> x * -0.5.
        assert_eq!(
            opt(input(0) / konst(-2.0)),
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Const(-0.5))
            )
        );
        // Non-power-of-two divisor stays a division (1/3 is inexact).
        assert!(matches!(opt(input(0) / konst(3.0)), ScalarExpr::Div(_, _)));
        // Zero / subnormal-reciprocal divisors stay divisions.
        assert!(matches!(opt(input(0) / konst(0.0)), ScalarExpr::Div(_, _)));
    }

    #[test]
    fn abs_and_relu_idempotents_collapse() {
        let abs = |e: ScalarExpr| ScalarExpr::Unary(UnaryOp::Abs, Box::new(e));
        let relu = |e: ScalarExpr| ScalarExpr::Unary(UnaryOp::Relu, Box::new(e));
        let x = ScalarExpr::Input(0);
        assert_eq!(optimize(&abs(abs(x.clone()))), abs(x.clone()));
        assert_eq!(optimize(&relu(relu(x.clone()))), relu(x.clone()));
        // |-y| == |y| bit-exactly (sign-bit op); relu(neg) is NOT rewritten.
        assert_eq!(optimize(&abs(neg(x.clone()))), abs(x.clone()));
        let rn = optimize(&relu(neg(x.clone())));
        assert!(
            matches!(&rn, ScalarExpr::Unary(UnaryOp::Relu, inner)
                if matches!(**inner, ScalarExpr::Unary(UnaryOp::Neg, _))),
            "relu(neg(x)) must stay, got {rn:?}"
        );
    }

    #[test]
    fn constants_fold() {
        assert_eq!(opt(konst(2.0) * konst(3.0)), ScalarExpr::Const(6.0));
        assert_eq!(opt(konst(2.0) + konst(5.0)), ScalarExpr::Const(7.0));
    }

    #[test]
    fn neg_neg_cancels() {
        assert_eq!(
            optimize(&neg(neg(ScalarExpr::Input(0)))),
            ScalarExpr::Input(0)
        );
    }

    #[test]
    fn redundant_chain_simplifies_under_an_op() {
        // relu(x*1 + (-0)) -> relu(x): the (sign-correct) identities propagate
        // under the Relu via the shared e-class, and extraction picks the
        // cheapest form.
        let body = (input(0) * konst(1.0) + konst(-0.0)).relu();
        assert_eq!(
            optimize(&body.0),
            ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
        );
    }

    #[test]
    fn transcendentals_are_not_const_folded() {
        // exp(1.0) is left symbolic (host-f64 vs device-f32), not folded to a const.
        let e = opt(konst(1.0).exp());
        assert_eq!(
            e,
            ScalarExpr::Unary(UnaryOp::Exp, Box::new(ScalarExpr::Const(1.0)))
        );
    }

    #[test]
    fn irreducible_body_is_unchanged() {
        let e = input(0) + input(1) * input(2);
        assert_eq!(opt(e.clone()), e.0);
    }

    #[test]
    fn binary_fn_folds_and_simplifies() {
        // const fold max/min; max(x,x) -> x.
        assert_eq!(opt(konst(2.0).max(konst(5.0))), ScalarExpr::Const(5.0));
        assert_eq!(opt(konst(2.0).min(konst(5.0))), ScalarExpr::Const(2.0));
        let max_xx = ScalarExpr::Binary(
            BinaryOp::Max,
            Box::new(ScalarExpr::Input(0)),
            Box::new(ScalarExpr::Input(0)),
        );
        assert_eq!(optimize(&max_xx), ScalarExpr::Input(0));
        // Pow is not const-folded (host/device divergence) — stays symbolic.
        let pow = opt(konst(2.0).pow(konst(3.0)));
        assert!(matches!(pow, ScalarExpr::Binary(BinaryOp::Pow, _, _)));
        // Rem is FLOORED (torch.remainder): -3 rem 2 = 1 (sign-of-divisor), not -1.
        let rem = optimize(&ScalarExpr::Binary(
            BinaryOp::Rem,
            Box::new(ScalarExpr::Const(-3.0)),
            Box::new(ScalarExpr::Const(2.0)),
        ));
        assert_eq!(rem, ScalarExpr::Const(1.0));
    }

    #[test]
    fn idempotent() {
        let body = (input(0) * konst(1.0) + konst(0.0)).relu().0;
        let once = optimize(&body);
        assert_eq!(optimize(&once), once);
    }

    #[test]
    fn trunc_const_fold_is_finite_gated() {
        use crate::ir::UnaryOp;
        let trunc = |v: f64| ScalarExpr::Unary(UnaryOp::Trunc, Box::new(ScalarExpr::Const(v)));
        // Exact on finite values: fold (round toward zero, both signs).
        assert_eq!(optimize(&trunc(-3.7)), ScalarExpr::Const(-3.0));
        assert_eq!(optimize(&trunc(2.9)), ScalarExpr::Const(2.0));
        // Non-finite stays symbolic (house lesson: no non-finite const folds).
        assert!(matches!(
            optimize(&trunc(f64::INFINITY)),
            ScalarExpr::Unary(UnaryOp::Trunc, _)
        ));
        assert!(matches!(
            optimize(&trunc(f64::NEG_INFINITY)),
            ScalarExpr::Unary(UnaryOp::Trunc, _)
        ));
        assert!(matches!(
            optimize(&trunc(f64::NAN)),
            ScalarExpr::Unary(UnaryOp::Trunc, _)
        ));
    }

    #[test]
    fn increment_0a_fns_are_never_const_folded() {
        use crate::ir::{BinaryOp, UnaryOp};
        // Device-approximate fns stay symbolic (the Rsqrt-fold lesson) — ALL
        // 17 approximate increment-0a unaries, so a host-fold added for any
        // one of them fails here (mutation-caught gap: sampling 5 let an
        // Exp2 host-fold pass the suite).
        for op in [
            UnaryOp::Erfc,
            UnaryOp::Exp2,
            UnaryOp::Expm1,
            UnaryOp::Log2,
            UnaryOp::Log10,
            UnaryOp::Log1p,
            UnaryOp::Sinh,
            UnaryOp::Cosh,
            UnaryOp::Tan,
            UnaryOp::Asin,
            UnaryOp::Acos,
            UnaryOp::Atan,
            UnaryOp::Asinh,
            UnaryOp::Acosh,
            UnaryOp::Atanh,
            UnaryOp::Cbrt,
            UnaryOp::Lgamma,
        ] {
            let e = optimize(&ScalarExpr::Unary(op, Box::new(ScalarExpr::Const(1.5))));
            assert!(
                matches!(e, ScalarExpr::Unary(o, _) if o == op),
                "{op:?}(const) must stay symbolic, got {e:?}"
            );
        }
        // …and so do ALL the new binaries — including the exact bit-level ones
        // (when-in-doubt-add-no-rule; Nextafter is dtype-lattice-dependent).
        for op in [
            BinaryOp::Atan2,
            BinaryOp::Copysign,
            BinaryOp::Nextafter,
            BinaryOp::FmaxIeee,
            BinaryOp::FminIeee,
            BinaryOp::RemTrunc,
        ] {
            let e = optimize(&ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Const(-3.0)),
                Box::new(ScalarExpr::Const(2.0)),
            ));
            assert!(
                matches!(e, ScalarExpr::Binary(o, _, _) if o == op),
                "{op:?}(const, const) must stay symbolic, got {e:?}"
            );
        }
        // No new identity rewrites either: the x,x forms of every new binary
        // stay as authored (unlike the pinned max(x,x)/min(x,x) -> x for the
        // NaN-propagating Max/Min). Pinning all three closes the mutation
        // where extending the Max|Min rule arm to FminIeee passed the suite.
        for op in [BinaryOp::FmaxIeee, BinaryOp::FminIeee, BinaryOp::RemTrunc] {
            let same = ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(0)),
            );
            assert_eq!(optimize(&same), same, "{op:?}(x, x) must stay as authored");
        }
    }

    #[test]
    fn cmp_predicates_are_never_folded_or_rewritten() {
        use crate::ir::BinaryOp;
        const CMPS: [BinaryOp; 6] = [
            BinaryOp::CmpEq,
            BinaryOp::CmpNe,
            BinaryOp::CmpLt,
            BinaryOp::CmpLe,
            BinaryOp::CmpGt,
            BinaryOp::CmpGe,
        ];
        // No const folds, even const-const (NaN gates + host-vs-device compare
        // width + when-in-doubt-add-no-rule): every pair stays symbolic —
        // including the tempting finite pair, the NaN pair, and the ±0 pair
        // (where -0 == +0 is TRUE, a fold bug magnet).
        for op in CMPS {
            for (x, y) in [(2.0, 3.0), (f64::NAN, 1.0), (-0.0, 0.0)] {
                let e = ScalarExpr::Binary(
                    op,
                    Box::new(ScalarExpr::Const(x)),
                    Box::new(ScalarExpr::Const(y)),
                );
                assert!(
                    matches!(optimize(&e), ScalarExpr::Binary(o, _, _) if o == op),
                    "{op:?}({x}, {y}) must stay symbolic"
                );
            }
        }
        // And NO identity rewrites: CmpEq(x, x) must NOT fold to 1.0 — it is
        // FALSE for NaN x (and CmpNe(x, x) is TRUE for NaN x); the reflexive
        // form of every predicate stays exactly as authored. This pins the
        // mutation the 0a review caught for Min|FminIeee: extending the
        // max(x,x)->x rule arm to any Cmp* must fail here.
        for op in CMPS {
            let same = ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(0)),
            );
            assert_eq!(optimize(&same), same, "{op:?}(x, x) must stay as authored");
        }
    }

    #[test]
    fn int_bitwise_logical_ops_are_never_folded_or_rewritten() {
        use crate::ir::BinaryOp;
        // The e-graph's consts are host f64 — a host-f64 "fold" of an int op
        // is WRONG twice over (f64 cannot represent all i64; wrapping
        // two's-complement differs from float arithmetic), and the e-graph
        // has no dtype context to even know the operand width. ZERO rules for
        // ALL EIGHT increment-0c ops, pinned exhaustively (the 0a lesson: pin
        // all, not a sample — a sampled pin let an Exp2 host-fold pass).
        const INT_OPS: [BinaryOp; 8] = [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::LogicalAnd,
            BinaryOp::LogicalOr,
            BinaryOp::LogicalXor,
        ];
        // No const folds, even const-const — including the tempting all-int
        // pair, a zero operand (x & 0, x << 0 are classic fold bait), and a
        // negative shift amount (device-inherited behavior, unknowable here).
        for op in INT_OPS {
            for (x, y) in [(6.0, 3.0), (5.0, 0.0), (1.0, -1.0)] {
                let e = ScalarExpr::Binary(
                    op,
                    Box::new(ScalarExpr::Const(x)),
                    Box::new(ScalarExpr::Const(y)),
                );
                assert!(
                    matches!(optimize(&e), ScalarExpr::Binary(o, _, _) if o == op),
                    "{op:?}({x}, {y}) must stay symbolic"
                );
            }
        }
        // And NO identity rewrites: the (x, x) forms stay exactly as authored
        // (BitAnd(x,x)=x / BitOr(x,x)=x / LogicalXor(x,x)=0 are all true on
        // device but the rule set stays empty — when-in-doubt-add-no-rule;
        // this pins the 0a-review mutation class where widening the
        // max(x,x)->x rule arm passed the suite).
        for op in INT_OPS {
            let same = ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(0)),
            );
            assert_eq!(optimize(&same), same, "{op:?}(x, x) must stay as authored");
        }
        // Cost entries exist (compare-select tier) so extraction still ranks
        // bodies containing them — weight, not rules, is the only e-graph
        // surface the 0c ops touch.
        for op in INT_OPS {
            let n = ENode::Binary(op, 0, 1);
            assert_eq!(weight(&n), 2, "{op:?} must sit in the compare-select tier");
        }
    }

    #[test]
    fn coord_is_an_opaque_leaf_with_no_rules() {
        use crate::ir::{BinaryOp, coord};
        // Representative Coord bodies round-trip optimize() UNCHANGED — the
        // triu-mask predicate multiply and the alibi relative-position body.
        let triu = (input(0) * coord(1).binary(BinaryOp::CmpGe, coord(0) + konst(0.0))).0;
        assert_eq!(optimize(&triu), triu, "triu-mask body must round-trip");
        let alibi = ((coord(1) - coord(0)) * crate::ir::param(0)).0;
        assert_eq!(optimize(&alibi), alibi, "alibi body must round-trip");
        // A bare Coord leaf is already minimal.
        assert_eq!(optimize(&coord(1).0), ScalarExpr::Coord(1));
        // No rule equates Coord(i) with anything else: Coord(0) - Coord(0)
        // stays symbolic (no x-x rule exists, and none may be added for
        // Coord), and the reflexive compare stays as authored (the same
        // NaN-honesty pin the Cmp* set carries — a widened rule arm matching
        // Coord must fail here).
        let sub_same = ScalarExpr::Sub(
            Box::new(ScalarExpr::Coord(0)),
            Box::new(ScalarExpr::Coord(0)),
        );
        assert_eq!(optimize(&sub_same), sub_same);
        for op in [
            BinaryOp::CmpEq,
            BinaryOp::CmpGe,
            BinaryOp::Max,
            BinaryOp::Min,
        ] {
            let e = ScalarExpr::Binary(
                op,
                Box::new(ScalarExpr::Coord(0)),
                Box::new(ScalarExpr::Coord(1)),
            );
            assert_eq!(optimize(&e), e, "{op:?}(c0, c1) must stay as authored");
        }
        // Coord(0) and Coord(1) never merge (distinct axes = distinct values):
        // c0 + c1 keeps two distinct leaves.
        let e = optimize(&(coord(0) + coord(1)).0);
        assert!(
            matches!(&e, ScalarExpr::Add(a, b)
                if matches!(**a, ScalarExpr::Coord(0)) && matches!(**b, ScalarExpr::Coord(1))),
            "Coord(0)/Coord(1) must stay distinct, got {e:?}"
        );
        // The VALUE-GENERIC bit-exact identities still apply to a Coord
        // operand (they are proofs about every value, not about Coord):
        // c1 * 1.0 -> c1. This is hash-cons/extraction, not a Coord rule.
        assert_eq!(optimize(&(coord(1) * konst(1.0)).0), ScalarExpr::Coord(1));
    }

    #[test]
    fn rem_const_fold_is_finite_gated() {
        use crate::ir::BinaryOp;
        let rem = |a: f64, b: f64| {
            ScalarExpr::Binary(
                BinaryOp::Rem,
                Box::new(ScalarExpr::Const(a)),
                Box::new(ScalarExpr::Const(b)),
            )
        };
        // The legitimate fold still fires…
        assert_eq!(optimize(&rem(7.0, 2.0)), ScalarExpr::Const(1.0));
        // …but NaN operands stay symbolic (a fold would canonicalize the
        // payload/sign the device op propagates)…
        for e in [rem(f64::NAN, 2.0), rem(5.0, f64::NAN)] {
            assert!(
                matches!(optimize(&e), ScalarExpr::Binary(BinaryOp::Rem, _, _)),
                "NaN-operand Rem must stay symbolic"
            );
        }
        // …as do infinite operands and finite operands whose floored-mod
        // composite overflows to ±inf (a -INFINITY literal is forbidden under
        // the headerless-nvrtc discipline).
        for e in [
            rem(f64::INFINITY, 2.0),
            rem(5.0, f64::NEG_INFINITY),
            rem(1e308, 1e-308),
        ] {
            assert!(
                matches!(optimize(&e), ScalarExpr::Binary(BinaryOp::Rem, _, _)),
                "non-finite-in or non-finite-out Rem must stay symbolic"
            );
        }
    }
}
