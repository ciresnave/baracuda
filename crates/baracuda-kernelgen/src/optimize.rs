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
        _ => return None, // Sin/Cos/Rsqrt + the activations: transcendental, skip
    })
}

/// Fold a non-infix binary op on two constants — `Max`/`Min` and integer-clean
/// `Rem`; `Pow` is skipped (host-f64 vs device-f32), `Rem` by zero is skipped.
fn eval_binary(op: BinaryOp, x: f64, y: f64) -> Option<f64> {
    // Max/Min only fold when neither operand is NaN — the kernel propagates NaN
    // (NaN-select), so folding a NaN operand away (host f64::max suppresses it)
    // would disagree with the device.
    Some(match op {
        BinaryOp::Max if !x.is_nan() && !y.is_nan() => x.max(y),
        BinaryOp::Min if !x.is_nan() && !y.is_nan() => x.min(y),
        // floored remainder (torch.remainder), matching the kernel — not `x % y`
        BinaryOp::Rem if y != 0.0 => x - (x / y).floor() * y,
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
                if eg.class_const(b) == Some(0.0) {
                    changed |= eg.union(nid, a);
                }
                if eg.class_const(a) == Some(0.0) {
                    changed |= eg.union(nid, b);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    let c = eg.add(ENode::Const((x + y).to_bits()));
                    changed |= eg.union(nid, c);
                }
            }
            ENode::Sub(a, b) => {
                if eg.class_const(b) == Some(0.0) {
                    changed |= eg.union(nid, a);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    let c = eg.add(ENode::Const((x - y).to_bits()));
                    changed |= eg.union(nid, c);
                }
            }
            ENode::Mul(a, b) => {
                if eg.class_const(b) == Some(1.0) {
                    changed |= eg.union(nid, a);
                }
                if eg.class_const(a) == Some(1.0) {
                    changed |= eg.union(nid, b);
                }
                if eg.class_const(a) == Some(0.0) || eg.class_const(b) == Some(0.0) {
                    let z = eg.add(ENode::Const(0.0_f64.to_bits()));
                    changed |= eg.union(nid, z);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    let c = eg.add(ENode::Const((x * y).to_bits()));
                    changed |= eg.union(nid, c);
                }
            }
            ENode::Div(a, b) => {
                if eg.class_const(b) == Some(1.0) {
                    changed |= eg.union(nid, a);
                }
                if let (Some(x), Some(y)) = (eg.class_const(a), eg.class_const(b)) {
                    if y != 0.0 {
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
                    let c = eg.add(ENode::Const((-v).to_bits()));
                    changed |= eg.union(nid, c);
                }
            }
            ENode::Binary(op, a, b) => {
                // max(x, x) = x ; min(x, x) = x
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
        ENode::Input(_) | ENode::Param(_) | ENode::Const(_) => 1,
        ENode::Add(..) | ENode::Sub(..) | ENode::Mul(..) => 2,
        ENode::Div(..) => 8,
        ENode::Binary(op, ..) => match op {
            BinaryOp::Max | BinaryOp::Min => 2,
            BinaryOp::Rem => 8,
            BinaryOp::Pow => 16,
        },
        ENode::Unary(op, _) => match op {
            UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Relu => 1,
            UnaryOp::Sqr
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Sign
            | UnaryOp::Step => 2,
            UnaryOp::Sqrt | UnaryOp::Rsqrt | UnaryOp::Recip => 8,
            UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Tanh
            | UnaryOp::Sigmoid
            | UnaryOp::Erf
            | UnaryOp::Gelu
            | UnaryOp::Silu
            | UnaryOp::Sin
            | UnaryOp::Cos => 16,
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
    fn add_zero_is_identity() {
        assert_eq!(opt(input(0) + konst(0.0)), ScalarExpr::Input(0));
        assert_eq!(opt(input(2) - konst(0.0)), ScalarExpr::Input(2));
    }

    #[test]
    fn mul_zero_collapses() {
        assert_eq!(opt(input(0) * konst(0.0)), ScalarExpr::Const(0.0));
    }

    #[test]
    fn constants_fold() {
        assert_eq!(opt(konst(2.0) * konst(3.0)), ScalarExpr::Const(6.0));
        assert_eq!(opt(konst(2.0) + konst(5.0)), ScalarExpr::Const(7.0));
    }

    #[test]
    fn neg_neg_cancels() {
        assert_eq!(optimize(&neg(neg(ScalarExpr::Input(0)))), ScalarExpr::Input(0));
    }

    #[test]
    fn redundant_chain_simplifies_under_an_op() {
        // relu(x*1 + 0) -> relu(x): the identities propagate under the Relu via
        // the shared e-class, and extraction picks the cheapest form.
        let body = (input(0) * konst(1.0) + konst(0.0)).relu();
        assert_eq!(
            optimize(&body.0),
            ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
        );
    }

    #[test]
    fn transcendentals_are_not_const_folded() {
        // exp(1.0) is left symbolic (host-f64 vs device-f32), not folded to a const.
        let e = opt(konst(1.0).exp());
        assert_eq!(e, ScalarExpr::Unary(UnaryOp::Exp, Box::new(ScalarExpr::Const(1.0))));
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
}
