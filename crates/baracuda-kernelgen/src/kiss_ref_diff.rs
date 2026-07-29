//! In-tree Baracuda ↔ kiss-ref differential converter + comparators — the
//! oracle→kiss-ref consolidation's test machinery. Ported verbatim (converter +
//! comparators) from the standalone `tools/kiss-ref-diff` harness, which is the
//! tested reference. The device legs (`device_*`, JIT-on-GPU) stay in that tool
//! — CI has no GPU; this module is the CPU converter + `eval_recipe` leg only.
//!
//! ```text
//! OpDef ──semantics_dag──▶ "reduce[sum,last,nokd](in0)" ──parse──▶ FlatDag
//!   │                                                                │
//!   └────────── (Task 5: kernelgen CPU oracle) ─────┐   ┌─ kiss-ref eval_recipe
//!                                                   ▼   ▼
//!                                          bit / §6.8-conforming compare
//! ```
//!
//! Op-name resolution joins on `kiss_ops_vocab::Op::from_token` — the SAME
//! closed KISS-Ops token set `semantics_dag` re-bases onto (no duplicated name
//! table — the drift the consolidation exists to kill). The module is
//! `#[cfg(test)]`-only (declared `#[cfg(test)] mod kiss_ref_diff;` in `lib.rs`):
//! migrated numerical tests build kiss-ref DAGs from Baracuda `OpDef`s and
//! assert generated behavior against `eval_recipe`, replacing oracle.rs's
//! parallel CPU semantics as the reference.

use std::collections::HashMap;

use crate::ir::{OpDef, ReadIndex, WriteIndex};
use crate::recipe::semantics_dag;
use kiss_ops_vocab::Op;
use kiss_ref_core::{
    Combine, FlatDag, IndexRef, Monoid, Node, OobPolicy, RecipeEval, Tensor, eval_recipe,
};

// ===========================================================================
// The converter: emitted Semantics text -> kiss-ref FlatDag.
// ===========================================================================

/// The §6.11 index-operand mapping. kiss-ref carries integer index tensors in
/// a SEPARATE `indices` slot space (`IndexRef::Slot`), while Baracuda counts
/// them among the op's inputs (`in<k>` in the emitted text). Value-lane binds
/// renumber past the index operands; the converter's harness passes the index
/// operands' data through `indices`, not `inputs`.
struct IndexMap {
    /// Ascending op-input indices that are index operands.
    index_ops: Vec<usize>,
}

impl IndexMap {
    fn from_op(op: &OpDef) -> Self {
        let mut v: Vec<usize> = op
            .read_index
            .iter()
            .filter_map(|r| match r {
                ReadIndex::Indexed { index_operand, .. } => Some(*index_operand as usize),
                ReadIndex::Direct => None,
            })
            .collect();
        if let WriteIndex::ScatterIndexed { index_operand, .. } = &op.write_index {
            v.push(*index_operand as usize);
        }
        v.sort_unstable();
        v.dedup();
        IndexMap { index_ops: v }
    }
    /// kiss-ref value-lane input slot for op input `i`, or Err if `i` is an
    /// index operand (illegal in a value position — the lanes never mix).
    fn value_slot(&self, i: usize) -> Result<usize, String> {
        if self.index_ops.contains(&i) {
            return Err(format!(
                "in{i} is an INDEX operand used in a value position"
            ));
        }
        Ok(i - self.index_ops.iter().filter(|&&k| k < i).count())
    }
    /// kiss-ref `indices` slot for op input `i` (which must be an index operand).
    fn index_slot(&self, i: usize) -> Result<usize, String> {
        self.index_ops
            .iter()
            .position(|&k| k == i)
            .ok_or_else(|| format!("in{i} is not an index operand"))
    }
    /// The number of value-lane inputs the op contributes (the scatter dest,
    /// when synthesized, binds the NEXT slot after these).
    fn n_value_inputs(&self, n_inputs: usize) -> usize {
        n_inputs - self.index_ops.iter().filter(|&&k| k < n_inputs).count()
    }
}

struct DagBuilder {
    nodes: Vec<Node>,
    /// Structural dedup: canonical subexpression text -> node id. Turns the
    /// tree-shaped text back into a DAG (repeated `in0` etc. intern once).
    memo: HashMap<String, usize>,
    /// Iteration/input rank — resolves the rank-relative `last` axes token
    /// (the same §6.7-0005-style sentinel discipline as the key codec).
    rank: usize,
    /// The §6.11 index-operand mapping (empty for non-indexed ops).
    imap: IndexMap,
    /// Total op inputs (for the synthesized scatter-dest slot).
    n_inputs: usize,
}

impl DagBuilder {
    fn push(&mut self, key: String, node: Node) -> usize {
        if let Some(&id) = self.memo.get(&key) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(node);
        self.memo.insert(key, id);
        id
    }
}

/// Split `s` on top-level commas (depth-0 w.r.t. parentheses AND brackets).
fn split_args(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() {
        out.push(last);
    }
    out
}

fn parse_const(v: &str) -> Result<f64, String> {
    match v {
        "nan" => Ok(f64::NAN),
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        _ => v.parse().map_err(|e| format!("const `{v}`: {e}")),
    }
}

fn parse_monoid(s: &str) -> Result<Monoid, String> {
    Ok(match s {
        "sum" => Monoid::Sum,
        "prod" => Monoid::Prod,
        "max" => Monoid::Max,
        "min" => Monoid::Min,
        _ => return Err(format!("unknown monoid `{s}`")),
    })
}

fn parse_oob(s: &str) -> Result<OobPolicy, String> {
    Ok(match s {
        "skip" => OobPolicy::Skip,
        "clamp" => OobPolicy::Clamp,
        "zero_fill" => OobPolicy::ZeroFill,
        _ => return Err(format!("unknown oob policy `{s}`")),
    })
}

/// A bare `in<k>` (the index-operand position of gather/scatter — never an
/// expression: the index lane is data, not computation).
fn parse_bare_bind(s: &str) -> Result<usize, String> {
    s.trim()
        .strip_prefix("in")
        .filter(|r| !r.is_empty() && r.bytes().all(|c| c.is_ascii_digit()))
        .ok_or_else(|| format!("index position must be a bare in<k>, got `{s}`"))?
        .parse()
        .map_err(|e| format!("bind `{s}`: {e}"))
}

/// Resolve a `reduce`/`reduced_count` axes token: `last` (rank-relative) or a
/// raw `0x<hex>` bitmask -> sorted axis indices.
fn parse_axes(s: &str, rank: usize) -> Result<Vec<usize>, String> {
    if s == "last" {
        if rank == 0 {
            return Err("`last` on a rank-0 space".into());
        }
        return Ok(vec![rank - 1]);
    }
    let hex = s
        .strip_prefix("0x")
        .ok_or_else(|| format!("axes token `{s}`"))?;
    let mask = u8::from_str_radix(hex, 16).map_err(|e| format!("axes `{s}`: {e}"))?;
    Ok((0..8).filter(|b| mask >> b & 1 == 1).collect())
}

/// Parse one expression of the emitted functional Semantics text into `b`,
/// returning its node id. Grammar: leaves `in<i>` / `const(v)` /
/// `runtime_scalar(slot)` / `iota(axis)` / `reduced_count(axes)`; calls
/// `name(args…)` via `Op::from_token`; fold nodes `name[attrs](args…)`.
fn parse_expr(s: &str, b: &mut DagBuilder) -> Result<usize, String> {
    let s = s.trim();
    // Leaf: `in<i>` (the §6.4-0009 Bind) — renumbered past index operands
    // (kiss-ref's value lane excludes them; they ride `indices`).
    if let Some(rest) = s.strip_prefix("in") {
        if !rest.is_empty() && rest.bytes().all(|c| c.is_ascii_digit()) {
            let i: usize = rest.parse().map_err(|e| format!("bind `{s}`: {e}"))?;
            let slot = b.imap.value_slot(i)?;
            return Ok(b.push(s.to_string(), Node::Bind(slot)));
        }
    }
    // Call: `head(args…)` where head is `name` or `name[attr,…]`.
    let open = s
        .find('(')
        .ok_or_else(|| format!("unrecognized leaf `{s}`"))?;
    if !s.ends_with(')') {
        return Err(format!("unbalanced call `{s}`"));
    }
    let (head, inner) = (&s[..open], &s[open + 1..s.len() - 1]);
    let (name, attrs): (&str, Vec<&str>) = match head.find('[') {
        None => (head, Vec::new()),
        Some(bo) => {
            if !head.ends_with(']') {
                return Err(format!("unbalanced attrs `{head}`"));
            }
            (&head[..bo], split_args(&head[bo + 1..head.len() - 1]))
        }
    };
    let args = split_args(inner);
    let rank = b.rank;
    let node = match name {
        "const" => {
            if args.len() != 1 {
                return Err(format!("const arity: `{s}`"));
            }
            Node::Const(parse_const(args[0])?)
        }
        "runtime_scalar" => {
            if args.len() != 1 {
                return Err(format!("runtime_scalar arity: `{s}`"));
            }
            Node::RuntimeScalar(args[0].parse().map_err(|e| format!("slot `{s}`: {e}"))?)
        }
        // `iota(axis)`: kiss-ref's Iota is shape-of-`like`; the elementwise
        // iteration shape is the (broadcast) input-0 shape, so `like` = Bind(0).
        "iota" => {
            if args.len() != 1 {
                return Err(format!("iota arity: `{s}`"));
            }
            let like = b.push("in0".to_string(), Node::Bind(0));
            Node::Iota {
                like,
                axis: args[0].parse().map_err(|e| format!("axis `{s}`: {e}"))?,
            }
        }
        // `reduced_count(<axes>)` — the shape-derived reduced-extent leaf
        // (float Mean's divisor; NOT a literal const).
        "reduced_count" => {
            if args.len() != 1 {
                return Err(format!("reduced_count arity: `{s}`"));
            }
            Node::ReducedCount(parse_axes(args[0], rank)?)
        }
        // `reduce[<monoid>,<axes>,<kd|nokd>](child)`.
        "reduce" => {
            if attrs.len() != 3 || args.len() != 1 {
                return Err(format!("reduce shape: `{s}`"));
            }
            let keepdim = match attrs[2] {
                "kd" => true,
                "nokd" => false,
                other => return Err(format!("keepdim token `{other}`")),
            };
            Node::Reduce {
                monoid: parse_monoid(attrs[0])?,
                axes: parse_axes(attrs[1], rank)?,
                keepdim,
                child: parse_expr(args[0], b)?,
            }
        }
        // `prefix_scan[<monoid>,<axis>,<excl|incl>](child)`.
        "prefix_scan" => {
            if attrs.len() != 3 || args.len() != 1 {
                return Err(format!("prefix_scan shape: `{s}`"));
            }
            let exclusive = match attrs[2] {
                "excl" => true,
                "incl" => false,
                other => return Err(format!("exclusive token `{other}`")),
            };
            Node::PrefixScan {
                monoid: parse_monoid(attrs[0])?,
                axis: attrs[1].parse().map_err(|e| format!("axis `{s}`: {e}"))?,
                exclusive,
                child: parse_expr(args[0], b)?,
            }
        }
        // `matmul[<roles>](lhs, rhs)` — the canonical rank-2 / batched rank-3
        // role vectors only (kiss-ref's Matmul node shape).
        "matmul" => {
            if attrs.len() != 1 || args.len() != 2 {
                return Err(format!("matmul shape: `{s}`"));
            }
            if attrs[0] != "mk.kn" && attrs[0] != "bmk.bkn" {
                return Err(format!(
                    "matmul roles `{}` not the canonical cell",
                    attrs[0]
                ));
            }
            Node::Matmul {
                lhs: parse_expr(args[0], b)?,
                rhs: parse_expr(args[1], b)?,
            }
        }
        // `gather[<axis>,<oob>,<index_dtype>](data, in<k>)` — child order
        // data-then-index (Fuel's pinned child_edges). The index child maps to
        // `IndexRef::Slot` (kiss-ref's separate index-lane slot space);
        // `base: None` + Skip is RULED LEGAL (Gap 1 = option 1, DYNAMIC
        // base requirement, 2026-07-23): the in-place skip-gather stands
        // as-is; only an ACTUAL OOB read without a base is the typed error
        // (kiss-ref Error::GatherSkipNoBase). Differentials keep gather
        // indices in-range or non-Skip so the error path stays untraveled.
        "gather" => {
            if attrs.len() != 3 || args.len() != 2 {
                return Err(format!("gather shape: `{s}`"));
            }
            if !["u32", "i32", "i64"].contains(&attrs[2]) {
                return Err(format!("gather index_dtype `{}`", attrs[2]));
            }
            let data = parse_expr(args[0], b)?;
            let slot = b.imap.index_slot(parse_bare_bind(args[1])?)?;
            Node::Gather {
                data,
                index: IndexRef::Slot(slot),
                axis: attrs[0].parse().map_err(|e| format!("axis `{s}`: {e}"))?,
                oob: parse_oob(attrs[1])?,
                base: None,
            }
        }
        // `scatter[<axis>,<combine>,<oob>,<index_dtype>](value, in<k>)`.
        // Baracuda's scatter is IN-PLACE (dest = the output buffer, implicit);
        // the pure kiss-ref form needs an explicit dest node — the converter
        // synthesizes a Bind of the NEXT value slot and the harness supplies
        // the initial-contents tensor there (zeros for the oracle-equivalent
        // form). This is exactly the §6.11 explicit-dest RFC surface, applied
        // converter-side pending the KISS ruling.
        "scatter" => {
            if attrs.len() != 4 || args.len() != 2 {
                return Err(format!("scatter shape: `{s}`"));
            }
            let combine = match attrs[1] {
                "atomic-add" => Combine::AtomicAdd,
                other => return Err(format!("scatter combine `{other}` not in the floor")),
            };
            // Baracuda scatter kernels are Skip-only, and kiss-ref PINS
            // skipped OOB writes (§6.11-0005) — the attr is validated, then
            // carried implicitly.
            if attrs[2] != "skip" {
                return Err(format!(
                    "scatter oob `{}` (kernels are Skip-only)",
                    attrs[2]
                ));
            }
            if !["u32", "i32", "i64"].contains(&attrs[3]) {
                return Err(format!("scatter index_dtype `{}`", attrs[3]));
            }
            let updates = parse_expr(args[0], b)?;
            let slot = b.imap.index_slot(parse_bare_bind(args[1])?)?;
            let dest_slot = b.imap.n_value_inputs(b.n_inputs);
            let dest = b.push(format!("__dest{dest_slot}"), Node::Bind(dest_slot));
            Node::Scatter {
                dest,
                index: IndexRef::Slot(slot),
                updates,
                axis: attrs[0].parse().map_err(|e| format!("axis `{s}`: {e}"))?,
                combine,
            }
        }
        // `flip[<axis>](x)` — kiss-ref `Node::Flip`: reverse along axis, a
        // raw-bit move, ExactByte. NOTE the asymmetry is deliberate: the
        // CONVERTER can consume flip text, but Baracuda's EMITTER keeps reverse
        // scans an honest miss until `flip` REGISTERS in the KISS-Ops closed set
        // via the #67 grammar row (the #68 anti-fork witness gates on the
        // registry, not on kiss-ref's node set).
        "flip" => {
            if attrs.len() != 1 || args.len() != 1 {
                return Err(format!("flip shape: `{s}`"));
            }
            Node::Flip {
                child: parse_expr(args[0], b)?,
                axis: attrs[0].parse().map_err(|e| format!("axis `{s}`: {e}"))?,
            }
        }
        _ if !attrs.is_empty() => {
            return Err(format!("unknown bracketed node `{name}[…]`"));
        }
        _ => {
            // The vocabulary join: the SAME closed KISS-Ops token set the
            // emitter re-based onto. An unknown token is a typed error.
            let op =
                Op::from_token(name).ok_or_else(|| format!("`{name}` is not a KISS-Ops token"))?;
            let children = args
                .iter()
                .map(|a| parse_expr(a, b))
                .collect::<Result<Vec<_>, _>>()?;
            Node::Apply { op, children }
        }
    };
    Ok(b.push(s.to_string(), node))
}

/// The emitted Semantics text of `op` -> a kiss-ref `FlatDag` (value lane).
/// `rank` = the input-0 rank (resolves the rank-relative `last` sentinel).
pub(crate) fn recipe_to_flatdag(op: &OpDef, rank: usize) -> Result<FlatDag, String> {
    let text = semantics_dag(op).ok_or("op has no recipe (honest miss)")?;
    let mut b = DagBuilder {
        nodes: Vec::new(),
        memo: HashMap::new(),
        rank,
        imap: IndexMap::from_op(op),
        n_inputs: op.n_inputs as usize,
    };
    let root = parse_expr(&text, &mut b)?;
    Ok(FlatDag::new(b.nodes, vec![root]))
}

// ===========================================================================
// The kiss-ref leg + the comparators (the migration's assertion surface).
// ===========================================================================

/// The kiss-ref leg: emitted recipe -> converter -> `eval_recipe`. `shapes` are
/// the value-lane input shapes (input-0 first); `params` feeds `runtime_scalar`
/// slots. Index operands are out of scope here (they ride kiss-ref's separate
/// `indices` lane — the gather/scatter differentials supply them directly).
pub(crate) fn eval_recipe_for(
    op: &OpDef,
    shapes: &[Vec<usize>],
    inputs: &[Vec<f32>],
    params: &[f32],
) -> RecipeEval<f32> {
    let rank = shapes[0].len();
    let dag = recipe_to_flatdag(op, rank).unwrap_or_else(|e| panic!("{}: converter: {e}", op.name));
    let tensors: Vec<Tensor<f32>> = inputs
        .iter()
        .zip(shapes)
        .map(|(v, sh)| Tensor::from_vec(v.clone(), sh).expect("tensor"))
        .collect();
    eval_recipe(&dag, &tensors, params, &[])
        .unwrap_or_else(|e| panic!("{}: eval_recipe: {e:?}", op.name))
}

/// Bit-exact comparator: every lane's raw f32 bit pattern must match. Use where
/// the values are exactly representable (integer-valued, or a fold order that
/// cannot reorder) so byte equality is the correct assertion.
pub(crate) fn assert_bits_eq(name: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len(), "{name}: length");
    for (i, (a, b)) in reference.iter().zip(candidate).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{name}: bit divergence at [{i}]: reference {a:?} (0x{:08x}) vs candidate {b:?} (0x{:08x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

/// The KISS-Conform §6.8 exact comparator: bit-identical, EXCEPT both-NaN is
/// conforming (0 distance) — IEEE 754 leaves NaN payload propagation optional,
/// and hardware genuinely differs (an sm_89 float add canonicalizes a produced
/// NaN to `0x7fffffff`; x86 propagates the input's `0x7fc00000`). Mirrors
/// kiss-ref's `ulp_distance_*` pin: both-NaN → 0, one-NaN → MAX, ±0 → 1 (the
/// signed-zero distinction — exactly what caught the max_prop tie bug — is
/// PRESERVED; only the NaN payload/sign is classed).
pub(crate) fn assert_conforming_eq(name: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len(), "{name}: length");
    for (i, (r, c)) in reference.iter().zip(candidate).enumerate() {
        if r.is_nan() && c.is_nan() {
            continue; // both-NaN: conforming (payload unpinned across hardware)
        }
        assert_eq!(
            r.to_bits(),
            c.to_bits(),
            "{name}: divergence at [{i}]: reference {r:?} (0x{:08x}) vs candidate {c:?} (0x{:08x})",
            r.to_bits(),
            c.to_bits()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, OpDef, input, param};
    use baracuda_kernel_vocab::ElementKind;
    use kiss_ref_core::DetClass;

    /// relu(add(in0, in1)) built as a raw kiss-ref FlatDag (no converter) — the
    /// Task-0 smoke test, folded in here: proves the dev-deps resolve and
    /// `eval_recipe` runs, and pins the tokens `add`/`relu` resolving via
    /// `Op::from_token`.
    #[test]
    fn relu_add_direct_flatdag_evaluates() {
        let add = Op::from_token("add").expect("`add` is a KISS-Ops token");
        let relu = Op::from_token("relu").expect("`relu` is a KISS-Ops token");
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Bind(1),
                Node::Apply {
                    op: add,
                    children: vec![0, 1],
                },
                Node::Apply {
                    op: relu,
                    children: vec![2],
                },
            ],
            vec![3],
        );
        let a = Tensor::from_vec(vec![-1.0f32, 2.0, -3.0], &[3]).expect("tensor a");
        let b = Tensor::from_vec(vec![0.5f32, -0.5, 1.0], &[3]).expect("tensor b");
        let r = eval_recipe(&dag, &[a, b], &[], &[]).expect("eval_recipe");
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_eq!(got, vec![0.0, 1.5, 0.0]);
        assert!(r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)));
    }

    /// The converter's own end-to-end round-trip: a REAL Baracuda `OpDef` whose
    /// emitted recipe text is parsed back into a FlatDag and evaluated. The
    /// values are exactly representable, so the assertion is bit-exact.
    #[test]
    fn relu_add_opdef_roundtrips_through_converter_bit_exact() {
        let op = OpDef::elementwise(
            "relu_add",
            2,
            &[ElementKind::F32],
            (input(0) + input(1)).relu(),
        );
        let shapes = [vec![3usize], vec![3usize]];
        let a = vec![-1.0f32, 2.0, -3.0];
        let b = vec![0.5f32, -0.5, 1.0];
        let r = eval_recipe_for(&op, &shapes, &[a, b], &[]);
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        // add: [-0.5, 1.5, -2.0]; relu clamps the negatives to +0.0.
        assert_bits_eq("relu_add", &[0.0, 1.5, 0.0], &got);
        assert!(r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)));
    }

    /// A NaN rides through the converter: relu(add(NaN, x)) = NaN. Compared with
    /// the §6.8 conforming comparator (both-NaN is a class match, not a payload
    /// match) — the comparator the migrated OrderInvariant/device legs rely on.
    #[test]
    fn relu_add_nan_lane_is_conforming() {
        let op = OpDef::elementwise(
            "relu_add",
            2,
            &[ElementKind::F32],
            (input(0) + input(1)).relu(),
        );
        let shapes = [vec![3usize], vec![3usize]];
        let a = vec![f32::NAN, 2.0, -3.0];
        let b = vec![0.5f32, -0.5, 1.0];
        let r = eval_recipe_for(&op, &shapes, &[a, b], &[]);
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        // lane 0: relu(NaN + 0.5) = NaN (payload unpinned); 1: 1.5; 2: 0.0.
        assert_conforming_eq("relu_add_nan", &[f32::NAN, 1.5, 0.0], &got);
    }

    // ---- Ported float-value guards (oracle→kiss-ref consolidation, Task 2) --
    // These are the equal-or-better replacements that must exist HERE before the
    // matching oracle.rs elementwise self-tests are retired: kiss-ref is now the
    // float value-semantics reference. Each pins a bit-visible edge the oracle
    // test pinned, now asserted against kiss-ref via the converter.

    /// relu's signed-zero + NaN edges: relu(-0.0) = -0.0 (NOT max(x,0)), relu(NaN)
    /// = NaN. The §6.8 comparator keeps -0.0 bit-exact (signed zero is a value
    /// distinction; only the NaN payload is classed). Ports oracle.rs
    /// `elementwise_relu_neg_zero_and_nan`.
    #[test]
    fn relu_signed_zero_and_nan_through_kiss_ref() {
        let op = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu());
        let r = eval_recipe_for(
            &op,
            &[vec![4usize]],
            &[vec![-3.0f32, -0.0, 2.0, f32::NAN]],
            &[],
        );
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        // -3 -> +0.0; -0.0 -> -0.0 (sign preserved); 2 -> 2; NaN -> NaN.
        assert_conforming_eq("relu_edges", &[0.0, -0.0, 2.0, f32::NAN], &got);
    }

    /// Signed-zero add: +0.0 + -0.0 = +0.0, -0.0 + -0.0 = -0.0 (IEEE). Ports
    /// oracle.rs `probe_classes_add_bit_exact_zeros`.
    #[test]
    fn signed_zero_add_through_kiss_ref() {
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        let r = eval_recipe_for(
            &op,
            &[vec![2usize], vec![2usize]],
            &[vec![0.0f32, -0.0], vec![-0.0f32, -0.0]],
            &[],
        );
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_bits_eq("signed_zero_add", &[0.0, -0.0], &got);
    }

    /// `max_prop`/`min_prop` keep operand A on numeric ties — bit-visible only on
    /// signed-zero ties: max_prop(-0,+0) = -0, max_prop(+0,-0) = +0 (never
    /// order-dependent-on-b). The a-on-ties spelling the kiss-ref differential
    /// caught (a b-on-ties spelling diverged). Ports oracle.rs
    /// `elementwise_maxmin_prop_signed_zero_ties_keep_a`.
    #[test]
    fn max_prop_min_prop_signed_zero_ties_keep_a() {
        let a = vec![-0.0f32, 0.0]; // a-operand: (-0, +0)
        let b = vec![0.0f32, -0.0]; // b-operand: (+0, -0)
        let shapes = [vec![2usize], vec![2usize]];
        let maxp = OpDef::elementwise(
            "maxp",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::Max, input(1)),
        );
        let rmax = eval_recipe_for(&maxp, &shapes, &[a.clone(), b.clone()], &[]);
        assert_bits_eq(
            "max_prop_ties",
            &[-0.0, 0.0],
            &rmax.outputs[0].clone().into_data(),
        );
        let minp = OpDef::elementwise(
            "minp",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::Min, input(1)),
        );
        let rmin = eval_recipe_for(&minp, &shapes, &[a, b], &[]);
        assert_bits_eq(
            "min_prop_ties",
            &[-0.0, 0.0],
            &rmin.outputs[0].clone().into_data(),
        );
    }

    /// Affine through `runtime_scalar` params: in*p0 + p1 — exercises the
    /// converter's `runtime_scalar` node + `eval_recipe`'s param slots. Ports
    /// oracle.rs `elementwise_affine_with_params`.
    #[test]
    fn affine_with_params_through_kiss_ref() {
        let op = OpDef::elementwise(
            "affine",
            1,
            &[ElementKind::F32],
            input(0) * param(0) + param(1),
        );
        let r = eval_recipe_for(&op, &[vec![3usize]], &[vec![1.0f32, 2.0, 3.0]], &[2.0, 0.5]);
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_bits_eq("affine", &[2.5, 4.5, 6.5], &got);
    }

    /// Dense add incl. INF and signed-zero lanes: 1+0=1, -0+0=+0, 2.5+(-1.5)=1,
    /// INF+1=INF. Ports oracle.rs `elementwise_add_contiguous`.
    #[test]
    fn add_contiguous_through_kiss_ref() {
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        let r = eval_recipe_for(
            &op,
            &[vec![4usize], vec![4usize]],
            &[
                vec![1.0f32, -0.0, 2.5, f32::INFINITY],
                vec![0.0f32, 0.0, -1.5, 1.0],
            ],
            &[],
        );
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_bits_eq("add_contiguous", &[1.0, 0.0, 1.0, f32::INFINITY], &got);
    }
}
