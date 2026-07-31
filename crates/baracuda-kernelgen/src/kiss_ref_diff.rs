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
use crate::oracle::{self, TypedBuffer};
use crate::plan::build_plan;
use crate::recipe::semantics_dag;
use baracuda_kernel_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{
    Combine, FlatDag, IndexRef, Monoid, Node, OobPolicy, RecipeEval, Tensor, eval_recipe,
    scalar_int, tensor_int,
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
    // Preconditions (clearer than a bare `shapes[0]` index-OOB or a silent
    // `.zip` truncation when a caller under-provisions tensors).
    assert!(
        !shapes.is_empty(),
        "{}: eval_recipe_for needs at least the input-0 shape",
        op.name
    );
    assert_eq!(
        inputs.len(),
        shapes.len(),
        "{}: eval_recipe_for got {} input(s) for {} shape(s) — zip would truncate",
        op.name,
        inputs.len(),
        shapes.len()
    );
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

// ===========================================================================
// The prove-then-retire oracle differential (Tasks 5–6). Runs an op through
// BOTH the kernelgen CPU oracle AND kiss-ref, so a complex op (rowreduce,
// matmul) can be shown oracle ≡ kiss-ref before its oracle value self-test
// retires. The `Access::*` arms stay alive (permanent Baracuda plumbing), so
// this keeps working as an ongoing cross-implementation differential.
// ===========================================================================

/// Row-major dense strides for `shape`.
fn dense_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Run `op` over `inputs` through the kernelgen CPU oracle AND kiss-ref (via the
/// converter), returning `(oracle_vals, kiss_ref_vals)` for the caller to
/// compare — bit-exact where representable, tolerance for transcendentals.
/// `in_shapes`/`out_shape` build the operand descriptions (inputs then output,
/// dense row-major); `cat` is the op's structure-key category.
pub(crate) fn oracle_and_kiss_ref(
    op: &OpDef,
    cat: OpCategory,
    in_shapes: &[Vec<i64>],
    out_shape: &[i64],
    inputs: &[Vec<f32>],
    params: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    // Alignment 256 matches the oracle/shape test helpers (`desc`, `od`) so the
    // structure-key classification + plan selection are consistent across the
    // test harnesses (the value the oracle computes is alignment-independent, but
    // this keeps the exercised plan variant identical to the retired tests').
    let mk = |sh: &[i64]| OperandDesc::new(sh.len(), sh, &dense_strides(sh), ElementKind::F32, 256);
    let mut operands: Vec<OperandDesc> = in_shapes.iter().map(|s| mk(s)).collect();
    operands.push(mk(out_shape));
    let key = structure_key(cat, &operands, ArchSku::Sm89);

    // Leg 1: the kernelgen CPU oracle (plan interpreter — no CUDA involved).
    let plan = build_plan(op, &key);
    let images: Vec<TypedBuffer> = inputs
        .iter()
        .zip(in_shapes)
        .map(|(v, sh)| {
            TypedBuffer::new(
                ElementKind::F32,
                sh.clone(),
                dense_strides(sh),
                f32_bytes(v),
            )
        })
        .collect();
    let params_f64: Vec<f64> = params.iter().map(|&p| f64::from(p)).collect();
    let oracle_out = oracle::evaluate(&plan, &operands, &images, &params_f64);
    let oracle_vals = bytes_f32(&oracle_out[0].bytes);

    // Leg 2: kiss-ref via the converter.
    let shapes_usize: Vec<Vec<usize>> = in_shapes
        .iter()
        .map(|s| s.iter().map(|&d| d as usize).collect())
        .collect();
    let r = eval_recipe_for(op, &shapes_usize, inputs, params);
    let ref_vals: Vec<f32> = r.outputs[0].clone().into_data();
    (oracle_vals, ref_vals)
}

/// Tolerance comparator for transcendental ops (softmax/layernorm) where the
/// oracle's f64-domain math and kiss-ref's compute-dtype math agree only to a
/// few ULP, not bit-for-bit. `rel`/`abs` mirror the oracle self-tests' bounds.
pub(crate) fn assert_close(name: &str, reference: &[f32], candidate: &[f32], rel: f32, abs: f32) {
    assert_eq!(reference.len(), candidate.len(), "{name}: length");
    for (i, (r, c)) in reference.iter().zip(candidate).enumerate() {
        if r.is_nan() && c.is_nan() {
            continue;
        }
        let tol = abs + rel * r.abs();
        assert!(
            (r - c).abs() <= tol,
            "{name}: [{i}] reference {r} vs candidate {c} (|Δ|={} > tol={tol})",
            (r - c).abs()
        );
    }
}

// ===========================================================================
// Integer reduction differential (S8/U8 sum/max/min/prod vs kiss-ref):
// bit-exact, wrap + empty-axis. Host-side / i128-backed — a SEMANTICS
// differential, NOT a CUDA run (the on-device leg is a separate later task).
//
// Dependency check (done before writing this section): the pinned dev-dep is
// `kiss-ref-core = "0.1.0"` (Cargo.lock resolves exactly 0.1.0; a 0.2.0 is
// present in the local registry cache but `^0.1.0` never selects it). NEITHER
// 0.1.0 NOR 0.2.0 exposes `eval_recipe_int` (grepped both trees) — the
// `FlatDag`/`Node`/`eval_recipe` machinery `recipe_to_flatdag`/
// `eval_recipe_for` above ride is generic over `ScalarFloat` only, so there is
// no int-flavored recipe DAG to build in the first place. The int lane's
// directly-callable entry point is `tensor_int::reduce` (confirmed present in
// 0.1.0, pinned by that crate's own `int_reduce_sum_wraps` unit test), so
// that is what the kiss-ref leg below calls — the plan's documented fallback.
//
// The "emitter" leg does NOT call `crate::oracle::evaluate`: `oracle.rs`'s
// `eval_reduction` hard-asserts int-accumulate reduction dtypes to
// `I32`/`I64` only (see `oracle.rs` ~1180-1188, `assert!(matches!(plan.dtype,
// ElementKind::I32 | ElementKind::I64), ...)`) — a gate that predates the
// S8/U8 reduction-emitter admission (commit ee66115d) and would panic given
// an S8/U8 plan. Patching that assert is out of this task's staged-file scope
// (only `kiss_ref_diff.rs` + test files). Instead, `emitter_reduce_int` below
// is a standalone reference implementing the documented emitted-kernel
// algorithm directly (per this task's own spec): accumulate UNWRAPPED in
// `i128` (models the CUDA kernel's widened `long long acc` running the WHOLE
// reduced extent before ever truncating — cuda.rs's `long long acc = 0; acc
// += in0[idx];` pattern for S8/U8 sum, same shape for max/min/prod), and only
// the comparator (`assert_int_bits_eq`) applies the final wrap-to-dtype-width
// truncation that the real kernel's `signed/unsigned char*` store performs.
// This is a genuinely independent algorithm from kiss-ref's `tensor_int::
// reduce` (which wraps to width at EVERY combine step, per KISS-OPS-6.2-0002
// — see its own `int_reduce_sum_wraps` doctest: 100+100 wraps mid-fold to
// -56) — the two agree on Sum/Prod because modular addition/multiplication
// are associative (wrap-every-step ≡ wrap-once-at-the-end, mod 2^n), and
// trivially agree on Max/Min (a monoid pick from within already-in-range
// operands never itself goes out of range). Proving that agreement IS the
// differential.
// ===========================================================================

/// `ElementKind` (Baracuda's dtype currency) -> `Dtype` (kiss-ref's), for the
/// four int dtypes this differential covers. `I32`/`I64` are included even
/// though only S8/U8 have dedicated tests below — the whole int lane
/// (`emitter_reduce_int`/`oracle_and_kiss_ref_int`/`assert_int_bits_eq`) is
/// dtype-generic, so I32/I64 fall out "for free" (per the task's "reuse for
/// I32/I64 if trivial").
fn to_kiss_int_dtype(dtype: ElementKind) -> Dtype {
    match dtype {
        ElementKind::S8 => Dtype::S8,
        ElementKind::U8 => Dtype::U8,
        ElementKind::I32 => Dtype::I32,
        ElementKind::I64 => Dtype::I64,
        other => panic!("kiss_ref_diff: no int Dtype mapping for {other:?} (S8/U8/I32/I64 only)"),
    }
}

/// `(bit width, signed)` for `dtype`, sourced from kiss-ref's OWN spec table
/// (`scalar_int::int_spec`) rather than a hand-duplicated copy — so the
/// comparator's notion of "width" can never drift from the same table
/// `tensor_int::reduce`'s internal wrap-to-width uses.
fn int_width(dtype: ElementKind) -> (u32, bool) {
    scalar_int::int_spec(to_kiss_int_dtype(dtype))
        .unwrap_or_else(|| panic!("kiss_ref_diff: no int_spec for {dtype:?}"))
}

/// Row-major unravel of linear index `lin` into per-axis coordinates over
/// `ext` (each axis' extent) — the usize/reduction-shape counterpart of this
/// file's `dense_strides` (which is i64/dense-layout-shaped, for the f32 leg).
fn unravel_coord(mut lin: usize, ext: &[usize]) -> Vec<usize> {
    let mut c = vec![0usize; ext.len()];
    for d in (0..ext.len()).rev() {
        let e = ext[d].max(1);
        c[d] = lin % e;
        lin /= e;
    }
    c
}

/// Row-major linear index of `coord` in a dense buffer shaped `shape`.
fn coord_to_lin(coord: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for d in (0..shape.len()).rev() {
        idx += coord[d] * stride;
        stride *= shape[d];
    }
    idx
}

/// The emitter's documented int-reduction fold (see the section doc above for
/// why this doesn't ride `crate::oracle::evaluate`): fold `axes` of `data`
/// (shaped `shape`) with `monoid`, seeded at the KISS-OPS-6.11-0002 monoid
/// identity (sum=0, prod=1, max=dtype-min, min=dtype-max — so an EMPTY
/// reduced extent yields the identity untouched), accumulating UNWRAPPED in
/// `i128` end-to-end (no intermediate width truncation).
fn emitter_reduce_int(
    data: &[i128],
    shape: &[usize],
    axes: &[usize],
    monoid: Monoid,
    dtype: ElementKind,
) -> Vec<i128> {
    let (bits, signed) = int_width(dtype);
    let ident = match monoid {
        Monoid::Sum => 0i128,
        Monoid::Prod => 1i128,
        Monoid::Max => {
            if signed {
                -(1i128 << (bits - 1))
            } else {
                0
            }
        }
        Monoid::Min => {
            if signed {
                (1i128 << (bits - 1)) - 1
            } else {
                (1i128 << bits) - 1
            }
        }
    };

    let rank = shape.len();
    let mut is_reduced = vec![false; rank];
    for &a in axes {
        assert!(
            a < rank,
            "emitter_reduce_int: axis {a} out of range for rank {rank}"
        );
        is_reduced[a] = true;
    }
    let kept: Vec<usize> = (0..rank).filter(|&d| !is_reduced[d]).collect();
    let reduced: Vec<usize> = (0..rank).filter(|&d| is_reduced[d]).collect();
    let kept_ext: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    let red_ext: Vec<usize> = reduced.iter().map(|&d| shape[d]).collect();
    let n_kept: usize = kept_ext.iter().product();
    let n_red: usize = red_ext.iter().product();

    let mut out = Vec::with_capacity(n_kept);
    for klin in 0..n_kept {
        let kc = unravel_coord(klin, &kept_ext);
        let mut acc = ident;
        for rlin in 0..n_red {
            let rc = unravel_coord(rlin, &red_ext);
            let mut coord = vec![0usize; rank];
            for (j, &a) in kept.iter().enumerate() {
                coord[a] = kc[j];
            }
            for (j, &a) in reduced.iter().enumerate() {
                coord[a] = rc[j];
            }
            let x = data[coord_to_lin(&coord, shape)];
            acc = match monoid {
                Monoid::Sum => acc + x,
                Monoid::Prod => acc * x,
                Monoid::Max => {
                    if x > acc {
                        x
                    } else {
                        acc
                    }
                }
                Monoid::Min => {
                    if x < acc {
                        x
                    } else {
                        acc
                    }
                }
            };
        }
        out.push(acc);
    }
    out
}

/// Run the same int reduction (`monoid` over `axes` of `data`/`shape`,
/// `dtype`-typed) through BOTH legs: [0] the emitter reference
/// (`emitter_reduce_int`, unwrapped i128 accumulate) and [1] kiss-ref
/// (`tensor_int::reduce`, wraps every combine step). Returns
/// `(emitter_vals, kiss_ref_vals)` for the caller to diff — bit-exact via
/// `assert_int_bits_eq` (both legs' raw i128 values are only guaranteed equal
/// AFTER masking to `dtype`'s stored width; see that comparator).
pub(crate) fn oracle_and_kiss_ref_int(
    dtype: ElementKind,
    monoid: Monoid,
    shape: &[usize],
    axes: &[usize],
    data: &[i128],
) -> (Vec<i128>, Vec<i128>) {
    let emitter_vals = emitter_reduce_int(data, shape, axes, monoid, dtype);
    let kdtype = to_kiss_int_dtype(dtype);
    let t = Tensor::from_vec(data.to_vec(), shape).expect("kiss_ref_diff: int tensor build");
    let r = tensor_int::reduce(&t.view(), kdtype, monoid, axes)
        .unwrap_or_else(|e| panic!("kiss_ref_diff: tensor_int::reduce: {e:?}"));
    let kiss_vals: Vec<i128> = r.into_data();
    (emitter_vals, kiss_vals)
}

/// Mask `x` to its low `bits` bits and reinterpret per `signed` — the "actual
/// stored byte value" a `dtype`-width store would hold. Two i128s that differ
/// only above `bits` (e.g. a raw `-1` and a raw `255` at `bits=8`) compare
/// equal: both carry the byte pattern `0xff`, which is `-1` read as S8 or
/// `255` read as U8 — exactly the wrap-to-width identification the emitted
/// kernel's narrow-pointer store performs.
fn wrap_to_width(x: i128, bits: u32, signed: bool) -> i128 {
    if bits >= 128 {
        return x;
    }
    let mask = (1i128 << bits) - 1;
    let masked = x & mask;
    if signed && masked & (1i128 << (bits - 1)) != 0 {
        masked - (1i128 << bits)
    } else {
        masked
    }
}

/// Bit-exact integer comparator: mask both `a` and `b` to `dtype`'s stored
/// width (§6.2-0002 two's-complement wrap) before comparing — no tolerance,
/// this is the KISS int lane (sum/prod/max/min are EXACT-BYTE, not
/// ULP-tolerant).
pub(crate) fn assert_int_bits_eq(a: &[i128], b: &[i128], dtype: ElementKind, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    let (bits, signed) = int_width(dtype);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let wx = wrap_to_width(x, bits, signed);
        let wy = wrap_to_width(y, bits, signed);
        assert_eq!(
            wx, wy,
            "{label}: int divergence at [{i}]: emitter {x} (wrapped {wx}) vs kiss-ref {y} (wrapped {wy})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        BinaryOp, ContractionAxes, OpDef, ReduceOp, ReduceStage, UnaryOp, input, konst, param,
        reduced,
    };
    use baracuda_kernel_vocab::{AxisMask, ElementKind};
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

    // ---- Reduction value guards (Task 3) -----------------------------------
    // Value semantics of the logical-axis folds (dense, exactly-representable so
    // bit-exact despite the OrderInvariant DetClass). Baracuda's physical/int
    // reduction plumbing (if any) stays on the oracle.

    /// Sum over the last axis: [2,3] row sums. Ports oracle.rs `reduction_sum_last_axis`.
    #[test]
    fn reduce_sum_last_axis_through_kiss_ref() {
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        let r = eval_recipe_for(
            &op,
            &[vec![2usize, 3]],
            &[vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[],
        );
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_bits_eq("reduce_sum_last", &[6.0, 15.0], &got);
    }

    /// Max reduction PROPAGATES NaN (a NaN in the run sticks). Ports oracle.rs
    /// `reduction_max_nan_sticks` — pins kiss-ref's reduce-Max as NaN-propagating.
    #[test]
    fn reduce_max_nan_sticks_through_kiss_ref() {
        let op = OpDef::reduction("max", 1, &[ElementKind::F32], input(0), ReduceOp::Max);
        let r = eval_recipe_for(
            &op,
            &[vec![2usize, 3]],
            &[vec![1.0f32, f32::NAN, 3.0, 4.0, 5.0, 6.0]],
            &[],
        );
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        // row 0 has a NaN -> NaN sticks; row 1 -> 6.
        assert_conforming_eq("reduce_max_nan", &[f32::NAN, 6.0], &got);
    }

    /// Mean = sum ÷ reduced_count (the shape-derived divisor). Ports oracle.rs
    /// `reduction_mean_divisor`.
    #[test]
    fn reduce_mean_divisor_through_kiss_ref() {
        let op = OpDef::reduction("mean", 1, &[ElementKind::F32], input(0), ReduceOp::Mean);
        let r = eval_recipe_for(
            &op,
            &[vec![2usize, 4]],
            &[vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0]],
            &[],
        );
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_bits_eq("reduce_mean", &[2.5, 10.0], &got);
    }

    /// Reduce over the OUTER axis (axis 0, a hex-bitmask axes token). Ports
    /// oracle.rs `reduction_outer_axis`.
    #[test]
    fn reduce_outer_axis_through_kiss_ref() {
        let mut mask = AxisMask::EMPTY;
        mask.set(0);
        let op = OpDef::reduction_axes(
            "sum0",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            mask,
            false,
        );
        let r = eval_recipe_for(
            &op,
            &[vec![2usize, 3]],
            &[vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[],
        );
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        // reduce axis 0 -> column sums [1+4, 2+5, 3+6].
        assert_bits_eq("reduce_outer_axis", &[5.0, 7.0, 9.0], &got);
    }

    // ---- Scan value guards (Task 4) ----------------------------------------
    // Forward + exclusive prefix scans (value semantics). REVERSE scan is an
    // emission-level honest miss (flip withdrawn — `semantics_dag` returns None),
    // so kiss-ref can't cover it: oracle.rs
    // `scan_cummax_forward_and_reverse_and_exclusive` STAYS whole.

    /// Forward inclusive cumsum over axis 1. Ports oracle.rs `scan_cumsum_forward`.
    #[test]
    fn scan_cumsum_forward_through_kiss_ref() {
        let op = OpDef::scan_simple(
            "cumsum",
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
        );
        let r = eval_recipe_for(&op, &[vec![1usize, 4]], &[vec![1.0f32, 2.0, 3.0, 4.0]], &[]);
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_bits_eq("cumsum_fwd", &[1.0, 3.0, 6.0, 10.0], &got);
    }

    /// Exclusive cumsum writes the additive identity (0) at the first position.
    /// Ports oracle.rs `scan_cumsum_exclusive_first_pos_identity`.
    #[test]
    fn scan_cumsum_exclusive_through_kiss_ref() {
        let op = OpDef::scan_simple(
            "cumsum_excl",
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            true,
        );
        let r = eval_recipe_for(&op, &[vec![1usize, 4]], &[vec![1.0f32, 2.0, 3.0, 4.0]], &[]);
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_bits_eq("cumsum_excl", &[0.0, 1.0, 3.0, 6.0], &got);
    }

    // ---- Select value guard ------------------------------------------------
    // The select converter path was NOT in the fuzzer's safe palette, so this
    // VERIFIES Baracuda's raw-bit select round-trips through the converter
    // (select(cond,a,b) child order + cmp_gt token vs kiss-ref Op::Select /
    // resolve.rs:58) before its oracle self-test retires. kiss-ref's select impl
    // is byte-identical between 0.1.0 and the strengthened test at 5d0538b.

    /// Raw-bit select preserves -0.0 verbatim: cond=false picks the b-arm
    /// (-0.0), moved bit-for-bit (not arithmetic). Ports oracle.rs
    /// `select_raw_bit_neg_zero_survives`.
    #[test]
    fn select_raw_bit_neg_zero_through_kiss_ref() {
        // cmp_gt(in0, 100) is false for in0 = 1.0 -> select picks arm b = -0.0.
        let body = input(0)
            .binary(BinaryOp::CmpGt, konst(100.0))
            .select(input(0), konst(-0.0));
        let op = OpDef::elementwise("sel", 1, &[ElementKind::F32], body);
        let r = eval_recipe_for(&op, &[vec![1usize]], &[vec![1.0f32]], &[]);
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_bits_eq("select_neg_zero", &[-0.0], &got);
    }

    // ---- Contraction (matmul) prove-then-retire differential (Task 6) -------
    // oracle ≡ kiss-ref for matmul + epilogues over exactly-representable
    // integer cells (bit-exact). Proves the pair agree BEFORE the oracle
    // contraction value self-tests retire; stays as the ongoing differential.

    /// [2,3]·[3,2] identity epilogue, integer cell -> [[58,64],[139,154]]. Ports
    /// oracle.rs `contraction_matmul_identity_epilogue`.
    #[test]
    fn matmul_identity_oracle_eq_kiss_ref() {
        let op = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        let (o, k) = oracle_and_kiss_ref(
            &op,
            OpCategory::Gemm,
            &[vec![2, 3], vec![3, 2]],
            &[2, 2],
            &[
                vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
            ],
            &[],
        );
        assert_bits_eq("matmul_identity_oracle_vs_kiss", &o, &k);
        assert_bits_eq("matmul_identity_expect", &[58.0, 64.0, 139.0, 154.0], &k);
    }

    /// [1,2]·[2,2] with a relu epilogue over the K-sum: raw [-10,2] -> [0,2].
    /// Ports oracle.rs `contraction_matmul_relu_epilogue_over_reduced0`.
    #[test]
    fn matmul_relu_epilogue_oracle_eq_kiss_ref() {
        let op = OpDef::contraction(
            "matmul_relu",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0).unary(UnaryOp::Relu),
        );
        let (o, k) = oracle_and_kiss_ref(
            &op,
            OpCategory::Gemm,
            &[vec![1, 2], vec![2, 2]],
            &[1, 2],
            &[vec![1.0f32, -3.0], vec![2.0f32, 5.0, 4.0, 1.0]],
            &[],
        );
        assert_bits_eq("matmul_relu_oracle_vs_kiss", &o, &k);
        assert_bits_eq("matmul_relu_expect", &[0.0, 2.0], &k);
    }

    // ---- RowReduce prove-then-retire differential (Task 5) ------------------
    // oracle ≈ kiss-ref for softmax / layernorm within tolerance (exp/div/rsqrt
    // are transcendental — bit-exactness isn't expected between the oracle's
    // f64-domain math and kiss-ref's compute-dtype math).

    /// Softmax over [1,4]. Ports oracle.rs `rowreduce_softmax`.
    #[test]
    fn softmax_oracle_eq_kiss_ref() {
        let stages = vec![
            ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Max,
            },
            ReduceStage {
                pre: (input(0) - reduced(0)).unary(UnaryOp::Exp).0,
                op: ReduceOp::Sum,
            },
        ];
        let epi = (input(0) - reduced(0)).unary(UnaryOp::Exp) / reduced(1);
        let op = OpDef::row_reduce("softmax", 1, &[ElementKind::F32], stages, epi);
        let (o, k) = oracle_and_kiss_ref(
            &op,
            OpCategory::Softmax,
            &[vec![1, 4]],
            &[1, 4],
            &[vec![1.0f32, 2.0, 3.0, 4.0]],
            &[],
        );
        assert_close("softmax", &o, &k, 1e-6, 1e-6);
        // Independent invariants on the kiss-ref output (ported from the oracle
        // test, so the differential is equal-or-better than a bare oracle≈kiss
        // agreement): the row sums to 1 and is monotone in the logits.
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row sums to 1, got {sum}");
        assert!(
            k[0] < k[1] && k[1] < k[2] && k[2] < k[3],
            "softmax monotone"
        );
    }

    /// Layernorm over [1,4]. Ports oracle.rs `rowreduce_layernorm`.
    #[test]
    fn layernorm_oracle_eq_kiss_ref() {
        let eps = 1e-5;
        let stages = vec![
            ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Mean,
            },
            ReduceStage {
                pre: (input(0) - reduced(0)).unary(UnaryOp::Sqr).0,
                op: ReduceOp::Mean,
            },
        ];
        let epi = (input(0) - reduced(0)) * (reduced(1) + konst(eps)).unary(UnaryOp::Rsqrt);
        let op = OpDef::row_reduce("layernorm", 1, &[ElementKind::F32], stages, epi);
        let (o, k) = oracle_and_kiss_ref(
            &op,
            OpCategory::Normalization,
            &[vec![1, 4]],
            &[1, 4],
            &[vec![1.0f32, 2.0, 3.0, 4.0]],
            &[],
        );
        assert_close("layernorm", &o, &k, 1e-5, 1e-5);
        // Independent invariants (ported): the hand-computed normalized values
        // (mean 2.5, var 1.25) and zero-mean output.
        let rstd = 1.0f64 / (1.25f64 + eps).sqrt();
        for (i, x) in [1.0f64, 2.0, 3.0, 4.0].iter().enumerate() {
            let expect = ((x - 2.5) * rstd) as f32;
            assert!(
                (k[i] - expect).abs() < 1e-5,
                "layernorm[{i}] {} vs {expect}",
                k[i]
            );
        }
        let mean: f32 = k.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "layernorm zero-mean");
    }

    /// [1,2]·[2,2] + per-column bias + relu: K-sums [13,16] + bias [-100,1] ->
    /// [-87,17] -> relu -> [0,17]. Ports oracle.rs
    /// `contraction_matmul_bias_relu_epilogue`.
    #[test]
    fn matmul_bias_relu_oracle_eq_kiss_ref() {
        let op = OpDef::contraction_bias(
            "matmul_bias_relu",
            &[ElementKind::F32],
            (reduced(0) + input(2)).unary(UnaryOp::Relu),
        );
        let (o, k) = oracle_and_kiss_ref(
            &op,
            OpCategory::Gemm,
            &[vec![1, 2], vec![2, 2], vec![2]],
            &[1, 2],
            &[
                vec![1.0f32, 2.0],
                vec![3.0f32, 4.0, 5.0, 6.0],
                vec![-100.0f32, 1.0],
            ],
            &[],
        );
        assert_bits_eq("matmul_bias_relu_oracle_vs_kiss", &o, &k);
        assert_bits_eq("matmul_bias_relu_expect", &[0.0, 17.0], &k);
    }

    /// Batched [2,1,2]·[2,2,2] -> [2,1,2] identity: batch0 [7,10], batch1
    /// [43,50]. Ports oracle.rs `contraction_batched_matmul_identity_epilogue`.
    #[test]
    fn batched_matmul_oracle_eq_kiss_ref() {
        let op = OpDef::contraction(
            "bmm",
            &[ElementKind::F32],
            ContractionAxes::batched_matmul(),
            reduced(0),
        );
        let (o, k) = oracle_and_kiss_ref(
            &op,
            OpCategory::Gemm,
            &[vec![2, 1, 2], vec![2, 2, 2]],
            &[2, 1, 2],
            &[
                vec![1.0f32, 2.0, 3.0, 4.0],
                vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
            &[],
        );
        assert_bits_eq("batched_matmul_oracle_vs_kiss", &o, &k);
        assert_bits_eq("batched_matmul_expect", &[7.0, 10.0, 43.0, 50.0], &k);
    }

    // ---- Integer reduction differential (Task 4): S8/U8 vs kiss-ref -------
    // Bit-exact (mask-to-width), host-side. Each (dtype, monoid) pair below
    // covers: a normal in-range case, a dtype-boundary edge case (the
    // wrap-around overflow for Sum/Prod — where wrapping is meaningful; the
    // dtype-extreme value for Max/Min — where wrapping can't occur but the
    // asymmetric two's-complement range, e.g. S8's lone -128 with no +128
    // counterpart, is the analogous footgun), and an EMPTY-AXIS case (reduced
    // extent 0 -> the KISS-OPS-6.11-0002 monoid identity).

    #[test]
    fn s8_sum_differential_normal_wrap_and_empty_axis() {
        // Normal: 1+2+3+4 = 10 (no wrap).
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Sum, &[4], &[0], &[1, 2, 3, 4]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_sum_normal");
        assert_int_bits_eq(&[10], &k, ElementKind::S8, "s8_sum_normal_expect");

        // Signed-overflow-wrap: 100 + 100 = 200, wraps to -56 (the exact case
        // kiss-ref's own `tensor_int::int_reduce_sum_wraps` doctest pins).
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Sum, &[2], &[0], &[100, 100]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_sum_wrap");
        assert_int_bits_eq(&[-56], &k, ElementKind::S8, "s8_sum_wrap_expect");

        // Empty axis: reduced extent 0 -> identity 0.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Sum, &[0], &[0], &[]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_sum_empty");
        assert_int_bits_eq(&[0], &k, ElementKind::S8, "s8_sum_empty_expect");
    }

    #[test]
    fn s8_prod_differential_normal_wrap_and_empty_axis() {
        // Normal: 2*3 = 6.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Prod, &[2], &[0], &[2, 3]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_prod_normal");
        assert_int_bits_eq(&[6], &k, ElementKind::S8, "s8_prod_normal_expect");

        // Signed-overflow-wrap: 10*10*10 = 1000; 1000 mod 256 = 232, which as
        // signed 8-bit (>=128) is 232-256 = -24.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Prod, &[3], &[0], &[10, 10, 10]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_prod_wrap");
        assert_int_bits_eq(&[-24], &k, ElementKind::S8, "s8_prod_wrap_expect");

        // Empty axis: identity 1.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Prod, &[0], &[0], &[]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_prod_empty");
        assert_int_bits_eq(&[1], &k, ElementKind::S8, "s8_prod_empty_expect");
    }

    #[test]
    fn s8_max_differential_normal_boundary_and_empty_axis() {
        // Normal.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Max, &[3], &[0], &[-5, 3, -1]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_max_normal");
        assert_int_bits_eq(&[3], &k, ElementKind::S8, "s8_max_normal_expect");

        // Boundary: the dtype minimum -128 (no positive counterpart in two's
        // complement) is present but NOT the max, proving the fold doesn't
        // mis-seed/mis-wrap the asymmetric extreme.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Max, &[3], &[0], &[-128, -100, -50]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_max_boundary");
        assert_int_bits_eq(&[-50], &k, ElementKind::S8, "s8_max_boundary_expect");

        // Empty axis: identity = dtype minimum, -128.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Max, &[0], &[0], &[]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_max_empty");
        assert_int_bits_eq(&[-128], &k, ElementKind::S8, "s8_max_empty_expect");
    }

    #[test]
    fn s8_min_differential_normal_boundary_and_empty_axis() {
        // Normal.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Min, &[3], &[0], &[-5, 3, -1]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_min_normal");
        assert_int_bits_eq(&[-5], &k, ElementKind::S8, "s8_min_normal_expect");

        // Boundary: -128 present and IS the min — the lone-extreme footgun.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Min, &[3], &[0], &[-128, 127, 0]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_min_boundary");
        assert_int_bits_eq(&[-128], &k, ElementKind::S8, "s8_min_boundary_expect");

        // Empty axis: identity = dtype maximum, 127.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::S8, Monoid::Min, &[0], &[0], &[]);
        assert_int_bits_eq(&e, &k, ElementKind::S8, "s8_min_empty");
        assert_int_bits_eq(&[127], &k, ElementKind::S8, "s8_min_empty_expect");
    }

    #[test]
    fn u8_sum_differential_normal_wrap_and_empty_axis() {
        // Normal: 10+20+30 = 60.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Sum, &[3], &[0], &[10, 20, 30]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_sum_normal");
        assert_int_bits_eq(&[60], &k, ElementKind::U8, "u8_sum_normal_expect");

        // Overflow-wrap: 200+100 = 300, wraps (unsigned, no sign bit) to
        // 300 mod 256 = 44 — the same case kiss-ref's own `element_map`
        // U8-wrap doctest exercises via `add`.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Sum, &[2], &[0], &[200, 100]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_sum_wrap");
        assert_int_bits_eq(&[44], &k, ElementKind::U8, "u8_sum_wrap_expect");

        // Empty axis: identity 0.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Sum, &[0], &[0], &[]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_sum_empty");
        assert_int_bits_eq(&[0], &k, ElementKind::U8, "u8_sum_empty_expect");
    }

    #[test]
    fn u8_prod_differential_normal_wrap_and_empty_axis() {
        // Normal: 2*3 = 6.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Prod, &[2], &[0], &[2, 3]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_prod_normal");
        assert_int_bits_eq(&[6], &k, ElementKind::U8, "u8_prod_normal_expect");

        // Overflow-wrap: 10*10*10 = 1000; unsigned 8-bit: 1000 mod 256 = 232
        // (no sign reinterpretation — stays 232, unlike the S8 case above).
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Prod, &[3], &[0], &[10, 10, 10]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_prod_wrap");
        assert_int_bits_eq(&[232], &k, ElementKind::U8, "u8_prod_wrap_expect");

        // Empty axis: identity 1.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Prod, &[0], &[0], &[]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_prod_empty");
        assert_int_bits_eq(&[1], &k, ElementKind::U8, "u8_prod_empty_expect");
    }

    #[test]
    fn u8_max_differential_normal_boundary_and_empty_axis() {
        // Normal.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Max, &[3], &[0], &[10, 250, 5]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_max_normal");
        assert_int_bits_eq(&[250], &k, ElementKind::U8, "u8_max_normal_expect");

        // Boundary: 255 present (the value a buggy signed-compare would
        // misread as -1 and lose to 128).
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Max, &[3], &[0], &[255, 0, 128]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_max_boundary");
        assert_int_bits_eq(&[255], &k, ElementKind::U8, "u8_max_boundary_expect");

        // Empty axis: identity = dtype minimum, 0.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Max, &[0], &[0], &[]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_max_empty");
        assert_int_bits_eq(&[0], &k, ElementKind::U8, "u8_max_empty_expect");
    }

    #[test]
    fn u8_min_differential_normal_boundary_and_empty_axis() {
        // Normal.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Min, &[3], &[0], &[10, 250, 5]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_min_normal");
        assert_int_bits_eq(&[5], &k, ElementKind::U8, "u8_min_normal_expect");

        // Boundary: 0 present and IS the min.
        let (e, k) =
            oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Min, &[3], &[0], &[255, 0, 128]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_min_boundary");
        assert_int_bits_eq(&[0], &k, ElementKind::U8, "u8_min_boundary_expect");

        // Empty axis: identity = dtype maximum, 255.
        let (e, k) = oracle_and_kiss_ref_int(ElementKind::U8, Monoid::Min, &[0], &[0], &[]);
        assert_int_bits_eq(&e, &k, ElementKind::U8, "u8_min_empty");
        assert_int_bits_eq(&[255], &k, ElementKind::U8, "u8_min_empty_expect");
    }

    // ---- I32/I64 fall out "for free" (dtype-generic entry points) ---------
    // Not required by the plan, but a cheap proof the S8/U8 differential
    // machinery genuinely generalizes rather than being S8/U8-special-cased.

    #[test]
    fn i32_sum_differential_wraps_at_32_bits() {
        // i32::MAX + 1 wraps to i32::MIN.
        let (e, k) = oracle_and_kiss_ref_int(
            ElementKind::I32,
            Monoid::Sum,
            &[2],
            &[0],
            &[i128::from(i32::MAX), 1],
        );
        assert_int_bits_eq(&e, &k, ElementKind::I32, "i32_sum_wrap");
        assert_int_bits_eq(
            &[i128::from(i32::MIN)],
            &k,
            ElementKind::I32,
            "i32_sum_wrap_expect",
        );
    }

    // ---- Compute-dtype cmp/select value guard ------------------------------
    // VERIFIES the previously-unproven cmp/select-const converter path before
    // its oracle self-test retires: cmp/select must decide in the COMPUTE dtype
    // (f32), so the const 0.1 narrows into the f32 lane. kiss-ref is monomorphic
    // on the compute dtype (eval_op::<f32>, resolve.rs:67), so it should agree.

    /// `CmpEq(x, 0.1)` at x=0.1f decides in f32 → TRUE (1.0), NOT f64 (0.1f as
    /// f64 != 0.1_f64 → 0.0). The Select condition shares the fix. Ports
    /// oracle.rs `cmp_and_select_decide_in_compute_dtype`.
    #[test]
    fn cmp_select_decide_in_compute_dtype_through_kiss_ref() {
        // (a) bare mask: the const must narrow to f32 before the compare.
        let cmp = OpDef::elementwise(
            "cmpeq",
            1,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpEq, konst(0.1)),
        );
        let r = eval_recipe_for(&cmp, &[vec![1usize]], &[vec![0.1f32]], &[]);
        assert_bits_eq("cmpeq_f32", &[1.0], &r.outputs[0].clone().into_data());

        // (b) the Select condition shares the compute-dtype decision.
        let sel = OpDef::elementwise(
            "selcmp",
            1,
            &[ElementKind::F32],
            input(0)
                .binary(BinaryOp::CmpEq, konst(0.1))
                .select(konst(11.0), konst(22.0)),
        );
        let r = eval_recipe_for(&sel, &[vec![1usize]], &[vec![0.1f32]], &[]);
        assert_bits_eq("selcmp_f32", &[11.0], &r.outputs[0].clone().into_data());
    }
}
