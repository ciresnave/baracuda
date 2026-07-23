//! Baracuda ↔ kiss-ref differential PoC — **step 2 (elementwise) + 2b (folds)**:
//! the converter from Baracuda's REAL emitted recipe surface
//! (`baracuda_kernelgen::recipe::semantics_dag` — the KISS-Contract §2.3
//! Semantics text every contract carries) into kiss-ref's `FlatDag`,
//! differentially evaluated:
//!
//! ```text
//! OpDef ──semantics_dag──▶ "reduce[sum,last,nokd](in0)" ──parse──▶ FlatDag
//!   │                                                                │
//!   └───────── kernelgen CPU oracle ────────┐   ┌── kiss-ref eval_recipe
//!                                           ▼   ▼
//!                                bit-exact compare (+ DetClass check)
//! ```
//!
//! Op-name resolution joins on `kiss_ops_vocab::Op::from_token` — the same
//! closed KISS-Ops token set `semantics_dag` re-bases onto: NO duplicated
//! name table (the drift the consolidation exists to kill). Targets the
//! kiss-ref 2026-07-23 lock packet @ `004e1a4`.
//!
//! Step-2b surface: `reduce[<monoid>,<axes>,<kd>]`, `prefix_scan[<monoid>,
//! <axis>,<excl>]`, `matmul[<roles>]`, the `reduced_count(<axes>)` leaf
//! (float Mean = sum-fold ÷ reduced_count). `flip[…]` (reverse scans) is a
//! typed converter miss — kiss-ref v1 has no flip node. Gather/scatter with
//! `IndexRef` are step 2c.
//!
//! Fold differentials use exactly-representable (integer-valued) floats so
//! any fold order/accumulator width gives identical bits — VALUE equality is
//! then assertable even where the honest DetClass says order-nondeterministic
//! (the class-vs-value separation the production comparator keys on).

use std::collections::HashMap;

use baracuda_kernel_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use baracuda_kernelgen::ir::{
    BinaryOp, ContractionAxes, OpDef, ReadIndex, ReduceOp, WriteIndex, input, konst, param,
    reduced,
};
use baracuda_kernelgen::oracle::{self, TypedBuffer};
use baracuda_kernelgen::plan::build_plan;
use baracuda_kernelgen::recipe::semantics_dag;
use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{
    eval_recipe, Combine, DetClass, FlatDag, IndexRef, IndexTensor, Monoid, Node, OobPolicy,
    RecipeEval, Tensor,
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
            return Err(format!("in{i} is an INDEX operand used in a value position"));
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
    let open = s.find('(').ok_or_else(|| format!("unrecognized leaf `{s}`"))?;
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
                return Err(format!("matmul roles `{}` not the canonical cell", attrs[0]));
            }
            Node::Matmul {
                lhs: parse_expr(args[0], b)?,
                rhs: parse_expr(args[1], b)?,
            }
        }
        // `gather[<axis>,<oob>,<index_dtype>](data, in<k>)` — child order
        // data-then-index (Fuel's pinned child_edges). The index child maps to
        // `IndexRef::Slot` (kiss-ref's separate index-lane slot space);
        // `base: None` = the pre-§6.11-ruling semantics (Skip on an
        // actually-OOB read stays kiss-ref's typed error — the seam is
        // Provisional, so the differentials keep gather indices in-range or
        // non-Skip).
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
                return Err(format!("scatter oob `{}` (kernels are Skip-only)", attrs[2]));
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
        // Reverse scans emit `flip[axis](…)` — kiss-ref v1 has no flip node:
        // a TYPED converter miss, never a silent mis-encode.
        "flip" => return Err("`flip` has no kiss-ref v1 node (reverse-scan miss)".into()),
        _ if !attrs.is_empty() => {
            return Err(format!("unknown bracketed node `{name}[…]`"));
        }
        _ => {
            // The vocabulary join: the SAME closed KISS-Ops token set the
            // emitter re-based onto. An unknown token is a typed error.
            let op = Op::from_token(name)
                .ok_or_else(|| format!("`{name}` is not a KISS-Ops token"))?;
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
fn recipe_to_flatdag(op: &OpDef, rank: usize) -> Result<FlatDag, String> {
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
// Step 3b (partial) — the ON-DEVICE FOLD leg: the block-tree last-axis reduce
// kernel `(in0, out, n_out, k)` (grid-stride rows, per-row tree; block must be
// a multiple of 32 for the warp shuffles). With exactly-representable values
// the honest OrderInvariantNondeterministic class COLLAPSES to bit-stable
// (kiss-ref's tolerance-basis preview: OIN cells are value-set-conditioned);
// the general-value tolerance leg joins when kiss-ref's fold-depth bound
// arrives.
// ===========================================================================

fn device_reduce_leg(
    dc: &DeviceCtx,
    op: &OpDef,
    rows: i64,
    cols: i64,
    input: &[f32],
) -> Vec<f32> {
    use baracuda_driver::{DeviceBuffer, Module, Stream};
    let ind = OperandDesc::new(2, &[rows, cols], &[cols, 1], ElementKind::F32, 16);
    let outd = OperandDesc::new(1, &[rows], &[1], ElementKind::F32, 16);
    let key = structure_key(OpCategory::Reduction, &[ind, outd], ArchSku::Sm89);
    let kernel = baracuda_kernelgen::generate(op, &key, &baracuda_kernelgen::Cuda);
    if std::env::var_os("POC_DUMP").is_some() {
        eprintln!("--- {} ({}) ---\n{}", op.name, kernel.name, kernel.source);
    }
    let ptx = baracuda_nvrtc::Program::compile(&kernel.source, "gen_reduce.cu", &[&dc.arch_flag])
        .unwrap_or_else(|e| panic!("{}: nvrtc: {e}\n{}", op.name, kernel.source));
    let stream = Stream::new(&dc.ctx).expect("stream");
    let module = Module::load_ptx(&dc.ctx, &ptx).expect("load_ptx");
    let func = module.get_function(&kernel.name).expect("get_function");
    let d_in = DeviceBuffer::from_slice(&dc.ctx, input).expect("upload");
    let d_out: DeviceBuffer<f32> = DeviceBuffer::new(&dc.ctx, rows as usize).expect("out");
    let (in_ptr, out_ptr) = (d_in.as_raw(), d_out.as_raw());
    // SAFETY: matches the generated `(const float* in0, float* out,
    // long long n_out, long long k)` signature; block = 64 (multiple of 32).
    unsafe {
        func.launch()
            .grid((rows as u32, 1, 1))
            .block((64, 1, 1))
            .stream(&stream)
            .arg(&in_ptr)
            .arg(&out_ptr)
            .arg(&rows)
            .arg(&cols)
            .launch()
            .expect("launch");
    }
    stream.synchronize().expect("sync");
    let mut host = vec![0.0f32; rows as usize];
    d_out.copy_to_host(&mut host).expect("copy back");
    host
}

// ===========================================================================
// The differential harness.
// ===========================================================================

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn dense_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// kiss-ref leg: emitted recipe -> converter -> eval_recipe.
fn kiss_ref_leg(
    op: &OpDef,
    shapes: &[Vec<usize>],
    inputs: &[Vec<f32>],
    params: &[f32],
) -> (FlatDag, RecipeEval<f32>) {
    let rank = shapes[0].len();
    let dag = recipe_to_flatdag(op, rank).unwrap_or_else(|e| panic!("{}: converter: {e}", op.name));
    let tensors: Vec<Tensor<f32>> = inputs
        .iter()
        .zip(shapes)
        .map(|(v, sh)| Tensor::from_vec(v.clone(), sh).expect("tensor"))
        .collect();
    let r = eval_recipe(&dag, &tensors, params, &[])
        .unwrap_or_else(|e| panic!("{}: eval_recipe: {e:?}", op.name));
    (dag, r)
}

fn assert_bits_eq(name: &str, oracle_vals: &[f32], ref_vals: &[f32]) {
    assert_eq!(oracle_vals.len(), ref_vals.len(), "{name}: length");
    for (i, (a, b)) in oracle_vals.iter().zip(ref_vals).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{name}: bit divergence at [{i}]: oracle {a:?} (0x{:08x}) vs kiss-ref {b:?} (0x{:08x})",
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
fn assert_conforming_eq(name: &str, reference: &[f32], candidate: &[f32]) {
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

/// Differential vs the kernelgen CPU oracle. `in_shapes`/`out_shape` build the
/// operand descriptions (inputs then output, dense row-major).
fn diff_vs_oracle(
    op: &OpDef,
    cat: OpCategory,
    in_shapes: &[Vec<i64>],
    out_shape: &[i64],
    inputs: &[Vec<f32>],
    params: &[f32],
) -> (FlatDag, RecipeEval<f32>) {
    let mk = |sh: &[i64]| OperandDesc::new(sh.len(), sh, &dense_strides(sh), ElementKind::F32, 16);
    let mut operands: Vec<OperandDesc> = in_shapes.iter().map(|s| mk(s)).collect();
    operands.push(mk(out_shape));
    let key = structure_key(cat, &operands, ArchSku::Sm89);

    // Leg 1: the kernelgen CPU oracle (plan interpreter — no CUDA involved).
    let plan = build_plan(op, &key);
    let images: Vec<TypedBuffer> = inputs
        .iter()
        .zip(in_shapes)
        .map(|(v, sh)| TypedBuffer::new(ElementKind::F32, sh.clone(), dense_strides(sh), f32_bytes(v)))
        .collect();
    let params_f64: Vec<f64> = params.iter().map(|&p| f64::from(p)).collect();
    let oracle_out = oracle::evaluate(&plan, &operands, &images, &params_f64);
    let oracle_vals = bytes_f32(&oracle_out[0].bytes);

    // Leg 2: kiss-ref via the converter.
    let shapes_usize: Vec<Vec<usize>> = in_shapes
        .iter()
        .map(|s| s.iter().map(|&d| d as usize).collect())
        .collect();
    let (dag, r) = kiss_ref_leg(op, &shapes_usize, inputs, params);
    let ref_vals: Vec<f32> = r.outputs[0].clone().into_data();
    assert_bits_eq(&op.name, &oracle_vals, &ref_vals);
    (dag, r)
}

fn show(name: &str, dag: &FlatDag, r: &RecipeEval<f32>, n_vals: usize) {
    println!(
        "  {name} OK — {n_vals} values bit-identical, {} nodes, dets {:?}",
        dag.nodes.len(),
        r.dets
    );
}

// ===========================================================================
// Step 3a — the ON-DEVICE leg: JIT the GENERATED CUDA kernel via NVRTC,
// launch on the real GPU, bit-compare against kiss-ref (the ExactByte
// elementwise lane; tolerance-class cells are step 3b).
// ===========================================================================

struct DeviceCtx {
    ctx: baracuda_driver::Context,
    arch_flag: String,
    name: String,
}

fn device_init() -> DeviceCtx {
    baracuda_driver::init().expect("driver init");
    let device = baracuda_driver::Device::get(0).expect("device 0");
    let (maj, min) = device.compute_capability().expect("cc");
    let name = device.name().expect("name");
    let ctx = baracuda_driver::Context::new(&device).expect("context");
    DeviceCtx {
        ctx,
        arch_flag: format!("--gpu-architecture=compute_{maj}{min}"),
        name,
    }
}

/// Generate the op's CUDA kernel for a SCALAR (v1) contiguous cell, NVRTC-JIT
/// it, launch over `inputs`, and return the output bits. `params` follow `n`
/// in the generated signature (`…, long long n, float p0, float p1`).
fn device_leg(
    dc: &DeviceCtx,
    op: &OpDef,
    cat: OpCategory,
    inputs: &[Vec<f32>],
    params: &[f32],
) -> Vec<f32> {
    use baracuda_driver::{DeviceBuffer, Module, Stream};
    let n = inputs[0].len() as i64;
    // align 4 + odd n => VecWidth::Scalar => count_unit is ELEMENTS.
    let od = OperandDesc::new(1, &[n], &[1], ElementKind::F32, 4);
    let operands: Vec<OperandDesc> = std::iter::repeat(od).take(inputs.len() + 1).collect();
    let key = structure_key(cat, &operands, ArchSku::Sm89);
    let kernel = baracuda_kernelgen::generate(op, &key, &baracuda_kernelgen::Cuda);
    if std::env::var_os("POC_DUMP").is_some() {
        eprintln!("--- {} ({}) ---\n{}", op.name, kernel.name, kernel.source);
    }

    let ptx = baracuda_nvrtc::Program::compile(&kernel.source, "gen.cu", &[&dc.arch_flag])
        .unwrap_or_else(|e| panic!("{}: nvrtc: {e}\n--- source ---\n{}", op.name, kernel.source));
    let stream = Stream::new(&dc.ctx).expect("stream");
    let module = Module::load_ptx(&dc.ctx, &ptx).expect("load_ptx");
    let func = module.get_function(&kernel.name).expect("get_function");

    let d_ins: Vec<DeviceBuffer<f32>> = inputs
        .iter()
        .map(|v| DeviceBuffer::from_slice(&dc.ctx, v).expect("upload"))
        .collect();
    let d_out: DeviceBuffer<f32> = DeviceBuffer::new(&dc.ctx, n as usize).expect("out");
    let in_ptrs: Vec<_> = d_ins.iter().map(DeviceBuffer::as_raw).collect();
    let out_ptr = d_out.as_raw();

    // SAFETY: argument order matches the generated signature
    // `(const float* in0[, in1…], float* out, long long n[, float p…])`.
    unsafe {
        let mut l = func.launch();
        l = l.grid((4, 1, 1)).block((32, 1, 1)).stream(&stream);
        for p in &in_ptrs {
            l = l.arg(p);
        }
        l = l.arg(&out_ptr).arg(&n);
        for p in params {
            l = l.arg(p);
        }
        l.launch().expect("launch");
    }
    stream.synchronize().expect("sync");
    let mut host = vec![0.0f32; n as usize];
    d_out.copy_to_host(&mut host).expect("copy back");
    host
}

fn main() {
    // Adversarial value set: ±0.0, NaN, ±inf, negatives, tiny/huge magnitudes.
    let a = vec![
        -1.0f32,
        2.0,
        -3.5,
        4.25,
        -0.0,
        0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0e-38,
        -1.0e38,
        3.402_823_5e38,
    ];
    let b = vec![
        0.5f32,
        -0.5,
        1.0,
        -2.0,
        0.0,
        -0.0,
        1.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        -1.0e-38,
        1.0e38,
        1.0,
    ];

    println!("step-2 elementwise differential (kernelgen oracle vs kiss-ref @ 004e1a4):");

    // (1) relu_add — the loop-closure op: relu(add(in0, in1)). The -0.0 / NaN
    // rows are the §6.15 relu-vs-naive-max seam.
    let relu_add = OpDef::elementwise(
        "relu_add",
        2,
        &[ElementKind::F32],
        (input(0) + input(1)).relu(),
    );
    let (dag, r) = diff_vs_oracle(
        &relu_add,
        OpCategory::BinaryElementwise,
        &[vec![12], vec![12]],
        &[12],
        &[a.clone(), b.clone()],
        &[],
    );
    assert!(r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)));
    show("relu_add", &dag, &r, 12);

    // (2) affine — runtime_scalar leaves: in0 * p0 + p1.
    let affine = OpDef::elementwise(
        "affine",
        1,
        &[ElementKind::F32],
        input(0) * param(0) + param(1),
    );
    let (dag, r) = diff_vs_oracle(
        &affine,
        OpCategory::UnaryElementwise,
        &[vec![12]],
        &[12],
        &[a.clone()],
        &[1.5, -0.25],
    );
    assert!(r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)));
    show("affine", &dag, &r, 12);

    // (3) maxprop_scaled — NaN-PROPAGATING elementwise max (KISS `max_prop`),
    // over a shared subexpression: max_prop(mul(in0, const(2)), in1). The
    // (-0.0, +0.0) tie row is the seam this differential caught on 2026-07-23
    // (Baracuda was b-on-ties in all three backends; spec + kiss-ref = a).
    let maxp = OpDef::elementwise(
        "maxprop_scaled",
        2,
        &[ElementKind::F32],
        (input(0) * konst(2.0)).binary(BinaryOp::Max, input(1)),
    );
    let (dag, r) = diff_vs_oracle(
        &maxp,
        OpCategory::BinaryElementwise,
        &[vec![12], vec![12]],
        &[12],
        &[a.clone(), b.clone()],
        &[],
    );
    assert!(r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)));
    show("maxprop_scaled", &dag, &r, 12);

    println!("step-2b fold differential:");

    // Rank-2 [4,8] integer-valued grid (exactly representable — any fold order
    // and accumulator width give identical bits), one NaN cell, one signed-zero
    // pair in a row (deliberate monoid-tie probe).
    let grid: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, //
        0.0, -0.0, 9.0, -9.0, 16.0, -16.0, 2.0, 4.0, //
        1.0, f32::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, //
    ];

    // (4) reduce[max,last,nokd] — ExactByte monoid; NaN row propagates; the
    // (0.0, -0.0) row exercises the monoid signed-zero handling (max = 16).
    let rmax = OpDef::reduction("rowmax", 1, &[ElementKind::F32], input(0), ReduceOp::Max);
    let (dag, r) = diff_vs_oracle(
        &rmax,
        OpCategory::Reduction,
        &[vec![4, 8]],
        &[4],
        &[grid.clone()],
        &[],
    );
    assert!(
        r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)),
        "max monoid folds ExactByte: {:?}",
        r.dets
    );
    show("reduce[max,last,nokd]", &dag, &r, 4);

    // (5) reduce[sum,last,nokd] — values bit-equal (exact construction) even
    // though the HONEST class of a float sum fold is order-nondeterministic:
    // the class-vs-value separation the production comparator keys on. (NaN
    // row: sum propagates NaN on both legs.)
    let rsum = OpDef::reduction("rowsum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
    let (dag, r) = diff_vs_oracle(
        &rsum,
        OpCategory::Reduction,
        &[vec![4, 8]],
        &[4],
        &[grid.clone()],
        &[],
    );
    let reduce_det = &r.dets[1]; // nodes: [Bind(0), Reduce]
    assert!(
        !matches!(reduce_det, DetClass::ExactByte),
        "a float sum fold must NOT class ExactByte: {reduce_det:?}"
    );
    show("reduce[sum,last,nokd]", &dag, &r, 4);

    // (6) rowmean — float Mean = div(reduce[sum,last,nokd](in0),
    // reduced_count(last)): the ReducedCount leaf + the composed div. Count 8
    // is a power of two, so the division is exact.
    let rmean = OpDef::reduction("rowmean", 1, &[ElementKind::F32], input(0), ReduceOp::Mean);
    let (dag, r) = diff_vs_oracle(
        &rmean,
        OpCategory::Reduction,
        &[vec![4, 8]],
        &[4],
        &[grid.clone()],
        &[],
    );
    assert!(
        dag.nodes
            .iter()
            .any(|n| matches!(n, Node::ReducedCount(ax) if ax == &vec![1usize])),
        "mean carries the reduced_count(last) leaf resolved to axis 1"
    );
    show("mean = div(reduce[sum], reduced_count)", &dag, &r, 4);

    // (7) prefix_scan[sum,1,incl] — cumulative sum along the last axis,
    // integer-valued (exact at any order); full-width output.
    let scan = OpDef::scan_simple("cumsum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
    let (dag, r) = diff_vs_oracle(
        &scan,
        OpCategory::Scan,
        &[vec![4, 8]],
        &[4, 8],
        &[grid.clone()],
        &[],
    );
    let scan_det = &r.dets[1]; // nodes: [Bind(0), PrefixScan]
    assert!(
        !matches!(scan_det, DetClass::ExactByte),
        "a float sum scan must NOT class ExactByte: {scan_det:?}"
    );
    show("prefix_scan[sum,1,incl]", &dag, &r, 32);

    // (8) matmul[mk.kn] — the kernelgen CPU oracle defers Contraction (a
    // panic), so this leg diffs kiss-ref against a LOCAL naive same-order
    // triple loop; integer-valued small cell [2,3]·[3,2] is exact at any
    // order. (The kernel-vs-kiss-ref matmul diff is the on-device step.)
    let mm = OpDef::contraction(
        "matmul",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0),
    );
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3,2]
    let (dag, r) = kiss_ref_leg(&mm, &[vec![2, 3], vec![3, 2]], &[lhs.clone(), rhs.clone()], &[]);
    let mut expect = vec![0.0f32; 4];
    for m in 0..2 {
        for n in 0..2 {
            let mut acc = 0.0f32;
            for k in 0..3 {
                acc += lhs[m * 3 + k] * rhs[k * 2 + n];
            }
            expect[m * 2 + n] = acc;
        }
    }
    let got: Vec<f32> = r.outputs[0].clone().into_data();
    assert_bits_eq("matmul", &expect, &got);
    let mm_det = &r.dets[2]; // nodes: [Bind(0), Bind(1), Matmul]
    assert!(
        !matches!(mm_det, DetClass::ExactByte),
        "a float matmul must NOT class ExactByte: {mm_det:?}"
    );
    show("matmul[mk.kn]", &dag, &r, 4);

    // (9) Reverse scan — the honest miss now lives at EMISSION (e3530a39: the
    // #68 anti-fork witness caught `flip` as a fabricated token, so
    // semantics_dag returns None until flip registers in KISS-Ops). The
    // converter's own flip guard stays as defense-in-depth for older text.
    let rev = OpDef::scan_simple("revsum", &[ElementKind::F32], ReduceOp::Sum, 1, true, false);
    assert_eq!(semantics_dag(&rev), None, "reverse scan is an emission-level honest miss");
    match recipe_to_flatdag(&rev, 2) {
        Err(e) if e.contains("honest miss") => {
            println!("  reverse-scan honest miss OK (at emission): {e}");
        }
        other => panic!("reverse scan must miss at emission, got {other:?}"),
    }

    // ---- Step 3a: on-device (generated CUDA kernel vs kiss-ref, bit-exact). --
    // Odd length (13) forces the scalar cell; adversarial values ride along.
    let mut a13 = a.clone();
    a13.push(-7.0);
    let mut b13 = b.clone();
    b13.push(7.0);
    let dc = device_init();
    println!("step-3a on-device differential ({} / {}):", dc.name, dc.arch_flag);

    for (op, cat, ins, params) in [
        (
            &relu_add,
            OpCategory::BinaryElementwise,
            vec![a13.clone(), b13.clone()],
            vec![],
        ),
        (
            &affine,
            OpCategory::UnaryElementwise,
            vec![a13.clone()],
            vec![1.5f32, -0.25],
        ),
        (
            &maxp,
            OpCategory::BinaryElementwise,
            vec![a13.clone(), b13.clone()],
            vec![],
        ),
    ] {
        // kiss-ref expectation over the same 13-length inputs.
        let shapes: Vec<Vec<usize>> = ins.iter().map(|v| vec![v.len()]).collect();
        let (_, r) = kiss_ref_leg(op, &shapes, &ins, &params);
        assert!(
            r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)),
            "3a is the ExactByte lane"
        );
        let expect: Vec<f32> = r.outputs[0].clone().into_data();
        let got = device_leg(&dc, op, cat, &ins, &params);
        // §6.8 exact comparator (both-NaN conforming): an ARITHMETIC-produced
        // NaN carries the device-canonical payload (0x7fffffff on sm_89) vs
        // x86's input-propagated 0x7fc00000 — a class match, not a bit match.
        assert_conforming_eq(&format!("device:{}", op.name), &expect, &got);
        println!(
            "  {} OK — GPU output §6.8-exact vs kiss-ref over {} adversarial values",
            op.name,
            got.len()
        );
    }

    // ---- Step 3b (partial): on-device folds vs kiss-ref. --------------------
    // The [4,8] exact-representable grid: the device block-tree fold order
    // differs from kiss-ref's serial fold, but every partial sum is exactly
    // representable, so the OIN class collapses to bit-stable — compared under
    // the §6.8 conforming comparator (the NaN row's payload may remint).
    println!("step-3b (partial) on-device fold differential:");
    for (op, label) in [
        (&rsum, "reduce[sum,last,nokd]"),
        (&rmax, "reduce[max,last,nokd]"),
    ] {
        let shapes = vec![vec![4usize, 8]];
        let (_, r) = kiss_ref_leg(op, &shapes, std::slice::from_ref(&grid), &[]);
        let expect: Vec<f32> = r.outputs[0].clone().into_data();
        let got = device_reduce_leg(&dc, op, 4, 8, &grid);
        assert_conforming_eq(&format!("device:{label}"), &expect, &got);
        println!("  {label} OK — GPU block-tree fold §6.8-exact vs kiss-ref: {got:?}");
    }

    // ---- Step 2c: gather/scatter through IndexRef (CPU lane). ---------------
    println!("step-2c indexed-op differential (emitted recipe -> IndexRef -> eval_recipe):");

    // (10) gather[0,clamp,i64](in0, in1): rank-1 data, one OOB index CLAMPED.
    let g = OpDef::gather(
        "gather",
        &[ElementKind::F32],
        0,
        baracuda_kernelgen::ir::OobPolicy::Clamp,
        ElementKind::I64,
    );
    assert_eq!(
        semantics_dag(&g).as_deref(),
        Some("gather[0,clamp,i64](in0, in1)"),
        "the emitted gather surface"
    );
    let dag = recipe_to_flatdag(&g, 1).expect("gather converts");
    let data = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], &[6]).unwrap();
    let idx = IndexTensor::new(vec![5, 0, 3, 3, 9], &[5], Dtype::I64).unwrap();
    let r = eval_recipe(&dag, &[data], &[], &[idx]).expect("gather eval");
    let got: Vec<f32> = r.outputs[0].clone().into_data();
    // idx 9 clamps to 5 -> 60. Hand-computed.
    assert_bits_eq("gather-clamp", &[60.0, 10.0, 40.0, 40.0, 60.0], &got);
    assert!(
        r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)),
        "an exact-index gather is ExactByte: {:?}",
        r.dets
    );
    println!("  gather[0,clamp,i64] OK — {got:?}, dets {:?}", r.dets);

    // (11) scatter_add — mirrors kiss-ref's R6 histogram corpus vector shape:
    // duplicate indices accumulate, the OOB write (7) is SKIPPED (§6.11-0005 =
    // Baracuda's Skip-only scatter kernels). The converter synthesizes the
    // explicit dest bind (slot 1 after the updates value slot); zeros dest =
    // the in-place kernel's oracle-equivalent pure form.
    let sc = OpDef::scatter_add("scatter_add", &[ElementKind::F32], 0, ElementKind::U32);
    assert_eq!(
        semantics_dag(&sc).as_deref(),
        Some("scatter[0,atomic-add,skip,u32](in0, in1)"),
        "the emitted scatter surface"
    );
    let dag = recipe_to_flatdag(&sc, 1).expect("scatter converts");
    assert!(
        dag.nodes
            .iter()
            .any(|n| matches!(n, Node::Scatter { dest, .. } if matches!(dag.nodes[*dest], Node::Bind(1)))),
        "synthesized dest binds the slot after the value inputs"
    );
    let updates = Tensor::from_vec(vec![1.0f32, 2.0, 0.5, 0.25, 4.0, 1.5], &[6]).unwrap();
    let dest = Tensor::from_vec(vec![0.0f32; 4], &[4]).unwrap();
    let idx = IndexTensor::new(vec![0, 2, 0, 3, 2, 7], &[6], Dtype::U32).unwrap();
    let r = eval_recipe(&dag, &[updates, dest], &[], &[idx]).expect("scatter eval");
    let got: Vec<f32> = r.outputs[0].clone().into_data();
    assert_bits_eq("scatter_add", &[1.5, 0.0, 6.0, 0.25], &got);
    let sc_det = r
        .dets
        .iter()
        .zip(&dag.nodes)
        .find(|(_, n)| matches!(n, Node::Scatter { .. }))
        .map(|(d, _)| d)
        .expect("scatter node det");
    assert!(
        !matches!(sc_det, DetClass::ExactByte),
        "a float atomic-add scatter must NOT class ExactByte: {sc_det:?}"
    );
    println!("  scatter[0,atomic-add,skip,u32] OK — {got:?} (OOB 7 skipped), scatter det {sc_det:?}");

    // (12) bincount — the self-indexing scatter (`in0` is BOTH the only op
    // input and the index operand; the value is const(1)): zero value-lane
    // inputs, dest binds slot 0, x rides `indices`. SEAM FINDING (flagged to
    // kiss-ref, 2026-07-23): Baracuda emits a RANK-0 `const(1)` as the scatter
    // updates (conceptually broadcast over the index count — the same rank-0
    // broadcast kiss-ref's own `Gather.base` performs), but kiss-ref's Scatter
    // requires updates shaped like the index -> typed ShapeMismatch. Whether
    // scalar updates kernel-broadcast is a §6.11/#67 grammar ruling; until
    // then this vector pins the CURRENT typed-decline behavior.
    let bc = OpDef::bincount("bincount", ElementKind::I32);
    let text = semantics_dag(&bc).expect("bincount recipe");
    println!("  bincount emits: {text}");
    let dag = recipe_to_flatdag(&bc, 1).expect("bincount converts");
    let dest = Tensor::from_vec(vec![0.0f32; 4], &[4]).unwrap();
    let x = IndexTensor::new(vec![0, 2, 0, 3, 2, 2], &[6], Dtype::I32).unwrap();
    match eval_recipe(&dag, &[dest], &[], &[x]) {
        Err(e) => println!(
            "  bincount: OPEN SEAM (typed, not a crash) — rank-0 const updates vs \
             index-shaped updates: {e:?} (routed to the §6.11/#67 ruling)"
        ),
        Ok(r) => {
            // If kiss-ref adopts rank-0 updates broadcast, this becomes the
            // golden path: counts [2,0,3,1].
            let got: Vec<f32> = r.outputs[0].clone().into_data();
            assert_bits_eq("bincount", &[2.0, 0.0, 3.0, 1.0], &got);
            println!("  bincount OK — {got:?} (broadcast-updates adopted)");
        }
    }

    println!(
        "steps 2+2b+3a+3b(partial)+2c OK: emitted-recipe -> FlatDag converter (elementwise + \
         folds + indexed IndexRef ops) matches, and the GENERATED CUDA kernels — elementwise \
         AND block-tree folds — match kiss-ref ON DEVICE."
    );
}
