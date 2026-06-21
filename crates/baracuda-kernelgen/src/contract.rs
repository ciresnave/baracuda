//! Emit a complete Fuel **FKC kernel contract** from an op + its structure cell.
//!
//! Where [`crate::pattern`] derives only the `pattern:` block (for a fused op),
//! this assembles the *whole* importable contract Fuel reads to bind a kernel:
//! the bundle front-matter plus the per-kernel `accept` / `return` / `op_params`
//! / `caps` / `cost` / `precision` / `determinism` blocks (and `pattern:` when
//! the op is a recognized fusion).
//!
//! The block field set is pinned to the Profile-v1 conformance matrix
//! (*Kernel-Seam Interop* §4.3) and the `ImplId` five-field tuple (FKC §4.11:
//! `backend`, op, `dtypes`, `kernel_source`, `kernel_revision_hash` — five
//! separable wire fields, never a hash). The admissibility predicate **is** the
//! structure key (per `docs/design/kernel-specialization.md`), so each contract
//! carries its [`StructureKey::to_token`] verbatim under `accept` — the planner's
//! miss signal stays honest by construction.
//!
//! # Scope (v1) / reconciliation
//!
//! Elementwise ops. The block *field names* follow §4.3 and the `ImplId` tuple;
//! a few leaf *spellings* (the dtype tokens — review item E5 — and the precision
//! / layout enums) are reconstructed from the FKC annex and reconciled when the
//! full `kernel-contract-format.md` is wired. They are isolated in the small
//! helpers below so reconciliation is a localized change.

use crate::backend::GeneratedKernel;
use crate::ir::{BinaryOp, OpDef, ScalarExpr, UnaryOp};
use crate::pattern::{derive_pattern, to_fkc, PatternNode};
use baracuda_kernels_types::{Contiguity, ElementKind, StructureKey, VecWidth};

/// Provider-wide FKC bundle front-matter (FKC §0/§3.1) — emitted once per bundle
/// file, above the per-kernel [`contract`] blocks. `revision_base` is the
/// provider source-tree revision the kernels were built from (the
/// `ImplId.kernel_revision_hash` base). Carries `seam_profiles: [1]` so an
/// importer can reject a contract outside the negotiated seam profile (§3.5).
#[must_use]
pub fn front_matter(backend_name: &str, revision_base: &str) -> String {
    format!(
        "---\n\
         fkc_version: 1\n\
         provider:\n  \
         name: baracuda\n  \
         backend: {backend_name}\n  \
         kernel_source: baracuda\n  \
         link_registry: baracuda_link_registry\n  \
         revision_base: \"{revision_base}\"\n\
         seam_profiles: [1]\n\
         ---\n"
    )
}

/// Emit the per-kernel FKC contract block (a ```` ```fkc ```` fenced section) for
/// `kernel`, generated for `op` at structure cell `key` and lowered by
/// `backend_name`.
///
/// A single graph-`Op` body advertises as a primitive (`op_kind`); a multi-op
/// body advertises as a recognized fusion (`fused_op` + a `pattern:` block).
#[must_use]
pub fn contract(
    op: &OpDef,
    key: &StructureKey,
    kernel: &GeneratedKernel,
    backend_name: &str,
) -> Option<String> {
    // Skip a cell whose dtype has no FKC §5 base-dtype spelling — an unbindable
    // contract would corrupt the planner's honest miss signal (§4.3).
    let dtype = fkc_dtype(key.dtype)?;

    let pattern = derive_pattern(op).ok();
    let n_ops = pattern.as_ref().map_or(0, count_ops);
    let is_fusion = n_ops > 1;
    let op_line = match &pattern {
        // exactly one graph op → a primitive identity (e.g. `Add`, `AddScalar`).
        Some(p) if n_ops == 1 => format!("op_kind: {}", root_op_name(p)),
        // ≥2 graph ops → a fused identity carried by the op's stable name.
        Some(_) => format!("fused_op: {}", op.name),
        // body not expressible as a pattern (Const / non-elementwise / bind
        // mismatch) → not advertisable; skip rather than fake an op_kind from
        // the op's free-form name (which is not an OpKind dispatch key).
        None => return None,
    };

    let out_idx = key.n_operands.saturating_sub(1) as usize;
    let params = params_used(&op.body);
    let (prec_mode, prec_ulp) = precision_of(&op.body);

    let mut s = String::from("```fkc\n");
    s.push_str(&format!("kernel: {}_{}\n", op.name, cell_suffix(key)));
    s.push_str(&op_line);
    s.push('\n');
    s.push_str(&format!("blurb: \"{}\"\n", blurb(op, key, dtype, is_fusion)));
    // ImplId tuple (FKC §4.11), kept as five separable fields.
    s.push_str(&format!("backend: {backend_name}\n"));
    s.push_str("kernel_source: baracuda\n");
    s.push_str(&format!("dtypes: [{dtype}]\n"));
    s.push_str(&format!("entry_point: {}\n", kernel.name));
    s.push_str(&format!(
        "kernel_revision_hash: \"{:016x}\"\n",
        revision_hash(&kernel.source)
    ));

    // accept — the admissibility predicate IS the structure key (the honesty
    // invariant); the per-input dtype/layout lines are a human-readable gloss.
    s.push_str("accept:\n");
    s.push_str(&format!("  structure_key: \"{}\"\n", key.to_token()));
    s.push_str("  inputs:\n");
    for i in 0..op.n_inputs as usize {
        s.push_str(&format!(
            "    - dtype: {dtype}\n      layout: {}\n",
            layout_token(key, i)
        ));
    }

    if !params.is_empty() {
        s.push_str("op_params:\n");
        for p in &params {
            // v1 scalar params are f32 launch arguments (the `extract:` carrier).
            s.push_str(&format!("  - name: param{p}\n    dtype: F32\n"));
        }
    }

    s.push_str("return:\n  outputs:\n");
    s.push_str(&format!(
        "    - dtype_rule: same_as_input(0)\n      \
         shape_rule: same_as_input(0)\n      \
         layout: {}\n      \
         aliasing: none\n",
        layout_token(key, out_idx)
    ));

    s.push_str("caps:\n");
    s.push_str("  in_place: allowed\n");
    s.push_str(&format!("  alignment_bytes: {}\n", required_align(key)));
    s.push_str(&format!("  awkward_layout: {}\n", awkward_layout(key)));

    s.push_str("cost:\n");
    s.push_str("  provenance: declared\n");
    s.push_str("  class: elementwise\n");
    s.push_str(&format!("  flops_per_elem: {}\n", count_flops(&op.body)));
    s.push_str(&format!("  bytes_per_elem: {}\n", bytes_per_elem(op, key)));

    s.push_str("precision:\n");
    s.push_str(&format!("  mode: {prec_mode}\n"));
    if let Some(u) = prec_ulp {
        s.push_str(&format!("  max_ulp: {u}\n"));
    }
    s.push_str("determinism: bitwise\n");

    if let (Some(p), true) = (&pattern, is_fusion) {
        s.push_str(&to_fkc(p));
    }

    s.push_str("```\n");
    Some(s)
}

// ---------------------------------------------------------------------------
// Pattern shape helpers
// ---------------------------------------------------------------------------

fn count_ops(node: &PatternNode) -> usize {
    match node {
        PatternNode::Bind(_) => 0,
        PatternNode::Op { operands, .. } => 1 + operands.iter().map(count_ops).sum::<usize>(),
    }
}

fn root_op_name(node: &PatternNode) -> String {
    match node {
        PatternNode::Op { op, .. } => op.clone(),
        PatternNode::Bind(_) => "Identity".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Body scans
// ---------------------------------------------------------------------------

/// Sorted, unique runtime-parameter indices referenced by the body.
fn params_used(body: &ScalarExpr) -> Vec<u8> {
    let mut out = Vec::new();
    scan_params(body, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

fn scan_params(e: &ScalarExpr, out: &mut Vec<u8>) {
    match e {
        ScalarExpr::Param(i) => out.push(*i),
        ScalarExpr::Input(_) | ScalarExpr::Const(_) => {}
        ScalarExpr::Unary(_, x) => scan_params(x, out),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => {
            scan_params(a, out);
            scan_params(b, out);
        }
    }
}

/// Declared flop count per output element: one per arithmetic / unary node.
fn count_flops(e: &ScalarExpr) -> u32 {
    match e {
        ScalarExpr::Input(_) | ScalarExpr::Const(_) | ScalarExpr::Param(_) => 0,
        ScalarExpr::Unary(_, x) => 1 + count_flops(x),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => 1 + count_flops(a) + count_flops(b),
    }
}

/// `true` if the body contains a transcendental that lowers to a hardware/library
/// approximation (a few ulp) rather than a correctly-rounded IEEE primitive.
fn has_transcendental(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Input(_) | ScalarExpr::Const(_) | ScalarExpr::Param(_) => false,
        ScalarExpr::Unary(op, x) => is_transcendental(*op) || has_transcendental(x),
        // `Pow` lowers to `powf` (a few ulp); `Max`/`Min`/`Rem` are exact.
        ScalarExpr::Binary(op, a, b) => {
            matches!(op, BinaryOp::Pow) || has_transcendental(a) || has_transcendental(b)
        }
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b) => has_transcendental(a) || has_transcendental(b),
    }
}

fn is_transcendental(op: UnaryOp) -> bool {
    matches!(
        op,
        UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Tanh
            | UnaryOp::Sigmoid
            | UnaryOp::Erf
            | UnaryOp::Gelu
            | UnaryOp::Silu
            | UnaryOp::Rsqrt
            | UnaryOp::Sin
            | UnaryOp::Cos
    )
}

/// Precision contract: arithmetic + `Relu`/`Abs`/`Neg`/`Sqrt` are correctly
/// rounded (0 ulp of the dtype, bit-reproducible); a transcendental relaxes to a
/// small ulp bound. (`F32Strict` would force `correctly_rounded` regardless — it
/// is a precision mode, not a wire dtype.)
fn precision_of(body: &ScalarExpr) -> (&'static str, Option<u32>) {
    if has_transcendental(body) {
        ("approximate", Some(2))
    } else {
        ("correctly_rounded", Some(0))
    }
}

// ---------------------------------------------------------------------------
// Structure-cell projections
// ---------------------------------------------------------------------------

fn layout_token(key: &StructureKey, i: usize) -> &'static str {
    match key.operands[i].contig {
        Contiguity::Contig => "contiguous",
        Contiguity::InnerContig => "inner_contiguous",
        Contiguity::Strided => "strided",
        Contiguity::Broadcast => "broadcast",
    }
}

/// A kernel specialized for a strided/broadcast cell handles awkward layouts;
/// a contiguous/vectorized cell requires the packed layout it was built for.
fn awkward_layout(key: &StructureKey) -> &'static str {
    match key.operands[0].contig {
        Contiguity::Strided | Contiguity::Broadcast => "handles_strided",
        Contiguity::Contig | Contiguity::InnerContig => "requires_contiguous",
    }
}

/// Required base-pointer alignment (bytes): a vectorized cell needs its vector
/// width; a scalar cell needs the dtype's natural alignment.
fn required_align(key: &StructureKey) -> u32 {
    let dsz = dtype_size(key.dtype);
    match key.operands[0].vec_width {
        VecWidth::V8 => (8 * dsz).min(16),
        VecWidth::V4 => (4 * dsz).min(16),
        VecWidth::V2 => 2 * dsz,
        VecWidth::Scalar => dsz,
    }
}

fn bytes_per_elem(op: &OpDef, key: &StructureKey) -> u32 {
    // inputs + one output, each one dtype-wide element (broadcast operands touch
    // fewer; this is a declared upper estimate).
    (u32::from(op.n_inputs) + 1) * dtype_size(key.dtype)
}

/// `<op>_<dtype>_<contig0>_<vec0>` cell discriminator for the readable `kernel`
/// name (the linkable symbol is `entry_point`).
fn cell_suffix(key: &StructureKey) -> String {
    let o = &key.operands[0];
    format!("{}_{}_{}", dtype_short(key.dtype), contig_short(o.contig), vec_short(o.vec_width))
}

fn blurb(op: &OpDef, key: &StructureKey, dtype: &str, is_fusion: bool) -> String {
    let kind = if is_fusion { "fused" } else { "elementwise" };
    format!("{} {} ({}, {} layout).", kind, op.name, dtype, layout_token(key, 0))
}

// ---------------------------------------------------------------------------
// Leaf spellings (reconciled against the FKC annex; review item E5 for dtypes)
// ---------------------------------------------------------------------------

/// FKC §5 logical-DType token, or `None` for a dtype with no §5 *base-dtype*
/// slot (so the caller skips the cell rather than emit an unbindable contract).
///
/// Reconciled to FKC rev-4 §5 (review item E5): `Bool` → `U8` (Fuel has no Bool
/// dtype — masks are U8), signed-8 → `I8`, `F32Strict` rides as `F32` (a
/// precision mode, not a wire dtype). Packed sub-byte / quant payloads
/// (`S4`/`U4`/`Bin`) ride the **FDX sidecar**, not a base dtype, so carry no
/// token here; `Fp8E5M2` and complex have no §5 slot yet — all return `None`.
fn fkc_dtype(dt: ElementKind) -> Option<&'static str> {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    Some(match dt {
        F32 | F32Strict => "F32",
        F16 => "F16",
        Bf16 => "BF16",
        F64 => "F64",
        I32 => "I32",
        I64 => "I64",
        S8 => "I8",        // §5: signed-8 spells I8
        U8 | Bool => "U8", // §5 (B5/E5): Fuel has no Bool — masks are U8
        Fp8E4M3 => "F8E4M3",
        // No §5 base-dtype slot: FDX-sidecar payloads + unlisted fp8 / complex.
        Fp8E5M2 | S4 | U4 | Bin | Complex32 | Complex64 => return None,
    })
}

fn dtype_short(dt: ElementKind) -> &'static str {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    match dt {
        F32 | F32Strict => "f32",
        F16 => "f16",
        Bf16 => "bf16",
        F64 => "f64",
        I32 => "i32",
        I64 => "i64",
        S8 => "s8",
        U8 => "u8",
        Bool => "bool",
        Fp8E4M3 => "e4m3",
        Fp8E5M2 => "e5m2",
        S4 => "s4",
        U4 => "u4",
        Bin => "b1",
        Complex32 => "c32",
        Complex64 => "c64",
    }
}

fn contig_short(c: Contiguity) -> &'static str {
    match c {
        Contiguity::Contig => "co",
        Contiguity::InnerContig => "ic",
        Contiguity::Strided => "st",
        Contiguity::Broadcast => "br",
    }
}

fn vec_short(v: VecWidth) -> &'static str {
    match v {
        VecWidth::Scalar => "v1",
        VecWidth::V2 => "v2",
        VecWidth::V4 => "v4",
        VecWidth::V8 => "v8",
    }
}

fn dtype_size(dt: ElementKind) -> u32 {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    match dt {
        S4 | U4 | Bin => 1, // sub-byte: round up to a byte for the declared estimate
        S8 | U8 | Bool | Fp8E4M3 | Fp8E5M2 => 1,
        F16 | Bf16 => 2,
        F32 | F32Strict | I32 => 4,
        F64 | I64 | Complex32 => 8,
        Complex64 => 16,
    }
}

/// 64-bit FNV-1a over the kernel source — the `ImplId.kernel_revision_hash` base.
/// Stable and dependency-free (unlike `DefaultHasher`, which is unspecified).
pub(crate) fn revision_hash(src: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in src.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{input, param, OpDef};
    use crate::{generate, Cuda};
    use baracuda_kernels_types::{
        structure_key, ArchSku, ElementKind, OpCategory, OperandDesc,
    };

    fn key_for(n_operands: usize, op_cat: OpCategory) -> StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(op_cat, &operands, ArchSku::Sm89)
    }

    #[test]
    fn front_matter_has_provider_and_seam_profiles() {
        let fm = front_matter("cuda", "abc123");
        assert!(fm.contains("fkc_version: 1"));
        assert!(fm.contains("name: baracuda"));
        assert!(fm.contains("link_registry: baracuda_link_registry"));
        assert!(fm.contains("seam_profiles: [1]"));
        assert!(fm.contains("revision_base: \"abc123\""));
    }

    #[test]
    fn primitive_add_uses_op_kind_and_carries_required_blocks() {
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        let key = key_for(3, OpCategory::BinaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, "cuda").unwrap();

        // primitive → op_kind, no fused_op, no pattern block.
        assert!(c.contains("op_kind: Add"));
        assert!(!c.contains("fused_op:"));
        assert!(!c.contains("pattern:"));
        // ImplId five fields all present + separable.
        assert!(c.contains("backend: cuda"));
        assert!(c.contains("kernel_source: baracuda"));
        assert!(c.contains("dtypes: [F32]"));
        assert!(c.contains("entry_point: "));
        assert!(c.contains("kernel_revision_hash: \""));
        // required §4.3 blocks.
        for block in [
            "accept:", "structure_key: \"sk1|", "return:", "caps:", "cost:", "precision:",
            "determinism: bitwise",
        ] {
            assert!(c.contains(block), "missing block: {block}");
        }
        // correctly-rounded arithmetic.
        assert!(c.contains("mode: correctly_rounded"));
    }

    #[test]
    fn fused_activation_uses_fused_op_with_pattern() {
        // relu(a + b) — two graph ops → a fused identity + a pattern block.
        let op = OpDef::elementwise(
            "relu_add",
            2,
            &[ElementKind::F32],
            (input(0) + input(1)).relu(),
        );
        let key = key_for(3, OpCategory::BinaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, "cuda").unwrap();
        assert!(c.contains("fused_op: relu_add"));
        assert!(!c.contains("op_kind:"));
        assert!(c.contains("pattern:"));
        assert!(c.contains("op: Relu"));
    }

    #[test]
    fn scalar_param_emits_op_params_and_transcendental_relaxes_precision() {
        // silu(x * p0 + p1): a transcendental (approximate) with two scalar params.
        let op = OpDef::elementwise(
            "affine_silu",
            1,
            &[ElementKind::F32],
            (input(0) * param(0) + param(1)).silu(),
        );
        let key = key_for(2, OpCategory::UnaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, "cuda").unwrap();
        assert!(c.contains("op_params:"));
        assert!(c.contains("name: param0"));
        assert!(c.contains("name: param1"));
        assert!(c.contains("mode: approximate"));
        assert!(c.contains("max_ulp: 2"));
    }

    #[test]
    fn revision_hash_is_source_sensitive() {
        assert_ne!(revision_hash("kernel a"), revision_hash("kernel b"));
        assert_eq!(revision_hash("stable"), revision_hash("stable"));
    }

    fn key_dtype(dt: ElementKind, n_operands: usize) -> StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
        let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89)
    }

    // The dtype-classification tests don't exercise CUDA codegen (which rightly
    // rejects Bool/Complex), only the contract's dtype channel — a stand-in kernel.
    fn stub_kernel() -> GeneratedKernel {
        GeneratedKernel { name: "k".into(), source: "s".into() }
    }

    #[test]
    fn bool_dtype_maps_to_u8_not_bool() {
        // §5 (B5/E5): Fuel has no Bool dtype — a provider's Bool rides as U8.
        let op = OpDef::elementwise("eq", 2, &[ElementKind::Bool], input(0) + input(1));
        let key = key_dtype(ElementKind::Bool, 3);
        let c = contract(&op, &key, &stub_kernel(), "cuda").unwrap();
        assert!(c.contains("dtypes: [U8]"));
        assert!(!c.contains("Bool"));
    }

    #[test]
    fn unsupported_dtype_yields_no_contract() {
        // Complex has no FKC §5 base-dtype slot — skip the cell (honest miss),
        // never emit an unbindable `dtypes: [C64]` contract.
        let op = OpDef::elementwise("add", 2, &[ElementKind::Complex64], input(0) + input(1));
        let key = key_dtype(ElementKind::Complex64, 3);
        assert!(contract(&op, &key, &stub_kernel(), "cuda").is_none());
    }
}
