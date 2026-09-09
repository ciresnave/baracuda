//! FKC contract tests that require a real backend lowering (`&Cuda`).
//!
//! Relocated from `baracuda-kernelgen` `src/contract.rs`'s `mod tests` during the
//! Unpopped carve (step 3): these exercise `contract()`/`generate()` over the CUDA
//! backend and non-elementwise op classes (contraction / reduction / gather / scan)
//! that only the CUDA emitter lowers, so they live with the CUDA backend crate.
//! The backend-free contract tests (bundle-framing, precision/flops helpers, the
//! primitive-op-kind mapping) stayed neutral in `baracuda-kernelgen`.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use baracuda_cuda_emit::Cuda;
use unpopped::backend::*;
use unpopped::contract::{
    AccuracyKey, bundle, bundle_kisc, cell_suffix, contract, contract_admissible, front_matter,
    fuel_primitive_op_kind, precision_of, root_op_name,
};
use unpopped::generate;
use unpopped::ir::*;
use unpopped::pattern::*;
use unpopped_vocab::*;

fn key_for(n_operands: usize, op_cat: OpCategory) -> StructureKey {
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
    structure_key(op_cat, &operands, ArchSku::Sm89)
}

fn gather_key(index_dt: ElementKind, one_d: bool) -> StructureKey {
    // [data F32, index `index_dt`, out F32], rank-2 axis-0 gather. `one_d`
    // keys the index 1-D (broadcast on axis 1 via stride 0) ⇒ index_select /
    // embedding; else full-shape ⇒ torch-gather.
    let data = OperandDesc::new(2, &[128, 64], &[64, 1], ElementKind::F32, 256);
    let idx = if one_d {
        OperandDesc::new(2, &[128, 64], &[1, 0], index_dt, 256)
    } else {
        OperandDesc::new(2, &[128, 64], &[64, 1], index_dt, 256)
    };
    let out = OperandDesc::new(2, &[128, 64], &[64, 1], ElementKind::F32, 256);
    structure_key(
        OpCategory::BinaryElementwise,
        &[data, idx, out],
        ArchSku::Sm89,
    )
}

/// An add cell with in0 dense and in1 FULLY broadcast (strides [0,0]). The
/// kernel hoists `in1[0]` (bakes the broadcast), so no truthful layout exists.
fn bias_add_key(in1_strides: &[i64]) -> (OpDef, StructureKey) {
    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let in0 = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let in1 = OperandDesc::new(2, &[128, 256], in1_strides, ElementKind::F32, 256);
    let out = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let key = structure_key(
        OpCategory::BinaryElementwise,
        &[in0, in1, out],
        ArchSku::Sm89,
    );
    (op, key)
}

fn key_dtype(dt: ElementKind, n_operands: usize) -> StructureKey {
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
    let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
    structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89)
}

// The dtype-classification tests don't exercise CUDA codegen (which rightly
// rejects Bool/Complex), only the contract's dtype channel — a stand-in kernel.
fn stub_kernel() -> GeneratedKernel {
    GeneratedKernel::new("k".into(), "s".into())
}

fn pred_key() -> StructureKey {
    // f32 inputs, u8 mask output — the elementwise_pred caller key shape.
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::U8, 256);
    structure_key(OpCategory::BinaryElementwise, &[a, a, o], ArchSku::Sm89)
}

#[test]
fn contraction_advertises_a_recipe_carrying_contract() {
    use unpopped::ir::{ContractionAxes, reduced};
    use unpopped::pattern::PatternError;
    // A contraction is NOT expressible as an elementwise pattern (derive_pattern
    // rejects it), but it carries a neutral KISS-Ops recipe — so it advertises a
    // recipe-carrying `fused_op` contract (matmul node + `from_recipe` shape),
    // admitted ONLY to a recipe-import peer. The old "no contract at all" wall
    // is replaced by the recipe-import withhold — the kernel still runs AOT.
    let mm = OpDef::contraction(
        "matmul",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0),
    );
    let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
    let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
    let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
    let kernel = generate(&mm, &key, &Cuda);
    let c = contract(&mm, &key, &kernel, &Cuda).expect("recipe-carrying contract");
    assert!(c.contains("fused_op: matmul"), "{c}");
    assert!(c.contains("semantics: matmul[mk.kn](in0, in1)"), "{c}");
    // The matmul output shape ≠ any input → no FKC shape_rule form fits, so
    // shape_rule is OMITTED (the recipe carries the shape). dtype is uniform →
    // passthrough(in0), a real form Fuel interprets.
    assert!(
        !c.contains("shape_rule"),
        "shape rides the recipe, omitted:\n{c}"
    );
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    // Honest-miss discipline preserved: WITHHELD from a non-recipe-import bundle,
    // admitted only to a recipe-import peer.
    assert!(
        !contract_admissible(&c, false),
        "withheld without recipe-import"
    );
    assert!(
        contract_admissible(&c, true),
        "admitted to a recipe-import peer"
    );
    // derive_pattern still rejects it — the recipe, not a pattern, is its identity.
    assert!(matches!(
        unpopped::derive_pattern(&mm),
        Err(PatternError::NotElementwise)
    ));
}

#[test]
fn contract_accumulation_type_matches_key_acc() {
    use unpopped::ir::{ContractionAxes, ReduceOp, reduced};
    // The sk3 RFC §4.2 / KISS-Contract §6.8 pin: a contraction cell's
    // contract MUST declare `accumulation_type` denoting the SAME dtype as
    // the key's `<acc>` coordinate, in the SAME closed dtype-token
    // spelling — one dtype, two surfaces.
    let mm = OpDef::contraction(
        "matmul",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0),
    );
    let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
    let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
    let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
    let acc = key.contraction.expect("gem cell").acc;
    let want = format!(
        "  accumulation_type: {}\n",
        unpopped_vocab::dtype_token(acc)
    );
    let c = contract(&mm, &key, &generate(&mm, &key, &Cuda), &Cuda).expect("contract");
    assert!(
        c.contains(&want),
        "contract must declare the key's <acc>: {c}"
    );
    assert!(c.contains("  accumulation_type: f32\n"), "{c}");
    // The generated skinny kernel actually accumulates in that dtype (a
    // `float acc`, never `double` — the F32Strict/binary32 discipline).
    let k = generate(&mm, &key, &Cuda);
    assert!(k.source.contains("float acc"), "{}", k.source);
    assert!(!k.source.contains("double acc"), "{}", k.source);

    // A reduction-bearing cell (no key <acc> coordinate — the §6.8 field is
    // the only surface) declares its fold dtype: an F32Strict sum folds in
    // double (the precision-first strict-reduce kernel), so it declares f64.
    let x = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32Strict, 256);
    let ro = OperandDesc::new(1, &[256], &[1], ElementKind::F32Strict, 256);
    let rk = structure_key(OpCategory::Reduction, &[x, ro], ArchSku::Sm89);
    let red = OpDef::reduction("s", 1, &[ElementKind::F32Strict], input(0), ReduceOp::Sum);
    let cr = contract(&red, &rk, &generate(&red, &rk, &Cuda), &Cuda).expect("contract");
    assert!(
        cr.contains("  accumulation_type: f64\n"),
        "strict reduce declares its double fold: {cr}"
    );

    // A pure elementwise cell has no fold — the field is ABSENT.
    let ew = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let ek = key_for(3, OpCategory::BinaryElementwise);
    let ce = contract(&ew, &ek, &generate(&ew, &ek, &Cuda), &Cuda).expect("contract");
    assert!(
        !ce.contains("accumulation_type"),
        "no fold, no declaration: {ce}"
    );
}

#[test]
fn rowreduce_advertises_a_recipe_carrying_contract() {
    use unpopped::ir::{ReduceOp, ReduceStage, UnaryOp, reduced};
    use unpopped::pattern::PatternError;
    // A RowReduce (softmax) is NOT an elementwise pattern (derive_pattern rejects
    // it), but it carries a neutral KISS-Ops recipe — staged `reduce[…]` folds
    // producing `Reduced(0..n)` + the row epilogue over them and the row-streamed
    // input — so it advertises a recipe-carrying `fused_op` contract, admitted
    // ONLY to a recipe-import peer (the same shape+dtype posture as the
    // contraction/scan arms). No contract.rs change was needed: `recipe_carrying`
    // auto-fires for a non-elementwise op the moment `semantics_dag` covers it.
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
    let sm = OpDef::row_reduce("softmax", 1, &[ElementKind::F32], stages, epi);
    let a = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Softmax, &[a, o], ArchSku::Sm89);
    let kernel = generate(&sm, &key, &Cuda);
    let c = contract(&sm, &key, &kernel, &Cuda).expect("recipe-carrying contract");
    assert!(c.contains("fused_op: softmax"), "{c}");
    assert!(
        c.contains(
            "semantics: div(exp(sub(in0, reduce[max,last,nokd](in0))), \
                 reduce[sum,last,nokd](exp(sub(in0, reduce[max,last,nokd](in0)))))"
        ),
        "{c}"
    );
    // The softmax output shape is the recipe's authority → no FKC shape_rule;
    // dtype is uniform → passthrough(in0), a real form Fuel interprets.
    assert!(
        !c.contains("shape_rule"),
        "shape rides the recipe, omitted:\n{c}"
    );
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    // Honest-miss discipline preserved: WITHHELD from a non-recipe-import bundle,
    // admitted only to a recipe-import peer.
    assert!(
        !contract_admissible(&c, false),
        "withheld without recipe-import"
    );
    assert!(
        contract_admissible(&c, true),
        "admitted to a recipe-import peer"
    );
    // End-to-end at the bundle seam: `bundle` (recipe_import=false) withholds the
    // free-form fused op; `bundle_kisc` admits it ONLY to a recipe-import peer.
    let withheld = bundle("baracuda", "cuda", "rev0", std::slice::from_ref(&c));
    assert!(!withheld.contains("fused_op: softmax"), "{withheld}");
    let admitted = bundle_kisc("baracuda", "cuda", "rev0", std::slice::from_ref(&c), true);
    assert!(
        admitted.contains(&unpopped::kisc::kisc_frame(&c)),
        "recipe-carrying RowReduce admitted for a recipe-import peer: {admitted}"
    );
    // derive_pattern still rejects it — the recipe, not a pattern, is its identity.
    assert!(matches!(
        unpopped::derive_pattern(&sm),
        Err(PatternError::NotElementwise)
    ));
}

#[test]
fn scan_advertises_a_recipe_carrying_contract() {
    use unpopped::ir::ReduceOp;
    use unpopped::pattern::PatternError;
    // A scan is not an elementwise pattern (derive_pattern rejects it), but it
    // carries a `prefix_scan` recipe — so it advertises a recipe-carrying
    // `fused_op` contract, admitted only to a recipe-import peer. (Its output
    // shape = input shape, but shape still defers to the recipe uniformly with
    // the other non-elementwise families.)
    let sc = OpDef::scan_simple(
        "cumsum",
        &[ElementKind::F32],
        ReduceOp::Sum,
        1,
        false,
        false,
    );
    let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    let kernel = generate(&sc, &key, &Cuda);
    let c = contract(&sc, &key, &kernel, &Cuda).expect("recipe-carrying contract");
    assert!(c.contains("fused_op: cumsum"), "{c}");
    assert!(c.contains("semantics: prefix_scan[sum,1,incl](in0)"), "{c}");
    // Shape rides the recipe (omitted); a scan's dtype is uniform → passthrough.
    assert!(!c.contains("shape_rule"), "{c}");
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    assert!(
        !contract_admissible(&c, false),
        "withheld without recipe-import"
    );
    assert!(
        contract_admissible(&c, true),
        "admitted to a recipe-import peer"
    );
    assert!(matches!(
        unpopped::derive_pattern(&sc),
        Err(PatternError::NotElementwise)
    ));
}

#[test]
fn window_is_an_honest_miss_no_contract() {
    use unpopped::ir::ReduceOp;
    use unpopped::pattern::PatternError;
    // Increment 7 WINDOW (pooling) is an AOT-only honest miss: Fuel exposes no
    // Pool/Window OpKind (the pool family rides bespoke cuDNN, opaque), and
    // neither contract.rs nor pattern.rs has any Window vocabulary, so a window
    // emits NO FKC contract (the kernel still generates + runs AOT) — the
    // Reduction/Scan/Contraction precedent. `derive_pattern` rejects it as
    // NotElementwise BEFORE any body walk; `contract()` then returns None.
    let p = OpDef::window_simple(
        "maxpool",
        &[ElementKind::F32],
        ReduceOp::Max,
        1,
        2,
        2,
        1,
        0,
        0,
        false,
    );
    let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    let kernel = generate(&p, &key, &Cuda);
    assert!(
        contract(&p, &key, &kernel, &Cuda).is_none(),
        "a window (pool) must emit NO contract (no Fuel Pool/Window OpKind; AOT-only honest miss)"
    );
    assert!(matches!(
        unpopped::derive_pattern(&p),
        Err(PatternError::NotElementwise)
    ));
}

#[test]
fn sort_is_an_honest_miss_no_contract() {
    use unpopped::ir::SortOrder;
    use unpopped::pattern::PatternError;
    // Increment 8 SORT_PERM is an AOT-only honest miss — a STRONGER miss than
    // scan/window: like pooling↔cuDNN, sort already rides bespoke kernels
    // (crates/baracuda-kernels/src/sort/*), so there is no Fuel Sort/ArgSort
    // OpTag and neither contract.rs nor pattern.rs has any sort vocabulary. The
    // kernel still generates + runs AOT; `derive_pattern` rejects it as
    // NotElementwise BEFORE any body walk; `contract()` then returns None.
    let sc = OpDef::row_sort("sort_rows", ElementKind::F32, SortOrder::Asc);
    let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    let kernel = generate(&sc, &key, &Cuda);
    assert!(
        contract(&sc, &key, &kernel, &Cuda).is_none(),
        "a sort must emit NO contract (no Fuel Sort OpTag; AOT-only honest miss)"
    );
    assert!(matches!(
        unpopped::derive_pattern(&sc),
        Err(PatternError::NotElementwise)
    ));
}

#[test]
fn argsort_is_an_honest_miss_no_contract() {
    use unpopped::ir::SortOrder;
    use unpopped::pattern::PatternError;
    // The argsort (I32 index output) is the same honest miss — generating +
    // running AOT, but no FKC contract.
    let sc = OpDef::row_argsort("argsort_rows", ElementKind::F32, SortOrder::Desc);
    let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::I32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    let kernel = generate(&sc, &key, &Cuda);
    assert!(
        contract(&sc, &key, &kernel, &Cuda).is_none(),
        "an argsort must emit NO contract (no Fuel ArgSort OpTag; AOT-only honest miss)"
    );
    assert!(matches!(
        unpopped::derive_pattern(&sc),
        Err(PatternError::NotElementwise)
    ));
}

#[test]
fn im2col_is_an_honest_miss_no_contract() {
    use unpopped::pattern::PatternError;
    // Increment 11 IM2COL is an AOT-only honest miss: Fuel treats convolution as a
    // first-class PRIMITIVE (the FKC whitelist has Conv2D/ConvTranspose2D, NO
    // Im2Col/Unfold/Pool) and im2col is only an internal lowering helper, never an
    // advertised OpKind — so it withholds via the same NotElementwise wall as
    // window/scan/sort. The kernel still generates + runs AOT; `derive_pattern`
    // rejects it as NotElementwise BEFORE any body walk; `contract()` then returns
    // None. `body == Input(0)` keeps n_outputs == 1, so the multi-output gate never
    // fires — NotElementwise withholds one step earlier regardless.
    let sc = OpDef::im2col_2d("unfold", ElementKind::F32, (3, 3), (1, 1), (1, 1), (1, 1));
    let a = OperandDesc::new(4, &[2, 3, 8, 8], &[192, 64, 8, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(3, &[2, 27, 64], &[27 * 64, 64, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    let kernel = generate(&sc, &key, &Cuda);
    assert!(
        contract(&sc, &key, &kernel, &Cuda).is_none(),
        "an im2col must emit NO contract (no Fuel Im2Col/Unfold OpKind — conv is a \
             first-class primitive; AOT-only honest miss)"
    );
    assert!(matches!(
        unpopped::derive_pattern(&sc),
        Err(PatternError::NotElementwise)
    ));
}

#[test]
fn viewed_op_is_an_honest_miss_no_contract() {
    use unpopped::ir::View;
    use unpopped::pattern::PatternError;
    // A fused transpose-elementwise (relu(x^T)) computes body(transpose(x)),
    // but the Op+Bind pattern grammar (no layout node, no attrs channel) can't
    // express the transpose — advertising `op_kind: Relu` would bind where
    // Fuel's graph has relu(transpose(x)). Honest miss (kernel still AOT-runs).
    let op = OpDef::elementwise("relu_t", 1, &[ElementKind::F32], input(0).relu())
        .with_views(vec![View::Permute { perm: vec![1, 0] }]);
    let key = key_for(2, OpCategory::UnaryElementwise);
    let kernel = generate(&op, &key, &Cuda);
    assert!(
        contract(&op, &key, &kernel, &Cuda).is_none(),
        "a viewed op must emit NO contract (the transpose is inexpressible)"
    );
    assert!(matches!(
        unpopped::derive_pattern(&op),
        Err(PatternError::ViewUnsupported)
    ));
}

#[test]
fn fused_body_gather_is_an_honest_miss_no_contract() {
    use unpopped::ir::{OobPolicy, ReadIndex, UnaryOp};
    use unpopped::pattern::PatternError;
    // The recipe wiring covers the IDENTITY gather `data[index]` only. A FUSED
    // gather body (elementwise-over-gather, e.g. `relu(gather)`) is not yet
    // expressible as a single `gather[…]` recipe node, so `semantics_dag` returns
    // None (never a mis-described recipe) and the op stays an honest miss (AOT-
    // only; the kernel still runs). Uses a NON-u32 (recipe-path) index so it does
    // not ride the u32 op_kind path — the pure recipe-scope guard.
    let op = OpDef::elementwise(
        "fused_gather",
        2,
        &[ElementKind::F32],
        input(0).unary(UnaryOp::Relu),
    )
    .with_indexed(vec![
        ReadIndex::Indexed {
            index_operand: 1,
            axis: 0,
            oob: OobPolicy::Skip,
            index_dtype: ElementKind::I32,
        },
        ReadIndex::Direct,
    ]);
    let key = gather_key(ElementKind::I32, false);
    let kernel = generate(&op, &key, &Cuda);
    assert!(
        contract(&op, &key, &kernel, &Cuda).is_none(),
        "a fused-body gather must emit NO contract (v1 covers the identity gather only)"
    );
    assert_eq!(unpopped::recipe::semantics_dag(&op), None);
    assert!(matches!(
        unpopped::derive_pattern(&op),
        Err(PatternError::GatherUnsupported)
    ));
}

#[test]
fn scattered_op_is_an_honest_miss_no_contract() {
    use unpopped::pattern::PatternError;
    // Scatter stays a full honest miss even at u32 (see
    // `u32_scatter_family_stays_honest_miss`): scatter (no bare Scatter
    // op_kind), scatter_add/index_add (Fuel's `[T,U32,T,T]` 4-operand key vs
    // Baracuda's in-place 3-tuple — an ARITY mismatch), and the FP-atomic
    // determinism block is unauthored. AOT-only for BOTH cases here (i32).
    let key = key_for(3, OpCategory::BinaryElementwise);
    for op in [
        OpDef::scatter("scatter", &[ElementKind::F32], 0, ElementKind::I32),
        OpDef::scatter_add("scatter_add", &[ElementKind::F32], 0, ElementKind::I32),
    ] {
        let kernel = generate(&op, &key, &Cuda);
        assert!(
            contract(&op, &key, &kernel, &Cuda).is_none(),
            "a scattered op must emit NO contract"
        );
        assert!(matches!(
            unpopped::derive_pattern(&op),
            Err(PatternError::ScatterUnsupported)
        ));
    }
}

#[test]
fn offsetted_op_is_an_honest_miss_no_contract() {
    use unpopped::ir::BaseOffset;
    use unpopped::pattern::PatternError;
    // A runtime-offsetted op's kernel ABI requires the trailing `long long
    // off{i}` scalars the FKC accept block cannot convey (`start_offset`
    // stays truthful `rejected`; the frozen envelope has no off slot) —
    // emitting a contract would advertise an ABI Fuel launches without the
    // off args (OOB base-pointer bump). Honest miss, dual-gated: the
    // pattern's `OffsetUnsupported` AND `contract()`'s own up-front
    // `op_has_offset` guard (load-bearing for the gather-advert path below,
    // which never consults the pattern).
    let op = OpDef::elementwise("addoff", 2, &[ElementKind::F32], input(0) + input(1))
        .with_base_offsets(
            vec![BaseOffset::Runtime, BaseOffset::Zero],
            BaseOffset::Zero,
        );
    let key = key_for(3, OpCategory::BinaryElementwise);
    let kernel = generate(&op, &key, &Cuda);
    assert!(
        kernel.name.contains("_off0"),
        "precondition: the lowered kernel really is the offsetted ABI: {}",
        kernel.name
    );
    assert!(
        contract(&op, &key, &kernel, &Cuda).is_none(),
        "an offsetted op must emit NO contract (the off-arg ABI is inexpressible)"
    );
    assert!(matches!(
        unpopped::derive_pattern(&op),
        Err(PatternError::OffsetUnsupported)
    ));
}

#[test]
fn offsetted_u32_gather_is_an_honest_miss_no_contract() {
    use unpopped::ir::{BaseOffset, OobPolicy};
    use unpopped::pattern::PatternError;
    // THE bypass this guard exists for: a u32-index gather is advertisable
    // (Model A, structural op_kind — `derive_pattern` is never consulted),
    // so without `contract()`'s own `op_has_offset` guard an offsetted u32
    // gather would emit a FULL contract — `op_kind: Gather`, the `_off0`
    // entry point, `start_offset: rejected` — for a kernel whose ABI needs
    // a `long long off0` Fuel will never pass. The offset-free twin (see
    // `u32_gather_emits_a_keyed_contract…`) proves the advert path is
    // otherwise green, so THIS op reaches the offset guard and dies there.
    let op = OpDef::gather(
        "gather",
        &[ElementKind::F32],
        0,
        OobPolicy::Skip,
        ElementKind::U32,
    )
    .with_base_offsets(
        vec![BaseOffset::Runtime, BaseOffset::Zero],
        BaseOffset::Zero,
    );
    let key = gather_key(ElementKind::U32, false);
    let kernel = generate(&op, &key, &Cuda);
    assert!(
        kernel.name.contains("_off0"),
        "precondition: the lowered kernel really is the offsetted ABI: {}",
        kernel.name
    );
    assert!(
        contract(&op, &key, &kernel, &Cuda).is_none(),
        "an offsetted u32 gather must emit NO contract (the gather advert \
             must not bypass the offset guard)"
    );
    // The pattern side misses too — as GatherUnsupported (gather precedes
    // offset in `derive_pattern`'s check order), which is exactly why the
    // pattern miss alone could never guard this path.
    assert!(matches!(
        unpopped::derive_pattern(&op),
        Err(PatternError::GatherUnsupported)
    ));
}

#[test]
fn u32_gather_emits_a_keyed_contract_with_per_operand_dtype_and_oob() {
    use unpopped::ir::OobPolicy;
    // A u32-index torch-gather (full-shape index) is now HONESTLY advertisable
    // (Model A): op_kind Gather, the accept block carries the mixed-dtype tuple
    // [F32, U32, F32] (index slot U32, data slot F32) so Fuel assembles the
    // key `[T, U32, T]`, and oob_policy declares the skip semantics.
    let op = OpDef::gather(
        "gather",
        &[ElementKind::F32],
        0,
        OobPolicy::Skip,
        ElementKind::U32,
    );
    let key = gather_key(ElementKind::U32, false);
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    // Verified Fuel op_kind (fuel-dispatch fkc/lower.rs lower_op_kind).
    assert!(c.contains("op_kind: Gather"), "{c}");
    // oob_policy field present + skip.
    assert!(c.contains("oob_policy: skip"), "{c}");
    // Per-operand accept dtypes: data F32 + index U32 (order = [data, index]).
    // PLURAL `dtypes: [..]` — the field Fuel's importer actually reads
    // (review-confirmed: singular `dtype:` is silently dropped → BadScalarType).
    // Each operand now carries a `name: in{i}` role (item 4) above its dtypes.
    assert!(
        c.contains("    - name: in0\n      dtypes: [F32]\n"),
        "data slot in0 F32: {c}"
    );
    assert!(
        c.contains("    - name: in1\n      dtypes: [U32]\n"),
        "index slot in1 U32: {c}"
    );
    // The ImplId dtype channel stays the DATA (cell) dtype.
    assert!(c.contains("dtypes: [F32]"));
    // entry_point carries the u32 index infix.
    assert!(
        c.contains("entry_point: baracuda_gen_gather_f32_u32_strided_r2"),
        "{c}"
    );
    // A gather forces the strided schedule ⇒ elements.
    assert!(c.contains("count_unit: elements"));
}

#[test]
fn u32_index_select_emits_index_select_op_kind() {
    use unpopped::ir::OobPolicy;
    // A 1-D u32 index ⇒ IndexSelect (structurally, from the index broadcast
    // mask), skip OOB.
    let op = OpDef::index_select(
        "isel",
        &[ElementKind::F32],
        0,
        OobPolicy::Skip,
        ElementKind::U32,
    );
    let key = gather_key(ElementKind::U32, true);
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    assert!(c.contains("op_kind: IndexSelect"), "{c}");
    assert!(c.contains("oob_policy: skip"), "{c}");
    assert!(c.contains("    - name: in1\n      dtypes: [U32]\n"), "{c}");
    // Output = index shape ≠ data shape ⇒ shape_rule omitted (same reason as
    // the u32 gather test); dtype stays the gathered data dtype.
    assert!(!c.contains("shape_rule"), "{c}");
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
}

#[test]
fn u32_embedding_emits_index_select_with_zero_fill() {
    // embedding is a 1-D-index row gather with ZeroFill OOB ⇒ IndexSelect +
    // oob_policy zero_fill (Fuel has no `Embedding` op_kind; the zero_fill vs
    // Fuel's `error` mismatch is made explicit in the field).
    let op = OpDef::embedding("emb", &[ElementKind::F32], ElementKind::U32);
    let key = gather_key(ElementKind::U32, true);
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    assert!(c.contains("op_kind: IndexSelect"), "{c}");
    assert!(c.contains("oob_policy: zero_fill"), "{c}");
    // Output = index shape ≠ data shape ⇒ shape_rule omitted; dtype = data.
    assert!(!c.contains("shape_rule"), "{c}");
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
}

#[test]
fn u32_gather_omits_shape_rule_output_is_the_index_shape() {
    use unpopped::ir::OobPolicy;
    // A u32-index gather rides the op_kind path (op_kind: Gather ⇒ NOT
    // recipe_carrying), so it takes the elementwise return branch. But a
    // gather's output shape is the INDEX shape, not the DATA shape, so
    // `shape_rule: same_as(in0=data)` is a FALSE claim — the SAME reason the
    // i32/i64 gather recipe path omits shape_rule. Fuel's now-live
    // `eval_shape_rule` (return_check.rs) would `ShapeRuleMismatch` a
    // resized-axis gather. So the u32 gather must OMIT shape_rule too; its
    // dtype_rule (passthrough(in0) = the gathered data dtype) stays correct.
    let op = OpDef::gather(
        "gather",
        &[ElementKind::F32],
        0,
        OobPolicy::Skip,
        ElementKind::U32,
    );
    let key = gather_key(ElementKind::U32, false);
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    // Still the op_kind primitive advert (unchanged).
    assert!(c.contains("op_kind: Gather"), "{c}");
    // Output shape = index shape ≠ data shape ⇒ NO same_as(in0) claim.
    assert!(
        !c.contains("shape_rule"),
        "u32 gather out shape = index shape, shape_rule must be omitted:\n{c}"
    );
    // dtype is still the gathered data dtype.
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
}

#[test]
fn i32_gather_advertises_a_recipe_carrying_contract() {
    use unpopped::ir::OobPolicy;
    use unpopped::pattern::PatternError;
    // A non-u32 (i32/i64) index gather is NOT a Fuel graph primitive — Fuel's
    // op_kind `Gather` keys the index as a FIXED U32 slot (`[T, U32, T]`), so an
    // i32/i64 index is unreachable from a Fuel graph node and carries NO
    // `op_kind: Gather` advert. But Fuel's pinned `gather` RECIPE schema admits
    // index_dtype ∈ {u32,i32,i64}, so it now advertises a recipe-carrying
    // `fused_op` contract (the `gather[…]` node), admitted ONLY to a recipe-import
    // peer — the previously honest-missed gather retired to the recipe-import path
    // (the kernel still runs AOT). Complements the u32 op_kind advert.
    let op = OpDef::gather(
        "gather",
        &[ElementKind::F32],
        0,
        OobPolicy::Skip,
        ElementKind::I32,
    );
    let key = gather_key(ElementKind::I32, false);
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).expect("recipe-carrying contract");
    // Recipe advert, NOT the u32 op_kind primitive.
    assert!(c.contains("fused_op: gather"), "{c}");
    assert!(
        !c.contains("op_kind: Gather"),
        "no op_kind for a non-u32 gather: {c}"
    );
    assert!(c.contains("semantics: gather[0,skip,i32](in0, in1)"), "{c}");
    // The gather output shape = the index shape ≠ same_as(in0) (the data), so no
    // FKC shape_rule form fits → OMITTED (the recipe carries the shape); dtype is
    // the gathered data dtype → passthrough(in0), a real form Fuel interprets.
    assert!(
        !c.contains("shape_rule"),
        "shape rides the recipe, omitted:\n{c}"
    );
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    // The index operand's accept slot still carries its REAL dtype (I32), never
    // the data dtype — an honest per-operand gloss on the recipe path too.
    assert!(
        c.contains("    - name: in1\n      dtypes: [I32]\n"),
        "index slot in1 I32: {c}"
    );
    // Honest-miss discipline preserved: WITHHELD from a non-recipe-import bundle,
    // admitted only to a recipe-import peer.
    assert!(
        !contract_admissible(&c, false),
        "withheld without recipe-import"
    );
    assert!(
        contract_admissible(&c, true),
        "admitted to a recipe-import peer"
    );
    // End-to-end at the bundle seam: `bundle` (recipe_import=false) withholds it;
    // `bundle_kisc` admits it ONLY to a recipe-import peer.
    let withheld = bundle("baracuda", "cuda", "rev0", std::slice::from_ref(&c));
    assert!(!withheld.contains("fused_op: gather"), "{withheld}");
    let admitted = bundle_kisc("baracuda", "cuda", "rev0", std::slice::from_ref(&c), true);
    assert!(
        admitted.contains(&unpopped::kisc::kisc_frame(&c)),
        "recipe-carrying i32 gather admitted for a recipe-import peer: {admitted}"
    );
    // derive_pattern still rejects it — the recipe, not a pattern, is its identity.
    assert!(matches!(
        unpopped::derive_pattern(&op),
        Err(PatternError::GatherUnsupported)
    ));
}

#[test]
fn select_fusion_contract_is_withheld() {
    use unpopped::ir::{BinaryOp, OobPolicy};
    use unpopped::pattern::PatternError;
    // WHERE/SELECT (M10's target): ANY select-containing body has its
    // contract withheld wholesale — the Where advert needs the Model-A
    // per-operand tuple (cond U8) / fuel-side matcher validation, neither
    // of which exists in v1.
    //
    // (a) A cmp-free select body (cond = a raw input): the CMP honesty
    // gate does not fire here, so the select guard is the withholding
    // layer on the plain-elementwise path too.
    let sel = OpDef::elementwise(
        "sel",
        3,
        &[ElementKind::F32],
        input(0).select(input(1), input(2)),
    );
    let key = key_for(4, OpCategory::TernaryElementwise);
    let kernel = generate(&sel, &key, &Cuda);
    assert!(
        contract(&sel, &key, &kernel, &Cuda).is_none(),
        "a select body must emit NO contract"
    );
    // The pattern side misses typed too — but the miss does NOT
    // substitute for the contract guard (see (c)).
    assert_eq!(
        unpopped::derive_pattern(&sel),
        Err(PatternError::SelectUnsupported)
    );
    // (b) The fused-cmp form withholds as well (dual-gated with the cmp
    // honesty gate).
    let fused = OpDef::elementwise(
        "sel_cmp",
        4,
        &[ElementKind::F32],
        input(0)
            .binary(BinaryOp::CmpGe, input(1))
            .select(input(2), input(3)),
    );
    let key5 = key_for(5, OpCategory::TernaryElementwise);
    let kf = generate(&fused, &key5, &Cuda);
    assert!(contract(&fused, &key5, &kf, &Cuda).is_none());
    // (c) The LOAD-BEARING path: the Model-A u32-index gather advert
    // derives its op_kind STRUCTURALLY (never consults the pattern), so
    // WITHOUT the expr_contains_select guard a select body inside a
    // u32 gather would sail past the pattern miss and ADVERTISE. Prove
    // the guard holds there — and that the select is the only reason
    // (the select-free sibling advertises).
    let mut gsel = OpDef::gather(
        "gather_sel",
        &[ElementKind::F32],
        0,
        OobPolicy::Skip,
        ElementKind::U32,
    );
    gsel.body = input(0).select(input(0), unpopped::ir::konst(0.0)).0;
    let gkey = gather_key(ElementKind::U32, false);
    let gk = generate(&gsel, &gkey, &Cuda);
    assert!(
        contract(&gsel, &gkey, &gk, &Cuda).is_none(),
        "a select body inside a u32 gather must NOT advertise (the gather \
             op_kind path never consults the pattern — the select guard is the \
             only layer)"
    );
    let plain = OpDef::gather(
        "gather",
        &[ElementKind::F32],
        0,
        OobPolicy::Skip,
        ElementKind::U32,
    );
    let pk = generate(&plain, &gkey, &Cuda);
    assert!(
        contract(&plain, &gkey, &pk, &Cuda).is_some(),
        "the select-free gather sibling must still advertise (the None above \
             comes from the select guard, not some other withhold)"
    );
}

#[test]
fn i32_and_i64_gather_advertise_recipe_carrying_not_op_kind() {
    use unpopped::ir::OobPolicy;
    // COMPLEMENT (not supersede) the u32 op_kind path: Fuel is U32-index for its
    // graph PRIMITIVE (`op_kind: Gather` keys `[T, U32, T]`), so an i32/i64 index
    // carries NO op_kind — but Fuel's pinned `gather` RECIPE schema admits
    // index_dtype ∈ {u32,i32,i64}, so both now advertise a recipe-carrying
    // `fused_op` contract with the index dtype in the `gather[…]` node, admitted
    // only to a recipe-import peer. (The u32 twin stays on op_kind — see
    // `u32_gather_emits_a_keyed_contract…`.)
    for (dt, tok) in [(ElementKind::I32, "i32"), (ElementKind::I64, "i64")] {
        let op = OpDef::gather("gather", &[ElementKind::F32], 0, OobPolicy::Skip, dt);
        let key = gather_key(dt, false);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, &Cuda)
            .unwrap_or_else(|| panic!("{dt:?} gather must advertise a recipe-carrying contract"));
        assert!(c.contains("fused_op: gather"), "{c}");
        assert!(
            !c.contains("op_kind: Gather"),
            "a non-u32 gather carries no op_kind: {c}"
        );
        assert!(
            c.contains(&format!("semantics: gather[0,skip,{tok}](in0, in1)")),
            "{c}"
        );
        assert!(
            !contract_admissible(&c, false) && contract_admissible(&c, true),
            "withheld pre-recipe, admitted to a recipe-import peer: {c}"
        );
    }
}

#[test]
fn u32_scatter_family_stays_honest_miss() {
    // The WRITE side is NOT lifted even at u32: scatter (no bare Scatter
    // op_kind), scatter_add/index_add (4-operand `[T,U32,T,T]` key vs
    // Baracuda's in-place 3-tuple — an operand-arity mismatch), bincount (no
    // Bincount op_kind). All honest misses.
    let key3 = key_for(3, OpCategory::BinaryElementwise);
    for op in [
        OpDef::scatter("scatter", &[ElementKind::F32], 0, ElementKind::U32),
        OpDef::scatter_add("scatter_add", &[ElementKind::F32], 0, ElementKind::U32),
        OpDef::index_add("index_add", &[ElementKind::F32], 0, ElementKind::U32),
    ] {
        let kernel = generate(&op, &key3, &Cuda);
        assert!(
            contract(&op, &key3, &kernel, &Cuda).is_none(),
            "a u32 scatter/scatter_add/index_add must stay an honest miss"
        );
    }
    // bincount (Const body, self-index) at u32 — also a miss.
    let x = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::U32, 256);
    let o = OperandDesc::new(1, &[256], &[1], ElementKind::I32, 256);
    let bk = structure_key(OpCategory::Indexing, &[x, o], ArchSku::Sm89);
    let bc = OpDef::bincount("bincount", ElementKind::U32);
    let bkern = generate(&bc, &bk, &Cuda);
    assert!(
        contract(&bc, &bk, &bkern, &Cuda).is_none(),
        "bincount stays a miss"
    );
}

#[test]
fn uniform_op_accept_block_is_unchanged_by_the_model_a_fix() {
    // The per-operand-dtype accept fix must be NEUTRAL for a non-gather op:
    // every input stays the uniform key dtype and NO oob_policy field appears
    // (only the shared bundle-schema framing — named inputs + layout map —
    // differs, never a gather's U32/oob channel).
    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let key = key_for(3, OpCategory::BinaryElementwise);
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    assert!(
        !c.contains("oob_policy"),
        "no oob_policy on a uniform op: {c}"
    );
    assert!(!c.contains("U32"), "no U32 slot on a uniform op: {c}");
    // Both inputs are F32 — named plural `dtypes: [F32]` (the Fuel-readable
    // form; 6-space indent under the `- name:` line).
    assert_eq!(
        c.matches("      dtypes: [F32]\n").count(),
        2,
        "both inputs F32: {c}"
    );
}

#[test]
fn advertised_gather_op_kind_is_a_verified_fuel_string() {
    use unpopped::ir::OobPolicy;
    // Contract-import sanity: the emitted op_kind must be one of the exact
    // strings Fuel's `lower_op_kind` table accepts (else the whole bundle
    // fails import). Gather + IndexSelect are both in that table.
    const FUEL_OK: [&str; 2] = ["Gather", "IndexSelect"];
    for (one_d, _want) in [(false, "Gather"), (true, "IndexSelect")] {
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::U32,
        );
        let key = gather_key(ElementKind::U32, one_d);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, &Cuda).unwrap();
        let line = c
            .lines()
            .find(|l| l.starts_with("op_kind: "))
            .expect("op_kind line");
        let spelled = line.trim_start_matches("op_kind: ");
        assert!(
            FUEL_OK.contains(&spelled),
            "op_kind '{spelled}' not a Fuel string"
        );
    }
}

#[test]
fn identity_view_still_advertises_a_contract() {
    use unpopped::ir::View;
    // The view guard is PRECISE to address-affecting views: an all-Identity
    // view (an identity linear map) leaves the body-over-inputs pattern exactly
    // correct, so the op still advertises — same as view-free.
    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1))
        .with_views(vec![View::Identity, View::Identity]);
    let key = key_for(3, OpCategory::BinaryElementwise);
    let kernel = generate(&op, &key, &Cuda);
    assert!(
        contract(&op, &key, &kernel, &Cuda).is_some(),
        "an all-Identity view must not suppress the contract"
    );
}

#[test]
fn broadcast_bias_add_cell_is_withheld() {
    // NEGATIVE PIN (finding 1): a Broadcast-class operand's stride-0 mask is
    // BAKED into the kernel (fully-broadcast → `in1[0]`), unspeakable in Fuel's
    // tri-state ⇒ the contract is WITHHELD. The kernel still generates.
    for strides in [&[0i64, 0][..], &[0, 1][..]] {
        let (op, key) = bias_add_key(strides);
        let kernel = generate(&op, &key, &Cuda);
        assert_eq!(
            key.operands[1].contig,
            Contiguity::Broadcast,
            "in1 strides {strides:?} must key Broadcast"
        );
        assert!(!kernel.source.is_empty(), "the kernel still lowers (AOT)");
        assert!(
            contract(&op, &key, &kernel, &Cuda).is_none(),
            "a baked-broadcast bias-add cell must emit NO contract (strides {strides:?})"
        );
    }
}

#[test]
fn flipped_operand_cell_is_withheld() {
    // NEGATIVE PIN (finding 3): a reverse-stride (flipped) operand keys the
    // reversed cell, but the Elementwise schedule reads it FORWARD — the kernel
    // does not implement the cell it is keyed to, and Fuel has no truthful
    // spelling ⇒ WITHHELD (so `reverse_strides: rejected` stays honest).
    let op = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu());
    let rev = OperandDesc::new(2, &[128, 256], &[-256, 1], ElementKind::F32, 256);
    let out = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[rev, out], ArchSku::Sm89);
    assert!(key.operands[0].flipped, "in0 must key flipped");
    let kernel = generate(&op, &key, &Cuda);
    assert!(!kernel.source.is_empty(), "the kernel still lowers (AOT)");
    assert!(
        contract(&op, &key, &kernel, &Cuda).is_none(),
        "a flipped-operand cell must emit NO contract"
    );
}

#[test]
fn strided_cell_layout_spec_accepts_strided_and_broadcast_stride0() {
    // FINDING 2 + 17: for an InnerContig/Strided operand the kernel walks full
    // runtime strides, so a stride-0 axis is handled → `strided: accepted,
    // broadcast_stride0: accepted` (projects Fuel `strided_input = true`). Pin
    // the exact inline map + negative-pin the old `broadcast_stride0: rejected`.
    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    // [8,4] strides [1,8] ⇒ inner axis 0 stride 1, not row-major ⇒ InnerContig.
    let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[t, t, t], ArchSku::Sm89);
    assert!(matches!(
        key.operands[0].contig,
        Contiguity::InnerContig | Contiguity::Strided
    ));
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    assert!(
        c.contains(
            "layout: { contiguous: accepted, strided: accepted, \
                 broadcast_stride0: accepted, start_offset: rejected, \
                 reverse_strides: rejected }"
        ),
        "strided operand accepts strided + broadcast_stride0: {c}"
    );
    assert!(
        !c.contains("strided: accepted, broadcast_stride0: rejected"),
        "the old under-claim must not leak: {c}"
    );
    // Fuel Rule-4 coherence: broadcast accepted ⇒ strided accepted (holds).
    // caps.awkward_layout_strategy for a strided operand-0 stays handles_strided.
    assert!(
        c.contains("awkward_layout_strategy: handles_strided"),
        "{c}"
    );
}

#[test]
fn gather_index_operand_layout_is_contiguous_required_not_baked_broadcast() {
    // FINDING 1 EXCEPTION + 17: the ONE Broadcast-class operand we advertise is
    // the u32 index_select INDEX. Its physical index buffer is emitted
    // truthfully as `contiguous: required` (conservative, Fuel `[T,U32,T]`),
    // NEVER the old over-accepting `broadcast_stride0: accepted`.
    use unpopped::ir::OobPolicy;
    let op = OpDef::index_select(
        "isel",
        &[ElementKind::F32],
        0,
        OobPolicy::Skip,
        ElementKind::U32,
    );
    let key = gather_key(ElementKind::U32, true);
    assert_eq!(
        key.operands[1].contig,
        Contiguity::Broadcast,
        "index keys Broadcast"
    );
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    // The index (in1) slot: U32 dtype, contiguous-required layout.
    assert!(
        c.contains(
            "- name: in1\n      dtypes: [U32]\n      layout: { contiguous: required, \
                 strided: rejected, broadcast_stride0: rejected, start_offset: rejected, \
                 reverse_strides: rejected }"
        ),
        "index operand layout is contiguous-required: {c}"
    );
    assert!(
        !c.contains("dtypes: [U32]\n      layout: { contiguous: accepted"),
        "the index must not over-accept a baked broadcast: {c}"
    );
}

#[test]
fn count_unit_matches_the_emitted_abi() {
    use baracuda_cuda_emit::Cuda;
    use unpopped::generate;
    let c = |op: &OpDef, key: &StructureKey| {
        let k = generate(op, key, &Cuda);
        contract(op, key, &k, &Cuda).unwrap()
    };
    let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    // f32 contiguous/aligned → float4 kernel: n counts 4-element vectors.
    let kf = key_for(3, OpCategory::BinaryElementwise);
    assert!(c(&add, &kf).contains("count_unit: vectors_x4"));
    // f16 contiguous/aligned → packed half2 V8 kernel: 8-element vectors.
    let h = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F16, 256);
    let addh = OpDef::elementwise("add", 2, &[ElementKind::F16], input(0) + input(1));
    let kh = structure_key(OpCategory::BinaryElementwise, &[h, h, h], ArchSku::Sm89);
    assert!(c(&addh, &kh).contains("count_unit: vectors_x8"));
    // i32 keys V4 but has no int vector/packed path → the SCALAR fallback:
    // the contract must say elements, mirroring the emitted ABI, not the key.
    let gi = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::I32, 256);
    let addi = OpDef::elementwise("add", 2, &[ElementKind::I32], input(0) + input(1));
    let ki = structure_key(OpCategory::BinaryElementwise, &[gi, gi, gi], ArchSku::Sm89);
    assert!(c(&addi, &ki).contains("count_unit: elements"));
    // Strided cell → elements.
    let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
    let kt = structure_key(OpCategory::BinaryElementwise, &[t, t, t], ArchSku::Sm89);
    assert!(c(&add, &kt).contains("count_unit: elements"));
}

#[test]
fn reduction_advertises_a_recipe_carrying_contract() {
    use unpopped::ir::ReduceOp;
    use unpopped::pattern::PatternError;
    use unpopped_vocab::AxisMask;
    // A general-path reduction is not an elementwise pattern, but it carries a
    // `reduce[…]` recipe — so it advertises a recipe-carrying contract admitted
    // only to a recipe-import peer. Its output shape+dtype ≠ its input (the axis
    // is reduced away), so BOTH defer to the recipe (`from_recipe`), never a
    // false `same_as(in0)`/`passthrough(in0)`.
    let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let op = OpDef::reduction_axes(
        "sum",
        1,
        &[ElementKind::F32],
        input(0),
        ReduceOp::Sum,
        AxisMask(0b01),
        false,
    );
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).expect("recipe-carrying contract");
    assert!(c.contains("fused_op: sum"), "{c}");
    assert!(c.contains("semantics: reduce[sum,0x1,nokd](in0)"), "{c}");
    assert!(
        !c.contains("shape_rule"),
        "shape rides the recipe, omitted:\n{c}"
    );
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    assert!(
        !contract_admissible(&c, false),
        "withheld without recipe-import"
    );
    assert!(
        contract_admissible(&c, true),
        "admitted to a recipe-import peer"
    );
    assert!(matches!(
        derive_pattern(&op),
        Err(PatternError::NotElementwise)
    ));
}

#[test]
fn prod_and_hetero_out_reductions_advertise_recipe_carrying_contracts() {
    use unpopped::ir::{BinaryOp, ReduceOp, konst, reduced};
    // The 0e reductions (Prod combiner; boolean/count hetero-out via a Cmp*
    // post) carry a `reduce[…]` recipe — so they advertise recipe-carrying
    // contracts (admitted only to a recipe-import peer). Fuel resolves the
    // recipe primitives even though it has no ProdReduce/Any OpKind: that's the
    // whole point of recipe-import. Shape rides the recipe (omitted); the
    // hetero-out dtype declares `fixed(<dtype>)`, where `passthrough(in0)`
    // would state the wrong type.
    let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);

    // (a) Prod (uniform f32 out).
    let prod_out = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
    let pk = structure_key(OpCategory::Reduction, &[a, prod_out], ArchSku::Sm89);
    let prod = OpDef::reduction("p", 1, &[ElementKind::F32], input(0), ReduceOp::Prod);
    let cp = contract(&prod, &pk, &generate(&prod, &pk, &Cuda), &Cuda)
        .expect("recipe-carrying contract");
    assert!(cp.contains("fused_op: p"), "{cp}");
    assert!(
        cp.contains("semantics: reduce[prod,last,nokd](in0)"),
        "{cp}"
    );
    assert!(!cp.contains("shape_rule"), "{cp}");
    assert!(cp.contains("dtype_rule: passthrough(in0)"), "{cp}");
    assert!(!contract_admissible(&cp, false) && contract_admissible(&cp, true));

    // (b) hetero-out any (Sum(x!=0) → u8 via a Cmp* post) — dtype from the recipe.
    let any_out = OperandDesc::new(1, &[256], &[1], ElementKind::U8, 256);
    let ak = structure_key(OpCategory::Reduction, &[a, any_out], ArchSku::Sm89);
    let mut any = OpDef::reduction_post(
        "any",
        1,
        &[ElementKind::F32],
        input(0).binary(BinaryOp::CmpNe, konst(0.0)),
        ReduceOp::Sum,
        reduced(0).binary(BinaryOp::CmpGt, konst(0.0)),
    );
    any.out_dtype = Some(ElementKind::U8);
    let ca =
        contract(&any, &ak, &generate(&any, &ak, &Cuda), &Cuda).expect("recipe-carrying contract");
    assert!(ca.contains("fused_op: any"), "{ca}");
    assert!(
        ca.contains("semantics: cmp_gt(reduce[sum,last,nokd](cmp_ne(in0, const(0))), const(0))"),
        "{ca}"
    );
    // Hetero U8 output → `fixed(U8)` (a real FKC dtype form Fuel interprets);
    // shape rides the recipe (omitted).
    assert!(ca.contains("dtype_rule: fixed(U8)"), "{ca}");
    assert!(!ca.contains("shape_rule"), "{ca}");
    assert!(!contract_admissible(&ca, false) && contract_admissible(&ca, true));
}

#[test]
fn front_matter_has_provider_and_seam_profiles() {
    let fm = front_matter("baracuda", "cuda", "abc123");
    assert!(fm.contains("fkc_version: 1"));
    assert!(fm.contains("name: baracuda"));
    assert!(fm.contains("link_registry: baracuda_link_registry"));
    assert!(fm.contains("seam_profiles: [1]"));
    assert!(fm.contains("revision_base: \"abc123\""));
    // Item-1 casing: the lowercase provider token is canonicalized to Fuel's
    // capitalized wire spelling (`lower_backend` accepts `Cuda`, not `cuda`).
    assert!(fm.contains("backend: Cuda\n"), "{fm}");
    assert!(
        !fm.contains("backend: cuda"),
        "lowercase backend must not leak: {fm}"
    );
}

#[test]
fn bundle_frames_each_contract_under_a_heading() {
    // Fuel's parser SILENTLY drops a headingless ```fkc block (a zero-kernel
    // file imports Ok-but-empty). `bundle()` frames every contract under its
    // own `## <kernel>` heading so the shared assembler can't reintroduce
    // that hazard. The heading title is the contract's `kernel:` name.
    let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let key = key_for(3, OpCategory::BinaryElementwise);
    let kernel = generate(&add, &key, &Cuda);
    let c = contract(&add, &key, &kernel, &Cuda).unwrap();
    let b = bundle("baracuda", "cuda", "rev0", std::slice::from_ref(&c));
    // Front matter first, then a `## <kernel>` heading, then the block.
    assert!(b.starts_with("---\n"), "front matter leads: {b}");
    let kname = format!("add_{}", cell_suffix(&key));
    assert!(
        b.contains(&format!("\n## {kname}\n")),
        "heading names the kernel: {b}"
    );
    // The heading precedes the fenced block it frames.
    let h = b.find(&format!("## {kname}")).unwrap();
    let fence = b.find("```fkc").unwrap();
    assert!(h < fence, "heading must precede the fkc block: {b}");
}

#[test]
fn contract_emits_the_semantics_recipe_for_an_elementwise_op() {
    let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let key = key_for(3, OpCategory::BinaryElementwise);
    let kernel = generate(&add, &key, &Cuda);
    let c = contract(&add, &key, &kernel, &Cuda).unwrap();
    assert!(
        c.contains("semantics: add(in0, in1)\n"),
        "the neutral KISS-Ops recipe is emitted: {c}"
    );
}

#[test]
fn primitive_add_uses_op_kind_and_carries_required_blocks() {
    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let key = key_for(3, OpCategory::BinaryElementwise);
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();

    // primitive → op_kind, no fused_op, no pattern block. The emitted
    // spelling is the Fuel-importer DISPATCH name (`AddElementwise`), not
    // the internal pattern root (`Add`) — the reconciled arithmetic spelling.
    assert!(c.contains("op_kind: AddElementwise"), "{c}");
    assert!(
        !c.contains("op_kind: Add\n"),
        "internal spelling must not leak: {c}"
    );
    assert!(!c.contains("fused_op:"));
    assert!(!c.contains("pattern:"));
    // ImplId five fields all present + separable. The backend is Fuel's
    // capitalized wire spelling (`Cuda`), NOT the lowercase provider token
    // (`cuda` fails `lower_backend` with UnknownBackend) — item-1 casing.
    assert!(c.contains("backend: Cuda"), "{c}");
    assert!(
        !c.contains("backend: cuda\n"),
        "lowercase backend must not leak: {c}"
    );
    assert!(c.contains("kernel_source: baracuda"));
    assert!(c.contains("dtypes: [F32]"));
    assert!(c.contains("entry_point: "));
    assert!(c.contains("kernel_revision_hash: \""));
    // Bundle-schema reconciliation pins (items 3/4/5), so a representative
    // contract's changed lines are asserted, not assumed:
    //  - item 4: named, index-based accept inputs.
    assert!(c.contains("    - name: in0\n"), "named input in0: {c}");
    assert!(c.contains("    - name: in1\n"), "named input in1: {c}");
    //  - item 3: the five-flag LayoutSpec inline map on the ACCEPT inputs
    //    (not a bare string). A contiguous cell requires contiguous input.
    assert!(
        c.contains(
            "layout: { contiguous: required, strided: rejected, \
                 broadcast_stride0: rejected, start_offset: rejected, \
                 reverse_strides: rejected }"
        ),
        "input layout is the inline LayoutSpec map: {c}"
    );
    assert!(
        !c.contains("layout: contiguous\n"),
        "bare layout string must not leak: {c}"
    );
    //  - item 5: passthrough(in0) output dtype rule (Fuel keys the output),
    //    NOT same_as_input(0) (parses to DtypeRule::Other, output dropped).
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    assert!(!c.contains("dtype_rule: same_as_input(0)"), "{c}");
    //  - item 9: shape_rule spells the §5.2 grammar `same_as(<role>)`, NOT the
    //    out-of-grammar `same_as_input(0)` (negative pin).
    assert!(c.contains("shape_rule: same_as(in0)"), "{c}");
    assert!(
        !c.contains("same_as_input(0)"),
        "old out-of-grammar shape_rule must not leak: {c}"
    );
    //  - item 7: the OUTPUT descriptor carries `layout_guarantee:` (Fuel's
    //    OutputDesc field), never the five-flag `layout:` map (which lives on
    //    accept-inputs only). A contiguous output guarantees contiguous.
    assert!(c.contains("layout_guarantee: contiguous"), "{c}");
    // The `layout:` map appears ONLY under accept.inputs, never in `return:`.
    let ret = &c[c.find("return:").unwrap()..c.find("caps:").unwrap()];
    assert!(
        !ret.contains("layout:"),
        "no five-flag layout: map under return: {ret}"
    );
    //  - item 4: in_place is the Fuel-schema boolean `false` for EVERY cell
    //    (out-of-place kernels; `aliasing: none`), never `true` (the §4.6
    //    inversion) nor the pre-reconcile string.
    assert!(c.contains("  in_place: false\n"), "in_place is false: {c}");
    assert!(
        !c.contains("in_place: true"),
        "no in_place: true (§4.6 inversion): {c}"
    );
    assert!(!c.contains("in_place: allowed"), "no string in_place: {c}");
    assert!(
        c.contains("awkward_layout_strategy: requires_contiguous"),
        "{c}"
    );
    assert!(
        !c.contains("  awkward_layout: "),
        "old awkward_layout key must not leak: {c}"
    );
    //  - item 6: cost carries Fuel's `flops` / `bytes_moved` EXPRESSION keys,
    //    never the silently-dropped `flops_per_elem` / `bytes_per_elem`.
    assert!(c.contains("  flops: \"1 * n\"\n"), "flops expression: {c}");
    assert!(
        c.contains("  bytes_moved: \"12 * n\"\n"),
        "bytes_moved expression: {c}"
    );
    assert!(
        !c.contains("flops_per_elem"),
        "old scalar cost key must not leak: {c}"
    );
    assert!(
        !c.contains("bytes_per_elem"),
        "old scalar cost key must not leak: {c}"
    );
    // required §4.3 blocks.
    for block in [
        "accept:",
        "structure_key: \"sk4|",
        "return:",
        "caps:",
        "cost:",
        "precision:",
        "determinism: bitwise",
    ] {
        assert!(c.contains(block), "missing block: {block}");
    }
    //  - item 5: precision uses ONLY Fuel PrecisionBlock keys — never `mode:`.
    //    Correctly-rounded arithmetic ⇒ bit-stable + max_ulp 0.
    assert!(
        !c.contains("mode:"),
        "non-schema mode: key must not leak: {c}"
    );
    assert!(c.contains("  bit_stable_on_same_hardware: true\n"), "{c}");
    assert!(
        c.contains("  max_ulp: 0\n"),
        "correctly-rounded ⇒ max_ulp 0: {c}"
    );
    assert!(c.contains("  audited: true\n"), "{c}");
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
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    // The BARE contract still names the fusion (`relu_add` has no Fuel FusedOp
    // constant) — it rides the JIT seam, where Fuel stores the text unparsed.
    assert!(c.contains("fused_op: relu_add"));
    assert!(!c.contains("op_kind:"));
    assert!(c.contains("pattern:"));
    assert!(c.contains("op: Relu"));
    // …but `bundle()` WITHHOLDS it (item 8): an unknown `fused_op:` name is
    // bundle-FATAL, so a bundle carrying a correct primitive BESIDE this fusion
    // must contain only the primitive (never fail import).
    let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let ka = key_for(3, OpCategory::BinaryElementwise);
    let ca = contract(&add, &ka, &generate(&add, &ka, &Cuda), &Cuda).unwrap();
    let b = bundle("baracuda", "cuda", "rev0", &[ca.clone(), c.clone()]);
    assert!(
        b.contains("op_kind: AddElementwise"),
        "primitive survives: {b}"
    );
    assert!(
        !b.contains("fused_op: relu_add"),
        "fused advert withheld from bundle: {b}"
    );
    assert!(
        !b.contains("relu_add_"),
        "no relu_add section framed in the bundle: {b}"
    );
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
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    assert!(c.contains("op_params:"));
    assert!(c.contains("name: param0"));
    assert!(c.contains("name: param1"));
    // Precision uses Fuel's PrecisionBlock vocabulary (never `mode:`): a
    // transcendental is bit-stable with a finite declared ULP bound.
    assert!(
        !c.contains("mode:"),
        "non-schema mode: key must not leak: {c}"
    );
    assert!(c.contains("  bit_stable_on_same_hardware: true\n"), "{c}");
    // silu(x*p0 + p1): the silu composite (~3 ulp); arithmetic is exact.
    assert!(c.contains("  max_ulp: 3\n"), "{c}");
    assert!(c.contains("  audited: true\n"), "{c}");
    // F32 op_params carry the `F32` token (byte-identical to the pre-F64
    // hardcode) — the regression pin for the honesty-only dtype-token change.
    assert!(
        c.contains("  - name: param0\n    dtype: F32\n"),
        "f32 param carries the F32 token: {c}"
    );
}

#[test]
fn f64_scalar_param_op_params_carry_the_f64_token() {
    // M6: the honesty-only op_params dtype-token change. A single-output f64
    // param op emits a contract whose op_params carry `dtype: F64` (the real
    // scalar COMPUTE dtype, reusing the accept block's `dtype` token), not the
    // stale hardcoded `F32`.
    let op = OpDef::elementwise(
        "affine_f64",
        1,
        &[ElementKind::F64],
        input(0) * param(0) + param(1),
    );
    let key = key_dtype(ElementKind::F64, 2);
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    assert!(c.contains("op_params:"), "{c}");
    assert!(
        c.contains("  - name: param0\n    dtype: F64\n"),
        "f64 param carries the F64 token: {c}"
    );
    assert!(
        c.contains("  - name: param1\n    dtype: F64\n"),
        "second f64 param also F64: {c}"
    );
    assert!(!c.contains("dtype: F32"), "no stale F32 token: {c}");
}

#[test]
fn bool_dtype_maps_to_u8_not_bool() {
    use unpopped::ir::BinaryOp;
    // §5 (B5/E5): Fuel has no Bool dtype — a provider's Bool rides as U8.
    // Use a LOGICAL op: unpopped 0.2.0 (723fdb0, "bool is its own kind") declines
    // ARITHMETIC at Bool (`true + true` = 2 isn't a value of the dtype) and admits
    // logical ops — the path that actually reaches the FKC contract for a Bool cell.
    let op = OpDef::elementwise(
        "and",
        2,
        &[ElementKind::Bool],
        input(0).binary(BinaryOp::LogicalAnd, input(1)),
    );
    let key = key_dtype(ElementKind::Bool, 3);
    let c = contract(&op, &key, &stub_kernel(), &Cuda).unwrap();
    assert!(c.contains("dtypes: [U8]"));
    assert!(!c.contains("Bool"));
}

#[test]
fn unsupported_dtype_yields_no_contract() {
    // Complex has no FKC §5 base-dtype slot — skip the cell (honest miss),
    // never emit an unbindable `dtypes: [C64]` contract.
    let op = OpDef::elementwise("add", 2, &[ElementKind::Complex128], input(0) + input(1));
    let key = key_dtype(ElementKind::Complex128, 3);
    assert!(contract(&op, &key, &stub_kernel(), &Cuda).is_none());
}

#[test]
fn int_ops_rate_zero_ulp_and_carry_recipe_contracts() {
    use unpopped::ir::BinaryOp;
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
    // Precision table: all eight rated 0 EXHAUSTIVELY (bit-exact wrapping
    // int semantics — no rounding step exists), pinned per op so a future
    // arm shuffle can't silently re-rate one.
    for op in INT_OPS {
        // ⚠️ `precision_of` gained an `&AccuracyKey` in unpopped 0.10.0, and the
        // key is NOT a formality here. `ulp_bound` returns INFINITY for any
        // non-`cuda` namespace BEFORE it walks the expression, and
        // `precision_of` maps infinity to ("approximate", None) — so a
        // wrong-target key would flip all eight of these to an unknown bound
        // and this test would be asserting the opposite of what it is named
        // for. sm_89 is the target baracuda emits against, so the CUDA table
        // is the correct one to consult rather than a convenient one.
        let acc = AccuracyKey::for_target(ArchSku::Sm89.into());
        assert_eq!(
            precision_of(&input(0).binary(op, input(1)).0, &acc),
            ("correctly_rounded", Some(0)),
            "{op:?}"
        );
    }
    // Contract (Brief 4): bitwise/logical ops derive NO pattern — neither
    // OpTag 0.10.2 nor `lower_op_kind` names them (`derive_pattern` → the SAME
    // `NoFkcName` Err as before) — but they carry a valid KISS-Ops recipe
    // (`bit_and`/`logical_and` are confirmed floor tokens: `binary_kiss_name`
    // maps them, and Fuel resolves any named floor op — grammar reply Q6). So
    // the pattern-miss withhold is RETIRED: they now advertise a RECIPE-CARRYING
    // elementwise contract (`fused_op:` + `semantics:`, KEEPING the true
    // `same_as(in0)` return block since out shape+dtype = the input's), withheld
    // from a non-recipe-import bundle and admitted to a recipe-import peer. The
    // kernel still generates (bitwise at i32, logical at u8).
    use unpopped::pattern::PatternError;
    let band = OpDef::elementwise(
        "band",
        2,
        &[ElementKind::I32],
        input(0).binary(BinaryOp::BitAnd, input(1)),
    );
    let ki = key_dtype(ElementKind::I32, 3);
    let k = generate(&band, &ki, &Cuda);
    assert!(
        k.source.contains("(in0[i] & in1[i])"),
        "the kernel still lowers"
    );
    let c = contract(&band, &ki, &k, &Cuda).expect("recipe-carrying contract");
    assert!(c.contains("fused_op: band"), "{c}");
    assert!(c.contains("semantics: bit_and(in0, in1)"), "{c}");
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    assert!(c.contains("shape_rule: same_as(in0)"), "{c}");
    assert!(
        !c.contains("op_kind:"),
        "a pattern miss carries no op_kind: {c}"
    );
    assert!(
        !contract_admissible(&c, false) && contract_admissible(&c, true),
        "withheld pre-recipe, admitted to a recipe-import peer: {c}"
    );
    assert!(matches!(
        derive_pattern(&band),
        Err(PatternError::NoFkcName { ref op }) if op == "BitAnd"
    ));
    let land = OpDef::elementwise(
        "land",
        2,
        &[ElementKind::U8],
        input(0).binary(BinaryOp::LogicalAnd, input(1)),
    );
    let ku = key_dtype(ElementKind::U8, 3);
    let kl = generate(&land, &ku, &Cuda);
    assert!(kl.source.contains("!= 0 &&"), "the kernel still lowers");
    let cl = contract(&land, &ku, &kl, &Cuda).expect("recipe-carrying contract");
    assert!(cl.contains("fused_op: land"), "{cl}");
    assert!(cl.contains("semantics: logical_and(in0, in1)"), "{cl}");
    assert!(cl.contains("shape_rule: same_as(in0)"), "{cl}");
    assert!(matches!(
        derive_pattern(&land),
        Err(PatternError::NoFkcName { ref op }) if op == "LogicalAnd"
    ));
}

#[test]
fn uniform_int_add_contracts_carry_the_audited_dtype() {
    // Increment 0c: uniform-U8/S8 COMPUTE is audited, so an infix Add at
    // U8/S8 emits a real contract — dtypes carry the FKC §5 spellings
    // (U8; S8 spells I8), correctly_rounded means exact WRAPPING
    // semantics, and count_unit stays elements (int cells never
    // vectorize — no int vector/packed path exists).
    let addu = OpDef::elementwise("add", 2, &[ElementKind::U8], input(0) + input(1));
    let ku = key_dtype(ElementKind::U8, 3);
    let k = generate(&addu, &ku, &Cuda);
    let c = contract(&addu, &ku, &k, &Cuda).unwrap();
    assert!(c.contains("op_kind: AddElementwise"), "{c}");
    assert!(c.contains("dtypes: [U8]"));
    // correctly-rounded wrapping ⇒ bit-stable + max_ulp 0 (Fuel PrecisionBlock).
    assert!(c.contains("  bit_stable_on_same_hardware: true\n"), "{c}");
    assert!(c.contains("  max_ulp: 0\n"), "{c}");
    assert!(c.contains("count_unit: elements"));
    // 2 u8 reads + 1 u8 write ⇒ bytes_moved expression "3 * n".
    assert!(c.contains("  bytes_moved: \"3 * n\"\n"), "{c}");
    let adds = OpDef::elementwise("add", 2, &[ElementKind::I8], input(0) + input(1));
    let ks = key_dtype(ElementKind::I8, 3);
    let k8 = generate(&adds, &ks, &Cuda);
    let c8 = contract(&adds, &ks, &k8, &Cuda).unwrap();
    assert!(c8.contains("dtypes: [I8]"), "S8 spells I8 on the FKC wire");
    assert!(c8.contains("count_unit: elements"));
}

#[test]
fn cmp_u8_contract_returns_fixed_u8_and_forbids_in_place() {
    use unpopped::ir::BinaryOp;
    let op = OpDef::elementwise_pred(
        "cmp_lt",
        2,
        &[ElementKind::F32],
        input(0).binary(BinaryOp::CmpLt, input(1)),
    );
    let key = pred_key();
    let kernel = generate(&op, &key, &Cuda);
    let c = contract(&op, &key, &kernel, &Cuda).unwrap();
    // Primitive identity: the DISPATCH OpKind spelling — the exact string
    // Fuel's lower_op_kind table accepts (`op_kind: Lt` would typed-reject
    // and fail the whole bundle import).
    assert!(c.contains("op_kind: LessElementwise"), "{c}");
    assert!(!c.contains("op_kind: Lt"), "{c}");
    // The return dtype is HONEST — the §5.1 constant rule Fuel's own
    // compare contracts use, never the input passthrough.
    assert!(c.contains("dtype_rule: fixed(U8)"));
    assert!(!c.contains("dtype_rule: same_as_input(0)"));
    // The ImplId dtype channel stays the key (input) dtype.
    assert!(c.contains("dtypes: [F32]"));
    // A 1-byte store can't alias a 4-byte input buffer — in_place is the
    // Fuel-schema boolean `false` (never the pre-reconcile string).
    assert!(c.contains("in_place: false"));
    assert!(
        !c.contains("in_place: forbidden"),
        "in_place must be a bool, not a string: {c}"
    );
    // Scalar path (no packed u8 store) => n counts elements…
    assert!(c.contains("count_unit: elements"));
    // …and the traffic estimate is 2 f32 reads + 1 u8 write = 9 B/elem.
    assert!(c.contains("  bytes_moved: \"9 * n\"\n"), "{c}");
    // The predicate is exact ⇒ bit-stable + max_ulp 0 (Fuel PrecisionBlock).
    assert!(
        !c.contains("mode:"),
        "non-schema mode: key must not leak: {c}"
    );
    assert!(c.contains("  bit_stable_on_same_hardware: true\n"), "{c}");
    assert!(c.contains("  max_ulp: 0\n"), "{c}");
    assert!(c.contains("determinism: bitwise"));
}

#[test]
fn float_mask_toplevel_cmp_has_no_contract() {
    use unpopped::ir::BinaryOp;
    // A top-level cmp with out_dtype = None stores 1.0f/0.0f in the KEY
    // dtype — not Fuel's "comparison → U8 mask" op. The kernel generates,
    // the pattern even derives (the vocabulary exists), but the contract is
    // withheld: advertising `op_kind: Gt` for a 4-byte-store kernel would
    // bind where Fuel expects a 1-byte mask.
    let op = OpDef::elementwise(
        "gt_mask",
        2,
        &[ElementKind::F32],
        input(0).binary(BinaryOp::CmpGt, input(1)),
    );
    let key = key_for(3, OpCategory::BinaryElementwise);
    let kernel = generate(&op, &key, &Cuda);
    assert!(
        kernel.source.contains("? 1.0f : 0.0f"),
        "the kernel still lowers"
    );
    assert!(
        contract(&op, &key, &kernel, &Cuda).is_none(),
        "but no contract"
    );
    assert!(
        derive_pattern(&op).is_ok(),
        "vocabulary exists; the gate is honesty"
    );
}

#[test]
fn nested_cmp_fusion_contract_is_withheld() {
    use unpopped::ir::BinaryOp;
    // relu-backward mask-multiply: dy * (x > z). The kernel is correct and
    // still generates — but the fused PATTERN would encode Gt as a direct
    // operand of Mul, an edge no constructible Fuel graph has (Fuel's
    // compare builders pin U8 output and its binary ops assert dtype
    // equality, so real graphs interpose Cast(U8→float); Cast is outside
    // the §4.1 pattern grammar and the §4.3 see-through set). Advertising
    // it would register a matcher that can never fire on the graphs it
    // means. Withheld until Cast joins the pattern vocabulary.
    let op = OpDef::elementwise(
        "relu_bw",
        3,
        &[ElementKind::F32],
        input(0) * input(1).binary(BinaryOp::CmpGt, input(2)),
    );
    let key = key_for(4, OpCategory::TernaryElementwise);
    let kernel = generate(&op, &key, &Cuda);
    assert!(
        kernel.source.contains("? 1.0f : 0.0f"),
        "the kernel still lowers"
    );
    assert!(
        contract(&op, &key, &kernel, &Cuda).is_none(),
        "nested-cmp fused contract is withheld (missing-Cast pattern gap)"
    );
    assert!(
        derive_pattern(&op).is_ok(),
        "vocabulary exists; the gate is honesty"
    );
}

#[test]
fn coord_bodies_carry_recipe_contracts_via_recipe_import() {
    use unpopped::ir::{BinaryOp, coord};
    use unpopped::pattern::PatternError;
    // OpTag::Iota exists (0.10.2), but the emitted PATTERN grammar cannot carry
    // its axis attribute, so a Coord body still derives NO pattern (the SAME
    // `CoordUnsupported` Err). The RECIPE, however, expresses it honestly —
    // `iota(axis)` / `cmp_ge` / `mul` are confirmed floor tokens Fuel resolves +
    // numerically verifies, and the recipe is dtype-agnostic (the nested cmp
    // rides the recipe path, exempt from the pattern-grammar missing-`Cast`
    // limit) — so the withhold is RETIRED (Brief 4): the cell advertises a
    // RECIPE-CARRYING elementwise contract (`fused_op:` + `semantics:`, KEEPING
    // the true `same_as(in0)` return block), withheld from a non-recipe-import
    // bundle and admitted to a recipe-import peer. The kernel still generates.
    let triu = OpDef::elementwise(
        "triu_mask",
        1,
        &[ElementKind::F32],
        input(0) * coord(1).binary(BinaryOp::CmpGe, coord(0) + unpopped::ir::konst(0.0)),
    );
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a], ArchSku::Sm89);
    let k = generate(&triu, &key, &Cuda);
    assert!(k.source.contains("(float)c1"), "the kernel still lowers");
    let c = contract(&triu, &key, &k, &Cuda).expect("recipe-carrying contract");
    assert!(c.contains("fused_op: triu_mask"), "{c}");
    assert!(
        c.contains("semantics: mul(in0, cmp_ge(iota(1), add(iota(0), const(0))))"),
        "{c}"
    );
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    assert!(c.contains("shape_rule: same_as(in0)"), "{c}");
    assert!(
        !contract_admissible(&c, false) && contract_admissible(&c, true),
        "withheld pre-recipe, admitted to a recipe-import peer: {c}"
    );
    assert!(matches!(
        derive_pattern(&triu),
        Err(PatternError::CoordUnsupported { .. })
    ));
}

#[test]
fn vocab_ops_carry_recipe_contracts_via_recipe_import() {
    use unpopped::ir::{BinaryOp, UnaryOp};
    use unpopped::pattern::PatternError;
    // Fuel's §4.1/OpTag vocabulary doesn't name the increment-0a fns, so
    // `derive_pattern` still returns the SAME `NoFkcName` Err — but they carry a
    // valid KISS-Ops recipe (`erfc`/`atan2` are confirmed floor tokens Fuel
    // resolves), so the pattern-miss withhold is RETIRED (Brief 4): they now
    // advertise a RECIPE-CARRYING elementwise contract (`fused_op:` +
    // `semantics:`, KEEPING the true `same_as(in0)` return block), withheld from
    // a non-recipe-import bundle and admitted to a recipe-import peer. The kernel
    // still generates (lowering is unaffected).
    let erfc = OpDef::elementwise(
        "erfc",
        1,
        &[ElementKind::F32],
        input(0).unary(UnaryOp::Erfc),
    );
    let ukey = key_for(2, OpCategory::UnaryElementwise);
    let uk = generate(&erfc, &ukey, &Cuda);
    assert!(uk.source.contains("erfcf("), "the kernel still lowers");
    let c = contract(&erfc, &ukey, &uk, &Cuda).expect("recipe-carrying contract");
    assert!(c.contains("fused_op: erfc"), "{c}");
    assert!(c.contains("semantics: erfc(in0)"), "{c}");
    assert!(c.contains("dtype_rule: passthrough(in0)"), "{c}");
    assert!(c.contains("shape_rule: same_as(in0)"), "{c}");
    assert!(
        !contract_admissible(&c, false) && contract_admissible(&c, true),
        "withheld pre-recipe, admitted to a recipe-import peer: {c}"
    );
    assert!(matches!(
        unpopped::derive_pattern(&erfc),
        Err(PatternError::NoFkcName { ref op }) if op == "Erfc"
    ));
    let at2 = OpDef::elementwise(
        "atan2",
        2,
        &[ElementKind::F32],
        input(0).binary(BinaryOp::Atan2, input(1)),
    );
    let bkey = key_for(3, OpCategory::BinaryElementwise);
    let bk = generate(&at2, &bkey, &Cuda);
    let cb = contract(&at2, &bkey, &bk, &Cuda).expect("recipe-carrying contract");
    assert!(cb.contains("fused_op: atan2"), "{cb}");
    assert!(cb.contains("semantics: atan2(in0, in1)"), "{cb}");
    assert!(cb.contains("shape_rule: same_as(in0)"), "{cb}");
    assert!(matches!(
        unpopped::derive_pattern(&at2),
        Err(PatternError::NoFkcName { ref op }) if op == "Atan2"
    ));
}

#[test]
fn every_mapped_primitive_root_emits_its_fuel_op_kind_through_the_emitter() {
    use unpopped::ir::{BinaryOp, Expr, UnaryOp};
    // UNARY single-op roots driven end-to-end through the real emitter.
    let unary: &[(UnaryOp, &str)] = &[
        (UnaryOp::Neg, "NegElementwise"),
        (UnaryOp::Abs, "AbsElementwise"),
        (UnaryOp::Sqr, "SqrElementwise"),
        (UnaryOp::Sqrt, "SqrtElementwise"),
        (UnaryOp::Rsqrt, "RsqrtElementwise"),
        (UnaryOp::Recip, "RecipElementwise"),
        (UnaryOp::Exp, "ExpElementwise"),
        (UnaryOp::Log, "LogElementwise"),
        (UnaryOp::Tanh, "TanhElementwise"),
        (UnaryOp::Sigmoid, "SigmoidElementwise"),
        // Relu maps again (2026-07-08): both sides pin NaN-propagating relu.
        (UnaryOp::Relu, "ReluElementwise"),
        (UnaryOp::Erf, "ErfElementwise"),
        (UnaryOp::Gelu, "GeluErfElementwise"), // exact-erf flavor, NOT tanh Gelu
        (UnaryOp::Silu, "SiluElementwise"),
        (UnaryOp::Sin, "SinElementwise"),
        (UnaryOp::Cos, "CosElementwise"),
        (UnaryOp::Floor, "FloorElementwise"),
        (UnaryOp::Ceil, "CeilElementwise"),
        (UnaryOp::Round, "RoundElementwise"),
        (UnaryOp::Sign, "SignElementwise"),
        (UnaryOp::Step, "StepElementwise"),
    ];
    for (uop, want) in unary {
        let op = OpDef::elementwise("u", 1, &[ElementKind::F32], input(0).unary(*uop));
        let key = key_for(2, OpCategory::UnaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, &Cuda)
            .unwrap_or_else(|| panic!("{uop:?} must emit a contract"));
        assert!(
            c.contains(&format!("op_kind: {want}\n")),
            "{uop:?} -> want op_kind: {want}, got:\n{c}"
        );
    }
    // BINARY single-op roots driven end-to-end through the real emitter.
    let binary: &[(Expr, &str)] = &[
        (input(0) + input(1), "AddElementwise"),
        (input(0) - input(1), "SubElementwise"),
        (input(0) * input(1), "MulElementwise"),
        (input(0) / input(1), "DivElementwise"),
        (input(0).max(input(1)), "MaximumElementwise"),
        (input(0).min(input(1)), "MinimumElementwise"),
        (input(0).pow(input(1)), "PowElementwise"),
        (input(0).binary(BinaryOp::Rem, input(1)), "RemElementwise"),
    ];
    for (body, want) in binary {
        let op = OpDef::elementwise("b", 2, &[ElementKind::F32], body.clone());
        let key = key_for(3, OpCategory::BinaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, &Cuda)
            .unwrap_or_else(|| panic!("want {want} must emit a contract"));
        assert!(
            c.contains(&format!("op_kind: {want}\n")),
            "want op_kind: {want}, got:\n{c}"
        );
    }
}

#[test]
fn standalone_scalar_param_op_is_an_honest_miss_not_a_poison_line() {
    // `x + p0` derives a single-op `AddScalar` root (n_ops == 1) — the
    // pre-fix bug emitted `op_kind: AddScalar`, which Fuel's importer rejects
    // (UnknownOpKind) and fails the WHOLE bundle. Fuel has no scalar-param
    // primitive OpKind (it lowers Op::AddScalar/MulScalar onto the `Affine`
    // kernel, whose scalar routing rides a `pattern:`/`fused_op` block), so
    // the standalone advert is withheld — the kernel still generates + lowers.
    let add_p = OpDef::elementwise("add_p", 1, &[ElementKind::F32], input(0) + param(0));
    let key = key_for(2, OpCategory::UnaryElementwise);
    let kernel = generate(&add_p, &key, &Cuda);
    assert!(!kernel.source.is_empty(), "the kernel still lowers");
    assert!(
        contract(&add_p, &key, &kernel, &Cuda).is_none(),
        "standalone AddScalar must be an honest miss (no poison op_kind line)"
    );
    // Prove the root really IS the unmapped AddScalar (so the miss is the
    // mapping's doing, not some earlier guard).
    let root = root_op_name(&unpopped::derive_pattern(&add_p).unwrap());
    assert_eq!(root, "AddScalar");
    assert_eq!(fuel_primitive_op_kind(&root), None);

    let mul_p = OpDef::elementwise("mul_p", 1, &[ElementKind::F32], input(0) * param(0));
    let km = generate(&mul_p, &key, &Cuda);
    assert!(contract(&mul_p, &key, &km, &Cuda).is_none());
    assert_eq!(
        root_op_name(&unpopped::derive_pattern(&mul_p).unwrap()),
        "MulScalar"
    );
}

#[test]
fn identity_copy_root_never_reaches_the_op_kind_line() {
    // Item-5 reachability proof: a bare `Input(0)` copy derives a `Bind` root
    // with n_ops == 0, so it can NEVER hit the `n_ops == 1` primitive branch
    // — it falls through to the `fused_op` arm. Thus `root_op_name`'s
    // "Identity" spelling (and any Bind) is unreachable on the op_kind line;
    // the ONLY spellings that reach it are the single-Op arithmetic roots.
    let copy = OpDef::elementwise("copy", 1, &[ElementKind::F32], input(0));
    let key = key_for(2, OpCategory::UnaryElementwise);
    let kernel = generate(&copy, &key, &Cuda);
    let c = contract(&copy, &key, &kernel, &Cuda).expect("a bare copy still contracts");
    assert!(
        !c.contains("op_kind:"),
        "a Bind/Identity root must not emit op_kind: {c}"
    );
    assert!(
        c.contains("fused_op: copy"),
        "it advertises as fused_op instead: {c}"
    );
    // And the pattern really is a bare Bind (n_ops == 0).
    assert!(matches!(
        unpopped::derive_pattern(&copy).unwrap(),
        PatternNode::Bind(0)
    ));
}
