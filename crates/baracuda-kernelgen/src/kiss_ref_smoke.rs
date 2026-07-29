//! Temporary smoke test: prove the kiss-ref `0.1.0` dev-dependencies resolve
//! from crates.io and `eval_recipe` runs in-tree, against the exact API surface
//! the standalone `tools/kiss-ref-diff` harness already compiled + ran against.
//! This is the compile-verification the consolidation gate exists for: if
//! `0.1.0` diverged from the harness-tested rev, this file fails to build and we
//! reconcile with kiss-ref before migrating anything. Folded into
//! `kiss_ref_diff.rs` in Task 1 — delete this file then.

use kiss_ops_vocab::Op;
use kiss_ref_core::{DetClass, FlatDag, Node, Tensor, eval_recipe};

#[test]
fn kiss_ref_dev_dep_resolves_and_evaluates() {
    // relu(add(in0, in1)) over 3 values. Nodes: [Bind(0), Bind(1), add, relu];
    // the single output is node 3 (relu). Ops resolve through `Op::from_token`
    // — the converter's own path (the emitted recipe carries KISS-Ops tokens),
    // not hypothetical direct enum variants.
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
    // eval_recipe(dag, inputs, params, indices): params + indices both empty.
    let r = eval_recipe(&dag, &[a, b], &[], &[]).expect("eval_recipe");
    let got: Vec<f32> = r.outputs[0].clone().into_data();
    // add: [-0.5, 1.5, -2.0]; relu clamps the negatives to +0.0.
    assert_eq!(got, vec![0.0, 1.5, 0.0]);
    // A pure elementwise add+relu has no float-reduction reorder, so every
    // output classes byte-deterministic (the harness's step-3a ExactByte lane).
    assert!(r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)));
}
