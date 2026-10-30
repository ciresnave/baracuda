//! Unpopped 0.32 added `ScalarExpr` nodes (integer leaves, packed values, kernel
//! locals, qualifiers) that this emitter does not spell. Each must be a TYPED decline,
//! never a panic and never emitted source. Through the public API the plan gate
//! (`InadmissiblePlan`) refuses these first; the emitter's own backstop arms in
//! `assert_coord_lowerable` are the second layer and are not reachable from here.
//! The positive control is a supported body at
//! the same key: a broken backend declines everything and would pass the decline legs.

use unpopped::backend::LowerError;
use unpopped::ir::{Expr, ScalarExpr};
use unpopped::{OpDef, input, try_generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

use baracuda_cuda_emit::Cuda;

fn key() -> unpopped_vocab::StructureKey {
    let a = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::I32, 1);
    structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
}

fn op(body: ScalarExpr) -> OpDef {
    OpDef::elementwise("new_node", 1, &[ElementKind::I32], Expr(body))
}

fn declined(body: ScalarExpr) -> LowerError {
    match try_generate(&op(body), &key(), &Cuda) {
        Err(e) => e,
        Ok(k) => panic!(
            "expected a typed decline, got source:
{}",
            k.source
        ),
    }
}

#[test]
fn integer_and_packed_nodes_are_typed_declines_and_a_plain_body_still_emits() {
    // Positive control: the identity body emits at the same key.
    let ok = try_generate(
        &OpDef::elementwise("ctl", 1, &[ElementKind::I32], input(0)),
        &key(),
        &Cuda,
    )
    .unwrap_or_else(|e| panic!("control body should emit, got {e:?}"));
    assert!(
        ok.source.contains("in0["),
        "control emitted no read:\n{}",
        ok.source
    );

    assert!(matches!(
        declined(ScalarExpr::ConstInt(7, ElementKind::I32)),
        LowerError::InadmissiblePlan { .. } | LowerError::UnsupportedOp { .. }
    ));
    // Unpopped 0.34: an execution-id leaf is opaque like `Local`.
    assert!(matches!(
        declined(ScalarExpr::Exec(unpopped::exec::ExecId::LaneId)),
        LowerError::InadmissiblePlan { .. } | LowerError::UnsupportedOp { .. }
    ));
    assert!(matches!(
        declined(ScalarExpr::Extract(
            Box::new(ScalarExpr::Pack(vec![
                ScalarExpr::Input(0),
                ScalarExpr::Input(0)
            ])),
            0,
        )),
        LowerError::InadmissiblePlan { .. } | LowerError::UnsupportedOp { .. }
    ));
}
