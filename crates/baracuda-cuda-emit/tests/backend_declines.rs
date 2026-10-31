//! Decline-contract tests for the sibling backends `CpuC` / `Slang` (published from
//! the Unpopped extraction). Each asserts a DOCUMENTED refusal **paired with a
//! positive control in the same test** — "backend X declines Y" is also satisfied by
//! a backend that declines EVERYTHING (a broken build, a backend that failed to
//! initialise), so without an "X emits for something it supports" in the same test a
//! uniformly-broken emitter passes vacuously. This is the same non-vacuity bracket as
//! Fuel's `emitted.len() < recognized.len()` next to its subset check.
//!
//! Three-way agreement including CUDA (the thing only this crate can carry) lives in
//! `lift.rs` / `fuzz.rs` / `convert.rs`; the CpuC×Slang pair is owned by
//! `unpopped-conformance`. This file adds the refusal contracts, not that coverage.

use unpopped::backend::LowerError;
use unpopped::ir::BinaryOp;
use unpopped::{OpDef, UnaryOp, input, try_generate};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use vulkane_slang_emit::Slang;

/// CpuC lowers `f16` / `bf16` (a CPU codec since unpopped-cpu-c 0.15.0: the halves
/// are carried in `unsigned short` and converted through integer helpers, where
/// they used to be a typed `UnsupportedDtype` decline). The v1 backend still
/// serves only the scalar contiguous schedule, so a vector-aligned half key is a
/// TYPED `UnsupportedSchedule` decline, never a panic (`generate` unwraps and would
/// panic; `try_generate` is the Result form). Three legs, so that no one of them is
/// satisfied by a backend that refuses everything:
///
/// 1. the halves EMIT a real host loop over an `unsigned short` carrier at a
///    scalar-schedule key (alignment 1);
/// 2. the same dtype at a vector-aligned key is a typed schedule decline;
/// 3. f32 still emits (the positive control for leg 2).
#[test]
fn cpuc_emits_halves_and_f32_and_declines_vectorized_schedule() {
    // A single-input elementwise body, reused at each dtype.
    let op_at =
        |dt: ElementKind| OpDef::elementwise("relu", 1, &[dt], input(0).unary(UnaryOp::Relu));
    let key_at = |dt: ElementKind, align: u32| {
        let a = OperandDesc::new(1, &[1 << 16], &[1], dt, align);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    };

    for dt in [ElementKind::F16, ElementKind::Bf16] {
        let op = op_at(dt);

        // 1. EMITS at a scalar-schedule key, through the integer carrier.
        let k = try_generate(&op, &key_at(dt, 1), &CpuC)
            .unwrap_or_else(|e| panic!("CpuC should EMIT for {dt:?}, but declined: {e:?}"));
        assert!(
            k.source.contains("for (long long i"),
            "CpuC emitted no host loop for {dt:?}:
{}",
            k.source
        );
        assert!(
            k.source.contains("unsigned short"),
            "CpuC {dt:?} kernel does not carry the half in `unsigned short`:
{}",
            k.source
        );

        // 2. A vector-aligned key picks Vectorized, which v1 does not serve.
        match try_generate(&op, &key_at(dt, 4), &CpuC) {
            Err(LowerError::UnsupportedSchedule { .. }) => {}
            other => {
                panic!("CpuC should typed-decline the {dt:?} Vectorized schedule, got {other:?}")
            }
        }
    }

    // 3. POSITIVE CONTROL: f32 emits at the scalar key, so legs 1-2 are not
    // satisfied by a CpuC that refuses everything.
    let f32 = op_at(ElementKind::F32);
    let k = try_generate(&f32, &key_at(ElementKind::F32, 1), &CpuC)
        .unwrap_or_else(|e| panic!("CpuC should EMIT for f32, but declined: {e:?}"));
    assert!(
        k.source.contains("for (long long i"),
        "CpuC emitted no host loop for f32:
{}",
        k.source
    );
}

/// Slang declines `Copysign` / `Nextafter` (no base-profile Slang intrinsic) and
/// emits for the binaries it supports. As of unpopped-slang 0.3.0 this op-level
/// refusal is a **typed `LowerError`**, not a panic: the seam returns
/// `Ok(Spelling::Declined(Decline::UnsupportedOp { op: DeclinedOp::Binary(..), .. }))`,
/// which `lower_dag` folds into `Err(LowerError::UnsupportedOp { detail })` at the top
/// of the lowering (`detail` carries the `{op:?}` prefix, so it names the refused op).
/// So it is asserted with a `Result` match, paired with a positive control in the same
/// test.
///
/// Durability note for whoever touches this next: this asserts the 0.3.0 **typed**
/// decline (both that it is `Err(LowerError::UnsupportedOp)` AND that the `detail`
/// names the refused op). If it fails after an `vulkane-slang-emit` upgrade, first check
/// whether the decline stopped being typed — a `panic!` boundary or a bare `String`
/// reason would be a REGRESSION here (it was already typed), the inverse of the
/// pre-0.3.0 note: a green→red now means the decline got WORSE, not better. (This
/// inverts the old `catch_unwind` panic-boundary assertion, which the 0.1.0 op-level
/// panic convention required and 0.3.0's typed-decline seam retired.)
#[test]
fn slang_declines_copysign_nextafter_and_emits_pow() {
    let a = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);

    // POSITIVE CONTROL: Slang emits a real store (`output[`) for a supported binary.
    let pow = OpDef::elementwise(
        "pow",
        2,
        &[ElementKind::F32],
        input(0).binary(BinaryOp::Pow, input(1)),
    );
    let k = try_generate(&pow, &key, &Slang)
        .unwrap_or_else(|e| panic!("Slang should EMIT for Pow (supported), but declined: {e:?}"));
    assert!(
        k.source.contains("output["),
        "Slang emitted no output store for Pow:\n{}",
        k.source
    );

    // DECLINE (typed `LowerError`): Copysign / Nextafter have no base-profile Slang
    // intrinsic and decline by design. Assert it is the DOCUMENTED op-level decline
    // that names the refused op — otherwise "declines Copysign" would be satisfied by
    // any unrelated `UnsupportedOp`.
    for op_kind in [BinaryOp::Copysign, BinaryOp::Nextafter] {
        let op = OpDef::elementwise(
            "d",
            2,
            &[ElementKind::F32],
            input(0).binary(op_kind, input(1)),
        );
        match try_generate(&op, &key, &Slang) {
            Err(LowerError::UnsupportedOp { detail }) => assert!(
                detail.contains(&format!("{op_kind:?}")),
                "Slang typed-declined {op_kind:?} but the detail did not name it \
                 (detail = {detail:?}); an unrelated UnsupportedOp must not satisfy this \
                 contract"
            ),
            other => panic!(
                "Slang should typed-decline {op_kind:?} as LowerError::UnsupportedOp \
                 (vulkane-slang-emit typed-decline seam), got {other:?}"
            ),
        }
    }
}

/// `Nextafter@f16` WIRING TEST — the concrete payoff of unpopped 0.6.0's plan-gate
/// decline channel. Before 0.6.0, `try_generate` called `build_plan`, so an
/// inadmissible (op, dtype) ABORTED — a panic indistinguishable from a crash. 0.6.0
/// routes `try_generate` through `try_build_plan`, so the same input now returns a
/// TYPED `LowerError::InadmissiblePlan { source }` that names the dtype.
///
/// The two legs are deliberately the SAME op at two dtypes, so the test proves the
/// two outcomes are DISTINGUISHABLE where 0.5.0 conflated one of them with a crash:
///   Nextafter @ f32  → EMITS (CUDA has a real f32 nextafter lowering)   [positive control]
///   Nextafter @ f16  → InadmissiblePlan naming F16 (no half lowering)   [the 0.6.0 win]
/// It reds if `try_generate` ever reverts to `build_plan` (the f16 leg would panic,
/// not match), which is exactly the regression the 0.6.0 wiring must not undo.
#[test]
fn nextafter_f16_is_a_typed_inadmissible_plan_not_a_panic() {
    use baracuda_cuda_emit::Cuda;
    let op_at = |dt: ElementKind| {
        OpDef::elementwise(
            "na",
            2,
            &[dt],
            input(0).binary(BinaryOp::Nextafter, input(1)),
        )
    };
    let key_at = |dt: ElementKind| {
        let a = OperandDesc::new(1, &[1 << 16], &[1], dt, 4);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    };

    // POSITIVE CONTROL: f32 has a real lowering — it MUST emit the nextafter intrinsic,
    // not merely succeed. Without this, the f16 decline below is satisfied by a backend
    // that refuses Nextafter at every dtype.
    let f32 = op_at(ElementKind::F32);
    let k = try_generate(&f32, &key_at(ElementKind::F32), &Cuda)
        .unwrap_or_else(|e| panic!("Nextafter@f32 should EMIT, but declined: {e:?}"));
    assert!(
        k.source.contains("nextafter"),
        "Nextafter@f32 emitted no nextafter intrinsic (vacuous positive control):\n{}",
        k.source
    );

    // THE 0.6.0 WIN: f16 has no half lowering → a TYPED InadmissiblePlan that names the
    // dtype, where 0.5.0 panicked. A generic InadmissiblePlan that did not name F16, or
    // a panic (an unmatched `other`), both fail this contract.
    let f16 = op_at(ElementKind::F16);
    match try_generate(&f16, &key_at(ElementKind::F16), &Cuda) {
        Err(LowerError::InadmissiblePlan { source }) => {
            let s = format!("{source:?}");
            assert!(
                s.contains("F16"),
                "Nextafter@f16 gave an InadmissiblePlan that did not name the f16 dtype \
                 (source = {s:?}); a generic plan decline must not satisfy this contract"
            );
        }
        other => panic!(
            "Nextafter@f16 should be a typed LowerError::InadmissiblePlan (0.6.0 plan-gate \
             decline; was a panic pre-0.6.0). If this is a panic, try_generate reverted to \
             build_plan. Got: {other:?}"
        ),
    }
}
