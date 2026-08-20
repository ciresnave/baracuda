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

use std::panic::{AssertUnwindSafe, catch_unwind};

use unpopped::backend::LowerError;
use unpopped::ir::BinaryOp;
use unpopped::{OpDef, UnaryOp, generate, input, try_generate};
use unpopped_cpu_c::CpuC;
use unpopped_slang::Slang;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// CpuC declines the half dtypes it has no CPU codec for — as a TYPED
/// `LowerError::UnsupportedDtype`, never a panic (`generate` unwraps and would panic;
/// `try_generate` is the Result form) — and emits a real host loop for the dtypes it
/// does compute. The positive control (f32) is what stops the declines being
/// satisfied by a CpuC that refuses everything.
#[test]
fn cpuc_declines_half_dtypes_and_emits_f32() {
    // A single-input elementwise body, reused at each dtype. CpuC's dtype gate fires
    // in `lower` BEFORE the body is examined, so the same body serves both legs.
    let op_at =
        |dt: ElementKind| OpDef::elementwise("relu", 1, &[dt], input(0).unary(UnaryOp::Relu));
    let key_at = |dt: ElementKind| {
        let a = OperandDesc::new(1, &[1 << 16], &[1], dt, 4);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    };

    // DECLINE (typed): f16 / bf16 — no CPU half codec in v1.
    for dt in [ElementKind::F16, ElementKind::Bf16] {
        let op = op_at(dt);
        match try_generate(&op, &key_at(dt), &CpuC) {
            Err(LowerError::UnsupportedDtype { dtype, .. }) => assert_eq!(
                dtype, dt,
                "CpuC declined the wrong dtype: expected {dt:?}, error names {dtype:?}"
            ),
            other => panic!("CpuC should typed-decline {dt:?}, got {other:?}"),
        }
    }

    // POSITIVE CONTROL: f32 is inside CpuC's compute allowlist — it MUST emit a real
    // host-loop store. Without this, the declines above hold for a CpuC that refuses
    // every dtype (a broken backend), which is the failure mode this pairing catches.
    let f32 = op_at(ElementKind::F32);
    let k = try_generate(&f32, &key_at(ElementKind::F32), &CpuC)
        .unwrap_or_else(|e| panic!("CpuC should EMIT for f32, but declined: {e:?}"));
    assert!(
        k.source.contains("for (long long i"),
        "CpuC emitted no host loop for f32:\n{}",
        k.source
    );
}

/// Slang declines `Copysign` / `Nextafter` (no base-profile Slang intrinsic) and
/// emits for the binaries it supports. Unlike CpuC's dtype decline, this op-level
/// refusal is a **`panic!`**, not a typed `Err` — the documented "honest
/// panic-boundary" (unpopped-slang 0.1.0 `lib.rs`: `Copysign | Nextafter =>
/// panic!(...)`). So it is asserted with `catch_unwind`, paired with a positive
/// control in the same test.
///
/// TWO durability notes for whoever touches this next:
///
/// 1. **`catch_unwind` requires `panic = "unwind"`.** Under a `panic = "abort"`
///    profile it never returns — the process dies instead of the assertion failing.
///    This workspace uses the default (unwind) and sets no abort profile; if you add
///    one for binary size, this test needs `panic = "unwind"` or a `#[cfg]` guard.
///
/// 2. **This asserts unpopped-slang 0.1.0's op-level PANIC convention** (both that it
///    panics AND that the panic carries the documented "declined" message). If it
///    fails after an `unpopped-slang` upgrade, first check whether the decline became
///    a typed `Err(LowerError::…)` — that is an upstream IMPROVEMENT, and the fix is
///    to convert this to an `Err` match, NOT to restore the panic. (A reworded panic
///    message would also trip the payload check; loosen the marker in that case.) A
///    green→red here can mean the boundary got better, not that something regressed.
///
/// (The expected panics print "thread panicked …" to stderr even though the test
/// passes — that is `catch_unwind`'s default hook. The hook is process-global, so it
/// is deliberately NOT suppressed: silencing it would risk masking a genuine panic in
/// a test running concurrently.)
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
    let emitted = catch_unwind(AssertUnwindSafe(|| generate(&pow, &key, &Slang).source))
        .expect("Slang should EMIT for Pow (supported), not panic");
    assert!(
        emitted.contains("output["),
        "Slang emitted no output store for Pow:\n{emitted}"
    );

    // DECLINE (panic boundary): Copysign / Nextafter have no base-profile Slang
    // intrinsic and panic-decline by design.
    for op_kind in [BinaryOp::Copysign, BinaryOp::Nextafter] {
        let op = OpDef::elementwise(
            "d",
            2,
            &[ElementKind::F32],
            input(0).binary(op_kind, input(1)),
        );
        let payload = catch_unwind(AssertUnwindSafe(|| generate(&op, &key, &Slang).source))
            .err()
            .unwrap_or_else(|| {
                panic!("Slang should panic-decline {op_kind:?} (v0.1.0 op-level boundary), but it emitted a kernel")
            });
        // Confirm it is the DOCUMENTED op-level decline, not some unrelated panic —
        // otherwise "declines Copysign" would be satisfied by any crash at all. The
        // v0.1.0 message is `slang backend v1: {op} … — declined (a follow-up)`.
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            msg.contains("declined") && msg.contains(&format!("{op_kind:?}")),
            "Slang panicked on {op_kind:?} but not with the documented op-level decline \
             (message was {msg:?}); an unrelated panic must not satisfy this contract"
        );
    }
}
