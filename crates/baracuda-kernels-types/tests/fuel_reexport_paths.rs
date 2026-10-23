//! Fuel reaches these seven items at the `baracuda_kernels_types::` path and cannot tell which of
//! them arrive through the `pub use unpopped_vocab::*;` glob. API-freeze R3 replaces that glob with
//! an explicit list; this test fails to COMPILE if any of the seven stops resolving at this path.
//! Source: Fuel lane ack, 2026-10-23 (`fuel-dispatch/src/telemetry/baracuda_provider.rs:70` and
//! its `ElementKind` / `OperandDesc` / `ArchSku` / `TargetId` uses elsewhere).

#[allow(
    unused_imports,
    reason = "the test is the import: it must keep resolving, whatever kind of item each name is"
)]
use baracuda_kernels_types::{
    ArchSku, ElementKind, MAX_RANK, OpCategory, OperandDesc, TargetId, structure_key_token,
};

#[test]
fn the_seven_fuel_names_resolve_at_the_crate_root() {
    // A value-level touch for the items whose kind is known, so a rename to a different kind of
    // item (const -> fn) also fails here rather than only at Fuel.
    let _max_rank: usize = MAX_RANK;
    let _key_fn: fn(OpCategory, &[OperandDesc], ArchSku) -> String = structure_key_token;
    assert!(core::mem::size_of::<ElementKind>() > 0);
}
