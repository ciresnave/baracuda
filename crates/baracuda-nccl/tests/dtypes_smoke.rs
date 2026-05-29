//! NcclScalar / NcclDataType dtype mapping smoke tests.
//!
//! These tests verify that the `NcclScalar` sealed trait (alias
//! `NcclDataType`) correctly maps each supported Rust scalar to its
//! `ncclDataType_t` enum variant. The tests are **compile-time +
//! dispatch-time only** — they don't call NCCL, so they run on every
//! host (Linux / Windows / no-GPU CI).
//!
//! Multi-rank validation is out of scope for Phase 52 (needs 2+ GPUs or
//! a process-spawning test harness).

use baracuda_nccl::NcclScalar;
use baracuda_nccl_sys::ncclDataType_t;

fn require_scalar<T: NcclScalar>() -> ncclDataType_t {
    T::raw()
}

#[test]
fn i8_maps_to_int8() {
    assert_eq!(require_scalar::<i8>(), ncclDataType_t::Int8);
}

#[test]
fn u8_maps_to_uint8() {
    assert_eq!(require_scalar::<u8>(), ncclDataType_t::Uint8);
}

#[test]
fn i32_maps_to_int32() {
    assert_eq!(require_scalar::<i32>(), ncclDataType_t::Int32);
}

#[test]
fn u32_maps_to_uint32() {
    assert_eq!(require_scalar::<u32>(), ncclDataType_t::Uint32);
}

#[test]
fn i64_maps_to_int64() {
    assert_eq!(require_scalar::<i64>(), ncclDataType_t::Int64);
}

#[test]
fn u64_maps_to_uint64() {
    assert_eq!(require_scalar::<u64>(), ncclDataType_t::Uint64);
}

#[test]
fn f32_maps_to_float32() {
    assert_eq!(require_scalar::<f32>(), ncclDataType_t::Float32);
}

#[test]
fn f64_maps_to_float64() {
    assert_eq!(require_scalar::<f64>(), ncclDataType_t::Float64);
}

// NCCL also supports `Float16` (half::f16) and `BFloat16` (half::bf16);
// the impls are gated behind the `half-crate` Cargo feature so the
// default build doesn't pull in the `half` dep. The companion
// `half_scalar_tests` module inside `lib.rs` covers them when the
// feature is active.

#[test]
fn ncclreduceop_alias_resolves() {
    // Compile-time alias check: NcclReduceOp must be re-exported as RedOp.
    use baracuda_nccl::{NcclReduceOp, RedOp};
    let _: RedOp = NcclReduceOp::Sum;
    let _: NcclReduceOp = RedOp::Avg;
}

#[test]
fn ncclunique_id_alias_resolves() {
    // Compile-time alias check: NcclUniqueId must be re-exported as UniqueId.
    use baracuda_nccl::{NcclUniqueId, UniqueId};
    fn takes_unique(_: &UniqueId) {}
    fn takes_nccl(_: &NcclUniqueId) {}
    // Trivially-constructible from the all-zero wire form (doesn't
    // require NCCL — we're only proving the type alias relationship).
    let id = UniqueId::from_bytes([0u8; 128]);
    takes_unique(&id);
    takes_nccl(&id);
}
