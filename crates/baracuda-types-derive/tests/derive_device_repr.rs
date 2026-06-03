//! Direct integration tests for the `#[derive(DeviceRepr)]` proc-macro
//! in `baracuda-types-derive`.
//!
//! # Why inline tests instead of `trybuild`?
//!
//! `trybuild` is not in the baracuda workspace and pulling it in would
//! also commit us to `.stderr` snapshots that drift with each compiler
//! release (we publish a wide MSRV…stable bracket; the `Vec`/`Box`
//! error messages from the `Copy` and `DeviceRepr` bounds change wording
//! across rustc versions, which means snapshot tests would either be
//! flaky or pinned to one toolchain). Inline `#[test]` blocks that
//! assert the macro accepts the right shapes, combined with
//! `compile_fail` doc-tests in the macro itself for the rejection
//! cases, give the same coverage without those snapshot fragility costs.
//!
//! The macro emits `::baracuda_types::DeviceRepr`, so we drive it via
//! the parent crate (dev-dep, `derive` feature). Pure-rust dev-dep
//! cycles back to the parent are legal in Cargo and only affect the
//! test build, not the publish artifact.

use baracuda_types::{BFloat16, Complex32, Complex64, DeviceRepr, Half, KernelArg};

fn assert_device_repr<T: DeviceRepr>() {}
fn assert_kernel_arg<T: KernelArg>(_: T) {}

// ---------------------------------------------------------------------------
// Compile-pass cases. Each `#[derive(DeviceRepr)]` here must succeed
// AND produce an `unsafe impl DeviceRepr` that satisfies the trait
// bound, which we check below in tests.
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(C)]
struct SimpleNamed {
    a: u32,
    b: f32,
    c: i64,
}

#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(C)]
struct Empty;

#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(C)]
struct TupleStruct(f32, u32, i16);

#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(transparent)]
struct Newtype(u64);

// PhantomData<T> is Copy + 'static and we need to prove the derive
// happily threads generics + a where-clause-augmented bound through.
// Note: `PhantomData<T>` is NOT itself DeviceRepr, so the marker has
// to ride alongside primitive fields; this is the realistic shape for
// downstream generic tensor wrappers.
#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(C)]
struct GenericPayload<T: Copy + 'static> {
    payload: T,
    tag: u32,
}

#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(C)]
struct WithArrays {
    matrix: [[f32; 4]; 4],
    flags: [u8; 16],
}

#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(C)]
struct WithNumericWrappers {
    h: Half,
    b: BFloat16,
    c32: Complex32,
    c64: Complex64,
}

#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(C)]
struct Nested {
    inner: SimpleNamed,
    extra: f64,
}

// `#[repr(C, packed)]` and `#[repr(C, align(N))]` both still spell `C`
// in the attribute list; the derive should accept both.
#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
#[repr(C, align(16))]
struct AlignedC {
    x: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn derive_accepts_named_struct() {
    assert_device_repr::<SimpleNamed>();
}

#[test]
fn derive_accepts_unit_struct() {
    assert_device_repr::<Empty>();
}

#[test]
fn derive_accepts_tuple_struct() {
    assert_device_repr::<TupleStruct>();
}

#[test]
fn derive_accepts_transparent_newtype() {
    assert_device_repr::<Newtype>();
}

#[test]
fn derive_threads_generics_with_field_bound() {
    // Concrete instantiations with various DeviceRepr field types.
    assert_device_repr::<GenericPayload<u32>>();
    assert_device_repr::<GenericPayload<f64>>();
    assert_device_repr::<GenericPayload<[f32; 8]>>();
    assert_device_repr::<GenericPayload<SimpleNamed>>();
}

#[test]
fn derive_accepts_arrays_and_nested_aggregates() {
    assert_device_repr::<WithArrays>();
    assert_device_repr::<WithNumericWrappers>();
    assert_device_repr::<Nested>();
}

#[test]
fn derive_accepts_repr_c_with_align() {
    assert_device_repr::<AlignedC>();
}

#[test]
fn derived_types_compose_with_kernelarg_blanket_impl() {
    // `KernelArg` is blanket-implemented for `&T where T: DeviceRepr`,
    // so the derive is sufficient to make a value passable to a kernel
    // launch via `&value`. Regression-test that handshake here so a
    // future change to either side surfaces immediately.
    let v = SimpleNamed { a: 1, b: 2.0, c: 3 };
    assert_kernel_arg(&v);

    let n = Newtype(42);
    assert_kernel_arg(&n);

    let g: GenericPayload<f64> = GenericPayload { payload: 1.5, tag: 7 };
    assert_kernel_arg(&g);
}

#[test]
fn derive_does_not_consume_copy_clone() {
    // Sanity: the macro must NOT take ownership of or shadow the
    // user-derived Copy/Clone. If the derive macro had a bug that ate
    // the other derives, this would fail to compile.
    let a = SimpleNamed { a: 9, b: 1.0, c: -1 };
    let b = a; // Copy
    let _c = a.clone(); // Clone
    let _ = b;
}
