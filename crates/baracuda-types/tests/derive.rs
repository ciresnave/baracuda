//! Integration tests for `#[derive(DeviceRepr)]`.
//!
//! Run with `cargo test -p baracuda-types --features derive`.

#![cfg(feature = "derive")]

use baracuda_types::{BFloat16, Complex32, DeviceRepr, Half, KernelArg};

fn assert_device_repr<T: DeviceRepr>() {}
fn assert_kernel_arg<T: KernelArg>(_: T) {}

#[derive(Copy, Clone, DeviceRepr)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    velocity: [f32; 3],
    mass: f32,
}

#[derive(Copy, Clone, DeviceRepr)]
#[repr(transparent)]
struct Wrapped(u32);

#[derive(Copy, Clone, DeviceRepr)]
#[repr(C)]
struct WithNumeric {
    a: Half,
    b: BFloat16,
    c: Complex32,
}

#[derive(Copy, Clone, DeviceRepr)]
#[repr(C)]
struct Empty;

#[derive(Copy, Clone, DeviceRepr)]
#[repr(C)]
struct Tuple(f32, f32, f32);

#[derive(Copy, Clone, DeviceRepr)]
#[repr(C)]
struct Generic<T: Copy + 'static> {
    inner: T,
    count: u32,
}

#[test]
fn concrete_structs_implement_device_repr() {
    assert_device_repr::<Particle>();
    assert_device_repr::<Wrapped>();
    assert_device_repr::<WithNumeric>();
    assert_device_repr::<Empty>();
    assert_device_repr::<Tuple>();
}

#[test]
fn generic_struct_implements_device_repr_when_t_does() {
    assert_device_repr::<Generic<u32>>();
    assert_device_repr::<Generic<f64>>();
    assert_device_repr::<Generic<Particle>>();
}

#[test]
fn derived_types_are_kernel_args_via_reference() {
    let p = Particle {
        position: [0.0; 3],
        velocity: [1.0; 3],
        mass: 0.5,
    };
    // Any `&T where T: DeviceRepr` is a KernelArg.
    assert_kernel_arg(&p);

    let w = Wrapped(42);
    assert_kernel_arg(&w);
}
