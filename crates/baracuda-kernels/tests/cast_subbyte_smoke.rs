//! Real-GPU smoke tests for `CastSubBytePlan<TIn, TOut>` (Phase 13.3).
//!
//! Coverage:
//!   - `S4 -> i32` (unpack, sign-extension)
//!   - `U4 -> i32` (unpack, zero-extension)
//!   - `Fp8E4M3 -> f32` and `f32 -> Fp8E4M3` round-trip
//!   - `U8 -> f32`... NOT covered here because U8 routes through the
//!     classic `CastPlan` via the `u8 -> f32` FFI symbol; the dedicated
//!     test for that lives elsewhere.
//!   - `Bool -> f32` (truthiness: 0/1/255)
//!   - `S8 -> i32` ... ditto, classic `CastPlan` via `i8 -> i32`.
//!
//! Bit-exact compares where the conversion is bit-stable (Fp8 round-trip
//! depends on the source value already being representable on the Fp8
//! lattice — we use a small grid that we know IS).
//!
//! `#[ignore]` by default — run with:
//!   `cargo test -p baracuda-kernels --release --features sm89 \
//!     --test cast_subbyte_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CastSubByteArgs, CastSubByteDescriptor, CastSubBytePlan, ElementKind,
    Fp8E4M3, PlanPreference, TensorMut, TensorRef, Workspace, Bool, S4, U4,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// =============================================================================
// S4 -> i32 (unpack, sign-extension)
// =============================================================================

#[test]
#[ignore]
fn cast_s4_to_i32_unpacks_with_sign_extension() {
    let (ctx, stream) = setup();
    // 8 elements packed in 4 bytes:
    //   byte 0: lo=0 hi=1                   -> [0, 1]
    //   byte 1: lo=-1 (0xF) hi=-8 (0x8)     -> [-1, -8]
    //   byte 2: lo=7 (0x7) hi=-2 (0xE)      -> [7, -2]
    //   byte 3: lo=-4 (0xC) hi=3 (0x3)      -> [-4, 3]
    let host_packed: Vec<u8> = vec![
        S4::pack(0, 1).0,
        S4::pack(-1, -8).0,
        S4::pack(7, -2).0,
        S4::pack(-4, 3).0,
    ];
    let numel = 8i32;
    let expected: Vec<i32> = vec![0, 1, -1, -8, 7, -2, -4, 3];

    // Upload packed bytes as a DeviceBuffer<S4>. S4 is #[repr(transparent)]
    // over u8 so we go through `DeviceBuffer<u8>` and reinterpret.
    let dev_x_u8 = DeviceBuffer::from_slice(&ctx, &host_packed).expect("upload");
    let dev_x: DeviceBuffer<S4> = unsafe { core::mem::transmute(dev_x_u8) };
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc");

    let desc = CastSubByteDescriptor {
        numel,
        input_element: ElementKind::S4,
        output_element: ElementKind::I32,
    };
    let plan = CastSubBytePlan::<S4, i32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    // S4 is nibble-packed: the tensor view's shape/numel uses the PACKED
    // slot count (one slot per byte = two logical elements).
    let packed_slots = (numel as i32) / 2;
    let args = CastSubByteArgs::<S4, i32> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [packed_slots],
            stride: contiguous_stride([packed_slots]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; numel as usize];
    dev_y.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "S4 -> i32 sign-extend");
}

// =============================================================================
// U4 -> i32 (unpack, zero-extension)
// =============================================================================

#[test]
#[ignore]
fn cast_u4_to_i32_unpacks_with_zero_extension() {
    let (ctx, stream) = setup();
    // 8 elements packed in 4 bytes.
    //   byte 0: lo=0 hi=15
    //   byte 1: lo=1 hi=8
    //   byte 2: lo=5 hi=14
    //   byte 3: lo=3 hi=12
    let host_packed: Vec<u8> = vec![
        U4::pack(0, 15).0,
        U4::pack(1, 8).0,
        U4::pack(5, 14).0,
        U4::pack(3, 12).0,
    ];
    let numel = 8i32;
    let expected: Vec<i32> = vec![0, 15, 1, 8, 5, 14, 3, 12];

    let dev_x_u8 = DeviceBuffer::from_slice(&ctx, &host_packed).expect("upload");
    let dev_x: DeviceBuffer<U4> = unsafe { core::mem::transmute(dev_x_u8) };
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc");

    let desc = CastSubByteDescriptor {
        numel,
        input_element: ElementKind::U4,
        output_element: ElementKind::I32,
    };
    let plan = CastSubBytePlan::<U4, i32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let packed_slots = (numel as i32) / 2;
    let args = CastSubByteArgs::<U4, i32> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [packed_slots],
            stride: contiguous_stride([packed_slots]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; numel as usize];
    dev_y.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "U4 -> i32 zero-extend");
}

// =============================================================================
// Fp8E4M3 -> f32 -> Fp8E4M3 round-trip.
//
// Pick values that are exactly representable on the E4M3 lattice so the
// round-trip is bit-exact. The grid {0, 0.5, 1, 2, 4, 8, 16, 32, 64} all
// fit cleanly in E4M3 (powers of two and one half-step).
// =============================================================================

#[test]
#[ignore]
fn cast_fp8e4m3_to_f32_lattice_grid() {
    let (ctx, stream) = setup();
    let grid: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
    let host_x: Vec<Fp8E4M3> = grid.iter().map(|&f| Fp8E4M3::from_f32(f)).collect();
    let numel = host_x.len() as i32;

    // Reinterpret as bytes for upload (Fp8E4M3 is repr(transparent) over u8).
    let host_x_bytes: Vec<u8> = host_x.iter().map(|fp| fp.0).collect();
    let dev_x_u8 = DeviceBuffer::from_slice(&ctx, &host_x_bytes).expect("upload");
    let dev_x: DeviceBuffer<Fp8E4M3> = unsafe { core::mem::transmute(dev_x_u8) };
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc");

    let desc = CastSubByteDescriptor {
        numel,
        input_element: ElementKind::Fp8E4M3,
        output_element: ElementKind::F32,
    };
    let plan =
        CastSubBytePlan::<Fp8E4M3, f32>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = CastSubByteArgs::<Fp8E4M3, f32> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel as usize];
    dev_y.copy_to_host(&mut got).expect("download");
    // Reference: host-side Fp8E4M3 -> f32 lookup.
    let expected: Vec<f32> = host_x.iter().map(|fp| fp.to_f32()).collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "fp8e4m3 -> f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn cast_f32_to_fp8e4m3_lattice_grid() {
    let (ctx, stream) = setup();
    // Same exactly-representable grid as the reverse test.
    let host_x: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
    let numel = host_x.len() as i32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let dev_y_bytes: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc");
    let mut dev_y: DeviceBuffer<Fp8E4M3> = unsafe { core::mem::transmute(dev_y_bytes) };

    let desc = CastSubByteDescriptor {
        numel,
        input_element: ElementKind::F32,
        output_element: ElementKind::Fp8E4M3,
    };
    let plan =
        CastSubBytePlan::<f32, Fp8E4M3>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = CastSubByteArgs::<f32, Fp8E4M3> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    // Reinterpret dev_y as u8 for download.
    let dev_y_bytes_back: DeviceBuffer<u8> = unsafe { core::mem::transmute(dev_y) };
    let mut got_bytes = vec![0u8; numel as usize];
    dev_y_bytes_back.copy_to_host(&mut got_bytes).expect("download");

    let expected: Vec<u8> = host_x.iter().map(|&f| Fp8E4M3::from_f32(f).0).collect();
    assert_eq!(got_bytes, expected, "f32 -> fp8e4m3");
}

// =============================================================================
// Bool -> f32 truthiness — 0 stays 0.0, any non-zero byte becomes 1.0.
// =============================================================================

#[test]
#[ignore]
fn cast_bool_to_f32_normalizes_truthy_bytes() {
    let (ctx, stream) = setup();
    // Mix of false (0), canonical true (1), and non-canonical truthy
    // bytes (255 = 0xFF, 7, 128). The kernel must normalize all
    // non-zero bytes to 1.0.
    let host_bytes: Vec<u8> = vec![0, 1, 255, 0, 7, 128, 0, 1];
    let numel = host_bytes.len() as i32;
    let expected: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let dev_x_u8 = DeviceBuffer::from_slice(&ctx, &host_bytes).expect("upload");
    let dev_x: DeviceBuffer<Bool> = unsafe { core::mem::transmute(dev_x_u8) };
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc");

    let desc = CastSubByteDescriptor {
        numel,
        input_element: ElementKind::Bool,
        output_element: ElementKind::F32,
    };
    let plan = CastSubBytePlan::<Bool, f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = CastSubByteArgs::<Bool, f32> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel as usize];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "bool -> f32 mismatch @ {i}");
    }
}

// =============================================================================
// Select-time validation: odd numel for S4/U4 endpoints must reject.
// =============================================================================

#[test]
fn select_rejects_odd_numel_for_s4() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let desc = CastSubByteDescriptor {
        numel: 7, // odd — must be rejected.
        input_element: ElementKind::S4,
        output_element: ElementKind::I32,
    };
    let res = CastSubBytePlan::<S4, i32>::select(&stream, &desc, PlanPreference::default());
    assert!(res.is_err(), "odd numel must be rejected for S4 endpoints");
    let _ = ctx;
}

#[test]
fn select_rejects_unsupported_pair() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    // f32 -> f64 isn't in the sub-byte plan's coverage (it's classic CastPlan turf).
    let desc = CastSubByteDescriptor {
        numel: 8,
        input_element: ElementKind::F32,
        output_element: ElementKind::F64,
    };
    let res = CastSubBytePlan::<f32, f64>::select(&stream, &desc, PlanPreference::default());
    assert!(res.is_err(), "f32 -> f64 must be rejected (not in 13.3 scope)");
    let _ = ctx;
}
