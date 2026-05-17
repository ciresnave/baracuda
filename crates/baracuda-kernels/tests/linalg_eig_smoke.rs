//! Real-GPU smoke test for `EigPlan` (cuSOLVER Xgeev wrap).
//!
//! Tests the LAPACK-packed-real output convention (real input → `[wr,
//! wi]` packed into a `2N` real array; complex input → `N` complex
//! array). See `crates/baracuda-kernels/src/linalg/eig.rs` module
//! docs for the full convention.
//!
//! Fixture: 2×2 rotation matrix
//! ```text
//!   A = [ 0 -1 ]   →   λ_{0,1} = ±i
//!       [ 1  0 ]
//! ```
//! For real input: `wr ≈ [0, 0]`, `|wi[0]| ≈ |wi[1]| ≈ 1` with opposite
//! signs. For complex input: `W` contains `±i` directly.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Complex32, Complex64, EigArgs, EigDescriptor, EigPlan, ElementKind,
    PlanPreference, TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Column-major view of
///
/// ```text
///   A = [ 0 -1 ]
///       [ 1  0 ]
/// ```
fn rotation_2x2_f32() -> Vec<f32> {
    vec![0.0, 1.0, -1.0, 0.0]
}

fn rotation_2x2_f64() -> Vec<f64> {
    vec![0.0, 1.0, -1.0, 0.0]
}

fn rotation_2x2_c32() -> Vec<Complex32> {
    [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 0.0)]
        .iter()
        .map(|&(r, i)| Complex32::new(r, i))
        .collect()
}

fn rotation_2x2_c64() -> Vec<Complex64> {
    [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 0.0)]
        .iter()
        .map(|&(r, i)| Complex64::new(r, i))
        .collect()
}

#[test]
#[ignore]
fn eig_f32_pm_i() {
    let (ctx, stream) = setup();
    let n: i32 = 2;
    let a_host = rotation_2x2_f32();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    // Real input → W is [2*N] real: [wr_0, wr_1, wi_0, wi_1].
    let mut dev_w: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (2 * n) as usize).expect("alloc w");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = EigDescriptor {
        n,
        compute_left: false,
        compute_right: false,
        element: ElementKind::F32,
    };
    let plan = EigPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select EigPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let ws_alloc = ws_bytes.max(1);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_alloc).expect("alloc ws");

    let args = EigArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [n, n],
            stride: contiguous_stride([n, n]),
        },
        w: TensorMut {
            data: dev_w.as_slice_mut(),
            shape: [2 * n],
            stride: [1],
        },
        vl: None,
        vr: None,
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run eig f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eig f32 info != 0");

    let mut w_host = vec![0f32; (2 * n) as usize];
    dev_w.copy_to_host(&mut w_host).expect("dl w");

    // wr ≈ [0, 0]; wi ≈ [+1, -1] (or [-1, +1] — order is implementation-
    // defined). Verify the multiset {(0, +1), (0, -1)}.
    let tol = 1e-5f32;
    let pairs = [(w_host[0], w_host[2]), (w_host[1], w_host[3])];
    for (re, im) in pairs.iter() {
        assert!(re.abs() <= tol, "eig f32: wr={re} not near 0");
        assert!(
            (im.abs() - 1.0).abs() <= tol,
            "eig f32: |wi|={} not near 1",
            im.abs()
        );
    }
    let signs_opposite = (pairs[0].1 * pairs[1].1) < 0.0;
    assert!(signs_opposite, "eig f32: wi pair {pairs:?} should have opposite signs");
}

#[test]
#[ignore]
fn eig_f64_pm_i() {
    let (ctx, stream) = setup();
    let n: i32 = 2;
    let a_host = rotation_2x2_f64();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_w: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (2 * n) as usize).expect("alloc w");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = EigDescriptor {
        n,
        compute_left: false,
        compute_right: false,
        element: ElementKind::F64,
    };
    let plan = EigPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select EigPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let ws_alloc = ws_bytes.max(1);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_alloc).expect("alloc ws");

    let args = EigArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [n, n],
            stride: contiguous_stride([n, n]),
        },
        w: TensorMut {
            data: dev_w.as_slice_mut(),
            shape: [2 * n],
            stride: [1],
        },
        vl: None,
        vr: None,
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run eig f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eig f64 info != 0");

    let mut w_host = vec![0f64; (2 * n) as usize];
    dev_w.copy_to_host(&mut w_host).expect("dl w");

    let tol = 1e-12f64;
    let pairs = [(w_host[0], w_host[2]), (w_host[1], w_host[3])];
    for (re, im) in pairs.iter() {
        assert!(re.abs() <= tol, "eig f64: wr={re} not near 0");
        assert!(
            (im.abs() - 1.0).abs() <= tol,
            "eig f64: |wi|={} not near 1",
            im.abs()
        );
    }
    let signs_opposite = (pairs[0].1 * pairs[1].1) < 0.0;
    assert!(signs_opposite, "eig f64: wi pair {pairs:?} should have opposite signs");
}

#[test]
#[ignore]
fn eig_complex32_pm_i() {
    let (ctx, stream) = setup();
    let n: i32 = 2;
    let a_host = rotation_2x2_c32();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_w: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, n as usize).expect("alloc w");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = EigDescriptor {
        n,
        compute_left: false,
        compute_right: false,
        element: ElementKind::Complex32,
    };
    let plan = EigPlan::<Complex32>::select(&stream, &desc, PlanPreference::default())
        .expect("select EigPlan<Complex32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let ws_alloc = ws_bytes.max(1);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_alloc).expect("alloc ws");

    let args = EigArgs::<Complex32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [n, n],
            stride: contiguous_stride([n, n]),
        },
        w: TensorMut {
            data: dev_w.as_slice_mut(),
            shape: [n],
            stride: [1],
        },
        vl: None,
        vr: None,
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run eig Complex32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eig Complex32 info != 0");

    let mut w_host = vec![Complex32::new(0.0, 0.0); n as usize];
    dev_w.copy_to_host(&mut w_host).expect("dl w");

    let tol = 1e-5f32;
    for c in w_host.iter() {
        assert!(c.re.abs() <= tol, "eig Complex32: re={} not near 0", c.re);
        assert!(
            (c.im.abs() - 1.0).abs() <= tol,
            "eig Complex32: |im|={} not near 1",
            c.im.abs()
        );
    }
    let signs_opposite = (w_host[0].im * w_host[1].im) < 0.0;
    assert!(
        signs_opposite,
        "eig Complex32: eigenvalues should be conjugate pair, got {:?}",
        w_host
    );
}

#[test]
#[ignore]
fn eig_complex64_pm_i() {
    let (ctx, stream) = setup();
    let n: i32 = 2;
    let a_host = rotation_2x2_c64();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_w: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, n as usize).expect("alloc w");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = EigDescriptor {
        n,
        compute_left: false,
        compute_right: false,
        element: ElementKind::Complex64,
    };
    let plan = EigPlan::<Complex64>::select(&stream, &desc, PlanPreference::default())
        .expect("select EigPlan<Complex64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let ws_alloc = ws_bytes.max(1);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_alloc).expect("alloc ws");

    let args = EigArgs::<Complex64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [n, n],
            stride: contiguous_stride([n, n]),
        },
        w: TensorMut {
            data: dev_w.as_slice_mut(),
            shape: [n],
            stride: [1],
        },
        vl: None,
        vr: None,
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run eig Complex64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eig Complex64 info != 0");

    let mut w_host = vec![Complex64::new(0.0, 0.0); n as usize];
    dev_w.copy_to_host(&mut w_host).expect("dl w");

    let tol = 1e-12f64;
    for c in w_host.iter() {
        assert!(c.re.abs() <= tol, "eig Complex64: re={} not near 0", c.re);
        assert!(
            (c.im.abs() - 1.0).abs() <= tol,
            "eig Complex64: |im|={} not near 1",
            c.im.abs()
        );
    }
    let signs_opposite = (w_host[0].im * w_host[1].im) < 0.0;
    assert!(
        signs_opposite,
        "eig Complex64: eigenvalues should be conjugate pair, got {:?}",
        w_host
    );
}
