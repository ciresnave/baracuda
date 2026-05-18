//! Real-GPU smoke test for `GridSamplePlan<T>` + `AffineGridPlan<T>`
//! (Phase 9 Category T). PyTorch defaults: bilinear, zeros pad,
//! `align_corners=false`. `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AffineGridArgs, AffineGridDescriptor, AffineGridPlan, ElementKind,
    GridSampleArgs, GridSampleBackwardArgs, GridSampleBackwardDescriptor,
    GridSampleBackwardPlan, GridSampleDescriptor, GridSamplePlan, PlanPreference, TensorMut,
    TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn grid_to_src(nx: f32, src: i32) -> f32 {
    ((nx + 1.0) * src as f32 - 1.0) * 0.5
}

#[test]
#[ignore]
fn grid_sample_identity_f32() {
    // 1×1×2×2 input; grid covers the 4 pixel centers → output should equal input.
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (2, 2);
    let host_in: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    // pixel centers for align_corners=false correspond to norm coords:
    //  (2 * o + 1) / size - 1
    let mut host_grid: Vec<f32> = vec![0.0; (n * oh * ow * 2) as usize];
    for ohh in 0..oh {
        let ny = (2.0 * ohh as f32 + 1.0) / oh as f32 - 1.0;
        for oww in 0..ow {
            let nx = (2.0 * oww as f32 + 1.0) / ow as f32 - 1.0;
            let off = (((0 * oh) + ohh) * ow + oww) as usize * 2;
            host_grid[off] = nx;
            host_grid[off + 1] = ny;
        }
    }
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("up in");
    let dev_grid = DeviceBuffer::from_slice(&ctx, &host_grid).expect("up grid");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * oh * ow) as usize).expect("alloc");
    let desc = GridSampleDescriptor {
        n, c, ih, iw, oh, ow,
        element: ElementKind::F32,
    };
    let plan = GridSamplePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GridSampleArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        grid: TensorRef {
            data: dev_grid.as_slice(),
            shape: [n, oh, ow, 2],
            stride: contiguous_stride([n, oh, ow, 2]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_in.len()];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(host_in.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "grid_sample identity f32 mismatch @ {i}: got {g} vs {e}");
    }
    // Suppress unused warning (sanity check helper is intentionally inline).
    let _ = grid_to_src(0.0, ih);
}

#[test]
#[ignore]
fn grid_sample_zero_pad_oob_f32() {
    // Sample at (-2, -2) normalized → way OOB → 0.
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (1, 1);
    let host_in: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let host_grid: Vec<f32> = vec![-2.0, -2.0];
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("up in");
    let dev_grid = DeviceBuffer::from_slice(&ctx, &host_grid).expect("up grid");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc");
    let desc = GridSampleDescriptor {
        n, c, ih, iw, oh, ow,
        element: ElementKind::F32,
    };
    let plan = GridSamplePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GridSampleArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        grid: TensorRef {
            data: dev_grid.as_slice(),
            shape: [n, oh, ow, 2],
            stride: contiguous_stride([n, oh, ow, 2]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 1];
    dev_out.copy_to_host(&mut got).expect("dl");
    assert!(got[0].abs() < 1e-6, "grid_sample OOB should be 0, got {}", got[0]);
}

#[test]
#[ignore]
fn grid_sample_backward_smoke_f32() {
    // Identity grid; constant dout=1 → dinput sum should equal dout sum.
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (2, 2);
    let host_in: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut host_grid: Vec<f32> = vec![0.0; (n * oh * ow * 2) as usize];
    for ohh in 0..oh {
        let ny = (2.0 * ohh as f32 + 1.0) / oh as f32 - 1.0;
        for oww in 0..ow {
            let nx = (2.0 * oww as f32 + 1.0) / ow as f32 - 1.0;
            let off = ((ohh * ow + oww) * 2) as usize;
            host_grid[off] = nx;
            host_grid[off + 1] = ny;
        }
    }
    let host_dout: Vec<f32> = vec![1.0; 4];
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("");
    let dev_grid = DeviceBuffer::from_slice(&ctx, &host_grid).expect("");
    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("");
    let mut dev_din: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("");
    let mut dev_dgrid: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 8).expect("");
    let desc = GridSampleBackwardDescriptor {
        n, c, ih, iw, oh, ow,
        element: ElementKind::F32,
    };
    let plan = GridSampleBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GridSampleBackwardArgs {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [n, c, oh, ow],
            stride: contiguous_stride([n, c, oh, ow]),
        },
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        grid: TensorRef {
            data: dev_grid.as_slice(),
            shape: [n, oh, ow, 2],
            stride: contiguous_stride([n, oh, ow, 2]),
        },
        dinput: TensorMut {
            data: dev_din.as_slice_mut(),
            shape: [n, c, ih, iw],
            stride: contiguous_stride([n, c, ih, iw]),
        },
        dgrid: TensorMut {
            data: dev_dgrid.as_slice_mut(),
            shape: [n, oh, ow, 2],
            stride: contiguous_stride([n, oh, ow, 2]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_di = vec![0f32; 4];
    dev_din.copy_to_host(&mut got_di).expect("");
    let sum: f32 = got_di.iter().sum();
    assert!((sum - 4.0).abs() < 1e-4, "BW dinput sum {sum} != 4");
}

#[test]
#[ignore]
fn affine_grid_identity_f32() {
    let (ctx, stream) = setup();
    let n = 1; let oh = 3; let ow = 4;
    // Identity affine: 1,0,0 ; 0,1,0
    let host_theta: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let dev_theta = DeviceBuffer::from_slice(&ctx, &host_theta).expect("");
    let mut dev_grid: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (n * oh * ow * 2) as usize).expect("");
    let desc = AffineGridDescriptor { n, oh, ow, element: ElementKind::F32 };
    let plan = AffineGridPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("");
    let args = AffineGridArgs {
        theta: TensorRef {
            data: dev_theta.as_slice(),
            shape: [n, 2, 3],
            stride: contiguous_stride([n, 2, 3]),
        },
        grid: TensorMut {
            data: dev_grid.as_slice_mut(),
            shape: [n, oh, ow, 2],
            stride: contiguous_stride([n, oh, ow, 2]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; (n * oh * ow * 2) as usize];
    dev_grid.copy_to_host(&mut got).expect("");
    // Identity → base coords passed through unchanged.
    for ohh in 0..oh {
        let by = (2.0 * ohh as f32 + 1.0) / oh as f32 - 1.0;
        for oww in 0..ow {
            let bx = (2.0 * oww as f32 + 1.0) / ow as f32 - 1.0;
            let off = ((ohh * ow + oww) * 2) as usize;
            assert!((got[off] - bx).abs() < 1e-5);
            assert!((got[off + 1] - by).abs() < 1e-5);
        }
    }
}
