#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for the rest of the Phase 11.8 pool plans:
//! AdaptiveAvgPool1d / 3d, AdaptiveMaxPool1d / 2d / 3d, and the stubbed
//! FractionalMaxPool* / LpPool* plans (which assert `select()` rejects).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AdaptiveAvgPool1dPlan, AdaptiveAvgPool3dPlan, AdaptiveMaxPool1dPlan,
    AdaptiveMaxPool2dPlan, AdaptiveMaxPool3dPlan, AdaptivePool1dDescriptor, AdaptivePool1dFwArgs,
    AdaptivePool2dDescriptor, AdaptivePool2dFwArgs, AdaptivePool3dDescriptor,
    AdaptivePool3dFwArgs, ElementKind, FractionalMaxPool2dDescriptor, FractionalMaxPool2dPlan,
    FractionalMaxPool3dDescriptor, FractionalMaxPool3dPlan, LpPool1dDescriptor, LpPool1dPlan,
    LpPool2dDescriptor, LpPool2dPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn adaptive_avg_pool1d_8_to_4_f32() {
    let (ctx, stream) = setup();
    let host_x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Divisible: kernel=2, stride=2 → output[i] = (x[2i] + x[2i+1]) / 2.
    let exp_y: Vec<f32> = vec![1.5, 3.5, 5.5, 7.5];
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("y");
    let desc = AdaptivePool1dDescriptor {
        batch: 1, channels: 1, l_in: 8, l_out: 4, element: ElementKind::F32,
    };
    let plan = AdaptiveAvgPool1dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    // `derived_kernel_stride` was deprecated in Phase 16.1 — the
    // bespoke kernel uses per-output-cell variable windows; no single
    // (kernel, stride) pair represents the op. The deprecated getter
    // returns `(0, 0)` for source-compat.
    #[allow(deprecated)]
    {
        assert_eq!(plan.derived_kernel_stride(), (0, 0));
    }
    plan.run_fw(&stream, Workspace::None, AdaptivePool1dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: [1, 1, 8], stride: contiguous_stride([1, 1, 8]) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1, 4], stride: contiguous_stride([1, 1, 4]) },
    }).expect("fw");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 4];
    dev_y.copy_to_host(&mut got).expect("dl");
    let tol = 32.0 * f32::EPSILON;
    for i in 0..4 {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got[i] - exp_y[i]).abs() <= t, "adaptive_avg_pool1d @ {i}");
    }
}

#[test]
#[ignore]
fn adaptive_avg_pool3d_2x2x2_to_1x1x1_f32() {
    let (ctx, stream) = setup();
    let host_x: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    // Divisible: kernel=2 per axis, stride=2 per axis → single output =
    // average of all 8.
    let exp_y: Vec<f32> = vec![4.5];
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("y");
    let desc = AdaptivePool3dDescriptor {
        batch: 1, channels: 1, d_in: 2, h_in: 2, w_in: 2,
        d_out: 1, h_out: 1, w_out: 1, element: ElementKind::F32,
    };
    let plan = AdaptiveAvgPool3dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run_fw(&stream, Workspace::None, AdaptivePool3dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: [1, 1, 2, 2, 2], stride: contiguous_stride([1, 1, 2, 2, 2]) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1, 1, 1, 1], stride: contiguous_stride([1, 1, 1, 1, 1]) },
    }).expect("fw");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 1];
    dev_y.copy_to_host(&mut got).expect("dl");
    let tol = 32.0 * f32::EPSILON;
    assert!((got[0] - exp_y[0]).abs() <= (exp_y[0].abs() * tol).max(tol),
        "adaptive_avg_pool3d: got={} want={}", got[0], exp_y[0]);
}

#[test]
#[ignore]
fn adaptive_max_pool1d_8_to_4_f32() {
    let (ctx, stream) = setup();
    let host_x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // kernel=2, stride=2 → output[i] = max(x[2i], x[2i+1]).
    let exp_y: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("y");
    let desc = AdaptivePool1dDescriptor {
        batch: 1, channels: 1, l_in: 8, l_out: 4, element: ElementKind::F32,
    };
    let plan = AdaptiveMaxPool1dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run_fw(&stream, Workspace::None, AdaptivePool1dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: [1, 1, 8], stride: contiguous_stride([1, 1, 8]) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1, 4], stride: contiguous_stride([1, 1, 4]) },
    }).expect("fw");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 4];
    dev_y.copy_to_host(&mut got).expect("dl");
    for i in 0..4 {
        assert!((got[i] - exp_y[i]).abs() <= 16.0 * f32::EPSILON,
            "adaptive_max_pool1d @ {i}: got={} want={}", got[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn adaptive_max_pool2d_4x4_to_2x2_f32() {
    let (ctx, stream) = setup();
    let host_x: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    // kernel=2, stride=2 → max of each disjoint 2×2 tile.
    let exp_y: Vec<f32> = vec![6.0, 8.0, 14.0, 16.0];
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("y");
    let desc = AdaptivePool2dDescriptor {
        batch: 1, channels: 1, h_in: 4, w_in: 4, h_out: 2, w_out: 2,
        element: ElementKind::F32,
    };
    let plan = AdaptiveMaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run_fw(&stream, Workspace::None, AdaptivePool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: [1, 1, 4, 4], stride: contiguous_stride([1, 1, 4, 4]) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1, 2, 2], stride: contiguous_stride([1, 1, 2, 2]) },
    }).expect("fw");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 4];
    dev_y.copy_to_host(&mut got).expect("dl");
    for i in 0..4 {
        assert!((got[i] - exp_y[i]).abs() <= 16.0 * f32::EPSILON,
            "adaptive_max_pool2d @ {i}: got={} want={}", got[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn adaptive_max_pool3d_2x2x2_to_1x1x1_f32() {
    let (ctx, stream) = setup();
    let host_x: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let exp_y: Vec<f32> = vec![8.0];
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("y");
    let desc = AdaptivePool3dDescriptor {
        batch: 1, channels: 1, d_in: 2, h_in: 2, w_in: 2,
        d_out: 1, h_out: 1, w_out: 1, element: ElementKind::F32,
    };
    let plan = AdaptiveMaxPool3dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run_fw(&stream, Workspace::None, AdaptivePool3dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: [1, 1, 2, 2, 2], stride: contiguous_stride([1, 1, 2, 2, 2]) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1, 1, 1, 1], stride: contiguous_stride([1, 1, 1, 1, 1]) },
    }).expect("fw");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 1];
    dev_y.copy_to_host(&mut got).expect("dl");
    assert!((got[0] - exp_y[0]).abs() <= 16.0 * f32::EPSILON,
        "adaptive_max_pool3d: got={} want={}", got[0], exp_y[0]);
}

// ---------------------------------------------------------------------------
// Stub-plan rejections — these don't need a real GPU, but we keep them
// behind `#[ignore]` for consistency with the rest of the smoke suite.
// ---------------------------------------------------------------------------

// Phase 16.3 — FractionalMaxPool is now implemented as a bespoke
// kernel with caller-supplied uniform random samples. `select()`
// accepts well-formed descriptors. Full FW + BW coverage lives in
// `fractional_max_pool_smoke.rs`.

#[test]
#[ignore]
fn fractional_max_pool2d_select_accepts() {
    let (_ctx, stream) = setup();
    let desc = FractionalMaxPool2dDescriptor {
        batch: 1, channels: 1, h_in: 8, w_in: 8,
        window_h: 2, window_w: 2, h_out: 4, w_out: 4, seed: 0,
        element: ElementKind::F32,
    };
    let r = FractionalMaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(r.is_ok(), "FractionalMaxPool2d should accept well-formed select");
}

#[test]
#[ignore]
fn fractional_max_pool3d_select_accepts() {
    let (_ctx, stream) = setup();
    let desc = FractionalMaxPool3dDescriptor {
        batch: 1, channels: 1, d_in: 4, h_in: 8, w_in: 8,
        window_d: 2, window_h: 2, window_w: 2,
        d_out: 2, h_out: 4, w_out: 4, seed: 0,
        element: ElementKind::F32,
    };
    let r = FractionalMaxPool3dPlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(r.is_ok(), "FractionalMaxPool3d should accept well-formed select");
}

// Phase 16.2 — LpPool is now implemented as a bespoke fused kernel.
// `select()` accepts well-formed descriptors and only rejects p=∞,
// non-finite p, or shape errors. Bit-exact FW + BW coverage lives in
// `lp_pool_smoke.rs`.

#[test]
#[ignore]
fn lp_pool1d_select_accepts() {
    let (_ctx, stream) = setup();
    let desc = LpPool1dDescriptor {
        batch: 1, channels: 1, l_in: 8, window: 2, stride: 2,
        p: 2.0, ceil_mode: false, element: ElementKind::F32,
    };
    let r = LpPool1dPlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(r.is_ok(), "LpPool1d should accept select for p=2");
}

#[test]
#[ignore]
fn lp_pool2d_select_accepts() {
    let (_ctx, stream) = setup();
    let desc = LpPool2dDescriptor {
        batch: 1, channels: 1, h_in: 8, w_in: 8,
        window_h: 2, window_w: 2, stride_h: 2, stride_w: 2,
        p: 2.0, ceil_mode: false, element: ElementKind::F32,
    };
    let r = LpPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(r.is_ok(), "LpPool2d should accept select for p=2");
}
