#![cfg(feature = "cudnn")]
//! Real-GPU smoke test for `AdaptiveAvgPool2dPlan` FW.
//!
//! Verifies the **bit-exact PyTorch** bespoke kernel (Phase 16.1)
//! matches a hand-computed expected for a 4×4 → 2×2 case. In this
//! divisible case (`4 % 2 == 0`) the PyTorch formula collapses to a
//! uniform `kernel=2, stride=2, pad=0` window per axis — same answer
//! the prior cuDNN approximation produced. Non-divisible regression
//! coverage lives in `adaptive_pool_bitexact_smoke.rs`.
//!
//! Each output cell averages a disjoint 2×2 input tile:
//!
//! ```text
//! Input 4×4 (C=0):              Output 2×2:
//!   1  2 |  3  4               (1+2+5+6)/4   (3+4+7+8)/4
//!   5  6 |  7  8                = 3.5         = 5.5
//!  ────  ─────             →   (9+10+13+14)/4 (11+12+15+16)/4
//!   9 10 | 11 12                = 11.5         = 13.5
//!  13 14 | 15 16
//! ```

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AdaptiveAvgPool2dPlan, AdaptivePool2dDescriptor, AdaptivePool2dFwArgs,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
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
fn adaptive_avg_pool2d_4x4_to_2x2_f32() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    let (h_out, w_out) = (2i32, 2i32);
    // Row-major 4×4 with values 1..=16.
    let host_x: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    // Expected: average of each disjoint 2×2 tile.
    let exp_y: Vec<f32> = vec![3.5, 5.5, 11.5, 13.5];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("y");

    let desc = AdaptivePool2dDescriptor {
        batch: n, channels: c, h_in, w_in, h_out, w_out,
        element: ElementKind::F32,
    };
    let plan = AdaptiveAvgPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");

    // `derived_kernel_stride` was deprecated in Phase 16.1 — the
    // bespoke kernel uses per-output-cell variable windows; no single
    // (kernel, stride) pair represents the op. The deprecated getter
    // now returns `((0,0), (0,0))` for source-compat.
    #[allow(deprecated)]
    {
        let ((kh, sh), (kw, sw)) = plan.derived_kernel_stride();
        assert_eq!((kh, sh), (0, 0));
        assert_eq!((kw, sw), (0, 0));
    }

    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    plan.run_fw(&stream, Workspace::None, AdaptivePool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; 4];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 32.0 * f32::EPSILON;
    for i in 0..4 {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "adaptive_avg_pool2d f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
}
