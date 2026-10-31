//! Measures the opt-in `Conv2dPlan` forward algorithm picker against the pinned `IMPLICIT_GEMM`
//! on typical shapes (SA-25: the evidence for or against making the picker the default).
//!
//! For each shape and dtype (f32, f16, bf16; NCHW) it runs the forward pass with the picker off and on, prints the
//! resolved algorithm and the mean wall time over `ITERS` stream-synchronised runs, and asserts only
//! that the two outputs agree within a tolerance relative to the output's scale. It asserts NO
//! speed: timings are printed for a human to read (`-- --nocapture`), one device, one run.
//!
//! Run with `cargo gpu-test -p baracuda-kernels --features cudnn,device-tests --test conv2d_picker_measure -- --nocapture`.
#![cfg(feature = "cudnn")]
#![cfg(feature = "device-tests")]

use std::time::Instant;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    Conv2dArgs, Conv2dDescriptor, Conv2dPlan, Element, ElementKind, PlanPreference, TensorMut,
    TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

const ITERS: u32 = 20;

/// `(label, n, c_in, h, w, c_out, kernel, pad, stride)`.
type Shape = (&'static str, i32, i32, i32, i32, i32, i32, i32, i32);

const SHAPES: [Shape; 5] = [
    ("3x3 s1 64ch 56x56", 8, 64, 56, 56, 64, 3, 1, 1),
    ("3x3 s1 128ch 28x28", 8, 128, 28, 28, 128, 3, 1, 1),
    ("3x3 s1 256ch 14x14", 8, 256, 14, 14, 256, 3, 1, 1),
    ("1x1 s1 256->64 56x56", 8, 256, 56, 56, 64, 1, 0, 1),
    ("7x7 s2 3->64 224x224", 8, 3, 224, 224, 64, 7, 3, 2),
];

fn fill<T>(len: usize, mul: usize, from: fn(f32) -> T) -> Vec<T> {
    (0..len)
        .map(|i| from(((i * mul % 101) as f32 - 50.0) / 50.0))
        .collect()
}

/// Per-dtype conversions and the agreement tolerance.
struct Cfg<T> {
    kind: ElementKind,
    from: fn(f32) -> T,
    to: fn(T) -> f32,
    tol: f32,
}

fn plan<T: Element>(stream: &Stream, s: &Shape, picker: bool, kind: ElementKind) -> Conv2dPlan<T> {
    let &(_, n, c_in, h, w, c_out, k, pad, stride) = s;
    let desc = Conv2dDescriptor::new(n, c_in, h, w, c_out, k, k, kind)
        .with_padding(pad, pad)
        .with_stride(stride, stride);
    Conv2dPlan::<T>::select(stream, &desc, PlanPreference::default())
        .expect("select")
        .with_fw_algo_picker(picker)
}

/// Runs `s` forward with the picker `on`/off: returns the output, the resolved algo and mean ms.
fn run<T: Element + 'static>(
    ctx: &Context,
    stream: &Stream,
    s: &Shape,
    picker: bool,
    cfg: &Cfg<T>,
) -> (Vec<f32>, Option<i32>, f64) {
    let &(_, n, c_in, h, w, c_out, k, _, _) = s;
    let p = plan::<T>(stream, s, picker, cfg.kind);
    let (h_out, w_out) = p.output_dims();
    let (xs, ws, ys) = (
        [n, c_in, h, w],
        [c_out, c_in, k, k],
        [n, c_out, h_out, w_out],
    );
    let numel = |d: [i32; 4]| d.iter().product::<i32>() as usize;
    let dev_x = DeviceBuffer::from_slice(ctx, &fill(numel(xs), 37, cfg.from)).expect("up x");
    let dev_w = DeviceBuffer::from_slice(ctx, &fill(numel(ws), 53, cfg.from)).expect("up w");
    let mut dev_y: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, numel(ys)).expect("alloc y");
    let ws_bytes = p.query_fw_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes.max(1)).expect("alloc ws");
    let mut once = || {
        let workspace = if ws_bytes == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws.as_slice_mut())
        };
        let args = Conv2dArgs::<T> {
            x: TensorRef::new(dev_x.as_slice(), xs, contiguous_stride(xs)).unwrap(),
            w: TensorRef::new(dev_w.as_slice(), ws, contiguous_stride(ws)).unwrap(),
            y: TensorMut::new(dev_y.as_slice_mut(), ys, contiguous_stride(ys)).unwrap(),
        };
        p.run_fw(stream, workspace, args).expect("run_fw");
    };
    once();
    stream.synchronize().expect("warm-up sync");
    let t = Instant::now();
    for _ in 0..ITERS {
        once();
    }
    stream.synchronize().expect("sync");
    let ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);
    let mut got = vec![(cfg.from)(0.0); numel(ys)];
    dev_y.copy_to_host(&mut got).expect("dl y");
    let got = got.into_iter().map(cfg.to).collect();
    (got, p.fw_algo(), ms)
}

/// Largest absolute difference, relative to the largest magnitude in `a`.
fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

/// Prints one table for dtype `name` and asserts the picker agrees with the pinned output.
fn measure<T: Element + 'static>(ctx: &Context, stream: &Stream, name: &str, cfg: &Cfg<T>) {
    eprintln!("== {name}: shape | pinned algo ms | picker algo ms | speedup | rel diff");
    for s in &SHAPES {
        let (pinned, pinned_algo, pinned_ms) = run(ctx, stream, s, false, cfg);
        let (picked, picked_algo, picked_ms) = run(ctx, stream, s, true, cfg);
        let diff = rel_diff(&pinned, &picked);
        eprintln!(
            "{name} {} | {:?} {:.3} | {:?} {:.3} | {:.2}x | {:.2e}",
            s.0,
            pinned_algo,
            pinned_ms,
            picked_algo,
            picked_ms,
            pinned_ms / picked_ms,
            diff
        );
        assert_eq!(pinned_algo, Some(0), "default must stay IMPLICIT_GEMM");
        assert!(diff < cfg.tol, "{name} {}: picker differs by {diff}", s.0);
    }
}

#[test]
fn picker_vs_pinned_forward_on_typical_shapes() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let f32c = Cfg::<f32> {
        kind: ElementKind::F32,
        from: |v| v,
        to: |v| v,
        tol: 1e-2,
    };
    measure(&ctx, &stream, "f32", &f32c);
    let f16c = Cfg::<f16> {
        kind: ElementKind::F16,
        from: f16::from_f32,
        to: f16::to_f32,
        tol: 5e-2,
    };
    measure(&ctx, &stream, "f16", &f16c);
    let bf16c = Cfg::<bf16> {
        kind: ElementKind::Bf16,
        from: bf16::from_f32,
        to: bf16::to_f32,
        tol: 1e-1,
    };
    measure(&ctx, &stream, "bf16", &bf16c);
}
