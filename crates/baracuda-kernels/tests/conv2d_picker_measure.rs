//! Measures the opt-in `Conv2dPlan` forward algorithm picker against the pinned `IMPLICIT_GEMM`
//! on typical shapes (SA-25: the evidence for or against making the picker the default).
//!
//! For each shape it runs the forward pass with the picker off and on (f32, NCHW), prints the
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
    Conv2dArgs, Conv2dDescriptor, Conv2dPlan, ElementKind, PlanPreference, TensorMut, TensorRef,
    Workspace, contiguous_stride,
};

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

fn fill(len: usize, mul: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * mul % 101) as f32 - 50.0) / 50.0)
        .collect()
}

fn plan(stream: &Stream, s: &Shape, picker: bool) -> Conv2dPlan<f32> {
    let &(_, n, c_in, h, w, c_out, k, pad, stride) = s;
    let desc = Conv2dDescriptor::new(n, c_in, h, w, c_out, k, k, ElementKind::F32)
        .with_padding(pad, pad)
        .with_stride(stride, stride);
    Conv2dPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select")
        .with_fw_algo_picker(picker)
}

/// Runs `s` forward with the picker `on`/off: returns the output, the resolved algo and mean ms.
fn run(ctx: &Context, stream: &Stream, s: &Shape, picker: bool) -> (Vec<f32>, Option<i32>, f64) {
    let &(_, n, c_in, h, w, c_out, k, _, _) = s;
    let p = plan(stream, s, picker);
    let (h_out, w_out) = p.output_dims();
    let (xs, ws, ys) = (
        [n, c_in, h, w],
        [c_out, c_in, k, k],
        [n, c_out, h_out, w_out],
    );
    let numel = |d: [i32; 4]| d.iter().product::<i32>() as usize;
    let dev_x = DeviceBuffer::from_slice(ctx, &fill(numel(xs), 37)).expect("up x");
    let dev_w = DeviceBuffer::from_slice(ctx, &fill(numel(ws), 53)).expect("up w");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, numel(ys)).expect("alloc y");
    let ws_bytes = p.query_fw_workspace_size(stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, ws_bytes.max(1)).expect("alloc ws");
    let mut once = || {
        let workspace = if ws_bytes == 0 {
            Workspace::None
        } else {
            Workspace::Borrowed(dev_ws.as_slice_mut())
        };
        let args = Conv2dArgs::<f32> {
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
    let mut got = vec![0f32; numel(ys)];
    dev_y.copy_to_host(&mut got).expect("dl y");
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

#[test]
fn picker_vs_pinned_forward_on_typical_shapes() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    eprintln!("shape | pinned algo ms | picker algo ms | speedup | rel diff");
    for s in &SHAPES {
        let (pinned, pinned_algo, pinned_ms) = run(&ctx, &stream, s, false);
        let (picked, picked_algo, picked_ms) = run(&ctx, &stream, s, true);
        let diff = rel_diff(&pinned, &picked);
        eprintln!(
            "{} | {:?} {:.3} | {:?} {:.3} | {:.2}x | {:.2e}",
            s.0,
            pinned_algo,
            pinned_ms,
            picked_algo,
            picked_ms,
            pinned_ms / picked_ms,
            diff
        );
        assert_eq!(pinned_algo, Some(0), "default must stay IMPLICIT_GEMM");
        assert!(diff < 1e-2, "{}: picker output differs by {diff}", s.0);
    }
}
