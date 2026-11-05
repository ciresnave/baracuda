//! Measures the opt-in `Conv2dPlan` forward algorithm picker against the pinned `IMPLICIT_GEMM`
//! on typical shapes (SA-25: the evidence for or against making the picker the default).
//!
//! For each shape and dtype (f32, f16, bf16; NCHW) it runs the forward pass with the picker off and on, prints the
//! resolved algorithm and the mean wall time over `ITERS` stream-synchronised runs, and asserts only
//! that the two outputs agree within a tolerance relative to the output's scale. It asserts NO
//! speed: timings are printed for a human to read (`-- --nocapture`), one device; each shape is repeated `REPEATS` times and min/median printed.
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

/// Untimed busy warm-up before each measurement, long enough for the GPU to leave its idle P-state.
const WARMUP: std::time::Duration = std::time::Duration::from_millis(500);

/// `(label, n, c_in, h, w, c_out, kernel, pad, stride, dilation, groups)`.
type Shape = (
    &'static str,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
);

/// Repeat runs per (shape, dtype); the table reports min and median over them (SA-49 item 5).
const REPEATS: usize = 3;

const SHAPES: [Shape; 13] = [
    ("3x3 s1 64ch 56x56", 8, 64, 56, 56, 64, 3, 1, 1, 1, 1),
    ("3x3 s1 128ch 28x28", 8, 128, 28, 28, 128, 3, 1, 1, 1, 1),
    ("3x3 s1 256ch 14x14", 8, 256, 14, 14, 256, 3, 1, 1, 1, 1),
    ("1x1 s1 256->64 56x56", 8, 256, 56, 56, 64, 1, 0, 1, 1, 1),
    ("7x7 s2 3->64 224x224", 8, 3, 224, 224, 64, 7, 3, 2, 1, 1),
    ("1x1 s1 64->256 56x56", 8, 64, 56, 56, 256, 1, 0, 1, 1, 1),
    ("1x1 s1 512->128 28x28", 8, 512, 28, 28, 128, 1, 0, 1, 1, 1),
    (
        "1x1 s1 1024->256 14x14",
        8,
        1024,
        14,
        14,
        256,
        1,
        0,
        1,
        1,
        1,
    ),
    ("1x1 s2 256->512 56x56", 8, 256, 56, 56, 512, 1, 0, 2, 1, 1),
    (
        "3x3 s1 g32 256ch 14x14",
        8,
        256,
        14,
        14,
        256,
        3,
        1,
        1,
        1,
        32,
    ),
    (
        "3x3 s1 depthwise 128ch 28x28",
        8,
        128,
        28,
        28,
        128,
        3,
        1,
        1,
        1,
        128,
    ),
    ("3x3 s1 d2 128ch 28x28", 8, 128, 28, 28, 128, 3, 2, 1, 2, 1),
    ("3x3 s1 d4 64ch 56x56", 8, 64, 56, 56, 64, 3, 4, 1, 4, 1),
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
    let &(_, n, c_in, h, w, c_out, k, pad, stride, dil, groups) = s;
    let desc = Conv2dDescriptor::new(n, c_in, h, w, c_out, k, k, kind)
        .with_padding(pad, pad)
        .with_stride(stride, stride)
        .with_dilation(dil, dil)
        .with_groups(groups);
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
    let &(_, n, c_in, h, w, c_out, k, _, _, _, groups) = s;
    let p = plan::<T>(stream, s, picker, cfg.kind);
    let (h_out, w_out) = p.output_dims();
    let (xs, ws, ys) = (
        [n, c_in, h, w],
        [c_out, c_in / groups, k, k],
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
    // Busy warm-up: the host side of this test (debug-build data fill, upload, download) leaves the GPU idle long
    // enough between runs to drop to a low P-state, and a 20-iteration timed loop is too short to ramp the clocks back.
    let warm = Instant::now();
    while warm.elapsed() < WARMUP {
        once();
        stream.synchronize().expect("warm-up sync");
    }
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

/// Min and median of `v` (sorted in place).
fn min_median(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[0], v[v.len() / 2])
}

/// Prints one table for dtype `name` and asserts the picker agrees with the pinned output.
/// Each shape runs `REPEATS` times per mode; times shown are min/median over the repeats.
fn measure<T: Element + 'static>(ctx: &Context, stream: &Stream, name: &str, cfg: &Cfg<T>) {
    eprintln!(
        "== {name}: shape | pinned algo ms min/med | picker algo ms min/med | speedup (med) | rel diff"
    );
    for s in &SHAPES {
        let (mut pin, mut pick) = (Vec::new(), Vec::new());
        let (mut diff, mut algos) = (0.0f32, (None, None));
        for _ in 0..REPEATS {
            let (pinned, pinned_algo, pinned_ms) = run(ctx, stream, s, false, cfg);
            let (picked, picked_algo, picked_ms) = run(ctx, stream, s, true, cfg);
            diff = diff.max(rel_diff(&pinned, &picked));
            algos = (pinned_algo, picked_algo);
            pin.push(pinned_ms);
            pick.push(picked_ms);
        }
        let ((pin_min, pin_med), (pick_min, pick_med)) =
            (min_median(&mut pin), min_median(&mut pick));
        eprintln!(
            "{name} {} | {:?} {pin_min:.3}/{pin_med:.3} | {:?} {pick_min:.3}/{pick_med:.3} | {:.2}x | {diff:.2e}",
            s.0,
            algos.0,
            algos.1,
            pin_med / pick_med,
        );
        assert_eq!(algos.0, Some(0), "default must stay IMPLICIT_GEMM");
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
