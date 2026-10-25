//! Real-GPU smoke test for `PadBackwardPlan<T, N>` in the non-constant
//! modes (Reflect / Replicate / Circular), ROADMAP SA-30. The backward is
//! the adjoint of the forward index map, so the CPU reference scatter-adds
//! every `dy` cell into the `dx` cell the forward read it from.
//!
//! `dy` holds small integers, so every partial sum is exact in f16 / bf16
//! too and the comparison is bit-exact whatever the summation order.
//!
//! Gated behind the `device-tests` feature; run with
//! `cargo gpu-test -p baracuda-kernels --features sm89,device-tests \
//!   --test pad_backward_modes_smoke`.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PadBackwardArgs, PadBackwardDescriptor, PadBackwardPlan, PadMode, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// The forward kernels' per-axis map (`pad_{reflect,replicate,circular}_kernel`).
fn fwd_axis(mode: PadMode, in_coord: i64, extent: i64) -> i64 {
    match mode {
        PadMode::Reflect => {
            let mut c = in_coord;
            if extent <= 1 {
                c = 0;
            } else if c < 0 {
                c = -c;
            } else if c >= extent {
                c = 2 * extent - 2 - c;
            }
            c.clamp(0, extent - 1)
        }
        PadMode::Replicate => in_coord.clamp(0, extent - 1),
        _ => in_coord.rem_euclid(extent),
    }
}

/// CPU adjoint: `dx[fwd(c - pad_low)] += dy[c]` over every `dy` cell.
fn cpu_adjoint<const N: usize>(
    mode: PadMode,
    dy: &[i32],
    in_shape: [i32; N],
    pad_low: [i32; N],
    dy_shape: [i32; N],
) -> Vec<i32> {
    let in_stride = contiguous_stride(in_shape);
    let in_numel: usize = in_shape.iter().map(|&d| d as usize).product();
    let mut dx = vec![0i32; in_numel];
    for (linear, &v) in dy.iter().enumerate() {
        let mut rem = linear as i64;
        let mut dst = 0i64;
        for d in (0..N).rev() {
            let c = rem % dy_shape[d] as i64;
            rem /= dy_shape[d] as i64;
            let src = fwd_axis(mode, c - pad_low[d] as i64, in_shape[d] as i64);
            dst += src * in_stride[d];
        }
        dx[dst as usize] += v;
    }
    dx
}

fn dy_shape_of<const N: usize>(s: [i32; N], lo: [i32; N], hi: [i32; N]) -> [i32; N] {
    let mut out = [0i32; N];
    for d in 0..N {
        out[d] = s[d] + lo[d] + hi[d];
    }
    out
}

/// Small-integer dy values, `(i * 5 + 3) % 7 - 3`, in `[-3, 3]`.
fn int_dy(numel: usize) -> Vec<i32> {
    (0..numel).map(|i| ((i * 5 + 3) % 7) as i32 - 3).collect()
}

macro_rules! modes_case {
    ($name:ident, $ty:ty, $kind:expr, $from:expr, $to:expr, $n:literal, $mode:expr,
     $shape:expr, $lo:expr, $hi:expr) => {
        #[test]
        fn $name() {
            let (ctx, stream) = setup();
            let (in_shape, pad_low, pad_high): ([i32; $n], [i32; $n], [i32; $n]) =
                ($shape, $lo, $hi);
            let dy_shape = dy_shape_of(in_shape, pad_low, pad_high);
            let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
            let dx_numel: usize = in_shape.iter().map(|&d| d as usize).product();
            let host_dy = int_dy(dy_numel);
            let expected = cpu_adjoint($mode, &host_dy, in_shape, pad_low, dy_shape);

            let dy_t: Vec<$ty> = host_dy.iter().map(|&v| $from(v)).collect();
            let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_t).expect("upload");
            let mut dev_dx: DeviceBuffer<$ty> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
            let desc = PadBackwardDescriptor {
                mode: $mode,
                input_shape: in_shape,
                pad_low,
                pad_high,
                element: $kind,
            };
            let plan =
                PadBackwardPlan::<$ty, $n>::select(&stream, &desc, PlanPreference::default())
                    .expect("select");
            let args = PadBackwardArgs::<$ty, $n> {
                dy: TensorRef::new(dev_dy.as_slice(), dy_shape, contiguous_stride(dy_shape))
                    .unwrap(),
                dx: TensorMut::new(dev_dx.as_slice_mut(), in_shape, contiguous_stride(in_shape))
                    .unwrap(),
            };
            plan.run(&stream, Workspace::None, args).expect("run");
            stream.synchronize().expect("sync");

            let mut got: Vec<$ty> = dy_t[..1].repeat(dx_numel);
            dev_dx.copy_to_host(&mut got).expect("download");
            let got: Vec<i32> = got.iter().map(|&v| $to(v)).collect();
            assert_eq!(got, expected, "pad BW mismatch");
        }
    };
}

fn f16_from(v: i32) -> f16 {
    f16::from_f32(v as f32)
}
fn f16_to(v: f16) -> i32 {
    v.to_f32() as i32
}
fn bf16_from(v: i32) -> bf16 {
    bf16::from_f32(v as f32)
}
fn bf16_to(v: bf16) -> i32 {
    v.to_f32() as i32
}

modes_case!(
    reflect_f32_2d,
    f32,
    ElementKind::F32,
    |v| v as f32,
    |v: f32| v as i32,
    2,
    PadMode::Reflect,
    [5, 6],
    [2, 1],
    [3, 2]
);
modes_case!(
    replicate_f32_2d,
    f32,
    ElementKind::F32,
    |v| v as f32,
    |v: f32| v as i32,
    2,
    PadMode::Replicate,
    [3, 4],
    [2, 3],
    [1, 4]
);
modes_case!(
    circular_f32_2d,
    f32,
    ElementKind::F32,
    |v| v as f32,
    |v: f32| v as i32,
    2,
    PadMode::Circular,
    [3, 4],
    [2, 1],
    [3, 5]
);
// Circular pad wider than the extent wraps more than once.
modes_case!(
    circular_f64_1d_multi_wrap,
    f64,
    ElementKind::F64,
    |v| v as f64,
    |v: f64| v as i32,
    1,
    PadMode::Circular,
    [3],
    [7],
    [8]
);
modes_case!(
    replicate_f64_3d,
    f64,
    ElementKind::F64,
    |v| v as f64,
    |v: f64| v as i32,
    3,
    PadMode::Replicate,
    [2, 3, 4],
    [1, 0, 2],
    [0, 2, 1]
);
modes_case!(
    reflect_f16_2d,
    f16,
    ElementKind::F16,
    f16_from,
    f16_to,
    2,
    PadMode::Reflect,
    [4, 5],
    [3, 2],
    [1, 3]
);
modes_case!(
    circular_bf16_2d,
    bf16,
    ElementKind::Bf16,
    bf16_from,
    bf16_to,
    2,
    PadMode::Circular,
    [3, 5],
    [1, 4],
    [2, 2]
);
modes_case!(
    replicate_bf16_3d,
    bf16,
    ElementKind::Bf16,
    bf16_from,
    bf16_to,
    3,
    PadMode::Replicate,
    [3, 2, 4],
    [2, 1, 0],
    [1, 2, 3]
);
// Extent 1: every pad cell reads element 0, so dx sums the whole axis.
modes_case!(
    reflect_f32_extent_one,
    f32,
    ElementKind::F32,
    |v| v as f32,
    |v: f32| v as i32,
    2,
    PadMode::Reflect,
    [1, 4],
    [3, 1],
    [2, 2]
);
modes_case!(
    replicate_f32_extent_one,
    f32,
    ElementKind::F32,
    |v| v as f32,
    |v: f32| v as i32,
    2,
    PadMode::Replicate,
    [4, 1],
    [1, 3],
    [2, 2]
);
// Zero padding is the identity in every mode.
modes_case!(
    reflect_f32_zero_pad,
    f32,
    ElementKind::F32,
    |v| v as f32,
    |v: f32| v as i32,
    2,
    PadMode::Reflect,
    [3, 4],
    [0, 0],
    [0, 0]
);

// Reflect beyond one reflection (pad > extent - 1) is declined, not mis-computed.
#[test]
fn reflect_over_pad_is_declined() {
    let (_ctx, stream) = setup();
    let desc = PadBackwardDescriptor {
        mode: PadMode::Reflect,
        input_shape: [4i32, 4],
        pad_low: [1i32, 4],
        pad_high: [1i32, 1],
        element: ElementKind::F32,
    };
    let res = PadBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default());
    assert!(
        res.is_err(),
        "Reflect with pad 4 on extent 4 must be declined"
    );
}
