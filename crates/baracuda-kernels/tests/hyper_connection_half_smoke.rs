//! Real-GPU test for `HyperConnectionPlan<f16 / bf16>` (SA-31 Tier 3).
//!
//! The half-precision plan widens `x_expanded` to f32, runs the vendored f32 path
//! unchanged, and narrows the result. So its outputs (forward and every gradient)
//! must equal the f32 plan's outputs on the widened inputs, rounded once to `T`.
//! The assertions are exact (bit equality), not a tolerance: the host `half`
//! rounding and the device cast are both round-to-nearest-even. The f32 plan is run
//! in the same test, and `HC_F32_FNV` (an FNV-1a hash of its output bits) is printed
//! so the f32 path can be compared before and after the change.
//!
//! Gated behind `device-tests` and `mhc`; needs a real CUDA device.

#![cfg(feature = "mhc")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    Element, ElementKind, HyperConnectionArgs, HyperConnectionBackwardArgs,
    HyperConnectionDescriptor, HyperConnectionPlan, PlanPreference, TensorMut, TensorRef,
    Workspace, contiguous_stride,
};
use half::{bf16, f16};

const B: i32 = 3;
const N: i32 = 4;
const C: i32 = 32;

/// Host-side conversion for the three wired activation types.
trait Act: Element + Copy + 'static {
    const ELEM: ElementKind;
    fn of(v: f32) -> Self;
    fn widen(self) -> f32;
}
macro_rules! act_impl {
    ($t:ty, $kind:expr, $of:expr, $widen:expr) => {
        impl Act for $t {
            const ELEM: ElementKind = $kind;
            fn of(v: f32) -> Self {
                $of(v)
            }
            fn widen(self) -> f32 {
                $widen(self)
            }
        }
    };
}
act_impl!(f32, ElementKind::F32, |v| v, |v| v);
act_impl!(f16, ElementKind::F16, f16::from_f32, f16::to_f32);
act_impl!(bf16, ElementKind::Bf16, bf16::from_f32, bf16::to_f32);

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn desc(element: ElementKind) -> HyperConnectionDescriptor {
    HyperConnectionDescriptor {
        batch: B,
        hidden_dim: C,
        n_streams: N,
        sinkhorn_iters: 20,
        eps: 1e-5,
        element,
    }
}

fn wave(len: usize, scale: f32, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.017 + phase).sin() * scale)
        .collect()
}

/// Inputs shared by every run. `x` and `d_out` are rounded to `T` first, so the f32
/// reference sees exactly the values the half plan widens.
struct Inputs {
    x: Vec<f32>,
    d_out: Vec<f32>,
    gamma: Vec<bf16>,
    h_pre: Vec<f32>,
    h_post: Vec<f32>,
    h_res: Vec<f32>,
}

fn inputs<T: Act>() -> Inputs {
    let round = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|a| T::of(a).widen()).collect() };
    Inputs {
        x: round(wave((B * N * C) as usize, 0.4, -0.5)),
        d_out: round(wave((B * N * C) as usize, 0.3, 0.7)),
        gamma: (0..C as usize)
            .map(|i| bf16::from_f32(0.9 + 0.2 * ((i as f32) * 0.31).sin()))
            .collect(),
        h_pre: wave(N as usize, 0.2, 0.1),
        h_post: wave(N as usize, 0.3, 0.4),
        h_res: (0..(N * N) as usize)
            .map(|i| if i % (N as usize + 1) == 0 { 0.05 } else { 0.0 })
            .collect(),
    }
}

/// One forward + backward, every output widened to f32.
struct Outputs {
    out: Vec<f32>,
    d_x: Vec<f32>,
    d_gamma: Vec<f32>,
    d_h_pre: Vec<f32>,
    d_h_post: Vec<f32>,
    d_h_res: Vec<f32>,
}

impl Outputs {
    /// Download the forward output, `d_x_expanded`, and the four small gradients
    /// (gamma, `h_pre`, `h_post`, `h_res`, in that order).
    fn collect<T: Act>(
        out: &DeviceBuffer<T>,
        gx: &DeviceBuffer<T>,
        small: [&DeviceBuffer<T>; 4],
    ) -> Self {
        let e = (B * N * C) as usize;
        Self {
            out: download(out, e),
            d_x: download(gx, e),
            d_gamma: download(small[0], C as usize),
            d_h_pre: download(small[1], N as usize),
            d_h_post: download(small[2], N as usize),
            d_h_res: download(small[3], (N * N) as usize),
        }
    }
}

fn upload<T: Act>(ctx: &Context, v: &[f32]) -> DeviceBuffer<T> {
    let t: Vec<T> = v.iter().map(|&a| T::of(a)).collect();
    DeviceBuffer::from_slice(ctx, &t).expect("upload")
}

fn download<T: Act>(b: &DeviceBuffer<T>, n: usize) -> Vec<f32> {
    let mut h = vec![T::of(0.0); n];
    b.copy_to_host(&mut h).expect("download");
    h.into_iter().map(Act::widen).collect()
}

fn run_plan<T: Act>(ctx: &Context, stream: &Stream, i: &Inputs) -> Outputs {
    let plan = HyperConnectionPlan::<T>::select_training(
        stream,
        &desc(T::ELEM),
        PlanPreference::default(),
    )
    .expect("select_training");
    let (sx, sg, sh, sr) = ([B, N, C], [C], [N], [N, N]);
    let (dx, dd) = (upload::<T>(ctx, &i.x), upload::<T>(ctx, &i.d_out));
    let dgamma = DeviceBuffer::from_slice(ctx, &i.gamma).expect("gamma");
    let dh_pre = DeviceBuffer::from_slice(ctx, &i.h_pre).expect("h_pre");
    let dh_post = DeviceBuffer::from_slice(ctx, &i.h_post).expect("h_post");
    let dh_res = DeviceBuffer::from_slice(ctx, &i.h_res).expect("h_res");
    let zeros = |n: usize| -> DeviceBuffer<T> { DeviceBuffer::zeros(ctx, n).expect("zeros") };
    let (mut out, mut gx) = (zeros((B * N * C) as usize), zeros((B * N * C) as usize));
    let (mut gg, mut gpre) = (zeros(C as usize), zeros(N as usize));
    let (mut gpost, mut gres) = (zeros(N as usize), zeros((N * N) as usize));
    plan.run(
        stream,
        Workspace::None,
        HyperConnectionArgs {
            x_expanded: TensorRef::new(dx.as_slice(), sx, contiguous_stride(sx)).unwrap(),
            rmsnorm_weight: TensorRef::new(dgamma.as_slice(), sg, contiguous_stride(sg)).unwrap(),
            h_pre: TensorRef::new(dh_pre.as_slice(), sh, contiguous_stride(sh)).unwrap(),
            h_post: TensorRef::new(dh_post.as_slice(), sh, contiguous_stride(sh)).unwrap(),
            h_res: TensorRef::new(dh_res.as_slice(), sr, contiguous_stride(sr)).unwrap(),
            out: TensorMut::new(out.as_slice_mut(), sx, contiguous_stride(sx)).unwrap(),
        },
    )
    .expect("forward");
    plan.run_backward(
        stream,
        HyperConnectionBackwardArgs {
            d_out: TensorRef::new(dd.as_slice(), sx, contiguous_stride(sx)).unwrap(),
            d_x_expanded: TensorMut::new(gx.as_slice_mut(), sx, contiguous_stride(sx)).unwrap(),
            d_rmsnorm_weight: TensorMut::new(gg.as_slice_mut(), sg, contiguous_stride(sg)).unwrap(),
            d_h_pre: TensorMut::new(gpre.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            d_h_post: TensorMut::new(gpost.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            d_h_res: TensorMut::new(gres.as_slice_mut(), sr, contiguous_stride(sr)).unwrap(),
        },
    )
    .expect("backward");
    stream.synchronize().expect("sync");
    Outputs::collect(&out, &gx, [&gg, &gpre, &gpost, &gres])
}

fn fnv(o: &Outputs) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in [
        &o.out,
        &o.d_x,
        &o.d_gamma,
        &o.d_h_pre,
        &o.d_h_post,
        &o.d_h_res,
    ] {
        for x in v.iter() {
            for byte in x.to_bits().to_le_bytes() {
                h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}

/// `got` must be `reference` rounded to `T`, bit for bit. Also requires the data to
/// be non-trivial, so an all-zero output cannot pass.
fn assert_rounded<T: Act>(name: &str, got: &[f32], reference: &[f32]) {
    assert_eq!(got.len(), reference.len(), "{name} length");
    assert!(
        reference.iter().any(|v| v.abs() > 1e-6),
        "{name}: reference is all ~zero, the comparison would be vacuous"
    );
    for (k, (g, r)) in got.iter().zip(reference).enumerate() {
        let want = T::of(*r).widen();
        assert_eq!(
            g.to_bits(),
            want.to_bits(),
            "{name}[{k}]: got {g}, want {want} (f32 ref {r})"
        );
    }
}

fn check<T: Act>() {
    let (ctx, stream) = setup();
    let ins = inputs::<T>();
    let reference = run_plan::<f32>(&ctx, &stream, &ins);
    // Control: the f32 plan is deterministic, so the equalities below are not noise.
    let again = run_plan::<f32>(&ctx, &stream, &ins);
    assert_eq!(fnv(&reference), fnv(&again), "f32 path must be bit-stable");
    println!("HC_F32_FNV={:016x}", fnv(&reference));
    let got = run_plan::<T>(&ctx, &stream, &ins);
    assert_rounded::<T>("out", &got.out, &reference.out);
    assert_rounded::<T>("d_x_expanded", &got.d_x, &reference.d_x);
    assert_rounded::<T>("d_rmsnorm_weight", &got.d_gamma, &reference.d_gamma);
    assert_rounded::<T>("d_h_pre", &got.d_h_pre, &reference.d_h_pre);
    assert_rounded::<T>("d_h_post", &got.d_h_post, &reference.d_h_post);
    assert_rounded::<T>("d_h_res", &got.d_h_res, &reference.d_h_res);
}

#[test]
fn hyper_connection_f16_matches_f32_path_on_widened_inputs() {
    check::<f16>();
}

#[test]
fn hyper_connection_bf16_matches_f32_path_on_widened_inputs() {
    check::<bf16>();
}

#[test]
fn hyper_connection_half_forward_only_plan_has_no_backward() {
    let (ctx, stream) = setup();
    let plan = HyperConnectionPlan::<f16>::select(
        &stream,
        &desc(ElementKind::F16),
        PlanPreference::default(),
    )
    .expect("select");
    let ins = inputs::<f16>();
    let (sx, sg, sh, sr) = ([B, N, C], [C], [N], [N, N]);
    let dd = upload::<f16>(&ctx, &ins.d_out);
    let mut gx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * N * C) as usize).unwrap();
    let mut gg: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, C as usize).unwrap();
    let mut gpre: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, N as usize).unwrap();
    let mut gpost: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, N as usize).unwrap();
    let mut gres: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (N * N) as usize).unwrap();
    let err = plan.run_backward(
        &stream,
        HyperConnectionBackwardArgs {
            d_out: TensorRef::new(dd.as_slice(), sx, contiguous_stride(sx)).unwrap(),
            d_x_expanded: TensorMut::new(gx.as_slice_mut(), sx, contiguous_stride(sx)).unwrap(),
            d_rmsnorm_weight: TensorMut::new(gg.as_slice_mut(), sg, contiguous_stride(sg)).unwrap(),
            d_h_pre: TensorMut::new(gpre.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            d_h_post: TensorMut::new(gpost.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            d_h_res: TensorMut::new(gres.as_slice_mut(), sr, contiguous_stride(sr)).unwrap(),
        },
    );
    assert!(err.is_err(), "a forward-only plan must refuse run_backward");
}
