//! Output-rounding-floor check for the f16 / bf16 dense GEMM at large `k`.
//!
//! cuBLAS's default math mode may split K across thread blocks and reduce the partials in the
//! output dtype, which loses the f32 accumulation (measured on an RTX 4070 Laptop, cuBLAS 13.5.1:
//! f16 4.1e-4 vs a 3.2e-4 floor at m32 n1024 k4096, bf16 3.4e-3 vs 1.8e-3 at m64 k8192). The facade
//! therefore sets `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` on its half entry points.
//!
//! This test is the repeatable form of that probe: against an f64 reference of the same half inputs
//! (uniform [-1, 1)) the max absolute error over `max|ref|` must stay within 5% of the error from
//! merely rounding the exact result to the output dtype. A last arm turns the opt-in on and prints
//! what the fast path does on this device (printed, not asserted: it depends on cuBLAS's
//! heuristics, and its job is to show the check can see the difference).
//!
//! One `#[test]` on purpose: the opt-in is process-wide, so the arms must run in order.
//!
//! Run with `cargo gpu-test -p baracuda-kernels --features sm89,device-tests --test dense_gemm_half_floor`.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    DenseGemmArgs, DenseGemmDescriptor, DenseGemmLayout, DenseGemmPlan, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
use half::{bf16, f16};

/// (m, n, k): the first and last cells are where the default mode was measured off the floor.
const SHAPES: [(usize, usize, usize); 3] = [(32, 512, 4096), (64, 512, 8192), (1, 512, 4096)];

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    }
}

/// A half storage dtype: conversion in and out of f32.
trait Half: baracuda_kernels::Element + Copy + Default + 'static {
    fn make(v: f32) -> Self;
    fn value(self) -> f32;
}

impl Half for f16 {
    fn make(v: f32) -> Self {
        f16::from_f32(v)
    }
    fn value(self) -> f32 {
        self.to_f32()
    }
}

impl Half for bf16 {
    fn make(v: f32) -> Self {
        bf16::from_f32(v)
    }
    fn value(self) -> f32 {
        self.to_f32()
    }
}

/// Half inputs plus the f64 reference of their product and its rounding floor.
struct Problem<T> {
    a: Vec<T>,
    b: Vec<T>,
    want: Vec<f64>,
    max_ref: f64,
    floor: f64,
}

fn problem<T: Half>((m, n, k): (usize, usize, usize)) -> Problem<T> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15 ^ (m * 31 + n * 7 + k) as u64);
    let a: Vec<T> = (0..m * k).map(|_| T::make(rng.next())).collect();
    let b: Vec<T> = (0..k * n).map(|_| T::make(rng.next())).collect();
    let af: Vec<f64> = a.iter().map(|&v| v.value() as f64).collect();
    let bf: Vec<f64> = b.iter().map(|&v| v.value() as f64).collect();
    let mut want = vec![0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = af[i * k + p];
            for j in 0..n {
                want[i * n + j] += av * bf[p * n + j];
            }
        }
    }
    let max_ref = want.iter().fold(0f64, |x, &v| x.max(v.abs()));
    let floor = want
        .iter()
        .map(|&v| (T::make(v as f32).value() as f64 - v).abs())
        .fold(0f64, f64::max)
        / max_ref;
    Problem {
        a,
        b,
        want,
        max_ref,
        floor,
    }
}

/// Run the facade-backed plan on the problem's inputs and download D.
fn run_plan<T: Half>(
    ctx: &Context,
    stream: &Stream,
    (m, n, k): (usize, usize, usize),
    p: &Problem<T>,
) -> Vec<T> {
    use baracuda_kernels::ScalarType;

    let dev_a = DeviceBuffer::from_slice(ctx, &p.a).expect("upload A");
    let dev_b = DeviceBuffer::from_slice(ctx, &p.b).expect("upload B");
    let mut dev_d: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, m * n).expect("alloc D");
    let desc = DenseGemmDescriptor {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        batch: 1,
        layout: DenseGemmLayout::Rrr,
    };
    let plan = DenseGemmPlan::<T>::select(stream, &desc, PlanPreference::default())
        .expect("select dense half plan");
    let args = DenseGemmArgs::<T> {
        a: MatrixRef::new(dev_a.as_slice(), m as i32, k as i32, k as i64).unwrap(),
        stride_a: 0,
        b: MatrixRef::new(dev_b.as_slice(), k as i32, n as i32, n as i64).unwrap(),
        stride_b: 0,
        d: MatrixMut::new(dev_d.as_slice_mut(), m as i32, n as i32, n as i64).unwrap(),
        stride_d: 0,
        alpha: <T::Scalar as ScalarType>::ONE,
        beta: <T::Scalar as ScalarType>::ZERO,
    };
    plan.run(stream, Workspace::None, args)
        .expect("dense half run");
    stream.synchronize().expect("stream sync");
    let mut got = vec![T::default(); m * n];
    dev_d.copy_to_host(&mut got).expect("download D");
    got
}

/// `(max abs err / max|ref|, rounding floor of the same measure)` for one shape.
fn measure<T: Half>(ctx: &Context, stream: &Stream, shape: (usize, usize, usize)) -> (f64, f64) {
    let p = problem::<T>(shape);
    let got = run_plan(ctx, stream, shape, &p);
    let err = got
        .iter()
        .zip(&p.want)
        .map(|(&g, &w)| (g.value() as f64 - w).abs())
        .fold(0f64, f64::max)
        / p.max_ref;
    (err, p.floor)
}

fn sweep(ctx: &Context, stream: &Stream, label: &str) -> Vec<(String, f64, f64)> {
    let mut out = Vec::new();
    for shape in SHAPES {
        let (e, f) = measure::<f16>(ctx, stream, shape);
        out.push((format!("f16 {label} {shape:?}"), e, f));
        let (e, f) = measure::<bf16>(ctx, stream, shape);
        out.push((format!("bf16 {label} {shape:?}"), e, f));
    }
    out
}
#[test]
fn half_gemm_large_k_reaches_output_rounding_floor() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_allow_reduced_precision_reduction(),
        0,
        "the accurate path must be the default"
    );
    for (what, err, floor) in sweep(&ctx, &stream, "accurate") {
        println!("{what}: err {err:.3e} floor {floor:.3e}");
        assert!(
            err <= floor * 1.05,
            "{what}: err {err:.3e} is above the output-rounding floor {floor:.3e}"
        );
    }

    baracuda_kernels_sys::baracuda_kernels_gemm_dense_set_allow_reduced_precision_reduction(1);
    assert_eq!(
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_allow_reduced_precision_reduction(),
        1
    );
    for (what, err, floor) in sweep(&ctx, &stream, "opt-in fast") {
        println!(
            "{what}: err {err:.3e} floor {floor:.3e} ratio {:.2}",
            err / floor
        );
    }
    baracuda_kernels_sys::baracuda_kernels_gemm_dense_set_allow_reduced_precision_reduction(0);
}
