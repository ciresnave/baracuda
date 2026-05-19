//! Real-GPU smoke test for `UnaryParamPlan + UnaryKind::PowI` (Phase 12.1).
//!
//! Forward: `y = x^n` for fixed integer `n` (power-by-squaring).
//! Backward: `dx = n · x^(n-1) · dy`, with special cases for `n = 0`
//! (`dx = 0`) and `n = 1` (`dx = dy`).
//!
//! FW coverage:
//!   * f32 with `n ∈ {0, 1, 2, 3, -1}` against host-side `f32::powi`.
//!   * f16 / bf16 / f64 spot-check at `n = 2` to verify all dtypes wired.
//!
//! BW coverage (f32):
//!   * `n = 2` — verify `dx = 2 · x · dy` analytically.
//!   * `n = 0` — verify `dx == 0`.
//!   * `n = 1` — verify `dx == dy`.
//!
//! `#[ignore]` by default; run with `--ignored` on a CUDA host.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryKind,
    UnaryParamArgs, UnaryParamBackwardArgs, UnaryParamBackwardDescriptor,
    UnaryParamBackwardPlan, UnaryParamDescriptor, UnaryParamPlan, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Input range that exercises negatives (PowI handles them correctly,
// unlike float-exponent Pow) but avoids exact zero (we test `n = -1`).
fn make_x_f32(numel: usize) -> Vec<f32> {
    (0..numel)
        .map(|i| {
            // Avoid 0 by offsetting; range roughly [-2.55, +2.45].
            let raw = (i as f32) * 0.005 - 2.55;
            if raw == 0.0 { 0.001 } else { raw }
        })
        .collect()
}

fn run_powi_f32(shape: [i32; 3], n: i32) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x = make_x_f32(numel);
    let host_expected: Vec<f32> = host_x.iter().map(|&x| x.powi(n)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape,
        element: ElementKind::F32,
        params: [n as f32, 0.0],
    };
    let plan = UnaryParamPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f32, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    // Power-by-squaring vs `f32::powi`: both reduce to the same chain
    // of multiplies for the same exponent. Tolerate one ULP of rounding
    // drift just in case the host implementation rearranges associations.
    let mut max_err = 0.0f32;
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        let denom = e.abs().max(1e-6);
        let rel = (g - e).abs() / denom;
        if rel > max_err { max_err = rel; }
        assert!(
            rel < 1e-5 || (g.is_nan() && e.is_nan()),
            "powi(n={n}) f32 @ {i}: x={}, got {g}, exp {e}, rel {rel}",
            host_x[i]
        );
    }
}

#[test]
#[ignore]
fn powi_f32_n0() { run_powi_f32([2, 16, 32], 0); }

#[test]
#[ignore]
fn powi_f32_n1() { run_powi_f32([2, 16, 32], 1); }

#[test]
#[ignore]
fn powi_f32_n2() { run_powi_f32([2, 16, 32], 2); }

#[test]
#[ignore]
fn powi_f32_n3() { run_powi_f32([2, 16, 32], 3); }

#[test]
#[ignore]
fn powi_f32_n_neg1() { run_powi_f32([2, 16, 32], -1); }

#[test]
#[ignore]
fn powi_f16_n2() {
    let (ctx, stream) = setup();
    let shape = [2i32, 16, 32];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = make_x_f32(numel).iter().map(|&v| f16::from_f32(v)).collect();
    // Host ref: do the math at f32, then round to f16 — matches the
    // kernel's f16→f32→pow→f16 convention.
    let host_expected: Vec<f16> = host_x
        .iter()
        .map(|x| f16::from_f32(x.to_f32().powi(2)))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape,
        element: ElementKind::F16,
        params: [2.0, 0.0],
    };
    let plan = UnaryParamPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        // Allow ±1 ULP for f16 rounding; the f32 product chain order is
        // not guaranteed to match the host one bit for bit.
        let gb = g.to_bits() as i32;
        let eb = e.to_bits() as i32;
        let diff = (gb - eb).abs();
        assert!(
            diff <= 1 || (g.to_f32().is_nan() && e.to_f32().is_nan()),
            "powi(n=2) f16 @ {i}: x={}, got {} (bits {:#x}), exp {} (bits {:#x})",
            host_x[i].to_f32(), g.to_f32(), g.to_bits(), e.to_f32(), e.to_bits()
        );
    }
}

#[test]
#[ignore]
fn powi_bf16_n2() {
    let (ctx, stream) = setup();
    let shape = [2i32, 16, 32];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = make_x_f32(numel).iter().map(|&v| bf16::from_f32(v)).collect();
    let host_expected: Vec<bf16> = host_x
        .iter()
        .map(|x| bf16::from_f32(x.to_f32().powi(2)))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape,
        element: ElementKind::Bf16,
        params: [2.0, 0.0],
    };
    let plan = UnaryParamPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<bf16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        let gb = g.to_bits() as i32;
        let eb = e.to_bits() as i32;
        let diff = (gb - eb).abs();
        assert!(
            diff <= 1 || (g.to_f32().is_nan() && e.to_f32().is_nan()),
            "powi(n=2) bf16 @ {i}: x={}, got {} (bits {:#x}), exp {} (bits {:#x})",
            host_x[i].to_f32(), g.to_f32(), g.to_bits(), e.to_f32(), e.to_bits()
        );
    }
}

#[test]
#[ignore]
fn powi_f64_n2() {
    let (ctx, stream) = setup();
    let shape = [2i32, 16, 32];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = (i as f64) * 0.005 - 2.55;
            if raw == 0.0 { 0.001 } else { raw }
        })
        .collect();
    // The kernel's power-by-squaring loop for n=2 collapses to a
    // single multiply, so the host reference is `x*x` directly — not
    // `x.powi(2)`, which on some platforms routes through libm's
    // `pow(x, 2.0)` and can differ by ≤ 1 ulp.
    let host_expected: Vec<f64> = host_x.iter().map(|&x| x * x).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape,
        element: ElementKind::F64,
        params: [2.0, 0.0],
    };
    let plan = UnaryParamPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f64, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        // f64 power-by-squaring with n=2 collapses to one multiply →
        // bit-exact against host `x.powi(2)` (which also reduces to
        // `x * x` in libm).
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "powi(n=2) f64 @ {i}: x={}, got {g}, exp {e}", host_x[i]
        );
    }
}

// =============================================================================
// Backward
// =============================================================================

fn run_powi_bw_f32(n: i32, expect: impl Fn(f32, f32) -> f32) {
    let (ctx, stream) = setup();
    let shape = [2i32, 16, 32];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x = make_x_f32(numel);
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 1.5).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamBackwardDescriptor {
        kind: UnaryKind::PowI,
        shape,
        element: ElementKind::F32,
        params: [n as f32, 0.0],
    };
    let plan = UnaryParamBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = expect(host_x[i], host_dy[i]);
        if n == 0 || n == 1 {
            // Special-cased branches: bit-exact (either 0 or pass-through dy).
            assert_eq!(
                got[i].to_bits(), exp.to_bits(),
                "powi-bw(n={n}) f32 @ {i}: x={}, dy={}, got {}, exp {}",
                host_x[i], host_dy[i], got[i], exp
            );
        } else {
            let denom = exp.abs().max(1e-6);
            let rel = (got[i] - exp).abs() / denom;
            assert!(
                rel < 1e-5,
                "powi-bw(n={n}) f32 @ {i}: x={}, dy={}, got {}, exp {}, rel {rel}",
                host_x[i], host_dy[i], got[i], exp
            );
        }
    }
}

#[test]
#[ignore]
fn powi_backward_f32_n2() {
    // dx = 2 · x · dy
    run_powi_bw_f32(2, |x, dy| 2.0 * x * dy);
}

#[test]
#[ignore]
fn powi_backward_f32_n0() {
    // dx = 0 (gradient of a constant)
    run_powi_bw_f32(0, |_x, _dy| 0.0);
}

#[test]
#[ignore]
fn powi_backward_f32_n1() {
    // dx = dy (identity gradient)
    run_powi_bw_f32(1, |_x, dy| dy);
}
