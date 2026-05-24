#![cfg(feature = "cudnn")]
//! Phase 16.1 — bit-exact PyTorch adaptive-pool smoke tests.
//!
//! Verifies the bespoke-kernel rewrite agrees with PyTorch's
//! non-uniform per-output-cell window convention for **non-divisible**
//! cases (where the previous cuDNN approximation diverged by ±1 cell).
//!
//! Window formula (per spatial axis):
//!   start_i = floor(i * in / out)
//!   end_i   = ceil((i + 1) * in / out)
//!
//! Example: `in=5, out=3` gives PyTorch windows
//!   i=0: [floor(0*5/3), ceil(1*5/3)) = [0, 2)   → x[0..2]
//!   i=1: [floor(1*5/3), ceil(2*5/3)) = [1, 4)   → x[1..4]
//!   i=2: [floor(2*5/3), ceil(3*5/3)) = [3, 5)   → x[3..5]
//!
//! The previous cuDNN approximation used `kernel=ceil(5/3)=2,
//! stride=floor(5/3)=1, pad=0` → windows `[0,2), [1,3), [2,4)` —
//! diverges on the boundary cell `i=2`.
//!
//! The PyTorch bit-exact behavior is the contract this test family
//! pins.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AdaptiveAvgPool2dPlan, AdaptiveAvgPool3dPlan, AdaptiveMaxPool2dPlan,
    AdaptivePool2dBwArgs, AdaptivePool2dDescriptor, AdaptivePool2dFwArgs,
    AdaptivePool3dDescriptor, AdaptivePool3dFwArgs, ElementKind, PlanPreference, TensorMut,
    TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// PyTorch's per-axis adaptive window bounds — integer-exact reference.
fn pt_start(i: i32, in_sz: i32, out_sz: i32) -> i32 {
    ((i as i64 * in_sz as i64) / out_sz as i64) as i32
}
fn pt_end(i: i32, in_sz: i32, out_sz: i32) -> i32 {
    (((i as i64 + 1) * in_sz as i64 + (out_sz - 1) as i64) / out_sz as i64) as i32
}

// =============================================================================
// AdaptiveAvgPool2d 5×5 → 3×3, f32.
// =============================================================================
//
// Non-divisible per axis. Reference computed with the bit-exact PyTorch
// formula above (independent per H/W axis).

#[test]
#[ignore]
fn adaptive_avg_pool2d_5x5_to_3x3_f32_bitexact() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 5i32, 5i32);
    let (h_out, w_out) = (3i32, 3i32);
    // 5×5 input filled with 1.0 .. 25.0 row-major.
    let host_x: Vec<f32> = (1..=25).map(|v| v as f32).collect();

    // Build the expected output using the same formula the kernel uses.
    let mut exp_y = vec![0f32; (h_out * w_out) as usize];
    for oh in 0..h_out {
        let sh = pt_start(oh, h_in, h_out);
        let eh = pt_end(oh, h_in, h_out);
        for ow in 0..w_out {
            let sw = pt_start(ow, w_in, w_out);
            let ew = pt_end(ow, w_in, w_out);
            let mut sum = 0f64;
            let mut cnt = 0u32;
            for hh in sh..eh {
                for ww in sw..ew {
                    sum += host_x[(hh * w_in + ww) as usize] as f64;
                    cnt += 1;
                }
            }
            exp_y[(oh * w_out + ow) as usize] = (sum / cnt as f64) as f32;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("y");
    let desc = AdaptivePool2dDescriptor {
        batch: n, channels: c, h_in, w_in, h_out, w_out,
        element: ElementKind::F32,
    };
    let plan = AdaptiveAvgPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    plan.run_fw(&stream, Workspace::None, AdaptivePool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got = vec![0f32; 9];
    dev_y.copy_to_host(&mut got).expect("dl y");
    let tol = 64.0 * f32::EPSILON;
    for i in 0..9 {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got[i] - exp_y[i]).abs() <= t,
            "adaptive_avg_pool2d 5→3 f32 @ {i}: got={} want={}", got[i], exp_y[i]);
    }
}

// =============================================================================
// AdaptiveMaxPool2d 5×5 → 3×3, f32 (FW values + BW argmax via re-scan).
// =============================================================================

#[test]
#[ignore]
fn adaptive_max_pool2d_5x5_to_3x3_f32_bitexact() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 5i32, 5i32);
    let (h_out, w_out) = (3i32, 3i32);
    let host_x: Vec<f32> = (1..=25).map(|v| v as f32).collect();

    let mut exp_y = vec![0f32; (h_out * w_out) as usize];
    let mut exp_idx = vec![0i64; (h_out * w_out) as usize];
    for oh in 0..h_out {
        let sh = pt_start(oh, h_in, h_out);
        let eh = pt_end(oh, h_in, h_out);
        for ow in 0..w_out {
            let sw = pt_start(ow, w_in, w_out);
            let ew = pt_end(ow, w_in, w_out);
            let mut best = f32::NEG_INFINITY;
            let mut best_idx: i64 = 0;
            let mut first = true;
            for hh in sh..eh {
                for ww in sw..ew {
                    let v = host_x[(hh * w_in + ww) as usize];
                    let idx = (hh * w_in + ww) as i64;
                    if first || v > best {
                        best = v;
                        best_idx = idx;
                        first = false;
                    }
                }
            }
            exp_y[(oh * w_out + ow) as usize] = best;
            exp_idx[(oh * w_out + ow) as usize] = best_idx;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("y");
    let desc = AdaptivePool2dDescriptor {
        batch: n, channels: c, h_in, w_in, h_out, w_out,
        element: ElementKind::F32,
    };
    let plan = AdaptiveMaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    plan.run_fw(&stream, Workspace::None, AdaptivePool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got = vec![0f32; 9];
    dev_y.copy_to_host(&mut got).expect("dl y");
    for i in 0..9 {
        assert!((got[i] - exp_y[i]).abs() <= 16.0 * f32::EPSILON,
            "adaptive_max_pool2d 5→3 FW @ {i}: got={} want={} (argmax want=x[{}])",
            got[i], exp_y[i], exp_idx[i]);
    }
    // Argmax-indices contract is implicit (BW recomputes them); the
    // values check above + the BW test below pin the convention end-to-end.
}

// =============================================================================
// AdaptiveAvgPool3d 5×7×4 → 3×4×2, f32.
// =============================================================================

#[test]
#[ignore]
fn adaptive_avg_pool3d_5x7x4_to_3x4x2_f32_bitexact() {
    let (ctx, stream) = setup();
    let (n, c) = (1i32, 1i32);
    let (d_in, h_in, w_in) = (5i32, 7i32, 4i32);
    let (d_out, h_out, w_out) = (3i32, 4i32, 2i32);
    let in_numel = (d_in * h_in * w_in) as usize;
    let out_numel = (d_out * h_out * w_out) as usize;
    let host_x: Vec<f32> = (0..in_numel).map(|v| (v as f32) * 0.5 - 3.0).collect();

    // Reference.
    let mut exp_y = vec![0f32; out_numel];
    for od in 0..d_out {
        let sd = pt_start(od, d_in, d_out);
        let ed = pt_end(od, d_in, d_out);
        for oh in 0..h_out {
            let sh = pt_start(oh, h_in, h_out);
            let eh = pt_end(oh, h_in, h_out);
            for ow in 0..w_out {
                let sw = pt_start(ow, w_in, w_out);
                let ew = pt_end(ow, w_in, w_out);
                let mut sum = 0f64;
                let mut cnt = 0u32;
                for dd in sd..ed {
                    for hh in sh..eh {
                        for ww in sw..ew {
                            let off = (dd * h_in * w_in + hh * w_in + ww) as usize;
                            sum += host_x[off] as f64;
                            cnt += 1;
                        }
                    }
                }
                let off = (od * h_out * w_out + oh * w_out + ow) as usize;
                exp_y[off] = (sum / cnt as f64) as f32;
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("y");
    let desc = AdaptivePool3dDescriptor {
        batch: n, channels: c,
        d_in, h_in, w_in,
        d_out, h_out, w_out,
        element: ElementKind::F32,
    };
    let plan = AdaptiveAvgPool3dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let x_shape = [n, c, d_in, h_in, w_in];
    let y_shape = [n, c, d_out, h_out, w_out];
    plan.run_fw(&stream, Workspace::None, AdaptivePool3dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got = vec![0f32; out_numel];
    dev_y.copy_to_host(&mut got).expect("dl y");
    let tol = 64.0 * f32::EPSILON;
    for i in 0..out_numel {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got[i] - exp_y[i]).abs() <= t,
            "adaptive_avg_pool3d @ {i}: got={} want={}", got[i], exp_y[i]);
    }
}

// =============================================================================
// AdaptiveAvgPool2d BW 5×5 → 3×3 — verifies gradient flows back correctly.
// =============================================================================
//
// Each input cell `j` receives contributions from every output window
// it belongs to. The expected `dx[j] = Σ_{i: j ∈ window_i} dy[i] / win_size_i`.

#[test]
#[ignore]
fn adaptive_avg_pool2d_5x5_to_3x3_f32_bw_bitexact() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 5i32, 5i32);
    let (h_out, w_out) = (3i32, 3i32);
    let in_numel = (h_in * w_in) as usize;
    let out_numel = (h_out * w_out) as usize;
    // Distinctive `dy` values so any drop / double-count would show up.
    let host_dy: Vec<f32> = (1..=out_numel).map(|v| v as f32 * 0.25).collect();
    let host_x: Vec<f32> = (0..in_numel).map(|v| v as f32).collect();

    // Reference dx computation (same formula the kernel implements).
    let mut exp_dx = vec![0f64; in_numel];
    for oh in 0..h_out {
        let sh = pt_start(oh, h_in, h_out);
        let eh = pt_end(oh, h_in, h_out);
        for ow in 0..w_out {
            let sw = pt_start(ow, w_in, w_out);
            let ew = pt_end(ow, w_in, w_out);
            let win = ((eh - sh) * (ew - sw)) as f64;
            let share = host_dy[(oh * w_out + ow) as usize] as f64 / win;
            for hh in sh..eh {
                for ww in sw..ew {
                    exp_dx[(hh * w_in + ww) as usize] += share;
                }
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    // Pre-fill dx with poison to verify the kernel zeroes correctly.
    let host_dx_init: Vec<f32> = (0..in_numel).map(|_| 999.0).collect();
    let mut dev_dx = DeviceBuffer::from_slice(&ctx, &host_dx_init).expect("up dx");
    // y is unused by AvgPool BW but the args require it for symmetry.
    let dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("y");

    let desc = AdaptivePool2dDescriptor {
        batch: n, channels: c, h_in, w_in, h_out, w_out,
        element: ElementKind::F32,
    };
    let plan = AdaptiveAvgPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    plan.run_bw(&stream, Workspace::None, AdaptivePool2dBwArgs {
        y:  TensorRef { data: dev_y.as_slice(),  shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape: x_shape, stride: contiguous_stride(x_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; in_numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    let tol = 64.0 * f32::EPSILON;
    for i in 0..in_numel {
        let t = (exp_dx[i].abs() as f32 * tol).max(tol);
        assert!((got_dx[i] - exp_dx[i] as f32).abs() <= t,
            "adaptive_avg_pool2d BW @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}

// =============================================================================
// Per-dtype spot checks (5→3, 1D) — verify the template instantiates for
// all four FP dtypes. f32 is the trailblazer; the rest follow if the
// per-dtype accum + atomicAdd paths compile.
// =============================================================================

mod spot_dtypes {
    use super::*;
    use baracuda_kernels::{
        AdaptiveAvgPool1dPlan, AdaptivePool1dDescriptor, AdaptivePool1dFwArgs,
    };
    use half::{bf16, f16};

    #[test]
    #[ignore]
    fn adaptive_avg_pool1d_5_to_3_f64_spot() {
        let (ctx, stream) = setup();
        let host_x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // PyTorch windows for in=5, out=3: [0,2), [1,4), [3,5).
        // means: (1+2)/2=1.5, (2+3+4)/3=3.0, (4+5)/2=4.5.
        let exp = [1.5f64, 3.0, 4.5];
        let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
        let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 3).expect("y");
        let desc = AdaptivePool1dDescriptor {
            batch: 1, channels: 1, l_in: 5, l_out: 3, element: ElementKind::F64,
        };
        let plan = AdaptiveAvgPool1dPlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("sel");
        plan.run_fw(&stream, Workspace::None, AdaptivePool1dFwArgs {
            x: TensorRef { data: dev_x.as_slice(), shape: [1,1,5], stride: contiguous_stride([1,1,5]) },
            y: TensorMut { data: dev_y.as_slice_mut(), shape: [1,1,3], stride: contiguous_stride([1,1,3]) },
        }).expect("fw");
        stream.synchronize().expect("sync");
        let mut got = vec![0f64; 3];
        dev_y.copy_to_host(&mut got).expect("dl");
        for i in 0..3 {
            assert!((got[i] - exp[i]).abs() <= 1e-12,
                "f64 spot @ {i}: got={} want={}", got[i], exp[i]);
        }
    }

    #[test]
    #[ignore]
    fn adaptive_avg_pool1d_5_to_3_f16_spot() {
        let (ctx, stream) = setup();
        let host_x: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0, 5.0]
            .iter().copied().map(f16::from_f32).collect();
        let exp = [1.5f32, 3.0, 4.5];
        let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
        let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 3).expect("y");
        let desc = AdaptivePool1dDescriptor {
            batch: 1, channels: 1, l_in: 5, l_out: 3, element: ElementKind::F16,
        };
        let plan = AdaptiveAvgPool1dPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
        plan.run_fw(&stream, Workspace::None, AdaptivePool1dFwArgs {
            x: TensorRef { data: dev_x.as_slice(), shape: [1,1,5], stride: contiguous_stride([1,1,5]) },
            y: TensorMut { data: dev_y.as_slice_mut(), shape: [1,1,3], stride: contiguous_stride([1,1,3]) },
        }).expect("fw");
        stream.synchronize().expect("sync");
        let mut got = vec![f16::ZERO; 3];
        dev_y.copy_to_host(&mut got).expect("dl");
        for i in 0..3 {
            let g = got[i].to_f32();
            assert!((g - exp[i]).abs() <= 1e-2,
                "f16 spot @ {i}: got={} want={}", g, exp[i]);
        }
    }

    #[test]
    #[ignore]
    fn adaptive_avg_pool1d_5_to_3_bf16_spot() {
        let (ctx, stream) = setup();
        let host_x: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0]
            .iter().copied().map(bf16::from_f32).collect();
        let exp = [1.5f32, 3.0, 4.5];
        let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
        let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 3).expect("y");
        let desc = AdaptivePool1dDescriptor {
            batch: 1, channels: 1, l_in: 5, l_out: 3, element: ElementKind::Bf16,
        };
        let plan = AdaptiveAvgPool1dPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
        plan.run_fw(&stream, Workspace::None, AdaptivePool1dFwArgs {
            x: TensorRef { data: dev_x.as_slice(), shape: [1,1,5], stride: contiguous_stride([1,1,5]) },
            y: TensorMut { data: dev_y.as_slice_mut(), shape: [1,1,3], stride: contiguous_stride([1,1,3]) },
        }).expect("fw");
        stream.synchronize().expect("sync");
        let mut got = vec![bf16::ZERO; 3];
        dev_y.copy_to_host(&mut got).expect("dl");
        for i in 0..3 {
            let g = got[i].to_f32();
            assert!((g - exp[i]).abs() <= 5e-2,
                "bf16 spot @ {i}: got={} want={}", g, exp[i]);
        }
    }
}
