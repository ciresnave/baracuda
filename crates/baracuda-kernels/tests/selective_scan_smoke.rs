//! Real-GPU smoke test for Phase 50b [`SelectiveScanPlan`] (FW).
//!
//! Validates the Mamba-1 selective_scan FW against a CPU reference loop
//! that walks the recurrence sequentially per (b, d). Covers:
//!   - Vanilla path (no D / no z / no delta_bias / no softplus)
//!   - With D skip-connection
//!   - With z gating (SiLU)
//!   - With delta_bias + softplus
//!   - All three FP dtypes (f32 / f16 / bf16) at the vanilla path

#![cfg(feature = "mamba")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SelectiveScanArgs,
    SelectiveScanDescriptor, SelectiveScanPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[allow(clippy::too_many_arguments)]
fn cpu_ref_selective_scan_f32(
    u: &[f32], delta: &[f32], a: &[f32], b: &[f32], c: &[f32],
    d_skip: Option<&[f32]>, z: Option<&[f32]>, delta_bias: Option<&[f32]>,
    delta_softplus: bool,
    bsz: usize, l: usize, dim: usize, dstate: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; bsz * l * dim];
    for bi in 0..bsz {
        for di in 0..dim {
            let mut h = vec![0.0f32; dstate];
            let db = delta_bias.map(|v| v[di]).unwrap_or(0.0);
            let ds = d_skip.map(|v| v[di]).unwrap_or(0.0);
            for t in 0..l {
                let mut dt = delta[bi * l * dim + t * dim + di] + db;
                if delta_softplus && dt <= 20.0 {
                    dt = (1.0 + dt.exp()).ln();
                }
                let u_t = u[bi * l * dim + t * dim + di];
                let mut y_state = 0.0f32;
                for n in 0..dstate {
                    let a_dn = a[di * dstate + n];
                    let b_n = b[bi * l * dstate + t * dstate + n];
                    let c_n = c[bi * l * dstate + t * dstate + n];
                    let da = (dt * a_dn).exp();
                    let dbu = dt * b_n * u_t;
                    h[n] = da * h[n] + dbu;
                    y_state += h[n] * c_n;
                }
                let mut y_val = y_state;
                if d_skip.is_some() {
                    y_val += ds * u_t;
                }
                if let Some(zv) = z {
                    let z_t = zv[bi * l * dim + t * dim + di];
                    let sig = 1.0 / (1.0 + (-z_t).exp());
                    y_val *= z_t * sig;
                }
                y[bi * l * dim + t * dim + di] = y_val;
            }
        }
    }
    y
}

fn check_close(a: &[f32], b: &[f32], tol: f32, tag: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", tag);
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1e-3);
        if diff > tol * scale {
            panic!("{}: mismatch at idx {} — got {}, expected {}, diff {}", tag, i, av, bv, diff);
        }
    }
}

const BSZ: usize = 1;
const L: usize = 8;
const DIM: usize = 4;
const DSTATE: usize = 4;

fn make_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let u: Vec<f32> = (0..BSZ * L * DIM).map(|i| (i as f32) * 0.01).collect();
    let delta: Vec<f32> = (0..BSZ * L * DIM).map(|i| 0.1 + (i as f32) * 0.005).collect();
    // A negative for stable recurrence (decay < 1).
    let a: Vec<f32> = (0..DIM * DSTATE).map(|i| -0.5 - (i as f32) * 0.02).collect();
    let b: Vec<f32> = (0..BSZ * L * DSTATE).map(|i| ((i as f32) * 0.07).sin() * 0.3).collect();
    let c: Vec<f32> = (0..BSZ * L * DSTATE).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();
    (u, delta, a, b, c)
}

#[allow(clippy::too_many_arguments)]
fn run_fw_f32(
    ctx: &Context, stream: &Stream,
    u_host: &[f32], delta_host: &[f32], a_host: &[f32],
    b_host: &[f32], c_host: &[f32],
    d_skip_host: Option<&[f32]>, z_host: Option<&[f32]>, db_host: Option<&[f32]>,
    delta_softplus: bool,
) -> Vec<f32> {
    let u_dev = DeviceBuffer::from_slice(ctx, u_host).expect("u");
    let delta_dev = DeviceBuffer::from_slice(ctx, delta_host).expect("delta");
    let a_dev = DeviceBuffer::from_slice(ctx, a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(ctx, b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(ctx, c_host).expect("c");
    let d_dev = d_skip_host.map(|v| DeviceBuffer::from_slice(ctx, v).expect("d"));
    let z_dev = z_host.map(|v| DeviceBuffer::from_slice(ctx, v).expect("z"));
    let db_dev = db_host.map(|v| DeviceBuffer::from_slice(ctx, v).expect("db"));
    let mut y_dev = DeviceBuffer::<f32>::zeros(ctx, BSZ * L * DIM).expect("y");

    let desc = SelectiveScanDescriptor {
        batch_size: BSZ as i32, seq_len: L as i32, dim: DIM as i32, dstate: DSTATE as i32,
        delta_softplus, element: ElementKind::F32,
    };
    let plan = SelectiveScanPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select");

    let s_ud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let s_a: [i32; 2] = [DIM as i32, DSTATE as i32];
    let s_bc: [i32; 3] = [BSZ as i32, L as i32, DSTATE as i32];
    let s_d: [i32; 1] = [DIM as i32];

    plan.run(stream, Workspace::None, SelectiveScanArgs {
        u: TensorRef { data: u_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        delta: TensorRef { data: delta_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        a: TensorRef { data: a_dev.as_slice(), shape: s_a, stride: contiguous_stride(s_a) },
        b: TensorRef { data: b_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        c: TensorRef { data: c_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        d_skip: d_dev.as_ref().map(|x| TensorRef {
            data: x.as_slice(), shape: s_d, stride: contiguous_stride(s_d) }),
        z: z_dev.as_ref().map(|x| TensorRef {
            data: x.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) }),
        delta_bias: db_dev.as_ref().map(|x| TensorRef {
            data: x.as_slice(), shape: s_d, stride: contiguous_stride(s_d) }),
        y: TensorMut { data: y_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
        last_state: None,
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0.0f32; BSZ * L * DIM];
    y_dev.copy_to_host(&mut got).expect("dl");
    got
}

#[test]
#[ignore]
fn selective_scan_f32_vanilla_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let (u, delta, a, b, c) = make_inputs();
    let got = run_fw_f32(&ctx, &stream, &u, &delta, &a, &b, &c, None, None, None, false);
    let expected = cpu_ref_selective_scan_f32(
        &u, &delta, &a, &b, &c, None, None, None, false,
        BSZ, L, DIM, DSTATE);
    check_close(&got, &expected, 1e-4, "selective_scan_f32_vanilla");
}

#[test]
#[ignore]
fn selective_scan_f32_with_d_skip() {
    let (ctx, stream) = setup();
    let (u, delta, a, b, c) = make_inputs();
    let d_skip: Vec<f32> = (0..DIM).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let got = run_fw_f32(&ctx, &stream, &u, &delta, &a, &b, &c, Some(&d_skip), None, None, false);
    let expected = cpu_ref_selective_scan_f32(
        &u, &delta, &a, &b, &c, Some(&d_skip), None, None, false,
        BSZ, L, DIM, DSTATE);
    check_close(&got, &expected, 1e-4, "selective_scan_f32_with_d_skip");
}

#[test]
#[ignore]
fn selective_scan_f32_with_z_gate() {
    let (ctx, stream) = setup();
    let (u, delta, a, b, c) = make_inputs();
    let z: Vec<f32> = (0..BSZ * L * DIM).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
    let got = run_fw_f32(&ctx, &stream, &u, &delta, &a, &b, &c, None, Some(&z), None, false);
    let expected = cpu_ref_selective_scan_f32(
        &u, &delta, &a, &b, &c, None, Some(&z), None, false,
        BSZ, L, DIM, DSTATE);
    check_close(&got, &expected, 1e-4, "selective_scan_f32_with_z_gate");
}

#[test]
#[ignore]
fn selective_scan_f32_with_delta_bias_and_softplus() {
    let (ctx, stream) = setup();
    let (u, delta, a, b, c) = make_inputs();
    let db: Vec<f32> = (0..DIM).map(|i| -2.0 + (i as f32) * 0.5).collect();
    let got = run_fw_f32(&ctx, &stream, &u, &delta, &a, &b, &c, None, None, Some(&db), true);
    let expected = cpu_ref_selective_scan_f32(
        &u, &delta, &a, &b, &c, None, None, Some(&db), true,
        BSZ, L, DIM, DSTATE);
    check_close(&got, &expected, 1e-4, "selective_scan_f32_with_delta_bias_and_softplus");
}

#[test]
#[ignore]
fn selective_scan_f32_full_combo_matches_cpu_ref() {
    // The full set of options simultaneously — D skip + z gating +
    // delta_bias + softplus. This is the path Mamba-1 actually uses
    // in production (Mamba-7B / Falcon-Mamba / Codestral-Mamba).
    let (ctx, stream) = setup();
    let (u, delta, a, b, c) = make_inputs();
    let d_skip: Vec<f32> = (0..DIM).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let z: Vec<f32> = (0..BSZ * L * DIM).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
    let db: Vec<f32> = (0..DIM).map(|i| -2.0 + (i as f32) * 0.5).collect();
    let got = run_fw_f32(&ctx, &stream, &u, &delta, &a, &b, &c,
        Some(&d_skip), Some(&z), Some(&db), true);
    let expected = cpu_ref_selective_scan_f32(
        &u, &delta, &a, &b, &c, Some(&d_skip), Some(&z), Some(&db), true,
        BSZ, L, DIM, DSTATE);
    check_close(&got, &expected, 1e-4, "selective_scan_f32_full_combo");
}

#[test]
#[ignore]
fn selective_scan_f16_vanilla_matches_cpu_ref_loose() {
    let (ctx, stream) = setup();
    let (u_f, delta_f, a_f, b_f, c_f) = make_inputs();
    // Convert to f16 host buffers.
    let u: Vec<f16> = u_f.iter().map(|&v| f16::from_f32(v)).collect();
    let delta: Vec<f16> = delta_f.iter().map(|&v| f16::from_f32(v)).collect();
    let a: Vec<f16> = a_f.iter().map(|&v| f16::from_f32(v)).collect();
    let b: Vec<f16> = b_f.iter().map(|&v| f16::from_f32(v)).collect();
    let c: Vec<f16> = c_f.iter().map(|&v| f16::from_f32(v)).collect();

    let u_dev = DeviceBuffer::from_slice(&ctx, &u).expect("u");
    let delta_dev = DeviceBuffer::from_slice(&ctx, &delta).expect("delta");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c).expect("c");
    let mut y_dev = DeviceBuffer::<f16>::zeros(&ctx, BSZ * L * DIM).expect("y");

    let desc = SelectiveScanDescriptor {
        batch_size: BSZ as i32, seq_len: L as i32, dim: DIM as i32, dstate: DSTATE as i32,
        delta_softplus: false, element: ElementKind::F16,
    };
    let plan = SelectiveScanPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let s_ud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let s_a: [i32; 2] = [DIM as i32, DSTATE as i32];
    let s_bc: [i32; 3] = [BSZ as i32, L as i32, DSTATE as i32];

    plan.run(&stream, Workspace::None, SelectiveScanArgs {
        u: TensorRef { data: u_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        delta: TensorRef { data: delta_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        a: TensorRef { data: a_dev.as_slice(), shape: s_a, stride: contiguous_stride(s_a) },
        b: TensorRef { data: b_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        c: TensorRef { data: c_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        d_skip: None, z: None, delta_bias: None,
        y: TensorMut { data: y_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
        last_state: None,
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got_h = vec![f16::ZERO; BSZ * L * DIM];
    y_dev.copy_to_host(&mut got_h).expect("dl");
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();

    let expected = cpu_ref_selective_scan_f32(
        &u_f, &delta_f, &a_f, &b_f, &c_f, None, None, None, false,
        BSZ, L, DIM, DSTATE);
    // f16 has ~3-4 decimal digits — loosen tolerance accordingly.
    check_close(&got, &expected, 5e-3, "selective_scan_f16_vanilla");
}

#[test]
#[ignore]
fn selective_scan_bf16_vanilla_matches_cpu_ref_loose() {
    let (ctx, stream) = setup();
    let (u_f, delta_f, a_f, b_f, c_f) = make_inputs();
    let u: Vec<bf16> = u_f.iter().map(|&v| bf16::from_f32(v)).collect();
    let delta: Vec<bf16> = delta_f.iter().map(|&v| bf16::from_f32(v)).collect();
    let a: Vec<bf16> = a_f.iter().map(|&v| bf16::from_f32(v)).collect();
    let b: Vec<bf16> = b_f.iter().map(|&v| bf16::from_f32(v)).collect();
    let c: Vec<bf16> = c_f.iter().map(|&v| bf16::from_f32(v)).collect();

    let u_dev = DeviceBuffer::from_slice(&ctx, &u).expect("u");
    let delta_dev = DeviceBuffer::from_slice(&ctx, &delta).expect("delta");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c).expect("c");
    let mut y_dev = DeviceBuffer::<bf16>::zeros(&ctx, BSZ * L * DIM).expect("y");

    let desc = SelectiveScanDescriptor {
        batch_size: BSZ as i32, seq_len: L as i32, dim: DIM as i32, dstate: DSTATE as i32,
        delta_softplus: false, element: ElementKind::Bf16,
    };
    let plan = SelectiveScanPlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let s_ud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let s_a: [i32; 2] = [DIM as i32, DSTATE as i32];
    let s_bc: [i32; 3] = [BSZ as i32, L as i32, DSTATE as i32];

    plan.run(&stream, Workspace::None, SelectiveScanArgs {
        u: TensorRef { data: u_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        delta: TensorRef { data: delta_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        a: TensorRef { data: a_dev.as_slice(), shape: s_a, stride: contiguous_stride(s_a) },
        b: TensorRef { data: b_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        c: TensorRef { data: c_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        d_skip: None, z: None, delta_bias: None,
        y: TensorMut { data: y_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
        last_state: None,
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got_h = vec![bf16::ZERO; BSZ * L * DIM];
    y_dev.copy_to_host(&mut got_h).expect("dl");
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();

    let expected = cpu_ref_selective_scan_f32(
        &u_f, &delta_f, &a_f, &b_f, &c_f, None, None, None, false,
        BSZ, L, DIM, DSTATE);
    // bf16 has ~2-3 decimal digits — looser tolerance than f16.
    check_close(&got, &expected, 2e-2, "selective_scan_bf16_vanilla");
}

#[test]
#[ignore]
fn selective_scan_dstate_too_large_rejected() {
    let (_ctx, stream) = setup();
    let desc = SelectiveScanDescriptor {
        batch_size: 1, seq_len: 4, dim: 4, dstate: 512,
        delta_softplus: false, element: ElementKind::F32,
    };
    let res = SelectiveScanPlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(res.is_err(), "dstate > 256 should be rejected");
}

#[test]
#[ignore]
fn selective_scan_with_last_state_saves_final_h() {
    // Sanity: when last_state is requested, the kernel writes a value
    // that matches the CPU reference's final h vector.
    let (ctx, stream) = setup();
    let (u, delta, a, b, c) = make_inputs();

    // Compute reference last state via CPU.
    let mut ref_last = vec![0.0f32; BSZ * DIM * DSTATE];
    for bi in 0..BSZ {
        for di in 0..DIM {
            let mut h = vec![0.0f32; DSTATE];
            for t in 0..L {
                let dt = delta[bi * L * DIM + t * DIM + di];
                let u_t = u[bi * L * DIM + t * DIM + di];
                for n in 0..DSTATE {
                    let a_dn = a[di * DSTATE + n];
                    let b_n = b[bi * L * DSTATE + t * DSTATE + n];
                    let da = (dt * a_dn).exp();
                    let dbu = dt * b_n * u_t;
                    h[n] = da * h[n] + dbu;
                }
            }
            for n in 0..DSTATE {
                ref_last[(bi * DIM + di) * DSTATE + n] = h[n];
            }
        }
    }

    let u_dev = DeviceBuffer::from_slice(&ctx, &u).expect("u");
    let delta_dev = DeviceBuffer::from_slice(&ctx, &delta).expect("delta");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c).expect("c");
    let mut y_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DIM).expect("y");
    let mut ls_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * DIM * DSTATE).expect("ls");

    let desc = SelectiveScanDescriptor {
        batch_size: BSZ as i32, seq_len: L as i32, dim: DIM as i32, dstate: DSTATE as i32,
        delta_softplus: false, element: ElementKind::F32,
    };
    let plan = SelectiveScanPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let s_ud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let s_a: [i32; 2] = [DIM as i32, DSTATE as i32];
    let s_bc: [i32; 3] = [BSZ as i32, L as i32, DSTATE as i32];
    let s_ls: [i32; 3] = [BSZ as i32, DIM as i32, DSTATE as i32];

    plan.run(&stream, Workspace::None, SelectiveScanArgs {
        u: TensorRef { data: u_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        delta: TensorRef { data: delta_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
        a: TensorRef { data: a_dev.as_slice(), shape: s_a, stride: contiguous_stride(s_a) },
        b: TensorRef { data: b_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        c: TensorRef { data: c_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
        d_skip: None, z: None, delta_bias: None,
        y: TensorMut { data: y_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
        last_state: Some(TensorMut { data: ls_dev.as_slice_mut(), shape: s_ls, stride: contiguous_stride(s_ls) }),
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; BSZ * DIM * DSTATE];
    ls_dev.copy_to_host(&mut got).expect("dl");
    check_close(&got, &ref_last, 1e-4, "selective_scan_last_state");
}
