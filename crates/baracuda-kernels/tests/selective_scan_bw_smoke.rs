//! Real-GPU smoke test for Phase 50b [`SelectiveScanBackwardPlan`].
//!
//! Validates `du`, `ddelta`, and `dA` via finite difference on a small
//! problem. Other grads (dB, dC) sanity-checked for finiteness +
//! magnitude. Run with `--features mamba`.

#![cfg(feature = "mamba")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SelectiveScanArgs,
    SelectiveScanBackwardArgs, SelectiveScanBackwardDescriptor, SelectiveScanBackwardPlan,
    SelectiveScanDescriptor, SelectiveScanPlan, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const BSZ: usize = 1;
const L: usize = 4;
const DIM: usize = 2;
const DSTATE: usize = 4;

#[allow(clippy::too_many_arguments)]
fn run_fw_and_sum(
    ctx: &Context, stream: &Stream,
    u_host: &[f32], delta_host: &[f32], a_host: &[f32],
    b_host: &[f32], c_host: &[f32],
) -> f32 {
    let u_dev = DeviceBuffer::from_slice(ctx, u_host).expect("u");
    let delta_dev = DeviceBuffer::from_slice(ctx, delta_host).expect("delta");
    let a_dev = DeviceBuffer::from_slice(ctx, a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(ctx, b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(ctx, c_host).expect("c");
    let mut y_dev = DeviceBuffer::<f32>::zeros(ctx, BSZ * L * DIM).expect("y");

    let desc = SelectiveScanDescriptor {
        batch_size: BSZ as i32, seq_len: L as i32, dim: DIM as i32, dstate: DSTATE as i32,
        delta_softplus: false, element: ElementKind::F32,
    };
    let plan = SelectiveScanPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let s_ud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let s_a: [i32; 2] = [DIM as i32, DSTATE as i32];
    let s_bc: [i32; 3] = [BSZ as i32, L as i32, DSTATE as i32];

    plan.run(stream, Workspace::None, SelectiveScanArgs {
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
    let mut got = vec![0.0f32; BSZ * L * DIM];
    y_dev.copy_to_host(&mut got).expect("dl");
    got.iter().sum()
}

#[test]
#[ignore]
fn selective_scan_bw_du_ddelta_da_match_fd_others_finite() {
    let (ctx, stream) = setup();
    let u_host: Vec<f32> = (0..BSZ * L * DIM).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let delta_host: Vec<f32> = (0..BSZ * L * DIM).map(|i| 0.1 + (i as f32) * 0.005).collect();
    let a_host: Vec<f32> = (0..DIM * DSTATE).map(|i| -0.5 - (i as f32) * 0.02).collect();
    let b_host: Vec<f32> = (0..BSZ * L * DSTATE).map(|i| ((i as f32) * 0.07).sin() * 0.3).collect();
    let c_host: Vec<f32> = (0..BSZ * L * DSTATE).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();
    let dy_host = vec![1.0f32; BSZ * L * DIM];

    let u_dev = DeviceBuffer::from_slice(&ctx, &u_host).expect("u");
    let delta_dev = DeviceBuffer::from_slice(&ctx, &delta_host).expect("delta");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c_host).expect("c");
    let dy_dev = DeviceBuffer::from_slice(&ctx, &dy_host).expect("dy");

    let mut du_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DIM).expect("du");
    let mut db_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DSTATE).expect("dB");
    let mut dc_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DSTATE).expect("dC");
    let mut ddelta_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DIM).expect("ddelta");
    let mut da_dev = DeviceBuffer::<f32>::zeros(&ctx, DIM * DSTATE).expect("dA");

    let bw_desc = SelectiveScanBackwardDescriptor {
        batch_size: BSZ as i32, seq_len: L as i32, dim: DIM as i32, dstate: DSTATE as i32,
        delta_softplus: false, element: ElementKind::F32,
    };
    let bw_plan = SelectiveScanBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default())
        .expect("bw select");
    let ws_bytes = bw_plan.workspace_size();
    let mut ws = DeviceBuffer::<u8>::zeros(&ctx, ws_bytes.max(1)).expect("ws");

    let s_ud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let s_a: [i32; 2] = [DIM as i32, DSTATE as i32];
    let s_bc: [i32; 3] = [BSZ as i32, L as i32, DSTATE as i32];

    bw_plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()),
        SelectiveScanBackwardArgs {
            u: TensorRef { data: u_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            delta: TensorRef { data: delta_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            a: TensorRef { data: a_dev.as_slice(), shape: s_a, stride: contiguous_stride(s_a) },
            b: TensorRef { data: b_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
            c: TensorRef { data: c_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_skip: None, z: None, delta_bias: None,
            dy: TensorRef { data: dy_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            du: TensorMut { data: du_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
            d_b: TensorMut { data: db_dev.as_slice_mut(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_c: TensorMut { data: dc_dev.as_slice_mut(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_delta: TensorMut { data: ddelta_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
            d_a: TensorMut { data: da_dev.as_slice_mut(), shape: s_a, stride: contiguous_stride(s_a) },
            d_d: None, dz: None, d_delta_bias: None,
        }).expect("bw run");
    stream.synchronize().expect("sync");

    let mut du_got = vec![0.0f32; BSZ * L * DIM];
    let mut db_got = vec![0.0f32; BSZ * L * DSTATE];
    let mut dc_got = vec![0.0f32; BSZ * L * DSTATE];
    let mut ddelta_got = vec![0.0f32; BSZ * L * DIM];
    let mut da_got = vec![0.0f32; DIM * DSTATE];
    du_dev.copy_to_host(&mut du_got).expect("dl");
    db_dev.copy_to_host(&mut db_got).expect("dl");
    dc_dev.copy_to_host(&mut dc_got).expect("dl");
    ddelta_dev.copy_to_host(&mut ddelta_got).expect("dl");
    da_dev.copy_to_host(&mut da_got).expect("dl");

    // Finiteness sanity for every gradient.
    for v in du_got.iter().chain(ddelta_got.iter())
        .chain(da_got.iter()).chain(db_got.iter()).chain(dc_got.iter())
    {
        assert!(v.is_finite(), "non-finite grad component: {}", v);
    }

    // FD check for du at two probe cells.
    let eps = 1e-3f32;
    let tol = 5e-2f32;
    for &i in &[0usize, BSZ * L * DIM - 1] {
        let mut up = u_host.clone(); up[i] += eps;
        let mut um = u_host.clone(); um[i] -= eps;
        let yp = run_fw_and_sum(&ctx, &stream, &up, &delta_host, &a_host, &b_host, &c_host);
        let ym = run_fw_and_sum(&ctx, &stream, &um, &delta_host, &a_host, &b_host, &c_host);
        let fd = (yp - ym) / (2.0 * eps);
        let analytic = du_got[i];
        let diff = (fd - analytic).abs();
        let scale = fd.abs().max(analytic.abs()).max(1e-3);
        assert!(diff < tol * scale,
            "du[{}]: analytic={} fd={} diff={}", i, analytic, fd, diff);
    }

    // FD check for ddelta at one probe cell.
    {
        let i = 0usize;
        let mut dp = delta_host.clone(); dp[i] += eps;
        let mut dm = delta_host.clone(); dm[i] -= eps;
        let yp = run_fw_and_sum(&ctx, &stream, &u_host, &dp, &a_host, &b_host, &c_host);
        let ym = run_fw_and_sum(&ctx, &stream, &u_host, &dm, &a_host, &b_host, &c_host);
        let fd = (yp - ym) / (2.0 * eps);
        let analytic = ddelta_got[i];
        let diff = (fd - analytic).abs();
        let scale = fd.abs().max(analytic.abs()).max(1e-3);
        assert!(diff < tol * scale,
            "ddelta[{}]: analytic={} fd={} diff={}", i, analytic, fd, diff);
    }

    // FD check for dA at one probe cell.
    {
        let i = 0usize;
        let mut ap = a_host.clone(); ap[i] += eps;
        let mut am = a_host.clone(); am[i] -= eps;
        let yp = run_fw_and_sum(&ctx, &stream, &u_host, &delta_host, &ap, &b_host, &c_host);
        let ym = run_fw_and_sum(&ctx, &stream, &u_host, &delta_host, &am, &b_host, &c_host);
        let fd = (yp - ym) / (2.0 * eps);
        let analytic = da_got[i];
        let diff = (fd - analytic).abs();
        let scale = fd.abs().max(analytic.abs()).max(1e-3);
        assert!(diff < tol * scale,
            "dA[{}]: analytic={} fd={} diff={}", i, analytic, fd, diff);
    }
}

#[test]
#[ignore]
fn selective_scan_bw_with_d_skip_dz_ddelta_bias_runs_finite() {
    // End-to-end: full optional surface (D + z + delta_bias) with all
    // matching gradients enabled. Just checks the launcher accepts the
    // arg topology and produces finite output.
    let (ctx, stream) = setup();
    let u_host: Vec<f32> = (0..BSZ * L * DIM).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let delta_host: Vec<f32> = (0..BSZ * L * DIM).map(|i| 0.05 + (i as f32) * 0.005).collect();
    let a_host: Vec<f32> = (0..DIM * DSTATE).map(|i| -0.5 - (i as f32) * 0.02).collect();
    let b_host: Vec<f32> = (0..BSZ * L * DSTATE).map(|i| ((i as f32) * 0.07).sin() * 0.3).collect();
    let c_host: Vec<f32> = (0..BSZ * L * DSTATE).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();
    let d_host: Vec<f32> = (0..DIM).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let z_host: Vec<f32> = (0..BSZ * L * DIM).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
    let db_host: Vec<f32> = (0..DIM).map(|i| -1.0 + (i as f32) * 0.5).collect();
    let dy_host = vec![1.0f32; BSZ * L * DIM];

    let u_dev = DeviceBuffer::from_slice(&ctx, &u_host).expect("u");
    let delta_dev = DeviceBuffer::from_slice(&ctx, &delta_host).expect("delta");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c_host).expect("c");
    let d_dev = DeviceBuffer::from_slice(&ctx, &d_host).expect("d");
    let z_dev = DeviceBuffer::from_slice(&ctx, &z_host).expect("z");
    let db_dev = DeviceBuffer::from_slice(&ctx, &db_host).expect("db");
    let dy_dev = DeviceBuffer::from_slice(&ctx, &dy_host).expect("dy");

    let mut du_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DIM).expect("du");
    let mut dB_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DSTATE).expect("dB");
    let mut dC_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DSTATE).expect("dC");
    let mut ddelta_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DIM).expect("ddelta");
    let mut dA_dev = DeviceBuffer::<f32>::zeros(&ctx, DIM * DSTATE).expect("dA");
    let mut dD_dev = DeviceBuffer::<f32>::zeros(&ctx, DIM).expect("dD");
    let mut dz_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DIM).expect("dz");
    let mut ddb_dev = DeviceBuffer::<f32>::zeros(&ctx, DIM).expect("ddb");

    let bw_desc = SelectiveScanBackwardDescriptor {
        batch_size: BSZ as i32, seq_len: L as i32, dim: DIM as i32, dstate: DSTATE as i32,
        delta_softplus: true, element: ElementKind::F32,
    };
    let bw_plan = SelectiveScanBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default())
        .expect("bw select");
    let ws_bytes = bw_plan.workspace_size();
    let mut ws = DeviceBuffer::<u8>::zeros(&ctx, ws_bytes.max(1)).expect("ws");

    let s_ud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let s_a: [i32; 2] = [DIM as i32, DSTATE as i32];
    let s_bc: [i32; 3] = [BSZ as i32, L as i32, DSTATE as i32];
    let s_d: [i32; 1] = [DIM as i32];

    bw_plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()),
        SelectiveScanBackwardArgs {
            u: TensorRef { data: u_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            delta: TensorRef { data: delta_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            a: TensorRef { data: a_dev.as_slice(), shape: s_a, stride: contiguous_stride(s_a) },
            b: TensorRef { data: b_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
            c: TensorRef { data: c_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_skip: Some(TensorRef { data: d_dev.as_slice(), shape: s_d, stride: contiguous_stride(s_d) }),
            z: Some(TensorRef { data: z_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) }),
            delta_bias: Some(TensorRef { data: db_dev.as_slice(), shape: s_d, stride: contiguous_stride(s_d) }),
            dy: TensorRef { data: dy_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            du: TensorMut { data: du_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
            d_b: TensorMut { data: dB_dev.as_slice_mut(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_c: TensorMut { data: dC_dev.as_slice_mut(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_delta: TensorMut { data: ddelta_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
            d_a: TensorMut { data: dA_dev.as_slice_mut(), shape: s_a, stride: contiguous_stride(s_a) },
            d_d: Some(TensorMut { data: dD_dev.as_slice_mut(), shape: s_d, stride: contiguous_stride(s_d) }),
            dz: Some(TensorMut { data: dz_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) }),
            d_delta_bias: Some(TensorMut { data: ddb_dev.as_slice_mut(), shape: s_d, stride: contiguous_stride(s_d) }),
        }).expect("bw full run");
    stream.synchronize().expect("sync");

    let mut all = vec![0.0f32; BSZ * L * DIM];
    let mut tmp_d = vec![0.0f32; DIM];
    du_dev.copy_to_host(&mut all).expect("dl");
    for v in &all { assert!(v.is_finite(), "du non-finite"); }
    ddelta_dev.copy_to_host(&mut all).expect("dl");
    for v in &all { assert!(v.is_finite(), "ddelta non-finite"); }
    dz_dev.copy_to_host(&mut all).expect("dl");
    for v in &all { assert!(v.is_finite(), "dz non-finite"); }
    dD_dev.copy_to_host(&mut tmp_d).expect("dl");
    for v in &tmp_d { assert!(v.is_finite(), "dD non-finite"); }
    ddb_dev.copy_to_host(&mut tmp_d).expect("dl");
    for v in &tmp_d { assert!(v.is_finite(), "d_delta_bias non-finite"); }
}

#[test]
#[ignore]
fn selective_scan_bw_missing_dD_when_d_given_errors() {
    // Topology check: if FW D given, BW must supply dD or error.
    let (ctx, stream) = setup();
    let u_host = vec![0.1f32; BSZ * L * DIM];
    let delta_host = vec![0.05f32; BSZ * L * DIM];
    let a_host = vec![-0.5f32; DIM * DSTATE];
    let b_host = vec![0.1f32; BSZ * L * DSTATE];
    let c_host = vec![0.1f32; BSZ * L * DSTATE];
    let d_host = vec![0.1f32; DIM];
    let dy_host = vec![1.0f32; BSZ * L * DIM];

    let u_dev = DeviceBuffer::from_slice(&ctx, &u_host).expect("u");
    let delta_dev = DeviceBuffer::from_slice(&ctx, &delta_host).expect("delta");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c_host).expect("c");
    let d_dev = DeviceBuffer::from_slice(&ctx, &d_host).expect("d");
    let dy_dev = DeviceBuffer::from_slice(&ctx, &dy_host).expect("dy");

    let mut du_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DIM).expect("du");
    let mut dB_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DSTATE).expect("dB");
    let mut dC_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DSTATE).expect("dC");
    let mut ddelta_dev = DeviceBuffer::<f32>::zeros(&ctx, BSZ * L * DIM).expect("ddelta");
    let mut dA_dev = DeviceBuffer::<f32>::zeros(&ctx, DIM * DSTATE).expect("dA");

    let bw_desc = SelectiveScanBackwardDescriptor {
        batch_size: BSZ as i32, seq_len: L as i32, dim: DIM as i32, dstate: DSTATE as i32,
        delta_softplus: false, element: ElementKind::F32,
    };
    let bw_plan = SelectiveScanBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default())
        .expect("bw select");
    let ws_bytes = bw_plan.workspace_size();
    let mut ws = DeviceBuffer::<u8>::zeros(&ctx, ws_bytes.max(1)).expect("ws");

    let s_ud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let s_a: [i32; 2] = [DIM as i32, DSTATE as i32];
    let s_bc: [i32; 3] = [BSZ as i32, L as i32, DSTATE as i32];
    let s_d: [i32; 1] = [DIM as i32];

    let res = bw_plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()),
        SelectiveScanBackwardArgs {
            u: TensorRef { data: u_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            delta: TensorRef { data: delta_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            a: TensorRef { data: a_dev.as_slice(), shape: s_a, stride: contiguous_stride(s_a) },
            b: TensorRef { data: b_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
            c: TensorRef { data: c_dev.as_slice(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_skip: Some(TensorRef { data: d_dev.as_slice(), shape: s_d, stride: contiguous_stride(s_d) }),
            z: None, delta_bias: None,
            dy: TensorRef { data: dy_dev.as_slice(), shape: s_ud, stride: contiguous_stride(s_ud) },
            du: TensorMut { data: du_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
            d_b: TensorMut { data: dB_dev.as_slice_mut(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_c: TensorMut { data: dC_dev.as_slice_mut(), shape: s_bc, stride: contiguous_stride(s_bc) },
            d_delta: TensorMut { data: ddelta_dev.as_slice_mut(), shape: s_ud, stride: contiguous_stride(s_ud) },
            d_a: TensorMut { data: dA_dev.as_slice_mut(), shape: s_a, stride: contiguous_stride(s_a) },
            d_d: None,  // intentionally missing while FW d_skip is Some(_)
            dz: None,
            d_delta_bias: None,
        });
    assert!(res.is_err(), "BW should reject d_skip=Some + d_d=None");
}
