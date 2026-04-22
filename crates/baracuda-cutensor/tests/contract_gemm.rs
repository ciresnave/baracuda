//! cuTENSOR integration test: a matmul via contraction.
//!
//! Runs `D[m,n] = A[m,k] * B[k,n]` and checks the result against a CPU
//! reference. Skips if cuTENSOR isn't installed.

use core::ffi::c_void;

use baracuda_cutensor::*;
use baracuda_runtime::{Device, DeviceBuffer, Stream};

#[test]
#[ignore = "requires cuTENSOR installed + NVIDIA GPU"]
fn contract_matmul_small() {
    if baracuda_cutensor::probe().is_err() {
        eprintln!("cuTENSOR not installed — skipping");
        return;
    }

    Device::from_ordinal(0).set_current().unwrap();
    baracuda_cutensor::set_log_level(2).ok();
    let stream = Stream::new().unwrap();
    let handle = Handle::new().unwrap();

    // Shapes: A = 8x4, B = 4x6, D = 8x6. Modes: m=0, n=1, k=2.
    let m = 8usize;
    let n = 6usize;
    let k = 4usize;

    let a_host: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
    let b_host: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.2).collect();

    // CPU reference (row-major): d[i,j] = Σ a[i,l] * b[l,j]
    let mut d_ref = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for l in 0..k {
                s += a_host[i * k + l] * b_host[l * n + j];
            }
            d_ref[i * n + j] = s;
        }
    }

    let d_a = DeviceBuffer::from_slice(&a_host).unwrap();
    let d_b = DeviceBuffer::from_slice(&b_host).unwrap();
    let d_d: DeviceBuffer<f32> = DeviceBuffer::new(m * n).unwrap();

    // cuTENSOR's default layout is column-major. Pass explicit
    // row-major strides (outer-dim first).
    let desc_a = TensorDescriptor::new(
        &handle,
        &[m as i64, k as i64],
        Some(&[k as i64, 1]),
        DataType::F32,
        128,
    )
    .unwrap();
    let desc_b = TensorDescriptor::new(
        &handle,
        &[k as i64, n as i64],
        Some(&[n as i64, 1]),
        DataType::F32,
        128,
    )
    .unwrap();
    let desc_d = TensorDescriptor::new(
        &handle,
        &[m as i64, n as i64],
        Some(&[n as i64, 1]),
        DataType::F32,
        128,
    )
    .unwrap();

    // Modes — mode indices are arbitrary ints; here 0=m, 1=n, 2=k.
    let modes_a: [i32; 2] = [0, 2];
    let modes_b: [i32; 2] = [2, 1];
    let modes_d: [i32; 2] = [0, 1];

    let compute = handle.compute_desc_32f().unwrap();
    let op = unsafe {
        Contraction::new(
            &handle, &desc_a, &modes_a, &desc_b, &modes_b,
            &desc_d, // C (we pass D's descriptor; β = 0 so it's write-only)
            &modes_d, &desc_d, &modes_d, compute,
        )
    }
    .unwrap();

    let pref = PlanPreference::default_for(&handle).unwrap();
    let ws = op
        .estimate_workspace(&pref, WorkspaceKind::Default)
        .unwrap();
    let workspace: DeviceBuffer<u8> = DeviceBuffer::new(ws as usize).unwrap();

    let plan = Plan::new(&op, &pref, ws).unwrap();

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        plan.contract(
            &alpha as *const _ as *const c_void,
            d_a.as_raw(),
            d_b.as_raw(),
            &beta as *const _ as *const c_void,
            d_d.as_raw(),
            d_d.as_raw(),
            workspace.as_raw(),
            ws,
            stream.as_raw(),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    let mut d_got = vec![0f32; m * n];
    d_d.copy_to_host(&mut d_got).unwrap();

    for (got, want) in d_got.iter().zip(&d_ref) {
        assert!(
            (got - want).abs() < 1e-3,
            "got={got}, want={want}, diff={}",
            (got - want).abs()
        );
    }
    eprintln!("cuTENSOR matmul OK — {m}x{k} · {k}x{n} matches CPU reference");
}

#[test]
#[ignore = "requires cuTENSOR installed + NVIDIA GPU"]
fn reduce_sum_axis() {
    if baracuda_cutensor::probe().is_err() {
        return;
    }
    Device::from_ordinal(0).set_current().unwrap();
    let stream = Stream::new().unwrap();
    let handle = Handle::new().unwrap();

    let rows = 4usize;
    let cols = 5usize;
    let a_host: Vec<f32> = (0..(rows * cols)).map(|i| (i + 1) as f32).collect();

    let d_a = DeviceBuffer::from_slice(&a_host).unwrap();
    let d_out: DeviceBuffer<f32> = DeviceBuffer::new(rows).unwrap();

    let desc_a = TensorDescriptor::new(
        &handle,
        &[rows as i64, cols as i64],
        Some(&[cols as i64, 1]),
        DataType::F32,
        128,
    )
    .unwrap();
    let desc_out =
        TensorDescriptor::new(&handle, &[rows as i64], None, DataType::F32, 128).unwrap();

    // Modes: 0 = rows, 1 = cols (reduced).
    let modes_a: [i32; 2] = [0, 1];
    let modes_out: [i32; 1] = [0];

    let compute = handle.compute_desc_32f().unwrap();
    let op = unsafe {
        Reduction::new(
            &handle,
            &desc_a,
            &modes_a,
            &desc_out,
            &modes_out,
            &desc_out,
            &modes_out,
            BinaryOp::Add,
            compute,
        )
    }
    .unwrap();
    let pref = PlanPreference::default_for(&handle).unwrap();
    let ws = op
        .estimate_workspace(&pref, WorkspaceKind::Default)
        .unwrap();
    let workspace: DeviceBuffer<u8> = DeviceBuffer::new(ws as usize).unwrap();
    let plan = Plan::new(&op, &pref, ws).unwrap();

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        plan.reduce(
            &alpha as *const _ as *const c_void,
            d_a.as_raw(),
            &beta as *const _ as *const c_void,
            d_out.as_raw(),
            d_out.as_raw(),
            workspace.as_raw(),
            ws,
            stream.as_raw(),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    let mut got = vec![0f32; rows];
    d_out.copy_to_host(&mut got).unwrap();
    for i in 0..rows {
        let want: f32 = (0..cols).map(|j| a_host[i * cols + j]).sum();
        assert!(
            (got[i] - want).abs() < 1e-4,
            "row {i}: got {}, want {}",
            got[i],
            want
        );
    }
}
