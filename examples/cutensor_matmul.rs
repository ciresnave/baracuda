//! cuTENSOR matmul via contraction.
//!
//! Runs `D[m,n] = Σₖ A[m,k] · B[k,n]` using cuTENSOR's `cutensorContract`
//! and verifies the result against a CPU reference.
//!
//! Run:
//! ```
//! cargo run --bin cutensor_matmul --release -- [m] [n] [k]
//! ```

use core::ffi::c_void;

use baracuda_cutensor::*;
use baracuda_runtime::{Device, DeviceBuffer, Stream};

fn main() -> Result<()> {
    // cuTENSOR's planner can blow a 1 MiB default Windows stack during
    // `cutensorCreatePlan`. Move the entire workload to a large-stack
    // thread.
    let t = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(run)
        .unwrap();
    t.join().unwrap()
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    println!("cuTENSOR matmul: A[{m}×{k}] · B[{k}×{n}] → D[{m}×{n}]");

    Device::from_ordinal(0)
        .set_current()
        .map_err(|_| Error::Status {
            status: baracuda_cutensor_sys::cutensorStatus_t::INTERNAL_ERROR,
        })?;
    let stream = Stream::new().map_err(|_| Error::Status {
        status: baracuda_cutensor_sys::cutensorStatus_t::INTERNAL_ERROR,
    })?;
    let handle = Handle::new()?;

    // Build inputs: A = arange(m*k)*0.01, B = arange(k*n)*0.02.
    let a_host: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let b_host: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.02).collect();

    // Upload to device.
    let d_a = DeviceBuffer::from_slice(&a_host).unwrap();
    let d_b = DeviceBuffer::from_slice(&b_host).unwrap();
    let d_d: DeviceBuffer<f32> = DeviceBuffer::new(m * n).unwrap();

    // Row-major strides — cuTENSOR defaults to column-major otherwise.
    let desc_a = TensorDescriptor::new(
        &handle,
        &[m as i64, k as i64],
        Some(&[k as i64, 1]),
        DataType::F32,
        128,
    )?;
    let desc_b = TensorDescriptor::new(
        &handle,
        &[k as i64, n as i64],
        Some(&[n as i64, 1]),
        DataType::F32,
        128,
    )?;
    let desc_d = TensorDescriptor::new(
        &handle,
        &[m as i64, n as i64],
        Some(&[n as i64, 1]),
        DataType::F32,
        128,
    )?;

    // Mode labels: 0=m, 1=n, 2=k.
    let ma = [0i32, 2];
    let mb = [2i32, 1];
    let md = [0i32, 1];

    let compute = handle.compute_desc_32f()?;
    let op = unsafe {
        Contraction::new(
            &handle, &desc_a, &ma, &desc_b, &mb, &desc_d, &md, &desc_d, &md, compute,
        )
    }?;

    let pref = PlanPreference::default_for(&handle)?;
    let ws = op.estimate_workspace(&pref, WorkspaceKind::Default)?;
    println!("  workspace: {} bytes", ws);

    let workspace: DeviceBuffer<u8> = DeviceBuffer::new(ws as usize).unwrap();
    let plan = Plan::new(&op, &pref, ws)?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let t0 = std::time::Instant::now();
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
        )?;
    }
    stream.synchronize().unwrap();
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // CPU reference.
    let t1 = std::time::Instant::now();
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
    let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Compare.
    let mut d_got = vec![0f32; m * n];
    d_d.copy_to_host(&mut d_got).unwrap();
    let max_abs_diff = d_got
        .iter()
        .zip(&d_ref)
        .map(|(g, r)| (g - r).abs())
        .fold(0f32, f32::max);
    let max_rel_diff = d_got
        .iter()
        .zip(&d_ref)
        .map(|(g, r)| (g - r).abs() / r.abs().max(1e-6))
        .fold(0f32, f32::max);

    println!("  GPU:   {:.3} ms", gpu_ms);
    println!("  CPU:   {:.3} ms", cpu_ms);
    println!(
        "  max |diff|: abs={:.2e}, rel={:.2e}",
        max_abs_diff, max_rel_diff
    );

    // F32 accumulation over k terms: 1e-5 relative is typical.
    assert!(max_rel_diff < 1e-3, "matmul rel error too large");
    println!("OK");
    Ok(())
}
