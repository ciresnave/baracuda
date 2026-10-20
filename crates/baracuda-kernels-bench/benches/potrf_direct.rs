//! Splits the Linalg `cholesky` gap vs PyTorch (ROADMAP "Linalg `cholesky` is
//! 4-6x slower than PyTorch") into its parts, on the same box, in the same run:
//!
//! * **library vs plan:** `CholeskyPlan::run` against a direct
//!   `cusolverDnSpotrf` call on the same buffers, stream and handle settings;
//! * **measurement discipline:** the bench column in `BENCHMARKS.md` times ONE
//!   call between two CUDA events with a host sync after each (so the span
//!   includes any gap while the host is still enqueueing), whereas the frozen
//!   PyTorch column (`tools/refresh_pytorch_baseline.py`) is the wall-clock of
//!   `INNER` back-to-back launches divided by `INNER`, with one sync at the end.
//!   Each launcher is therefore timed BOTH ways.
//!
//! Prints a table; writes no CSV and feeds no rollup. Needs a GPU; run it on a
//! quiet box under the `gpu-run` lock:
//!
//! `pwsh scripts/gpu-run.ps1 -Project baracuda -- cargo bench -p baracuda-kernels-bench --bench potrf_direct --features sm89`
//!
//! Input is the same SPD matrix as `benches/linalg.rs` (0.5 off-diagonal, `n` on
//! the diagonal), lower factor, f32, `n` in {256, 512}.
use std::cell::RefCell;
use std::time::Instant;

use baracuda_driver::{Context, Device, DeviceBuffer, Event, Stream, init};
use baracuda_kernels::{
    CholeskyArgs, CholeskyDescriptor, CholeskyPlan, ElementKind, PlanPreference, TensorMut,
    Workspace, contiguous_stride,
};
use baracuda_kernels_bench::measure_spread_restored;
use baracuda_kernels_sys::{
    CUBLAS_FILL_MODE_UPPER, cusolverDnCreate, cusolverDnHandle_t, cusolverDnSetStream,
    cusolverDnSpotrf, cusolverDnSpotrf_bufferSize,
};

/// Launches per timed batch in the pipelined (PyTorch-style) discipline.
const INNER: usize = 50;
/// Samples (batches) per cell.
const SAMPLES: usize = 21;

fn spd_host(n: usize) -> Vec<f32> {
    let mut a = vec![0.5_f32; n * n];
    for i in 0..n {
        a[i * n + i] = n as f32;
    }
    a
}

fn pct(sorted: &[f64], p: usize) -> f64 {
    sorted[(sorted.len() * p / 100).min(sorted.len() - 1)]
}

fn report(label: &str, mut v: Vec<f64>) {
    v.sort_by(f64::total_cmp);
    println!(
        "    {label:<44} median {:>9.1} us   min {:>9.1}   p90 {:>9.1}   max {:>9.1}   (n={})",
        pct(&v, 50) / 1e3,
        v[0] / 1e3,
        pct(&v, 90) / 1e3,
        v[v.len() - 1] / 1e3,
        v.len()
    );
}

/// Run `batch` `SAMPLES` times; each returns total ns for `INNER` launches.
fn pipelined(mut batch: impl FnMut() -> f64) -> Vec<f64> {
    (0..SAMPLES).map(|_| batch() / INNER as f64).collect()
}

fn one_size(ctx: &Context, stream: &Stream, n: i32) {
    let nu = n as usize;
    let host = spd_host(nu);
    println!("\n  cholesky f32 N{n} (lower):");

    // --- direct cuSOLVER handle + workspace ---------------------------------
    let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
    assert_eq!(
        unsafe { cusolverDnCreate(&mut handle) },
        0,
        "cusolverDnCreate"
    );
    assert_eq!(
        unsafe { cusolverDnSetStream(handle, stream.as_raw()) },
        0,
        "cusolverDnSetStream"
    );
    let mut lwork: i32 = 0;
    assert_eq!(
        unsafe {
            cusolverDnSpotrf_bufferSize(
                handle,
                CUBLAS_FILL_MODE_UPPER,
                n,
                core::ptr::null_mut(),
                n,
                &mut lwork,
            )
        },
        0
    );
    let ws: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, lwork.max(1) as usize).expect("ws");
    let mut info: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, 1).expect("info");

    // INNER distinct pristine inputs + INNER work buffers (potrf is in place, so
    // back-to-back launches need distinct data or every call after the first
    // would factor a non-SPD matrix and take the failure path).
    let pristine = DeviceBuffer::from_slice(ctx, &host).expect("pristine");
    let mut work: Vec<DeviceBuffer<f32>> = (0..INNER)
        .map(|_| DeviceBuffer::from_slice(ctx, &host).expect("work"))
        .collect();
    let single = RefCell::new(DeviceBuffer::from_slice(ctx, &host).expect("single"));

    let restore_all = |work: &Vec<DeviceBuffer<f32>>| {
        for w in work {
            pristine.copy_to_device_async(w, stream).expect("restore");
        }
        stream.synchronize().expect("sync");
    };

    // --- the plan -----------------------------------------------------------
    let desc = CholeskyDescriptor {
        matrix_size: n,
        batch_size: 1,
        lower: true,
        element: ElementKind::F32,
    };
    let plan = CholeskyPlan::<f32>::select(stream, &desc, PlanPreference::default()).expect("plan");
    let plan_ws_bytes = plan.query_workspace_size(stream).expect("plan ws");
    let mut plan_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, plan_ws_bytes.max(1)).expect("pws");
    let sh = [1, n, n];
    let st = contiguous_stride(sh);

    // warm both paths
    for _ in 0..10 {
        pristine
            .copy_to_device_async(&single.borrow(), stream)
            .unwrap();
        unsafe {
            cusolverDnSpotrf(
                handle,
                CUBLAS_FILL_MODE_UPPER,
                n,
                single.borrow().as_raw().0 as *mut f32,
                n,
                ws.as_raw().0 as *mut f32,
                lwork,
                info.as_raw().0 as *mut i32,
            );
        }
    }
    stream.synchronize().unwrap();

    // ---- per-call events (what BENCHMARKS.md times) ------------------------
    let direct_event = |s: &mut Vec<f64>| {
        let sp = measure_spread_restored(
            ctx,
            stream,
            SAMPLES,
            5,
            || {
                pristine
                    .copy_to_device_async(&single.borrow(), stream)
                    .expect("restore");
            },
            || unsafe {
                cusolverDnSpotrf(
                    handle,
                    CUBLAS_FILL_MODE_UPPER,
                    n,
                    single.borrow().as_raw().0 as *mut f32,
                    n,
                    ws.as_raw().0 as *mut f32,
                    lwork,
                    info.as_raw().0 as *mut i32,
                );
            },
        );
        s.extend([sp.min_ns, sp.median_ns, sp.p90_ns, sp.max_ns]);
    };
    let mut d = Vec::new();
    direct_event(&mut d);
    println!(
        "    {:<44} median {:>9.1} us   min {:>9.1}   p90 {:>9.1}   max {:>9.1}",
        "DIRECT potrf, per-call events",
        d[1] / 1e3,
        d[0] / 1e3,
        d[2] / 1e3,
        d[3] / 1e3
    );

    let sp = measure_spread_restored(
        ctx,
        stream,
        SAMPLES,
        5,
        || {
            pristine
                .copy_to_device_async(&single.borrow(), stream)
                .expect("restore");
        },
        || {
            let mut single_m = single.borrow_mut();
            let args = CholeskyArgs::<f32> {
                a: TensorMut::new(single_m.as_slice_mut(), sh, st).unwrap(),
                info: TensorMut::new(info.as_slice_mut(), [1], [1]).unwrap(),
            };
            plan.run(stream, Workspace::Borrowed(plan_ws.as_slice_mut()), args)
                .expect("plan run");
        },
    );
    println!(
        "    {:<44} median {:>9.1} us   min {:>9.1}   p90 {:>9.1}   max {:>9.1}",
        "PLAN run, per-call events (= BENCHMARKS.md)",
        sp.median_ns / 1e3,
        sp.min_ns / 1e3,
        sp.p90_ns / 1e3,
        sp.max_ns / 1e3
    );

    // ---- pipelined wall-clock (what the PyTorch column times) --------------
    let v = pipelined(|| {
        restore_all(&work);
        let t = Instant::now();
        for w in &work {
            unsafe {
                cusolverDnSpotrf(
                    handle,
                    CUBLAS_FILL_MODE_UPPER,
                    n,
                    w.as_raw().0 as *mut f32,
                    n,
                    ws.as_raw().0 as *mut f32,
                    lwork,
                    info.as_raw().0 as *mut i32,
                );
            }
        }
        stream.synchronize().unwrap();
        t.elapsed().as_nanos() as f64
    });
    report("DIRECT potrf, pipelined wall-clock x50", v);

    let v = pipelined(|| {
        restore_all(&work);
        let t = Instant::now();
        for w in work.iter_mut() {
            let args = CholeskyArgs::<f32> {
                a: TensorMut::new(w.as_slice_mut(), sh, st).unwrap(),
                info: TensorMut::new(info.as_slice_mut(), [1], [1]).unwrap(),
            };
            plan.run(stream, Workspace::Borrowed(plan_ws.as_slice_mut()), args)
                .expect("plan run");
        }
        stream.synchronize().unwrap();
        t.elapsed().as_nanos() as f64
    });
    report("PLAN run, pipelined wall-clock x50 (= PyTorch's)", v);

    // ---- host launch cost alone (enqueue only, no sync inside the span) ----
    let v = pipelined(|| {
        restore_all(&work);
        let t = Instant::now();
        for w in &work {
            unsafe {
                cusolverDnSpotrf(
                    handle,
                    CUBLAS_FILL_MODE_UPPER,
                    n,
                    w.as_raw().0 as *mut f32,
                    n,
                    ws.as_raw().0 as *mut f32,
                    lwork,
                    info.as_raw().0 as *mut i32,
                );
            }
        }
        let enqueue = t.elapsed().as_nanos() as f64;
        stream.synchronize().unwrap();
        enqueue
    });
    report("DIRECT potrf, host enqueue time only x50", v);

    // One more control: the same event-pair discipline around the 50-launch batch.
    let start = Event::new(ctx).unwrap();
    let end = Event::new(ctx).unwrap();
    let v = pipelined(|| {
        restore_all(&work);
        start.record(stream).unwrap();
        for w in &work {
            unsafe {
                cusolverDnSpotrf(
                    handle,
                    CUBLAS_FILL_MODE_UPPER,
                    n,
                    w.as_raw().0 as *mut f32,
                    n,
                    ws.as_raw().0 as *mut f32,
                    lwork,
                    info.as_raw().0 as *mut i32,
                );
            }
        }
        end.record(stream).unwrap();
        end.synchronize().unwrap();
        f64::from(Event::elapsed_time_ms(&start, &end).unwrap()) * 1e6
    });
    report("DIRECT potrf, GPU-event span x50 (device time)", v);
}

fn main() {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    println!("potrf_direct: library-vs-plan and discipline split (see file header)");
    for n in [256, 512] {
        one_size(&ctx, &stream, n);
    }
}
