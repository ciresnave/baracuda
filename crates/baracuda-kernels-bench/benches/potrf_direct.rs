//! Splits the Linalg `cholesky` gap vs PyTorch (ROADMAP "Linalg `cholesky` vs PyTorch")
//! into its parts, on the same box, in the same run:
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
/// Launches per sample in the per-call-events discipline (as `benches/linalg.rs`).
const EVENT_INNER: u64 = 5;

fn spd_host(n: usize) -> Vec<f32> {
    let mut a = vec![0.5_f32; n * n];
    for i in 0..n {
        a[i * n + i] = n as f32;
    }
    a
}

/// Print one row: label and median / min / p90 / max in microseconds.
fn row(label: &str, mut v: Vec<f64>) {
    v.sort_by(f64::total_cmp);
    let at = |p: usize| v[(v.len() * p / 100).min(v.len() - 1)] / 1e3;
    println!(
        "    {label:<50} median {:>9.1} us   min {:>9.1}   p90 {:>9.1}   max {:>9.1}   (n={})",
        at(50),
        v[0] / 1e3,
        at(90),
        v[v.len() - 1] / 1e3,
        v.len()
    );
}

/// Everything one matrix size needs, so each measurement is a short method.
struct Rig<'a> {
    ctx: &'a Context,
    stream: &'a Stream,
    n: i32,
    handle: cusolverDnHandle_t,
    lwork: i32,
    ws: DeviceBuffer<f32>,
    info: RefCell<DeviceBuffer<i32>>,
    pristine: DeviceBuffer<f32>,
    /// `INNER` distinct work buffers: potrf is in place, so back-to-back launches
    /// need distinct data or every call after the first factors a non-SPD matrix.
    work: Vec<RefCell<DeviceBuffer<f32>>>,
    plan: CholeskyPlan<f32>,
    plan_ws: RefCell<DeviceBuffer<u8>>,
}

impl<'a> Rig<'a> {
    fn new(ctx: &'a Context, stream: &'a Stream, n: i32) -> Self {
        let host = spd_host(n as usize);
        let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
        // SAFETY: `handle` is a valid out-pointer; the handle lives for the whole bench process.
        assert_eq!(
            unsafe { cusolverDnCreate(&mut handle) },
            0,
            "cusolverDnCreate"
        );
        // SAFETY: `handle` was just created and `stream` outlives it.
        assert_eq!(
            unsafe { cusolverDnSetStream(handle, stream.as_raw()) },
            0,
            "SetStream"
        );
        let mut lwork: i32 = 0;
        // SAFETY: bufferSize only reads `handle`/`n` and writes `lwork`; a null matrix pointer is allowed.
        let status = unsafe {
            cusolverDnSpotrf_bufferSize(
                handle,
                CUBLAS_FILL_MODE_UPPER,
                n,
                core::ptr::null_mut(),
                n,
                &mut lwork,
            )
        };
        assert_eq!(status, 0, "bufferSize");
        let desc = CholeskyDescriptor {
            matrix_size: n,
            batch_size: 1,
            lower: true,
            element: ElementKind::F32,
        };
        let plan =
            CholeskyPlan::<f32>::select(stream, &desc, PlanPreference::default()).expect("plan");
        let plan_ws_bytes = plan.query_workspace_size(stream).expect("plan ws").max(1);
        Self {
            ctx,
            stream,
            n,
            handle,
            lwork,
            ws: DeviceBuffer::zeros(ctx, lwork.max(1) as usize).expect("ws"),
            info: RefCell::new(DeviceBuffer::zeros(ctx, 1).expect("info")),
            pristine: DeviceBuffer::from_slice(ctx, &host).expect("pristine"),
            work: (0..INNER)
                .map(|_| RefCell::new(DeviceBuffer::from_slice(ctx, &host).expect("work")))
                .collect(),
            plan,
            plan_ws: RefCell::new(DeviceBuffer::zeros(ctx, plan_ws_bytes).expect("plan ws buf")),
        }
    }

    /// The only direct cuSOLVER factorization call: factor `buf` (n x n, f32) in place.
    fn potrf(&self, buf: &DeviceBuffer<f32>) {
        let info = self.info.borrow();
        // SAFETY: `buf` holds n*n f32, `ws` holds `lwork` f32 (from bufferSize), `info` holds one i32,
        // all live device buffers on `self.stream`, which `handle` is bound to; the data is SPD so
        // the factorization succeeds, and nothing else touches `buf` until the stream is synchronized.
        let status = unsafe {
            cusolverDnSpotrf(
                self.handle,
                CUBLAS_FILL_MODE_UPPER,
                self.n,
                buf.as_raw().0 as *mut f32,
                self.n,
                self.ws.as_raw().0 as *mut f32,
                self.lwork,
                info.as_raw().0 as *mut i32,
            )
        };
        assert_eq!(status, 0, "cusolverDnSpotrf");
    }

    fn plan_run(&self, buf: &mut DeviceBuffer<f32>) {
        let sh = [1, self.n, self.n];
        let st = contiguous_stride(sh);
        let mut info = self.info.borrow_mut();
        let args = CholeskyArgs::<f32> {
            a: TensorMut::new(buf.as_slice_mut(), sh, st).unwrap(),
            info: TensorMut::new(info.as_slice_mut(), [1], [1]).unwrap(),
        };
        let mut ws = self.plan_ws.borrow_mut();
        self.plan
            .run(self.stream, Workspace::Borrowed(ws.as_slice_mut()), args)
            .expect("plan run");
    }

    fn restore_all(&self) {
        for w in &self.work {
            self.pristine
                .copy_to_device_async(&w.borrow(), self.stream)
                .expect("restore");
        }
        self.stream.synchronize().expect("sync");
    }

    /// Per-call CUDA events with the restore fenced out (the `BENCHMARKS.md` method).
    /// Returns `[min, median, p90, max]` ns.
    fn per_call_events(&self, launch: impl FnMut()) -> [f64; 4] {
        let one = &self.work[0];
        let sp = measure_spread_restored(
            self.ctx,
            self.stream,
            SAMPLES,
            EVENT_INNER,
            || {
                self.pristine
                    .copy_to_device_async(&one.borrow(), self.stream)
                    .expect("restore");
            },
            launch,
        );
        [sp.min_ns, sp.median_ns, sp.p90_ns, sp.max_ns]
    }

    /// `SAMPLES` batches; each restores all buffers, then `batch` returns total ns for `INNER` launches.
    fn pipelined(&self, mut batch: impl FnMut() -> f64) -> Vec<f64> {
        (0..SAMPLES)
            .map(|_| {
                self.restore_all();
                batch() / INNER as f64
            })
            .collect()
    }

    /// Wall-clock over `INNER` launches + one sync (the PyTorch-column method).
    fn wall_clock(&self, mut launch_all: impl FnMut()) -> f64 {
        let t = Instant::now();
        launch_all();
        self.stream.synchronize().unwrap();
        t.elapsed().as_nanos() as f64
    }

    fn warm_up(&self) {
        for _ in 0..10 {
            self.pristine
                .copy_to_device_async(&self.work[0].borrow(), self.stream)
                .unwrap();
            self.potrf(&self.work[0].borrow());
        }
        self.stream.synchronize().unwrap();
    }

    fn print_events(&self, label: &str, e: [f64; 4]) {
        println!(
            "    {label:<50} median {:>9.1} us   min {:>9.1}   p90 {:>9.1}   max {:>9.1}",
            e[1] / 1e3,
            e[0] / 1e3,
            e[2] / 1e3,
            e[3] / 1e3
        );
    }

    fn report(&self) {
        println!("\n  cholesky f32 N{} (lower):", self.n);
        self.warm_up();
        let one = &self.work[0];
        let e = self.per_call_events(|| self.potrf(&one.borrow()));
        self.print_events("DIRECT potrf, per-call events", e);
        let e = self.per_call_events(|| self.plan_run(&mut one.borrow_mut()));
        self.print_events("PLAN run, per-call events (= BENCHMARKS.md)", e);
        self.report_pipelined();
    }

    fn report_pipelined(&self) {
        let direct_all = || self.work.iter().for_each(|w| self.potrf(&w.borrow()));
        let v = self.pipelined(|| self.wall_clock(direct_all));
        row("DIRECT potrf, pipelined wall-clock x50", v);
        let plan_all = || {
            self.work
                .iter()
                .for_each(|w| self.plan_run(&mut w.borrow_mut()))
        };
        let v = self.pipelined(|| self.wall_clock(plan_all));
        row("PLAN run, pipelined wall-clock x50 (= PyTorch's)", v);
        let v = self.pipelined(|| {
            let t = Instant::now();
            direct_all();
            let enqueue = t.elapsed().as_nanos() as f64;
            self.stream.synchronize().unwrap();
            enqueue
        });
        row("DIRECT potrf, host enqueue time only x50", v);
        let (start, end) = (Event::new(self.ctx).unwrap(), Event::new(self.ctx).unwrap());
        let v = self.pipelined(|| {
            start.record(self.stream).unwrap();
            direct_all();
            end.record(self.stream).unwrap();
            end.synchronize().unwrap();
            f64::from(Event::elapsed_time_ms(&start, &end).unwrap()) * 1e6
        });
        row("DIRECT potrf, GPU-event span x50 (device time)", v);
    }
}

fn main() {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    println!("potrf_direct: library-vs-plan and discipline split (see file header)");
    for n in [256, 512] {
        Rig::new(&ctx, &stream, n).report();
    }
}
