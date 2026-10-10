//! `baracuda-bench run --report` — board #162 slice 2b.
//!
//! Runs a fixed kernel sweep under CUDA-event timing, computes the seven-number
//! [`TimingStats`] per cell, captures the machine state the run happened under,
//! and writes one self-describing [`BenchReport`] JSON file.
//!
//! ```text
//! baracuda-bench run --report out.json --machine-id cires-laptop-4070 \
//!     [--note "browsers closed, quiet run"] [--samples 101] [--inner 50]
//! ```
//!
//! GPU-touching: run it through `scripts/gpu-run.ps1 -Project baracuda`.
//!
//! A report from anyone else is untrusted data on receipt (design doc item 6);
//! this binary only produces reports, it never merges one into a baseline.
//!
//! The sweep is deliberately a small table ([`run_sweep`]) — adding a kernel is
//! adding a `cell::<T>` call, not changing the report format.

use std::process::ExitCode;

use baracuda_bench_stats::{BaselineKey, BenchReport, ReportEntry, compute_stats};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    PlanPreference, RMSNormArgs, RMSNormDescriptor, RMSNormPlan, TensorMut, TensorRef, Workspace,
    contiguous_stride,
};
use baracuda_kernels_bench::machine_state::capture_machine_state;
use baracuda_kernels_bench::{
    CROSS_HIDDEN_SWEEP, CROSS_SEQLEN_SWEEP, DEVICE_TIMING_LOCK, LiveScalar, assert_cell_live,
    setup_device, time_with_events, warmup,
};
use half::f16;

const DEFAULT_SAMPLES: usize = 101;
const DEFAULT_INNER: u64 = 50;

struct Options {
    report: std::path::PathBuf,
    machine_id: String,
    note: Option<String>,
    samples: usize,
    inner: u64,
}

const USAGE: &str = "usage: baracuda-bench run --report <out.json> --machine-id <id> \
[--note <text>] [--samples <n>] [--inner <n>]";

fn parse_args(args: &[String]) -> Result<Options, String> {
    if args.first().map(String::as_str) != Some("run") {
        return Err(format!("expected subcommand `run`\n{USAGE}"));
    }
    let (mut report, mut machine_id, mut note) = (None, None, None);
    let (mut samples, mut inner) = (DEFAULT_SAMPLES, DEFAULT_INNER);
    let mut it = args[1..].iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--report" => report = Some(value("--report")?),
            "--machine-id" => machine_id = Some(value("--machine-id")?),
            "--note" => note = Some(value("--note")?),
            "--samples" => {
                samples = value("--samples")?
                    .parse()
                    .map_err(|e| format!("--samples: {e}"))?;
            }
            "--inner" => {
                inner = value("--inner")?
                    .parse()
                    .map_err(|e| format!("--inner: {e}"))?;
            }
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    let machine_id = machine_id.ok_or_else(|| format!("--machine-id is required\n{USAGE}"))?;
    // The id ends up in a filename and a shared directory name; keep it to a
    // charset that cannot carry a path or a username-looking surprise.
    if machine_id.is_empty()
        || !machine_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err("--machine-id must be non-empty and only [A-Za-z0-9_-]".into());
    }
    if samples == 0 || inner == 0 {
        return Err("--samples and --inner must be >= 1".into());
    }
    Ok(Options {
        report: report
            .ok_or_else(|| format!("--report is required\n{USAGE}"))?
            .into(),
        machine_id,
        note,
        samples,
        inner,
    })
}

/// One RMSNorm cell: `rows x hidden`, dtype `T`. Returns the per-sample
/// per-launch times in ns, or `None` if the cell could not be set up (an
/// unsupported shape/dtype is skipped and said so, never recorded as zero).
fn rmsnorm_cell<T>(
    ctx: &Context,
    stream: &Stream,
    rows: i32,
    hidden: i32,
    fill: T,
    o: &Options,
) -> Option<Vec<f64>>
where
    T: baracuda_kernels::Element + Copy + 'static + LiveScalar,
{
    let numel = (rows * hidden) as usize;
    let dev_x = DeviceBuffer::from_slice(ctx, &vec![fill; numel]).ok()?;
    let dev_g = DeviceBuffer::from_slice(ctx, &vec![fill; hidden as usize]).ok()?;
    let mut dev_y: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, numel).ok()?;
    let mut dev_rms: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, rows as usize).ok()?;
    let desc = RMSNormDescriptor::<2> {
        input_shape: [rows, hidden],
        norm_axes_mask: 0b10,
        eps: 1e-5,
        has_gamma: true,
        element: T::KIND,
    };
    let plan = RMSNormPlan::<T, 2>::select(stream, &desc, PlanPreference::default()).ok()?;

    let (xs, rs, gs) = ([rows, hidden], [rows, 1], [hidden]);
    let (stx, st_rms, stg) = (
        contiguous_stride(xs),
        contiguous_stride(rs),
        contiguous_stride(gs),
    );
    let launch = |dev_y: &mut DeviceBuffer<T>, dev_rms: &mut DeviceBuffer<T>| {
        let args = RMSNormArgs::<T, 2> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: xs,
                stride: stx,
            },
            gamma: Some(TensorRef {
                data: dev_g.as_slice(),
                shape: gs,
                stride: stg,
            }),
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: xs,
                stride: stx,
            },
            rms: TensorMut {
                data: dev_rms.as_slice_mut(),
                shape: rs,
                stride: st_rms,
            },
        };
        plan.run(stream, Workspace::None, args)
            .expect("baracuda rmsnorm");
    };

    warmup(stream, || launch(&mut dev_y, &mut dev_rms));
    // A report over NaN/garbage output is fast and meaningless.
    assert_cell_live("rmsnorm (report)", &dev_y, numel);
    let mut out = Vec::with_capacity(o.samples);
    for _ in 0..o.samples {
        let d = time_with_events(ctx, stream, o.inner, || launch(&mut dev_y, &mut dev_rms));
        out.push(d.as_secs_f64() * 1e9 / o.inner as f64);
    }
    Some(out)
}

fn run_sweep(
    ctx: &Context,
    stream: &Stream,
    arch: &str,
    o: &Options,
) -> (Vec<ReportEntry>, Vec<String>) {
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    let mut record = |dtype: &str, rows: i32, hidden: i32, samples: Option<Vec<f64>>| {
        let shape_class = format!("R{rows}_H{hidden}");
        let Some(samples) = samples else {
            skipped.push(format!("rmsnorm/{dtype}/{shape_class}"));
            return;
        };
        let stats = compute_stats(&samples).expect("samples is non-empty (--samples >= 1)");
        entries.push(ReportEntry {
            key: BaselineKey {
                kernel: "rmsnorm".into(),
                dtype: dtype.into(),
                shape_class,
                target_arch: arch.into(),
                machine_id: o.machine_id.clone(),
            },
            stats,
        });
    };
    for &rows in CROSS_SEQLEN_SWEEP {
        for &hidden in CROSS_HIDDEN_SWEEP {
            let s = rmsnorm_cell::<f32>(ctx, stream, rows, hidden, 1.0f32, o);
            record("f32", rows, hidden, s);
            let s = rmsnorm_cell::<f16>(ctx, stream, rows, hidden, f16::from_f32(1.0), o);
            record("f16", rows, hidden, s);
        }
    }
    (entries, skipped)
}

fn run(o: &Options) -> Result<(), String> {
    // Two concurrent device-timing runs on one GPU give a wrong result, not a
    // slow one (see the lock's doc comment).
    let _guard = DEVICE_TIMING_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let (ctx, stream) = setup_device();
    let device = Device::get(0).map_err(|e| format!("Device::get(0): {e}"))?;
    let (major, minor) = device
        .compute_capability()
        .map_err(|e| format!("compute_capability: {e}"))?;
    let name = device.name().map_err(|e| format!("device name: {e}"))?;
    let toolkit = baracuda_driver::version()
        .map(|v| v.to_string())
        .map_err(|e| format!("cuda version: {e}"))?;
    let arch = format!("cuda:sm{major}{minor}");

    // Captured BEFORE the sweep so it describes the box as the run found it;
    // the sweep itself heats the GPU and would otherwise mask a noisy start.
    let machine =
        capture_machine_state(&o.machine_id, &name, [major as u32, minor as u32], &toolkit);

    let (entries, skipped) = run_sweep(&ctx, &stream, &arch, o);
    for s in &skipped {
        eprintln!("skipped (could not set up): {s}");
    }
    let report = BenchReport::new(machine, o.note.clone(), entries);
    report.validate().map_err(|e| e.to_string())?;
    let json = report.to_json().map_err(|e| e.to_string())?;
    std::fs::write(&o.report, json).map_err(|e| format!("write {}: {e}", o.report.display()))?;
    eprintln!(
        "wrote {} ({} entries, {} skipped)",
        o.report.display(),
        report.entries.len(),
        skipped.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args).and_then(|o| run(&o)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("baracuda-bench: {e}");
            ExitCode::FAILURE
        }
    }
}
