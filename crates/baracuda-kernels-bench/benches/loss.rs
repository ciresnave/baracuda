//! Loss-family throughput — baracuda vs PyTorch (Wave-1 proof increment).
//!
//! Phase 29 cross-impl bench for the loss family: `mse`, `l1`,
//! `cross_entropy`, `nll`. Each op is timed against the frozen PyTorch
//! baseline (`bench-baselines/pytorch_*.json`), looked up by the shared
//! `R{rows}_C{cols}` shape key produced by
//! `tools/refresh_pytorch_baseline.py`.
//!
//! Scope: **f32 only** — this is the pipeline proof (refresh → baseline →
//! bench → rollup → drift-assert). Dtype fanout (f16 / bf16) and the wider
//! op set are the Loss-20 wave.
//!
//! Reduction is `Mean` on both sides (baracuda `LossReduction::Mean`,
//! PyTorch default `reduction='mean'`) so the two time the same work: a
//! per-cell / per-row pass plus a tree reduction to a scalar. Timing is
//! data-independent (dense kernels), so deterministic fills suffice; the
//! fills are chosen non-degenerate (loss ≠ 0) so `assert_cell_live` proves
//! a live scalar rather than an all-zero one.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    CrossEntropyLossArgs, CrossEntropyLossDescriptor, CrossEntropyLossPlan, CrossEntropyTargetKind,
    ElementKind, L1LossArgs, L1LossDescriptor, L1LossPlan, LossReduction, MseLossArgs,
    MseLossDescriptor, MseLossPlan, NllLossArgs, NllLossDescriptor, NllLossPlan, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};
use baracuda_kernels_bench::{
    LOSS_COL_SWEEP, LOSS_ROW_SWEEP, PhaseTwentyNineRow, PytorchBaseline, append_csv_row,
    assert_cell_live, measure_median_ns, setup_device, time_with_events, warmup,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const BENCH_NAME: &str = "loss";

/// Shared tail: warm the just-populated scalar, assert it is live, record
/// the baracuda-vs-PyTorch CSV row, then run the criterion timing. `$run`
/// is the launch block (re-expanded per closure; borrows the device
/// buffers in scope). `$dev_y` is the output scalar (numel 1).
macro_rules! bench_tail {
    ($ctx:expr, $stream:expr, $group:expr, $op:literal, $shape:expr, $dev_y:expr, $baseline:expr, $run:block) => {{
        warmup(&$stream, || $run);
        // Liveness (NOT a reference): the warmup launch populated the scalar
        // loss; assert it is finite before timing it — a bench over a
        // NaN/garbage scalar is fast and meaningless.
        assert_cell_live(&format!(concat!($op, "/f32/{}"), $shape), &$dev_y, 1);
        let baracuda_ns = measure_median_ns(&$ctx, &$stream, 11, 50, || $run);
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: $op,
                shape: ($shape).clone(),
                dtype: "f32",
                baracuda_ns,
                reference_ns: None,
                reference: "",
                pytorch_ns: $baseline.and_then(|b| b.lookup($op, &$shape, "f32")),
            },
        );
        $group.bench_with_input(BenchmarkId::from_parameter(&$shape), &(), |bb, _| {
            bb.iter_custom(|iters| time_with_events(&$ctx, &$stream, iters, || $run))
        });
    }};
}

/// `mse` / `l1` share one API shape (per-cell `pred`/`target` → scalar).
macro_rules! percell_loss_bench {
    ($fn_name:ident, $op:literal, $Plan:ident, $Desc:ident, $Args:ident) => {
        fn $fn_name(c: &mut Criterion, baseline: Option<&PytorchBaseline>) {
            let (ctx, stream) = setup_device();
            let mut group = c.benchmark_group(concat!($op, "/f32"));
            for &rows in LOSS_ROW_SWEEP {
                for &cols in LOSS_COL_SWEEP {
                    let shape = format!("R{rows}_C{cols}");
                    let ishape = [rows, cols];
                    let st = contiguous_stride(ishape);
                    let numel = (rows * cols) as usize;

                    let dev_p = match DeviceBuffer::from_slice(&ctx, &vec![1.0_f32; numel]) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let dev_t = match DeviceBuffer::from_slice(&ctx, &vec![0.5_f32; numel]) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let mut dev_y: DeviceBuffer<f32> = match DeviceBuffer::zeros(&ctx, 1) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let desc = $Desc::<2> {
                        input_shape: ishape,
                        reduction: LossReduction::Mean,
                        element: ElementKind::F32,
                    };
                    let plan =
                        match $Plan::<f32, 2>::select(&stream, &desc, PlanPreference::default()) {
                            Ok(p) => p,
                            Err(_) => continue,
                        };
                    let mut dev_ws: DeviceBuffer<u8> =
                        match DeviceBuffer::zeros(&ctx, plan.workspace_size().max(1)) {
                            Ok(b) => b,
                            Err(_) => continue,
                        };
                    bench_tail!(ctx, stream, group, $op, shape, dev_y, baseline, {
                        plan.run(
                            &stream,
                            Workspace::Borrowed(dev_ws.as_slice_mut()),
                            $Args::<f32, 2> {
                                pred: TensorRef {
                                    data: dev_p.as_slice(),
                                    shape: ishape,
                                    stride: st,
                                },
                                target: TensorRef {
                                    data: dev_t.as_slice(),
                                    shape: ishape,
                                    stride: st,
                                },
                                out: TensorMut {
                                    data: dev_y.as_slice_mut(),
                                    shape: [1, 1],
                                    stride: [1, 1],
                                },
                            },
                        )
                        .expect(concat!("baracuda ", $op));
                    });
                }
            }
            group.finish();
        }
    };
}

percell_loss_bench!(
    bench_mse,
    "mse",
    MseLossPlan,
    MseLossDescriptor,
    MseLossArgs
);
percell_loss_bench!(bench_l1, "l1", L1LossPlan, L1LossDescriptor, L1LossArgs);

/// CrossEntropy: logits `[n_rows, class_extent]` + i64 class-index target.
fn bench_cross_entropy(c: &mut Criterion, baseline: Option<&PytorchBaseline>) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("cross_entropy/f32");
    for &rows in LOSS_ROW_SWEEP {
        for &cols in LOSS_COL_SWEEP {
            let (n_rows, class_extent) = (rows, cols);
            let shape = format!("R{rows}_C{cols}");
            let ishape = [n_rows, class_extent];
            let ist = contiguous_stride(ishape);
            let tshape = [n_rows];
            let tst = contiguous_stride(tshape);
            let numel = (n_rows * class_extent) as usize;

            let dev_inp = match DeviceBuffer::from_slice(&ctx, &vec![0.1_f32; numel]) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let host_t: Vec<i64> = (0..n_rows as usize)
                .map(|i| (i as i64) % (class_extent as i64))
                .collect();
            let dev_t = match DeviceBuffer::from_slice(&ctx, &host_t) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<f32> = match DeviceBuffer::zeros(&ctx, 1) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let desc = CrossEntropyLossDescriptor {
                n_rows,
                class_extent,
                reduction: LossReduction::Mean,
                target_kind: CrossEntropyTargetKind::ClassIndex,
                element: ElementKind::F32,
            };
            let plan = match CrossEntropyLossPlan::<f32>::select(
                &stream,
                &desc,
                PlanPreference::default(),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let mut dev_ws: DeviceBuffer<u8> =
                match DeviceBuffer::zeros(&ctx, plan.workspace_size().max(1)) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
            bench_tail!(
                ctx,
                stream,
                group,
                "cross_entropy",
                shape,
                dev_y,
                baseline,
                {
                    plan.run(
                        &stream,
                        Workspace::Borrowed(dev_ws.as_slice_mut()),
                        CrossEntropyLossArgs {
                            input: TensorRef {
                                data: dev_inp.as_slice(),
                                shape: ishape,
                                stride: ist,
                            },
                            target: Some(TensorRef {
                                data: dev_t.as_slice(),
                                shape: tshape,
                                stride: tst,
                            }),
                            soft_target: None,
                            out: TensorMut {
                                data: dev_y.as_slice_mut(),
                                shape: [1],
                                stride: [1],
                            },
                        },
                    )
                    .expect("baracuda cross_entropy");
                }
            );
        }
    }
    group.finish();
}

/// NLL: log-probabilities `[n_rows, class_extent]` + i64 class-index target.
fn bench_nll(c: &mut Criterion, baseline: Option<&PytorchBaseline>) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("nll/f32");
    for &rows in LOSS_ROW_SWEEP {
        for &cols in LOSS_COL_SWEEP {
            let (n_rows, class_extent) = (rows, cols);
            let shape = format!("R{rows}_C{cols}");
            let ishape = [n_rows, class_extent];
            let ist = contiguous_stride(ishape);
            let tshape = [n_rows];
            let tst = contiguous_stride(tshape);
            let numel = (n_rows * class_extent) as usize;

            // Input is log-probabilities (≤ 0); a constant -1.0 is a valid
            // log-prob and yields a finite non-zero NLL.
            let dev_inp = match DeviceBuffer::from_slice(&ctx, &vec![-1.0_f32; numel]) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let host_t: Vec<i64> = (0..n_rows as usize)
                .map(|i| (i as i64) % (class_extent as i64))
                .collect();
            let dev_t = match DeviceBuffer::from_slice(&ctx, &host_t) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<f32> = match DeviceBuffer::zeros(&ctx, 1) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let desc = NllLossDescriptor {
                n_rows,
                class_extent,
                reduction: LossReduction::Mean,
                element: ElementKind::F32,
            };
            let plan = match NllLossPlan::<f32>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let mut dev_ws: DeviceBuffer<u8> =
                match DeviceBuffer::zeros(&ctx, plan.workspace_size().max(1)) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
            bench_tail!(ctx, stream, group, "nll", shape, dev_y, baseline, {
                plan.run(
                    &stream,
                    Workspace::Borrowed(dev_ws.as_slice_mut()),
                    NllLossArgs {
                        input: TensorRef {
                            data: dev_inp.as_slice(),
                            shape: ishape,
                            stride: ist,
                        },
                        target: TensorRef {
                            data: dev_t.as_slice(),
                            shape: tshape,
                            stride: tst,
                        },
                        out: TensorMut {
                            data: dev_y.as_slice_mut(),
                            shape: [1],
                            stride: [1],
                        },
                    },
                )
                .expect("baracuda nll");
            });
        }
    }
    group.finish();
}

/// Top-level criterion entry - invoked by criterion_main!.
fn benches(c: &mut Criterion) {
    let baseline = PytorchBaseline::load_default();
    let baseline_ref = baseline.as_ref();
    bench_mse(c, baseline_ref);
    bench_l1(c, baseline_ref);
    bench_cross_entropy(c, baseline_ref);
    bench_nll(c, baseline_ref);
}

// `criterion_group!` expands into a `pub fn benches_grp` whose signature
// is fixed by the macro - can't doc-comment it directly, so suppress the
// workspace `missing_docs = deny` lint on the generated fn.
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}
criterion_main!(criterion_glue::benches_grp);
