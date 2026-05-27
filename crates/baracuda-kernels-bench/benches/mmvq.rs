//! GGUF MMVQ throughput — baracuda self-bench across block formats,
//! shapes, and activation dtypes.
//!
//! No reference equivalent: cuBLAS / cuDNN don't expose a GGUF MMVQ
//! primitive. This bench exists as the **objective measurement basis**
//! for Phase 27's deferred multi-M MMVQ port and the Tier A k-quant
//! micro-optimizations.
//!
//! Sweeps:
//! - `nrows × ncols ∈ {(4096, 4096), (11008, 4096), (32000, 4096)}`.
//!   These mirror the three matmul shapes in Llama-2 7B's transformer
//!   layers: Q/K/V projections (square 4096), FFN up_proj (11008×4096),
//!   and the LM head (32000×4096).
//! - Block formats: `{Q4_0, Q4_K, Q6_K, Q8_0}`. Covers two 4-bit, one
//!   6-bit k-quant, and one 8-bit. Bench `ncols = 4096` cleanly divides
//!   every k-quant block size (256).
//! - Activations: `{f32, f16, bf16}`. Phase 18 fanout.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::quantize::gguf::mmvq::GgufMmvqActivation;
use baracuda_kernels::{
    GgufBlockFormat, GgufMmvqArgs, GgufMmvqDescriptor, GgufMmvqMultiMArgs,
    GgufMmvqMultiMDescriptor, GgufMmvqMultiMPlan, GgufMmvqPlan, PlanPreference, TensorMut,
    TensorRef, Workspace, U8,
};
use baracuda_kernels_bench::{
    append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, CROSS_MMVQ_FORMATS, CROSS_MMVQ_SHAPES,
};
use baracuda_kernels_types::contiguous_stride;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::{bf16, f16};

const BENCH_NAME: &str = "mmvq";

fn block_label(fmt: GgufBlockFormat) -> &'static str {
    match fmt {
        GgufBlockFormat::Q4_0 => "q4_0",
        GgufBlockFormat::Q4_1 => "q4_1",
        GgufBlockFormat::Q5_0 => "q5_0",
        GgufBlockFormat::Q5_1 => "q5_1",
        GgufBlockFormat::Q8_0 => "q8_0",
        GgufBlockFormat::Q2K => "q2_k",
        GgufBlockFormat::Q3K => "q3_k",
        GgufBlockFormat::Q4K => "q4_k",
        GgufBlockFormat::Q5K => "q5_k",
        GgufBlockFormat::Q6K => "q6_k",
        GgufBlockFormat::Q8K => "q8_k",
        _ => "unknown",
    }
}

/// Generic MMVQ bench body — one (block format × activation dtype) group.
fn bench_mmvq<T>(c: &mut Criterion, act_label: &str, fill: T)
where
    T: GgufMmvqActivation + Copy + 'static,
{
    let (ctx, stream) = setup_device();

    for &fmt in CROSS_MMVQ_FORMATS {
        let fmt_lbl = block_label(fmt);
        let mut group = c.benchmark_group(format!("mmvq/{act_label}/{fmt_lbl}"));

        for &(nrows, ncols) in CROSS_MMVQ_SHAPES {
            let bs = fmt.block_size() as i32;
            if ncols % bs != 0 {
                continue;
            }
            let blocks_per_row = ncols / bs;
            let weight_bytes = (nrows as usize) * (blocks_per_row as usize) * fmt.type_size();
            let shape = format!("N{nrows}_C{ncols}");

            let host_w: Vec<U8> = vec![U8(0x10); weight_bytes];
            let host_act: Vec<T> = vec![fill; ncols as usize];
            let dev_w = match DeviceBuffer::from_slice(&ctx, &host_w) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_act = match DeviceBuffer::from_slice(&ctx, &host_act) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_out: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, nrows as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let desc = GgufMmvqDescriptor {
                nrows,
                ncols,
                block_format: fmt,
                w_start_byte_offset: 0,
            };
            let plan = match GgufMmvqPlan::<T>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let w_shape = [weight_bytes as i32];
            let act_shape = [ncols];
            let out_shape = [nrows];
            let stw = contiguous_stride(w_shape);
            let sta = contiguous_stride(act_shape);
            let sto = contiguous_stride(out_shape);

            warmup(&stream, || {
                let args = GgufMmvqArgs::<T> {
                    weight: TensorRef {
                        data: dev_w.as_slice(),
                        shape: w_shape,
                        stride: stw,
                    },
                    activation: TensorRef {
                        data: dev_act.as_slice(),
                        shape: act_shape,
                        stride: sta,
                    },
                    output: TensorMut {
                        data: dev_out.as_slice_mut(),
                        shape: out_shape,
                        stride: sto,
                    },
                };
                plan.run(&stream, Workspace::None, args).expect("mmvq warmup");
            });

            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 100, || {
                let args = GgufMmvqArgs::<T> {
                    weight: TensorRef {
                        data: dev_w.as_slice(),
                        shape: w_shape,
                        stride: stw,
                    },
                    activation: TensorRef {
                        data: dev_act.as_slice(),
                        shape: act_shape,
                        stride: sta,
                    },
                    output: TensorMut {
                        data: dev_out.as_slice_mut(),
                        shape: out_shape,
                        stride: sto,
                    },
                };
                plan.run(&stream, Workspace::None, args).expect("mmvq run");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "mmvq",
                    shape: format!("{}_{}", block_label(fmt), shape),
                    dtype: leak_str(act_label),
                    baracuda_ns,
                    reference_ns: None,
                    reference: "",
                },
            );

            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = GgufMmvqArgs::<T> {
                            weight: TensorRef {
                                data: dev_w.as_slice(),
                                shape: w_shape,
                                stride: stw,
                            },
                            activation: TensorRef {
                                data: dev_act.as_slice(),
                                shape: act_shape,
                                stride: sta,
                            },
                            output: TensorMut {
                                data: dev_out.as_slice_mut(),
                                shape: out_shape,
                                stride: sto,
                            },
                        };
                        plan.run(&stream, Workspace::None, args).expect("mmvq run");
                    })
                });
            });
        }
        group.finish();
    }
}

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

// =============================================================================
// Phase 33 + 34 — multi-M MMVQ bench.
//
// Compares the `GgufMmvqMultiMPlan` (Q8_1 staging + multi-M dot)
// against the per-token loop over `GgufMmvqPlan`. The latter re-reads
// the full weight tensor M times; the former reads it once and
// amortizes across all M activation vectors.
//
// M ∈ {1, 2, 4, 8}.
// Shapes: CROSS_MMVQ_SHAPES (Llama-2 7B layer matmul shapes).
// Phase 33: Q8_0 only.
// Phase 34: extended to Q4_0/Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K.
// =============================================================================

const MULTIM_VALUES: &[i32] = &[1, 2, 4, 8];

const MULTIM_FORMATS: &[GgufBlockFormat] = &[
    GgufBlockFormat::Q8_0,
    GgufBlockFormat::Q4_0,
    GgufBlockFormat::Q4_1,
    GgufBlockFormat::Q5_0,
    GgufBlockFormat::Q5_1,
    GgufBlockFormat::Q2K,
    GgufBlockFormat::Q3K,
    GgufBlockFormat::Q4K,
    GgufBlockFormat::Q5K,
    GgufBlockFormat::Q6K,
];

fn bench_mmvq_multim_for(c: &mut Criterion, fmt: GgufBlockFormat) {
    let (ctx, stream) = setup_device();
    let fmt_lbl = block_label(fmt);

    let mut group = c.benchmark_group(format!("mmvq_multim/f32/{fmt_lbl}"));

    for &(nrows, ncols) in CROSS_MMVQ_SHAPES {
        let bs = fmt.block_size() as i32;
        if ncols % bs != 0 {
            continue;
        }
        let blocks_per_row = ncols / bs;
        let weight_bytes = (nrows as usize) * (blocks_per_row as usize) * fmt.type_size();

        let host_w: Vec<U8> = vec![U8(0x10); weight_bytes];
        let dev_w = match DeviceBuffer::from_slice(&ctx, &host_w) {
            Ok(b) => b,
            Err(_) => continue,
        };

        for &m in MULTIM_VALUES {
            let shape = format!("M{m}_N{nrows}_C{ncols}");

            // Allocate activations [M, ncols] and output [M, nrows].
            let host_act: Vec<f32> = vec![0.5_f32; (m * ncols) as usize];
            let dev_act = match DeviceBuffer::from_slice(&ctx, &host_act) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_out: DeviceBuffer<f32> =
                match DeviceBuffer::zeros(&ctx, (m * nrows) as usize) {
                    Ok(b) => b,
                    Err(_) => continue,
                };

            // ----- baracuda multi-M path -----
            let desc = GgufMmvqMultiMDescriptor {
                nrows,
                ncols,
                m,
                block_format: fmt,
                w_start_byte_offset: 0,
            };
            let plan_multim = match GgufMmvqMultiMPlan::<f32>::select(
                &stream,
                &desc,
                PlanPreference::default(),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let ws_bytes = plan_multim.workspace_size();
            let mut dev_ws: DeviceBuffer<u8> = match DeviceBuffer::zeros(&ctx, ws_bytes) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let w_shape = [weight_bytes as i32];
            let act_shape = [m, ncols];
            let out_shape = [m, nrows];
            let stw = contiguous_stride(w_shape);
            let sta = contiguous_stride(act_shape);
            let sto = contiguous_stride(out_shape);

            warmup(&stream, || {
                let args = GgufMmvqMultiMArgs::<f32> {
                    weight: TensorRef {
                        data: dev_w.as_slice(),
                        shape: w_shape,
                        stride: stw,
                    },
                    activations: TensorRef {
                        data: dev_act.as_slice(),
                        shape: act_shape,
                        stride: sta,
                    },
                    output: TensorMut {
                        data: dev_out.as_slice_mut(),
                        shape: out_shape,
                        stride: sto,
                    },
                };
                plan_multim
                    .run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
                    .expect("multim warmup");
            });

            let multim_ns = measure_median_ns(&ctx, &stream, 11, 100, || {
                let args = GgufMmvqMultiMArgs::<f32> {
                    weight: TensorRef {
                        data: dev_w.as_slice(),
                        shape: w_shape,
                        stride: stw,
                    },
                    activations: TensorRef {
                        data: dev_act.as_slice(),
                        shape: act_shape,
                        stride: sta,
                    },
                    output: TensorMut {
                        data: dev_out.as_slice_mut(),
                        shape: out_shape,
                        stride: sto,
                    },
                };
                plan_multim
                    .run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
                    .expect("multim run");
            });

            // ----- baseline: per-token loop over the M=1 GgufMmvqPlan -----
            let desc_baseline = GgufMmvqDescriptor {
                nrows,
                ncols,
                block_format: fmt,
                w_start_byte_offset: 0,
            };
            let plan_baseline = match GgufMmvqPlan::<f32>::select(
                &stream,
                &desc_baseline,
                PlanPreference::default(),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };
            // Allocate dedicated per-row activation + output buffers for the
            // baseline path; we want each M=1 launch to dispatch independently
            // so the per-token loop dominates the measured time. Using the
            // same buffer M times is fine — the kernel re-reads the weight
            // each call regardless.
            let host_act_1: Vec<f32> = vec![0.5_f32; ncols as usize];
            let dev_act_1 = match DeviceBuffer::from_slice(&ctx, &host_act_1) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_out_1: DeviceBuffer<f32> =
                match DeviceBuffer::zeros(&ctx, nrows as usize) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
            let act1_shape = [ncols];
            let out1_shape = [nrows];
            let sta1 = contiguous_stride(act1_shape);
            let sto1 = contiguous_stride(out1_shape);

            let baseline_ns = measure_median_ns(&ctx, &stream, 11, 100, || {
                // Per-token loop: M calls to the M=1 plan, each re-reading
                // the full weight tensor from gmem. This is exactly the
                // cost the multi-M path amortizes away.
                for _mi in 0..(m as usize) {
                    let args = GgufMmvqArgs::<f32> {
                        weight: TensorRef {
                            data: dev_w.as_slice(),
                            shape: w_shape,
                            stride: stw,
                        },
                        activation: TensorRef {
                            data: dev_act_1.as_slice(),
                            shape: act1_shape,
                            stride: sta1,
                        },
                        output: TensorMut {
                            data: dev_out_1.as_slice_mut(),
                            shape: out1_shape,
                            stride: sto1,
                        },
                    };
                    plan_baseline
                        .run(&stream, Workspace::None, args)
                        .expect("baseline run");
                }
            });

            // CSV row: format = "<qtype>_<M>_<shape>", reference = M=1 baseline ns.
            append_csv_row(
                "mmvq_multim",
                &PhaseTwentyNineRow {
                    op: "mmvq_multim",
                    shape: shape.clone(),
                    dtype: "f32",
                    baracuda_ns: multim_ns,
                    reference_ns: Some(baseline_ns),
                    reference: "mmvq_per_token_loop",
                },
            );

            eprintln!(
                "mmvq_multim {fmt_lbl} f32 {shape}: multim {multim_ns} ns  vs  per-token loop {baseline_ns} ns  → {:.2}× speedup",
                baseline_ns as f64 / multim_ns as f64
            );

            group.bench_with_input(
                BenchmarkId::new("multim", &shape),
                &(),
                |bb, _| {
                    bb.iter_custom(|iters| {
                        time_with_events(&ctx, &stream, iters, || {
                            let args = GgufMmvqMultiMArgs::<f32> {
                                weight: TensorRef {
                                    data: dev_w.as_slice(),
                                    shape: w_shape,
                                    stride: stw,
                                },
                                activations: TensorRef {
                                    data: dev_act.as_slice(),
                                    shape: act_shape,
                                    stride: sta,
                                },
                                output: TensorMut {
                                    data: dev_out.as_slice_mut(),
                                    shape: out_shape,
                                    stride: sto,
                                },
                            };
                            plan_multim
                                .run(
                                    &stream,
                                    Workspace::Borrowed(dev_ws.as_slice_mut()),
                                    args,
                                )
                                .expect("multim run");
                        })
                    });
                },
            );
        }
    }
    group.finish();
}

fn mmvq_benches(c: &mut Criterion) {
    bench_mmvq::<f32>(c, "f32", 1.0_f32);
    bench_mmvq::<f16>(c, "f16", f16::ONE);
    bench_mmvq::<bf16>(c, "bf16", bf16::ONE);
    // Phase 33 + 34: multi-M MMVQ sweep across all supported GGUF formats.
    for &fmt in MULTIM_FORMATS {
        bench_mmvq_multim_for(c, fmt);
    }
}

criterion_group!(benches_grp, mmvq_benches);
criterion_main!(benches_grp);
