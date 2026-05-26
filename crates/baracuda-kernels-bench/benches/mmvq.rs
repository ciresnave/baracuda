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
    GgufBlockFormat, GgufMmvqArgs, GgufMmvqDescriptor, GgufMmvqPlan, PlanPreference, TensorMut,
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

fn mmvq_benches(c: &mut Criterion) {
    bench_mmvq::<f32>(c, "f32", 1.0_f32);
    bench_mmvq::<f16>(c, "f16", f16::ONE);
    bench_mmvq::<bf16>(c, "bf16", bf16::ONE);
}

criterion_group!(benches_grp, mmvq_benches);
criterion_main!(benches_grp);
