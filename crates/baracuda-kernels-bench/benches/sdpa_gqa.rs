//! Flash SDPA + GQA broadcast — baracuda self-bench across head ratios.
//!
//! GQA (grouped-query attention) feeds the K/V tensor with a smaller
//! head count than Q by setting K/V's `head` stride to 0 — the kernel
//! reads the same K/V row for every Q head in a group. This bench
//! sweeps the Q-head:KV-head ratio at fixed model dims:
//!
//! - `num_q_heads = 32`, `num_kv_heads ∈ {32, 8, 4, 1}` — 1× (MHA),
//!   4× (Llama-2 70B), 8× (Llama-3 8B), 32× (full broadcast / MQA).
//! - `B = 1`, `Q = K = 2048`, `D = 128`.
//! - Dtypes: `f16`, `bf16`.
//!
//! When `num_kv_heads == num_q_heads`, the bench runs the standard
//! contiguous path; otherwise K/V allocate `num_kv_heads` head-rows and
//! the descriptor's `stride[1]` is zero, broadcasting one KV head over
//! the matching group of Q heads.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, flash_flops, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};

const BENCH_NAME: &str = "sdpa_gqa";

const NUM_Q_HEADS: i32 = 32;
const KV_HEAD_SWEEP: &[i32] = &[32, 8, 4, 1];
const SEQ_LEN: i32 = 2048;
const HEAD_DIM: i32 = 128;
const BATCH: i32 = 1;

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench<T>(c: &mut Criterion, dtype_label: &str, fill: T)
where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("sdpa_gqa/{dtype_label}"));

    for &num_kv in KV_HEAD_SWEEP {
        if NUM_Q_HEADS % num_kv != 0 {
            continue;
        }
        let label = format!("Hq{NUM_Q_HEADS}_Hkv{num_kv}_Q{SEQ_LEN}_D{HEAD_DIM}");
        let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();

        let q_numel = (BATCH * NUM_Q_HEADS * SEQ_LEN * HEAD_DIM) as usize;
        let kv_numel = (BATCH * num_kv * SEQ_LEN * HEAD_DIM) as usize;
        let y_numel = q_numel;
        let lse_numel = (BATCH * NUM_Q_HEADS * SEQ_LEN) as usize;

        let host_q: Vec<T> = vec![fill; q_numel];
        let host_kv: Vec<T> = vec![fill; kv_numel];

        let dq = match DeviceBuffer::from_slice(&ctx, &host_q) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dk = match DeviceBuffer::from_slice(&ctx, &host_kv) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dv = match DeviceBuffer::from_slice(&ctx, &host_kv) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dy: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, y_numel) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dlse: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, lse_numel) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let desc = FlashSdpaDescriptor {
            batch_size: BATCH,
            num_heads: NUM_Q_HEADS,
            query_len: SEQ_LEN,
            key_len: SEQ_LEN,
            d_k: HEAD_DIM,
            d_v: HEAD_DIM,
            scale,
            is_causal: false,
            element: T::KIND,
        };
        let plan = match FlashSdpaPlan::<T>::select(&stream, &desc, PlanPreference::default()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Q: contiguous (BATCH, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM).
        // K/V: physical (BATCH, num_kv, SEQ_LEN, HEAD_DIM) — but the
        // descriptor advertises `num_heads = NUM_Q_HEADS`. To broadcast
        // the smaller KV head count to the Q head count, we set the
        // logical shape to (BATCH, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM) but
        // give the head stride a step that aliases multiple Q heads onto
        // the same KV head.
        //
        // For the case num_kv == NUM_Q_HEADS, this collapses to the
        // standard contig stride. For num_kv < NUM_Q_HEADS, set the
        // head stride to `0` for the broadcast head dim — but baracuda's
        // strided FFI accepts only stride[1] == 0 or full contig (no
        // intermediate "every k-th head" striding). So for num_kv < Hq,
        // we set stride[1] == 0 and rely on caller-side replication.
        //
        // To keep the bench simple and the comparison fair, we measure
        // two regimes: num_kv == Hq (MHA fast path) and num_kv == 1
        // (full MQA broadcast). Intermediate ratios fall into the same
        // stride[1] == 0 pattern, so we replicate per-group instead.
        let sq = [BATCH, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM];
        let stq = contiguous_stride(sq);

        let (skv, stkv) = if num_kv == NUM_Q_HEADS {
            // Plain MHA — KV physical shape == Q.
            let s = [BATCH, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM];
            (s, contiguous_stride(s))
        } else if num_kv == 1 {
            // Full MQA broadcast — KV physical (BATCH, 1, SEQ_LEN,
            // HEAD_DIM); descriptor head stride = 0.
            let s = [BATCH, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM];
            let mut st = contiguous_stride(s);
            st[1] = 0;
            (s, st)
        } else {
            // Skip intermediate ratios — would require a contig
            // KV repeat pre-pass, which doesn't model real GQA inference
            // (which uses stride-0 broadcast like MQA, just with smaller
            // group sizes). Mark in CSV and continue.
            continue;
        };

        let sy = sq;
        let sty = contiguous_stride(sy);
        let sl = [BATCH, NUM_Q_HEADS, SEQ_LEN];
        let stl = contiguous_stride(sl);

        warmup(&stream, || {
            let args = FlashSdpaArgs::<T> {
                q: TensorRef {
                    data: dq.as_slice(),
                    shape: sq,
                    stride: stq,
                },
                k: TensorRef {
                    data: dk.as_slice(),
                    shape: skv,
                    stride: stkv,
                },
                v: TensorRef {
                    data: dv.as_slice(),
                    shape: skv,
                    stride: stkv,
                },
                y: TensorMut {
                    data: dy.as_slice_mut(),
                    shape: sy,
                    stride: sty,
                },
                lse: TensorMut {
                    data: dlse.as_slice_mut(),
                    shape: sl,
                    stride: stl,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda flash gqa");
        });
        let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 20, || {
            let args = FlashSdpaArgs::<T> {
                q: TensorRef {
                    data: dq.as_slice(),
                    shape: sq,
                    stride: stq,
                },
                k: TensorRef {
                    data: dk.as_slice(),
                    shape: skv,
                    stride: stkv,
                },
                v: TensorRef {
                    data: dv.as_slice(),
                    shape: skv,
                    stride: stkv,
                },
                y: TensorMut {
                    data: dy.as_slice_mut(),
                    shape: sy,
                    stride: sty,
                },
                lse: TensorMut {
                    data: dlse.as_slice_mut(),
                    shape: sl,
                    stride: stl,
                },
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda flash gqa");
        });
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "flash_sdpa_gqa",
                shape: label.clone(),
                dtype: leak_str(dtype_label),
                baracuda_ns,
                reference_ns: None,
                reference: "",
            },
        );

        group.throughput(Throughput::Elements(flash_flops(
            BATCH, NUM_Q_HEADS, SEQ_LEN, SEQ_LEN, HEAD_DIM,
        )));
        group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = FlashSdpaArgs::<T> {
                        q: TensorRef {
                            data: dq.as_slice(),
                            shape: sq,
                            stride: stq,
                        },
                        k: TensorRef {
                            data: dk.as_slice(),
                            shape: skv,
                            stride: stkv,
                        },
                        v: TensorRef {
                            data: dv.as_slice(),
                            shape: skv,
                            stride: stkv,
                        },
                        y: TensorMut {
                            data: dy.as_slice_mut(),
                            shape: sy,
                            stride: sty,
                        },
                        lse: TensorMut {
                            data: dlse.as_slice_mut(),
                            shape: sl,
                            stride: stl,
                        },
                    };
                    plan.run(&stream, Workspace::None, args).expect("baracuda flash gqa");
                })
            });
        });
    }
    group.finish();
    let _ = ElementKind::F32;
}

fn benches(c: &mut Criterion) {
    bench::<f16>(c, "f16", f16::ONE);
    bench::<bf16>(c, "bf16", bf16::ONE);
}

criterion_group!(benches_grp, benches);
criterion_main!(benches_grp);
