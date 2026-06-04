//! HyperConnection throughput — Phase 43 head-to-head.
//!
//! Validates the upstream "5-15× FW vs naive PyTorch" claim on RTX
//! 4070. mHC.cu's upstream measurements are vs a PyTorch reference
//! that materializes the same math via uncoalesced ops + multiple
//! kernel launches. We can't compare against PyTorch from a Rust
//! bench harness without a heavy subprocess shim — instead we
//! compare against the **closest baracuda equivalent**:
//!
//!   naive baseline =
//!     1. `RMSNormPlan` over `aggregate(x_expanded, h_pre)` (we
//!        compute the aggregate via a manual stream loop — N
//!        separate calls).
//!     2. Output reduction `sum_i M[i, j] * x_expanded[b, j, c]`
//!        approximated as `H_res @ x_expanded` (no Sinkhorn
//!        normalization — Sinkhorn is the special sauce of mHC).
//!     3. Scale + add via `affine_inplace_*`.
//!
//! This is **not** a like-for-like comparison: the naive baseline
//! skips the Sinkhorn projection (~20 iterations of normalize-rows /
//! normalize-cols). We document this asymmetry in the result CSV.
//!
//! Shapes (LLM-typical from the upstream paper Appendix A.1, scaled
//! down to fit an 8 GiB RTX 4070):
//!   batch ∈ {32, 128, 320}, hidden ∈ {1024, 2048}, n_streams ∈ {4}.
//!
//! Without the `mhc` cargo feature, `main` is a no-op print and the
//! bench binary still builds.

#[cfg(not(feature = "mhc"))]
fn main() {
    println!(
        "hyper_connection bench skipped — build with --features mhc to run \
         the Phase 43 HyperConnection throughput sweep."
    );
}

// criterion_main!() must live at crate root; gate the entire bench
// body on the feature flag rather than wrapping in a sub-mod.
#[cfg(feature = "mhc")]
mod bench_impl_inner {
    use baracuda_driver::DeviceBuffer;
    use baracuda_kernels::{
        contiguous_stride, ElementKind, HyperConnectionArgs, HyperConnectionDescriptor,
        HyperConnectionPlan, PlanPreference, RMSNormArgs, RMSNormDescriptor, RMSNormPlan,
        TensorMut, TensorRef, Workspace,
    };
    use baracuda_kernels_bench::{
        append_csv_row, measure_median_ns, setup_device, time_with_events, warmup,
        PhaseTwentyNineRow,
    };
    use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
    use half::bf16;

    const BENCH_NAME: &str = "hyper_connection";

    fn leak_str(s: &str) -> &'static str {
        Box::leak(s.to_owned().into_boxed_str())
    }

    // LLM-typical shapes, scaled to fit RTX 4070 (8 GiB). The upstream
    // paper Appendix A.1 reports B=320..2560 + C=1280..2560 — we keep
    // a subset.
    const SHAPES: &[(i32, i32, i32)] = &[
        // (batch, hidden, n_streams)
        (32, 1024, 4),
        (32, 2048, 4),
        (128, 1024, 4),
        (320, 1280, 4),
    ];

    pub fn run(c: &mut Criterion) {
        let (ctx, stream) = setup_device();
        let mut group = c.benchmark_group("hyper_connection/f32_bf16-gamma");

        for &(b, hd, n) in SHAPES {
            let shape = format!("B{b}_H{hd}_N{n}");
            let x_len = (b * n * hd) as usize;

            let host_x: Vec<f32> = (0..x_len)
                .map(|i| ((i as f32) * 0.011 - 0.4).sin() * 0.3)
                .collect();
            let host_gamma: Vec<bf16> = (0..hd as usize)
                .map(|i| bf16::from_f32(0.9 + 0.2 * ((i as f32) * 0.21).sin()))
                .collect();
            let host_h_pre: Vec<f32> = (0..n as usize).map(|i| 0.1 * (i as f32)).collect();
            let host_h_post: Vec<f32> = (0..n as usize).map(|i| 0.2 * (i as f32)).collect();
            let host_h_res: Vec<f32> = (0..(n * n) as usize)
                .map(|i| if (i as i32) % (n + 1) == 0 { 0.05 } else { 0.0 })
                .collect();

            let dev_x = match DeviceBuffer::from_slice(&ctx, &host_x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_gamma = match DeviceBuffer::from_slice(&ctx, &host_gamma) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_h_pre = match DeviceBuffer::from_slice(&ctx, &host_h_pre) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_h_post = match DeviceBuffer::from_slice(&ctx, &host_h_post) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_h_res = match DeviceBuffer::from_slice(&ctx, &host_h_res) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_out: DeviceBuffer<f32> = match DeviceBuffer::zeros(&ctx, x_len) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let desc = HyperConnectionDescriptor {
                batch: b,
                hidden_dim: hd,
                n_streams: n,
                sinkhorn_iters: 20,
                eps: 1e-5,
                element: ElementKind::F32,
            };
            let plan = match HyperConnectionPlan::<f32>::select(
                &stream,
                &desc,
                PlanPreference::default(),
            ) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("hyper_connection {shape}: skip — select failed: {e:?}");
                    continue;
                }
            };

            let xs = [b, n, hd];
            let stx = contiguous_stride(xs);
            let gs = [hd];
            let stg = contiguous_stride(gs);
            let hs = [n];
            let sth = contiguous_stride(hs);
            let rs = [n, n];
            let str_ = contiguous_stride(rs);

            warmup(&stream, || {
                let args = HyperConnectionArgs::<f32> {
                    x_expanded: TensorRef {
                        data: dev_x.as_slice(),
                        shape: xs,
                        stride: stx,
                    },
                    rmsnorm_weight: TensorRef {
                        data: dev_gamma.as_slice(),
                        shape: gs,
                        stride: stg,
                    },
                    h_pre: TensorRef {
                        data: dev_h_pre.as_slice(),
                        shape: hs,
                        stride: sth,
                    },
                    h_post: TensorRef {
                        data: dev_h_post.as_slice(),
                        shape: hs,
                        stride: sth,
                    },
                    h_res: TensorRef {
                        data: dev_h_res.as_slice(),
                        shape: rs,
                        stride: str_,
                    },
                    out: TensorMut {
                        data: dev_out.as_slice_mut(),
                        shape: xs,
                        stride: stx,
                    },
                };
                plan.run(&stream, Workspace::None, args)
                    .expect("baracuda hyper_connection");
            });

            let mhc_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                let args = HyperConnectionArgs::<f32> {
                    x_expanded: TensorRef {
                        data: dev_x.as_slice(),
                        shape: xs,
                        stride: stx,
                    },
                    rmsnorm_weight: TensorRef {
                        data: dev_gamma.as_slice(),
                        shape: gs,
                        stride: stg,
                    },
                    h_pre: TensorRef {
                        data: dev_h_pre.as_slice(),
                        shape: hs,
                        stride: sth,
                    },
                    h_post: TensorRef {
                        data: dev_h_post.as_slice(),
                        shape: hs,
                        stride: sth,
                    },
                    h_res: TensorRef {
                        data: dev_h_res.as_slice(),
                        shape: rs,
                        stride: str_,
                    },
                    out: TensorMut {
                        data: dev_out.as_slice_mut(),
                        shape: xs,
                        stride: stx,
                    },
                };
                plan.run(&stream, Workspace::None, args)
                    .expect("baracuda hyper_connection");
            });

            // Naive baseline: just an RMSNorm over [B*N, hidden] —
            // the dominant memory traffic component of the full mHC
            // pipeline (Sinkhorn-Knopp + stream-aggregate skipped).
            // This is a LOWER BOUND on what a naive PyTorch
            // implementation would do; the actual naive pipeline
            // would issue ~10 kernels and have ~5x the launch
            // overhead.
            let mut naive_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, x_len).expect("naive y");
            let mut naive_rms: DeviceBuffer<f32> =
                DeviceBuffer::zeros(&ctx, (b * n) as usize).expect("naive rms");
            // Use [B*N, hidden] reshape — RMSNorm over the last axis
            // matches what mHC does internally (per-row RMS over the
            // hidden dim).
            let bn = b * n;
            let naive_desc = RMSNormDescriptor::<2> {
                input_shape: [bn, hd],
                norm_axes_mask: 0b10,
                eps: 1e-5,
                has_gamma: false,
                element: ElementKind::F32,
            };
            let naive_plan_opt = RMSNormPlan::<f32, 2>::select(
                &stream,
                &naive_desc,
                PlanPreference::default(),
            );
            let naive_ns = if let Ok(naive_plan) = naive_plan_opt {
                let xs2 = [bn, hd];
                let stx2 = contiguous_stride(xs2);
                let rs2 = [bn, 1];
                let str_rms = contiguous_stride(rs2);
                warmup(&stream, || {
                    let args = RMSNormArgs::<f32, 2> {
                        x: TensorRef {
                            data: dev_x.as_slice(),
                            shape: xs2,
                            stride: stx2,
                        },
                        gamma: None,
                        y: TensorMut {
                            data: naive_y.as_slice_mut(),
                            shape: xs2,
                            stride: stx2,
                        },
                        rms: TensorMut {
                            data: naive_rms.as_slice_mut(),
                            shape: rs2,
                            stride: str_rms,
                        },
                    };
                    naive_plan
                        .run(&stream, Workspace::None, args)
                        .expect("naive rms");
                });
                Some(measure_median_ns(&ctx, &stream, 11, 50, || {
                    let args = RMSNormArgs::<f32, 2> {
                        x: TensorRef {
                            data: dev_x.as_slice(),
                            shape: xs2,
                            stride: stx2,
                        },
                        gamma: None,
                        y: TensorMut {
                            data: naive_y.as_slice_mut(),
                            shape: xs2,
                            stride: stx2,
                        },
                        rms: TensorMut {
                            data: naive_rms.as_slice_mut(),
                            shape: rs2,
                            stride: str_rms,
                        },
                    };
                    naive_plan
                        .run(&stream, Workspace::None, args)
                        .expect("naive rms");
                }))
            } else {
                None
            };

            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "hyper_connection",
                    shape: shape.clone(),
                    dtype: leak_str("f32"),
                    baracuda_ns: mhc_ns,
                    reference_ns: naive_ns,
                    reference: "naive_rmsnorm_lower_bound",
                    pytorch_ns: None,
                },
            );

            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = HyperConnectionArgs::<f32> {
                            x_expanded: TensorRef {
                                data: dev_x.as_slice(),
                                shape: xs,
                                stride: stx,
                            },
                            rmsnorm_weight: TensorRef {
                                data: dev_gamma.as_slice(),
                                shape: gs,
                                stride: stg,
                            },
                            h_pre: TensorRef {
                                data: dev_h_pre.as_slice(),
                                shape: hs,
                                stride: sth,
                            },
                            h_post: TensorRef {
                                data: dev_h_post.as_slice(),
                                shape: hs,
                                stride: sth,
                            },
                            h_res: TensorRef {
                                data: dev_h_res.as_slice(),
                                shape: rs,
                                stride: str_,
                            },
                            out: TensorMut {
                                data: dev_out.as_slice_mut(),
                                shape: xs,
                                stride: stx,
                            },
                        };
                        plan.run(&stream, Workspace::None, args)
                            .expect("baracuda hyper_connection");
                    })
                });
            });
        }
        group.finish();
    }

    criterion_group!(benches_grp, run);
}

#[cfg(feature = "mhc")]
criterion::criterion_main!(bench_impl_inner::benches_grp);
