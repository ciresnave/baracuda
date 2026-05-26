//! GEMM throughput — baracuda vs cuBLAS.
//!
//! For each `(M, K=N, dtype)` shape, runs two criterion bench groups:
//! `gemm_vs_cublas/baracuda/<dtype>` and `gemm_vs_cublas/cublas/<dtype>`
//! at the same shapes. Side-by-side BenchmarkId strings make criterion's
//! HTML report directly comparable.
//!
//! Phase 29 cross-impl bench. The Phase-10 `gemm.rs` already covers
//! baracuda-only sweeps at a wider shape grid; this file's value is
//! the head-to-head against cuBLAS at a smaller (faster-to-run) grid.
//!
//! ## Dtype coverage
//!
//! - `f32` via `cublasSgemm` (baracuda's CUTLASS RCR f32 sibling).
//! - `f16` via `cublasGemmEx(R_16F, R_16F, R_16F, Compute32F)` — cuBLAS's
//!   tensor-core path with f32 accumulator. baracuda's f16 RCR plan.
//! - `bf16` via `cublasGemmEx(R_16BF, R_16BF, R_16BF, Compute32F)`. baracuda's
//!   bf16 RCR plan.

use core::ffi::c_void;

use baracuda_cublas::{gemm, gemm_ex, Handle as CublasHandle, Op};
use baracuda_cublas_sys::functions::{cublasComputeType_t, cudaDataType_t};
use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, gemm_flops, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow, CROSS_GEMM_KN_SWEEP, CROSS_GEMM_M_SWEEP,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};

const BENCH_NAME: &str = "gemm_vs_cublas";

/// baracuda f32 RCR GEMM bench.
fn bench_baracuda_f32(c: &mut Criterion) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("gemm_vs_cublas/baracuda/f32");

    for &kn in CROSS_GEMM_KN_SWEEP {
        for &m in CROSS_GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");

            let host_a: Vec<f32> = vec![1.0; (m * k) as usize];
            let host_b: Vec<f32> = vec![1.0; (k * n) as usize];
            let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_d: DeviceBuffer<f32> = match DeviceBuffer::zeros(&ctx, (m * n) as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let desc = GemmDescriptor {
                m,
                n,
                k,
                layout: LayoutSku::Rcr,
                epilogue: EpilogueKind::Identity,
            };
            let plan = match GemmPlan::<f32>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let run = || {
                let args = GemmArgs::<f32> {
                    a: MatrixRef {
                        data: dev_a.as_slice(),
                        rows: m,
                        cols: k,
                        ld: k as i64,
                    },
                    b: MatrixRef {
                        data: dev_b.as_slice(),
                        rows: k,
                        cols: n,
                        ld: k as i64,
                    },
                    c: None,
                    d: MatrixMut {
                        data: dev_d.as_slice_mut(),
                        rows: m,
                        cols: n,
                        ld: n as i64,
                    },
                    bias: None,
                    alpha: 1.0,
                    beta: 0.0,
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda f32 gemm");
            };

            warmup(&stream, run);
            // run() can't be cloned for measure_median_ns; rebuild via a closure
            // capturing &dev_a etc.
            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                let args = GemmArgs::<f32> {
                    a: MatrixRef {
                        data: dev_a.as_slice(),
                        rows: m,
                        cols: k,
                        ld: k as i64,
                    },
                    b: MatrixRef {
                        data: dev_b.as_slice(),
                        rows: k,
                        cols: n,
                        ld: k as i64,
                    },
                    c: None,
                    d: MatrixMut {
                        data: dev_d.as_slice_mut(),
                        rows: m,
                        cols: n,
                        ld: n as i64,
                    },
                    bias: None,
                    alpha: 1.0_f32,
                    beta: 0.0_f32,
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda f32 gemm");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "gemm",
                    shape: shape.clone(),
                    dtype: "f32",
                    baracuda_ns,
                    reference_ns: None,
                    reference: "baracuda",
                },
            );

            group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = GemmArgs::<f32> {
                            a: MatrixRef {
                                data: dev_a.as_slice(),
                                rows: m,
                                cols: k,
                                ld: k as i64,
                            },
                            b: MatrixRef {
                                data: dev_b.as_slice(),
                                rows: k,
                                cols: n,
                                ld: k as i64,
                            },
                            c: None,
                            d: MatrixMut {
                                data: dev_d.as_slice_mut(),
                                rows: m,
                                cols: n,
                                ld: n as i64,
                            },
                            bias: None,
                            alpha: 1.0,
                            beta: 0.0,
                        };
                        plan.run(&stream, Workspace::None, args).expect("baracuda f32 gemm");
                    })
                });
            });
        }
    }
    group.finish();
}

/// cuBLAS f32 GEMM bench. baracuda's `LayoutSku::Rcr` means A is row-
/// major (M×K, lda=K) and B is column-major (K×N, ldb=K) — i.e. baracuda
/// computes `D = A · B^T`-ish, but the layout sku encoding is
/// "A row-major, B col-major, C row-major" with logical `M, N, K`.
///
/// cuBLAS is column-major. To get the same product, call `gemm(N, T, ...)`
/// in column-major terms but with cuBLAS's convention that the inputs
/// represent transposes — i.e. treat the row-major A as column-major
/// A^T, etc. The canonical row-major-from-cuBLAS trick: compute
/// `C^T = B^T · A^T` in column-major, which lets us pass A/B straight
/// through and read C in row-major.
fn bench_cublas_f32(c: &mut Criterion) {
    let (ctx, stream) = setup_device();
    let handle = CublasHandle::new().expect("cublas handle");
    handle.set_stream(&stream).expect("cublas set_stream");

    let mut group = c.benchmark_group("gemm_vs_cublas/cublas/f32");

    for &kn in CROSS_GEMM_KN_SWEEP {
        for &m in CROSS_GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");

            let host_a: Vec<f32> = vec![1.0; (m * k) as usize];
            let host_b: Vec<f32> = vec![1.0; (k * n) as usize];
            let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_c: DeviceBuffer<f32> = match DeviceBuffer::zeros(&ctx, (m * n) as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };

            // Row-major A·B in col-major cuBLAS: compute C^T = B^T · A^T.
            // Pass B as the first operand with shape (N, K, ldb=N), A as
            // the second with shape (K, M, lda=K). C output is (N, M)
            // in col-major == (M, N) in row-major.
            let warmup_call = || {
                gemm(
                    &handle,
                    Op::N,
                    Op::N,
                    n,
                    m,
                    k,
                    1.0_f32,
                    &dev_b,
                    n,
                    &dev_a,
                    k,
                    0.0_f32,
                    &mut dev_c,
                    n,
                )
                .expect("cublas sgemm");
            };
            warmup(&stream, warmup_call);

            let cublas_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                gemm(
                    &handle,
                    Op::N,
                    Op::N,
                    n,
                    m,
                    k,
                    1.0_f32,
                    &dev_b,
                    n,
                    &dev_a,
                    k,
                    0.0_f32,
                    &mut dev_c,
                    n,
                )
                .expect("cublas sgemm");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "gemm",
                    shape: shape.clone(),
                    dtype: "f32",
                    baracuda_ns: 0.0,
                    reference_ns: Some(cublas_ns),
                    reference: "cuBLAS",
                },
            );

            group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        gemm(
                            &handle,
                            Op::N,
                            Op::N,
                            n,
                            m,
                            k,
                            1.0_f32,
                            &dev_b,
                            n,
                            &dev_a,
                            k,
                            0.0_f32,
                            &mut dev_c,
                            n,
                        )
                        .expect("cublas sgemm");
                    })
                });
            });
        }
    }
    group.finish();
}

/// baracuda RCR GEMM for f16 / bf16. `alpha`/`beta` are passed as
/// `T::Scalar` because the generic body can't construct `1.0` of an
/// associated-type scalar from inside; mirrors `bench_float_gemm` in
/// `gemm.rs`.
fn bench_baracuda_half<T>(
    c: &mut Criterion,
    dtype_label: &str,
    fill: T,
    alpha: T::Scalar,
    beta: T::Scalar,
) where
    T: baracuda_kernels::Element + Copy + 'static,
    T::Scalar: Copy,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("gemm_vs_cublas/baracuda/{dtype_label}"));

    for &kn in CROSS_GEMM_KN_SWEEP {
        for &m in CROSS_GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");

            let host_a: Vec<T> = vec![fill; (m * k) as usize];
            let host_b: Vec<T> = vec![fill; (k * n) as usize];
            let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_d: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, (m * n) as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let desc = GemmDescriptor {
                m,
                n,
                k,
                layout: LayoutSku::Rcr,
                epilogue: EpilogueKind::Identity,
            };
            let plan = match GemmPlan::<T>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };

            warmup(&stream, || {
                let args = GemmArgs::<T> {
                    a: MatrixRef {
                        data: dev_a.as_slice(),
                        rows: m,
                        cols: k,
                        ld: k as i64,
                    },
                    b: MatrixRef {
                        data: dev_b.as_slice(),
                        rows: k,
                        cols: n,
                        ld: k as i64,
                    },
                    c: None,
                    d: MatrixMut {
                        data: dev_d.as_slice_mut(),
                        rows: m,
                        cols: n,
                        ld: n as i64,
                    },
                    bias: None,
                    alpha,
                    beta,
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda half gemm");
            });

            let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
                let args = GemmArgs::<T> {
                    a: MatrixRef {
                        data: dev_a.as_slice(),
                        rows: m,
                        cols: k,
                        ld: k as i64,
                    },
                    b: MatrixRef {
                        data: dev_b.as_slice(),
                        rows: k,
                        cols: n,
                        ld: k as i64,
                    },
                    c: None,
                    d: MatrixMut {
                        data: dev_d.as_slice_mut(),
                        rows: m,
                        cols: n,
                        ld: n as i64,
                    },
                    bias: None,
                    alpha,
                    beta,
                };
                plan.run(&stream, Workspace::None, args).expect("baracuda half gemm");
            });
            // Stash with reference=None — `bench_cublas_half` will emit
            // the matching reference-side row.
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "gemm",
                    shape: shape.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns,
                    reference_ns: None,
                    reference: "baracuda",
                },
            );

            group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = GemmArgs::<T> {
                            a: MatrixRef {
                                data: dev_a.as_slice(),
                                rows: m,
                                cols: k,
                                ld: k as i64,
                            },
                            b: MatrixRef {
                                data: dev_b.as_slice(),
                                rows: k,
                                cols: n,
                                ld: k as i64,
                            },
                            c: None,
                            d: MatrixMut {
                                data: dev_d.as_slice_mut(),
                                rows: m,
                                cols: n,
                                ld: n as i64,
                            },
                            bias: None,
                            alpha,
                            beta,
                        };
                        plan.run(&stream, Workspace::None, args).expect("baracuda half gemm");
                    })
                });
            });
        }
    }
    group.finish();
}

/// cuBLAS `gemmEx` for half-precision dtypes.
fn bench_cublas_half(
    c: &mut Criterion,
    dtype_label: &str,
    data_type: cudaDataType_t,
    elt_bytes: usize,
) {
    let (ctx, stream) = setup_device();
    let handle = CublasHandle::new().expect("cublas handle");
    handle.set_stream(&stream).expect("cublas set_stream");

    let mut group = c.benchmark_group(format!("gemm_vs_cublas/cublas/{dtype_label}"));

    for &kn in CROSS_GEMM_KN_SWEEP {
        for &m in CROSS_GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");

            // Zero-fill — for fp16/bf16 the bit pattern 0x0000 is +0.0,
            // and the FMA path is hit identically to a 1.0-fill (cuBLAS
            // doesn't early-exit). We still warmup before timing.
            let bytes_a = (m * k) as usize * elt_bytes;
            let bytes_b = (k * n) as usize * elt_bytes;
            let bytes_d = (m * n) as usize * elt_bytes;
            let dev_a: DeviceBuffer<u8> = match DeviceBuffer::zeros(&ctx, bytes_a) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_b: DeviceBuffer<u8> = match DeviceBuffer::zeros(&ctx, bytes_b) {
                Ok(b) => b,
                Err(_) => continue,
            };
            // gemm_ex writes through the raw `*mut c_void` we pass in,
            // so we don't need `&mut` for the wrapper API — but the
            // buffer itself must outlive the launch loop. Declare
            // non-mut; the `as_raw().0 as *mut c_void` cast crosses
            // the Rust-side immutability.
            let dev_d: DeviceBuffer<u8> = match DeviceBuffer::zeros(&ctx, bytes_d) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            // C^T = B^T · A^T trick (row-major-from-col-major).
            // SAFETY: every pointer comes from a live DeviceBuffer and the
            // type tags + leading dimensions match the byte allocations.
            let launch = || unsafe {
                gemm_ex(
                    &handle,
                    Op::N,
                    Op::N,
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const c_void,
                    dev_b.as_raw().0 as *const c_void,
                    data_type,
                    n,
                    dev_a.as_raw().0 as *const c_void,
                    data_type,
                    k,
                    &beta as *const f32 as *const c_void,
                    dev_d.as_raw().0 as *mut c_void,
                    data_type,
                    n,
                    cublasComputeType_t::Compute32F,
                    0,
                )
                .expect("cublas gemm_ex");
            };
            warmup(&stream, launch);

            let cublas_ns = measure_median_ns(&ctx, &stream, 11, 50, || unsafe {
                gemm_ex(
                    &handle,
                    Op::N,
                    Op::N,
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const c_void,
                    dev_b.as_raw().0 as *const c_void,
                    data_type,
                    n,
                    dev_a.as_raw().0 as *const c_void,
                    data_type,
                    k,
                    &beta as *const f32 as *const c_void,
                    dev_d.as_raw().0 as *mut c_void,
                    data_type,
                    n,
                    cublasComputeType_t::Compute32F,
                    0,
                )
                .expect("cublas gemm_ex");
            });
            append_csv_row(
                BENCH_NAME,
                &PhaseTwentyNineRow {
                    op: "gemm",
                    shape: shape.clone(),
                    dtype: leak_str(dtype_label),
                    baracuda_ns: 0.0,
                    reference_ns: Some(cublas_ns),
                    reference: "cuBLAS",
                },
            );

            group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || unsafe {
                        gemm_ex(
                            &handle,
                            Op::N,
                            Op::N,
                            n,
                            m,
                            k,
                            &alpha as *const f32 as *const c_void,
                            dev_b.as_raw().0 as *const c_void,
                            data_type,
                            n,
                            dev_a.as_raw().0 as *const c_void,
                            data_type,
                            k,
                            &beta as *const f32 as *const c_void,
                            dev_d.as_raw().0 as *mut c_void,
                            data_type,
                            n,
                            cublasComputeType_t::Compute32F,
                            0,
                        )
                        .expect("cublas gemm_ex");
                    })
                });
            });
        }
    }
    group.finish();
}

/// `String → &'static str` via `Box::leak`. Acceptable in bench code:
/// the static set of dtype labels is fixed and tiny, the alternative
/// (threading `String` through `PhaseTwentyNineRow`) would balloon the
/// struct definition without buying anything for a one-shot CSV emit.
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn benches(c: &mut Criterion) {
    bench_baracuda_f32(c);
    bench_cublas_f32(c);
    bench_baracuda_half::<f16>(c, "f16", f16::ONE, 1.0_f32, 0.0_f32);
    bench_cublas_half(c, "f16", cudaDataType_t::R_16F, 2);
    bench_baracuda_half::<bf16>(c, "bf16", bf16::ONE, 1.0_f32, 0.0_f32);
    bench_cublas_half(c, "bf16", cudaDataType_t::R_16BF, 2);
}

criterion_group!(benches_grp, benches);
criterion_main!(benches_grp);
