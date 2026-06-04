//! GEMM throughput bench — TFLOPS across LLM-typical shapes × dtypes.
//!
//! Sweep:
//! - `M ∈ {1, 8, 32, 128, 512}` × `K = N ∈ {2048, 4096, 8192}`.
//! - Dtypes: `f32` / `f16` / `bf16` / `fp8e4m3` (sm89-gated) / `int8`.
//!
//! Reports `Throughput::Elements(flops)` so criterion prints
//! `flops/sec`. Divide by `1e12` for TFLOPS.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    ElementKind, EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, IntGemmArgs, IntGemmDescriptor,
    IntGemmPlan, LayoutSku, MatrixMut, MatrixRef, PlanPreference, Workspace, S8,
};
use baracuda_kernels_bench::{
    gemm_flops, setup_device, time_with_events, warmup, GEMM_KN_SWEEP, GEMM_M_SWEEP,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};

#[cfg(feature = "sm89")]
use baracuda_kernels::{Fp8E4M3, Fp8GemmArgs, Fp8GemmDescriptor, Fp8GemmPlan};

/// Float GEMM bench — generic over `T: Element` so the same body covers
/// `f32`, `f16`, `bf16` against the CUTLASS-RCR plan.
///
/// The `make_alpha` / `make_beta` callbacks let the caller produce the
/// right scalar type for the dtype: `T::Scalar` is `f32` for
/// f16/bf16/f32 and `f64` for f64, and there's no zero-cost way to
/// `1.0`-construct a generic `ScalarType` from inside the generic body.
///
/// Uses `EpilogueKind::Identity` (no bias) so the measurement is the
/// pure GEMM kernel, not the epilogue overhead.
fn bench_float_gemm<T>(
    c: &mut Criterion,
    dtype_label: &str,
    fill: T,
    alpha: T::Scalar,
    beta: T::Scalar,
) where
    T: baracuda_kernels::Element + Copy + 'static,
{
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("gemm/{dtype_label}"));

    for &kn in GEMM_KN_SWEEP {
        for &m in GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");

            // Per-shape device allocation. Done once per (M, K, N), not
            // per iteration. Filled with `fill` so the kernel exercises
            // real FMA paths (zero-filled buffers can be optimised by
            // some kernels' early-exit paths).
            let host_a: Vec<T> = vec![fill; (m * k) as usize];
            let host_b: Vec<T> = vec![fill; (k * n) as usize];
            let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
                Ok(b) => b,
                Err(_) => continue, // OOM — skip this shape.
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
                Err(_) => continue, // SKU unsupported — skip silently.
            };

            // GemmArgs is rebuilt per launch because `MatrixMut<'_, T>`
            // borrows `dev_d` mutably and criterion's closure can't keep
            // the borrow alive across calls. View construction is host-
            // side and zero-cost.
            group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
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
                    plan.run(&stream, Workspace::None, args).expect("gemm warmup run");
                });

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
                        plan.run(&stream, Workspace::None, args).expect("gemm run");
                    })
                });
            });
        }
    }
    group.finish();
}

/// Int8 GEMM bench — `S8 × Identity × RCR` (the CUTLASS path).
fn bench_int8_gemm(c: &mut Criterion) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("gemm/int8");

    for &kn in GEMM_KN_SWEEP {
        for &m in GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");

            let host_a: Vec<S8> = vec![S8(1); (m * k) as usize];
            let host_b: Vec<S8> = vec![S8(1); (k * n) as usize];
            let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_d: DeviceBuffer<S8> = match DeviceBuffer::zeros(&ctx, (m * n) as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };

            let desc = IntGemmDescriptor {
                m,
                n,
                k,
                layout: LayoutSku::Rcr,
                epilogue: EpilogueKind::Identity,
            };
            let plan =
                match IntGemmPlan::<S8, f32>::select(&stream, &desc, PlanPreference::default()) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
            // Int GEMM may need workspace bytes (CUTLASS split-K cases).
            let ws_bytes = plan.workspace_size();
            let mut dev_ws: Option<DeviceBuffer<u8>> = if ws_bytes > 0 {
                Some(DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws"))
            } else {
                None
            };

            group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                warmup(&stream, || {
                    let workspace = match &mut dev_ws {
                        Some(w) => Workspace::Borrowed(w.as_slice_mut()),
                        None => Workspace::None,
                    };
                    let args = IntGemmArgs::<S8, f32> {
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
                    plan.run(&stream, workspace, args).expect("int8 gemm warmup");
                });

                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let workspace = match &mut dev_ws {
                            Some(w) => Workspace::Borrowed(w.as_slice_mut()),
                            None => Workspace::None,
                        };
                        let args = IntGemmArgs::<S8, f32> {
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
                        plan.run(&stream, workspace, args).expect("int8 gemm run");
                    })
                });
            });
        }
    }
    group.finish();
}

/// FP8 (E4M3) GEMM bench — gated behind the `sm89` feature. The bench
/// binary still compiles without `sm89`; this function just becomes a
/// no-op (no group registered).
#[cfg(feature = "sm89")]
fn bench_fp8_gemm(c: &mut Criterion) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("gemm/fp8e4m3");

    for &kn in GEMM_KN_SWEEP {
        for &m in GEMM_M_SWEEP {
            let n = kn;
            let k = kn;
            let shape = format!("M{m}_N{n}_K{k}");

            // Fp8E4M3(0x38) ≈ 1.0 (exponent bias 7, mantissa 0).
            let host_a: Vec<Fp8E4M3> = vec![Fp8E4M3(0x38); (m * k) as usize];
            let host_b: Vec<Fp8E4M3> = vec![Fp8E4M3(0x38); (k * n) as usize];
            let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_d: DeviceBuffer<Fp8E4M3> =
                match DeviceBuffer::zeros(&ctx, (m * n) as usize) {
                    Ok(b) => b,
                    Err(_) => continue,
                };

            let desc = Fp8GemmDescriptor {
                m,
                n,
                k,
                layout: LayoutSku::Rcr,
                epilogue: EpilogueKind::Identity,
            };
            let plan =
                match Fp8GemmPlan::<Fp8E4M3>::select(&stream, &desc, PlanPreference::default()) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

            group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
            group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
                warmup(&stream, || {
                    let args = Fp8GemmArgs::<Fp8E4M3> {
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
                    plan.run(&stream, Workspace::None, args).expect("fp8 gemm warmup");
                });

                bb.iter_custom(|iters| {
                    time_with_events(&ctx, &stream, iters, || {
                        let args = Fp8GemmArgs::<Fp8E4M3> {
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
                        plan.run(&stream, Workspace::None, args).expect("fp8 gemm run");
                    })
                });
            });
        }
    }
    group.finish();
}

#[cfg(not(feature = "sm89"))]
fn bench_fp8_gemm(_c: &mut Criterion) {
    // FP8 path requires `--features sm89`. Skip silently.
}

/// Entry — registers one group per dtype.
fn gemm_benches(c: &mut Criterion) {
    bench_float_gemm::<f32>(c, "f32", 1.0_f32, 1.0_f32, 0.0_f32);
    bench_float_gemm::<f16>(c, "f16", f16::ONE, 1.0_f32, 0.0_f32);
    bench_float_gemm::<bf16>(c, "bf16", bf16::ONE, 1.0_f32, 0.0_f32);
    bench_fp8_gemm(c);
    bench_int8_gemm(c);
    // Silence unused-binding lint for ElementKind in sm80 builds.
    let _ = ElementKind::F32;
}

// `criterion_group!` expands into a `pub fn benches` whose
// signature is fixed by the macro - can't doc-comment it directly, so
// suppress the workspace `missing_docs = deny` lint on the generated fn.
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches, gemm_benches);
}
criterion_main!(criterion_glue::benches);
