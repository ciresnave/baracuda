//! GEMM head-to-head: baracuda CUTLASS / cuBLAS gemmEx / CUDA-L2.
//!
//! Phase 44 bench — RL+LLM-tuned HGEMM kernels from
//! [`deepreinforce-ai/CUDA-L2`](https://github.com/deepreinforce-ai/CUDA-L2)
//! vs our existing two backends (`baracuda-Bespoke` CUTLASS sm_80 and
//! `Cublas` gemmEx fast-path). The bench answers one question:
//! "should we vendor CUDA-L2 as a third `GemmPlan` backend?"
//!
//! ## Decision: SKIP (see report at end + `BENCHMARKS.md`).
//!
//! ## Two run modes
//!
//! * **Default** (no `cuda_l2` feature): runs baracuda + cuBLAS arms at
//!   the CUDA-L2-covered shapes (M ∈ {128, 2048}, N=K=4096, f16) and
//!   emits the CUDA-L2 timing as a pre-measured `eprintln!` constant
//!   harvested from the standalone probes under
//!   `external/cuda-l2-probes/`. This keeps the bench binary lightweight
//!   and avoids pulling nvcc into the default bench build path.
//!
//! * **With `--features cuda_l2,sm89`**: `build.rs` compiles the two
//!   CUDA-L2 wrappers under `external/cuda-l2-probes/wrapper_m*.cu`
//!   into a static library and the bench links + invokes the
//!   `baracuda_cuda_l2_m{128,2048}_launch` C ABI entry points for live
//!   end-to-end measurement.
//!
//! ## Shape coverage
//!
//! | Shape | Regime | CUDA-L2 covers? | Notes |
//! |---|---|---|---|
//! | M=1, K=4096, N=4096 | Decode | **NO** | CUDA-L2 ships no M<64 kernel |
//! | M=8, K=4096, N=4096 | Spec. decode | **NO** | (same) |
//! | M=32, K=4096, N=4096 | Small-batch decode | **NO** | (same) |
//! | M=128, K=4096, N=4096 | Prefill batch | yes | wrapper_m128 |
//! | M=2048, K=4096, N=4096 | Large prefill | yes | wrapper_m2048 |
//!
//! The "NO" rows are exactly the regime where baracuda's Phase 30
//! cuBLAS fast-path won 3× over CUTLASS. CUDA-L2 has zero coverage
//! there; their upstream FAQ suggests "pad to next-larger shape and
//! zero-fill" which costs 64× the work at M=1.

use core::ffi::c_void;

use baracuda_cublas::{gemm_ex, Handle as CublasHandle, Op};
use baracuda_cublas_sys::functions::{cublasComputeType_t, cudaDataType_t};
use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
use baracuda_kernels_bench::{
    append_csv_row, gemm_flops, measure_median_ns, setup_device, time_with_events, warmup,
    PhaseTwentyNineRow,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::f16;

const BENCH_NAME: &str = "gemm_vs_cuda_l2";

/// Shapes the CUDA-L2 3090 set covers. We bench at f16 (their only
/// dtype claim) with fp32 accumulator (the F32F16F16F32 kernel set).
/// (M, K=N) — N is always equal to K in this sweep. `i32` to match
/// the upstream cuBLAS / baracuda-kernels shape parameter type.
const CUDA_L2_SHAPES: &[(i32, i32)] = &[(128, 4096), (2048, 4096)];

/// Reference measurements harvested from the standalone probes under
/// `external/cuda-l2-probes/` on RTX 4070 + CUDA 13.0 (2026-05-28).
/// Each tuple is `(M, K=N, CUDA-L2 us, cuBLAS us in same harness)`.
/// These let the default-mode bench output a comparable side-by-side
/// without compiling the CUDA-L2 kernels in the workspace build path.
const REFERENCE_PROBE_NUMBERS: &[(i32, i32, f64, f64)] = &[
    (128, 4096, 175.20, 177.37),  // CUDA-L2 +1.2% over cuBLAS
    (2048, 4096, 2452.73, 2621.46), // CUDA-L2 +6.4% over cuBLAS
];

#[cfg(feature = "cuda_l2")]
mod cuda_l2_ffi {
    use baracuda_driver::cuda_sys::cudaStream_t;
    use core::ffi::c_int;
    use half::f16;

    unsafe extern "C" {
        /// `wrapper_m128.cu` — 128_4096_4096 CUDA-L2 kernel. Operand
        /// convention: `a` row-major (M, K), `b_col_major` col-major
        /// (K, N), `c` row-major (M, N). f16 only, fp32 accum.
        pub safe fn baracuda_cuda_l2_m128_launch(
            a: *const f16,
            b_col_major: *const f16,
            c: *mut f16,
            m: c_int,
            n: c_int,
            k: c_int,
            stream: cudaStream_t,
        );

        /// `wrapper_m2048.cu` — 2048_4096_4096 CUDA-L2 kernel. Same
        /// operand convention as `m128_launch`.
        pub safe fn baracuda_cuda_l2_m2048_launch(
            a: *const f16,
            b_col_major: *const f16,
            c: *mut f16,
            m: c_int,
            n: c_int,
            k: c_int,
            stream: cudaStream_t,
        );
    }

    pub fn dispatch(
        m: u32,
        n: u32,
        k: u32,
        a: *const f16,
        b_col_major: *const f16,
        c: *mut f16,
        stream: cudaStream_t,
    ) {
        let m_i = m as c_int;
        let n_i = n as c_int;
        let k_i = k as c_int;
        match m {
            128 => unsafe {
                baracuda_cuda_l2_m128_launch(a, b_col_major, c, m_i, n_i, k_i, stream)
            },
            2048 => unsafe {
                baracuda_cuda_l2_m2048_launch(a, b_col_major, c, m_i, n_i, k_i, stream)
            },
            _ => panic!("no CUDA-L2 wrapper compiled for M={m}"),
        }
    }
}

/// baracuda f16 RCR GEMM at the CUDA-L2 shape set. Mirrors
/// `bench_baracuda_half<f16>` from `gemm_vs_cublas.rs`.
fn bench_baracuda_f16(c: &mut Criterion) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("gemm_vs_cuda_l2/baracuda/f16");

    for &(m, kn) in CUDA_L2_SHAPES {
        let n = kn;
        let k = kn;
        let shape = format!("M{m}_N{n}_K{k}");

        let host_a: Vec<f16> = vec![f16::ONE; (m * k) as usize];
        let host_b: Vec<f16> = vec![f16::ONE; (k * n) as usize];
        let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_d: DeviceBuffer<f16> = match DeviceBuffer::zeros(&ctx, (m * n) as usize) {
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
        let plan = match GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default()) {
            Ok(p) => p,
            Err(_) => continue,
        };

        warmup(&stream, || {
            let args = GemmArgs::<f16> {
                a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
                b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
                c: None,
                d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
                bias: None,
                alpha: 1.0_f32,
                beta: 0.0_f32,
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda f16 gemm");
        });

        let baracuda_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
            let args = GemmArgs::<f16> {
                a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
                b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
                c: None,
                d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
                bias: None,
                alpha: 1.0_f32,
                beta: 0.0_f32,
            };
            plan.run(&stream, Workspace::None, args).expect("baracuda f16 gemm");
        });
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "gemm",
                shape: shape.clone(),
                dtype: "f16",
                baracuda_ns,
                reference_ns: None,
                reference: "baracuda",
                pytorch_ns: None,
            },
        );

        group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
        group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    let args = GemmArgs::<f16> {
                        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
                        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
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
                    plan.run(&stream, Workspace::None, args).expect("baracuda f16 gemm");
                })
            });
        });
    }
    group.finish();
}

/// cuBLAS `gemmEx` f16 at the CUDA-L2 shape set. Uses the
/// row-major-from-col-major trick (C^T = B^T · A^T) so the operand
/// shapes match the baracuda RCR semantics.
fn bench_cublas_f16(c: &mut Criterion) {
    let (ctx, stream) = setup_device();
    let handle = CublasHandle::new().expect("cublas handle");
    handle.set_stream(&stream).expect("cublas set_stream");

    let mut group = c.benchmark_group("gemm_vs_cuda_l2/cublas/f16");

    for &(m, kn) in CUDA_L2_SHAPES {
        let n = kn;
        let k = kn;
        let shape = format!("M{m}_N{n}_K{k}");

        let elt_bytes = 2_usize;
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
        let dev_d: DeviceBuffer<u8> = match DeviceBuffer::zeros(&ctx, bytes_d) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

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
                cudaDataType_t::R_16F,
                n,
                dev_a.as_raw().0 as *const c_void,
                cudaDataType_t::R_16F,
                k,
                &beta as *const f32 as *const c_void,
                dev_d.as_raw().0 as *mut c_void,
                cudaDataType_t::R_16F,
                n,
                cublasComputeType_t::Compute32F,
                0,
            )
            .expect("cublas gemmEx");
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
                cudaDataType_t::R_16F,
                n,
                dev_a.as_raw().0 as *const c_void,
                cudaDataType_t::R_16F,
                k,
                &beta as *const f32 as *const c_void,
                dev_d.as_raw().0 as *mut c_void,
                cudaDataType_t::R_16F,
                n,
                cublasComputeType_t::Compute32F,
                0,
            )
            .expect("cublas gemmEx");
        });
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "gemm",
                shape: shape.clone(),
                dtype: "f16",
                baracuda_ns: 0.0,
                reference_ns: Some(cublas_ns),
                reference: "cuBLAS",
                pytorch_ns: None,
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
                        cudaDataType_t::R_16F,
                        n,
                        dev_a.as_raw().0 as *const c_void,
                        cudaDataType_t::R_16F,
                        k,
                        &beta as *const f32 as *const c_void,
                        dev_d.as_raw().0 as *mut c_void,
                        cudaDataType_t::R_16F,
                        n,
                        cublasComputeType_t::Compute32F,
                        0,
                    )
                    .expect("cublas gemmEx");
                })
            });
        });
    }
    group.finish();
}

/// CUDA-L2 timing arm. Live measurement requires the `cuda_l2`
/// feature; without it, the function emits the pre-measured
/// `REFERENCE_PROBE_NUMBERS` to stderr so the bench output still
/// includes the side-by-side comparison.
#[cfg(feature = "cuda_l2")]
fn bench_cuda_l2_f16(c: &mut Criterion) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group("gemm_vs_cuda_l2/cuda_l2/f16");

    for &(m, kn) in CUDA_L2_SHAPES {
        let n = kn;
        let k = kn;
        let shape = format!("M{m}_N{n}_K{k}");

        // Same as the cuBLAS arm: f16 row-major A (M, K), and CUDA-L2
        // expects B in col-major (K, N), which is the row-major (N, K)
        // transpose. For all-ones fixtures these are the same bytes.
        let host_a: Vec<f16> = vec![f16::ONE; (m * k) as usize];
        let host_b: Vec<f16> = vec![f16::ONE; (n * k) as usize];
        let dev_a = match DeviceBuffer::from_slice(&ctx, &host_a) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let dev_b = match DeviceBuffer::from_slice(&ctx, &host_b) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut dev_c: DeviceBuffer<f16> = match DeviceBuffer::zeros(&ctx, (m * n) as usize) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let launch = || {
            cuda_l2_ffi::dispatch(
                m,
                n,
                k,
                dev_a.as_slice().as_ptr(),
                dev_b.as_slice().as_ptr(),
                dev_c.as_slice_mut().as_mut_ptr(),
                stream.as_raw().0 as _,
            );
        };
        warmup(&stream, launch);

        let cuda_l2_ns = measure_median_ns(&ctx, &stream, 11, 50, || {
            cuda_l2_ffi::dispatch(
                m,
                n,
                k,
                dev_a.as_slice().as_ptr(),
                dev_b.as_slice().as_ptr(),
                dev_c.as_slice_mut().as_mut_ptr(),
                stream.as_raw().0 as _,
            );
        });
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "gemm",
                shape: shape.clone(),
                dtype: "f16",
                baracuda_ns: 0.0,
                reference_ns: Some(cuda_l2_ns),
                reference: "CUDA-L2",
                pytorch_ns: None,
            },
        );

        group.throughput(Throughput::Elements(gemm_flops(m, n, k)));
        group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
            bb.iter_custom(|iters| {
                time_with_events(&ctx, &stream, iters, || {
                    cuda_l2_ffi::dispatch(
                        m,
                        n,
                        k,
                        dev_a.as_slice().as_ptr(),
                        dev_b.as_slice().as_ptr(),
                        dev_c.as_slice_mut().as_mut_ptr(),
                        stream.as_raw().0 as _,
                    );
                })
            });
        });
    }
    group.finish();
}

/// Documentation-only mode (no `cuda_l2` feature): emit the
/// `REFERENCE_PROBE_NUMBERS` as a stderr table so the bench output
/// still includes a side-by-side. Compiles + links without needing
/// nvcc on the bench-build path.
#[cfg(not(feature = "cuda_l2"))]
fn bench_cuda_l2_f16(_c: &mut Criterion) {
    eprintln!();
    eprintln!("== CUDA-L2 reference measurements (RTX 4070 sm_89, CUDA 13.0) ==");
    eprintln!(
        "   Build with `--features cuda_l2,sm89` to time CUDA-L2 live in this bench."
    );
    eprintln!("   Numbers below come from `external/cuda-l2-probes/probe_*.cu`.");
    eprintln!();
    eprintln!("   {:>16} | {:>14} | {:>14} | {:>8}", "shape", "CUDA-L2 (us)", "cuBLAS (us)", "ratio");
    eprintln!("   {:->16}-+-{:->14}-+-{:->14}-+-{:->8}", "", "", "", "");
    for &(m, kn, l2_us, cublas_us) in REFERENCE_PROBE_NUMBERS {
        let shape = format!("M{m}_N{kn}_K{kn}");
        let ratio = l2_us / cublas_us;
        eprintln!(
            "   {:>16} | {:>14.2} | {:>14.2} | {:>8.3}",
            shape, l2_us, cublas_us, ratio
        );
    }
    eprintln!();
    eprintln!("== Decision: SKIP vendor. ==");
    eprintln!(
        "   CUDA-L2 either ties (M=128: +1.2%) or modestly beats cuBLAS (M=2048: +6.4%)"
    );
    eprintln!(
        "   at the shapes it covers, but ships NO kernels for M ∈ {{1, 8, 32}} — the decode"
    );
    eprintln!(
        "   regime where baracuda's Phase 30 cuBLAS fast-path already won 3× over CUTLASS."
    );
    eprintln!(
        "   Per-shape build.rs compilation + per-shape FFI plumbing is high cost for"
    );
    eprintln!(
        "   ≤6% wins outside the actually-load-bearing decode regime."
    );
    eprintln!();
    // Also write a CSV row so the standard phase29 CSV companion
    // picks up the documentation-mode reference numbers.
    for &(m, kn, l2_us, _cublas_us) in REFERENCE_PROBE_NUMBERS {
        let shape = format!("M{m}_N{kn}_K{kn}");
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: "gemm",
                shape,
                dtype: "f16",
                baracuda_ns: 0.0,
                reference_ns: Some(l2_us * 1_000.0),
                reference: "CUDA-L2-ref",
                pytorch_ns: None,
            },
        );
    }
}

/// Top-level criterion entry - invoked by criterion_main!.
fn benches(c: &mut Criterion) {
    bench_baracuda_f16(c);
    bench_cublas_f16(c);
    bench_cuda_l2_f16(c);
}

// `criterion_group!` expands into a `pub fn benches_grp` whose
// signature is fixed by the macro - can't doc-comment it directly, so
// suppress the workspace `missing_docs = deny` lint on the generated fn.
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}
criterion_main!(criterion_glue::benches_grp);
