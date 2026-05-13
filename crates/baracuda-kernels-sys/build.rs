//! Build script for `baracuda-kernels-sys`.
//!
//! Compiles the curated bespoke kernel sources under `kernels/` into a
//! static library via `baracuda-forge`. Mirrors the pattern of
//! `baracuda-cutlass-kernels-sys/build.rs` but does **not** depend on
//! CUTLASS — every kernel in this crate is hand-rolled on top of plain
//! CUDA PTX intrinsics (`mma.sync`, `cp.async`, `ldmatrix`).
//!
//! The set of files compiled is selected by Cargo features:
//!
//! - `sm80` (default): Ampere baseline (also runs on Ada / Hopper as
//!   forward-compatible fallback).
//! - `sm89`: Ada Lovelace specializations (adds FP8 kernels).
//! - `sm90a`: Hopper-specialized kernels.
//!
//! When no arch features are enabled the build is a no-op (the safe
//! layer will return an `Unsupported` error at runtime).

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    if env::var_os("DOCS_RS").is_some() {
        println!(
            "cargo:warning=baracuda-kernels-sys: DOCS_RS=1 detected; skipping nvcc build."
        );
        return;
    }

    let kernels = collect_kernel_files();

    if kernels.is_empty() {
        println!(
            "cargo:warning=baracuda-kernels-sys: no kernel sources to compile (no arch feature, or kernel files not present yet)."
        );
        return;
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
    let lib_name = "baracuda_kernels";
    let lib_file = lib_static_path(&out_dir, lib_name);

    use baracuda_forge::KernelBuilder;
    let mut builder = KernelBuilder::new()
        .source_files(kernels.iter().map(|k| format!("kernels/{k}")))
        // Bespoke header includes (cp.async / mma.sync helpers, dtype
        // wrappers, warp-reduction primitives, ...).
        .include_path("kernels/include")
        .out_dir(&out_dir);

    if cfg!(feature = "sm90a") {
        builder = builder.compute_cap(90).arg("-DBARACUDA_KERNELS_HAS_SM90A");
    } else if cfg!(feature = "sm89") {
        builder = builder.compute_cap(89).arg("-DBARACUDA_KERNELS_HAS_SM89");
    } else if cfg!(feature = "sm80") {
        builder = builder.compute_cap(80);
    }

    builder
        .build_lib(&lib_file)
        .expect("baracuda-kernels-sys: nvcc build failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static={lib_name}");

    // CUDA Runtime — kernel objects reference cudart helpers (streams etc.).
    println!("cargo:rustc-link-lib=dylib=cudart");

    // C++ runtime on Linux — the static lib pulls in libstdc++ via
    // template instantiations.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn collect_kernel_files() -> Vec<&'static str> {
    let mut kernels: Vec<&'static str> = Vec::new();

    // Phase 1 deliverables: int8 GEMM RRR, sm_80 baseline. Files land
    // as the kernel source is implemented; build.rs gracefully skips any
    // not-yet-present file so the workspace stays buildable during
    // Phase 0 scaffolding.
    if cfg!(any(feature = "sm80", feature = "sm89", feature = "sm90a")) {
        if cfg!(any(feature = "sm80", feature = "sm89")) {
            for f in &[
                "gemm/gemm_s8_rrr_sm80.cu",
                "gemm/gemm_u8_rrr_sm80.cu",
                "gemm/gemm_s8_rrr_sm80_bias.cu",
                "gemm/gemm_u8_rrr_sm80_bias.cu",
            ] {
                if std::path::Path::new(&format!("kernels/{f}")).exists() {
                    kernels.push(*f);
                }
            }
        }

        // Phase 2 deliverables: FP8 / int4 / bin GEMM, sm_89 (Ada
        // Lovelace) baseline. Selected only when `sm89` is enabled —
        // the FP8 m16n8k32 instruction is sm_89+ and won't assemble
        // against an `sm_80` target.
        //
        // Full FP8 SKU matrix: {E4M3, E5M2} × {RCR, RRR} × {Identity,
        // Bias, BiasRelu, BiasGelu, BiasSilu} = 20 SKUs across 8 .cu
        // files (4 Identity + 4 bias multi-instantiation files).
        if cfg!(feature = "sm89") {
            for f in &[
                "gemm/gemm_fp8_e4m3_rcr_sm89.cu",
                "gemm/gemm_fp8_e4m3_rrr_sm89.cu",
                "gemm/gemm_fp8_e5m2_rcr_sm89.cu",
                "gemm/gemm_fp8_e5m2_rrr_sm89.cu",
                "gemm/gemm_fp8_e4m3_rcr_sm89_bias.cu",
                "gemm/gemm_fp8_e4m3_rrr_sm89_bias.cu",
                "gemm/gemm_fp8_e5m2_rcr_sm89_bias.cu",
                "gemm/gemm_fp8_e5m2_rrr_sm89_bias.cu",
                // int4 GEMM — S4 / U4 × {RCR, RRR} × Identity
                // (alpha.17). The bias-family variants land in
                // subsequent fanout commits.
                // `mma.sync.m16n8k64.{s4|u4}.{s4|u4}.s32` requires
                // sm_89+. RRR header gathers two nibbles from two
                // gmem K-rows into one packed-pair smem byte (B is
                // pair-packed along N in gmem, K-contig in smem).
                "gemm/gemm_s4_rcr_sm89.cu",
                "gemm/gemm_u4_rcr_sm89.cu",
                "gemm/gemm_s4_rrr_sm89.cu",
                "gemm/gemm_u4_rrr_sm89.cu",
                // int4 bias-family multi-instantiation files. Each file
                // ships 8 `_run` symbols for the
                // `{Bias, BiasRelu, BiasGelu, BiasSilu} × {f32, i32}` bias
                // matrix at a fixed `(element, layout)` pair. The
                // Identity SKU above ships the shared `_workspace_size`
                // and `_can_implement` for each `(element, layout)`.
                "gemm/gemm_s4_rcr_sm89_bias.cu",
                "gemm/gemm_u4_rcr_sm89_bias.cu",
                "gemm/gemm_s4_rrr_sm89_bias.cu",
                "gemm/gemm_u4_rrr_sm89_bias.cu",
                // bin (B1) GEMM — RCR + RRR Identity-only. `xor.popc`
                // model; output is raw int32 popcount sum (no
                // α/β/bias/activation chain).
                // `mma.sync.m16n8k256.b1.b1.s32.xor.popc` (sm_80+;
                // gated to sm_89 here for build-set parity with
                // int4/FP8). The RRR header bit-gathers 8 K-row gmem
                // bytes into one K-pair smem byte per output column
                // (B is N-bit-packed in gmem, K-bit-packed in smem).
                "gemm/gemm_bin_rcr_sm89.cu",
                "gemm/gemm_bin_rrr_sm89.cu",
            ] {
                if std::path::Path::new(&format!("kernels/{f}")).exists() {
                    kernels.push(*f);
                }
            }
        }
    }

    kernels
}

fn lib_static_path(out_dir: &str, name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{out_dir}/{name}.lib")
    } else {
        format!("{out_dir}/lib{name}.a")
    }
}
