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
