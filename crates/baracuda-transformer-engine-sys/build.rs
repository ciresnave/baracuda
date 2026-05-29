//! Build script for `baracuda-transformer-engine-sys` (Phase 55).
//!
//! Compiles the vendored TransformerEngine FP8 cast + delayed-scaling
//! recipe sources together with the baracuda C-ABI shim into a single
//! static archive via `baracuda-forge`.
//!
//! The shim TU (`csrc/baracuda_te_shim.cu`) hides the C++-template-
//! heavy upstream surface behind a flat `extern "C"` API — same
//! pattern as `baracuda-ozimmu-sys/cuda/baracuda_shim.cu` and
//! `baracuda-optim/csrc/baracuda_optim_shim.cu`. The Rust side never
//! sees TE's `Tensor`/`NVTEDType` types directly; it works in terms
//! of `cudaStream_t` + raw device pointers.
//!
//! Docs.rs short-circuit, build-vendor / arch-feature gating, and the
//! Linux-vs-Windows `lib*.a` / `*.lib` naming convention all follow
//! the pattern established by the other baracuda `-sys` crates.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=vendor/transformer-engine");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    if env::var_os("DOCS_RS").is_some() {
        println!(
            "cargo:warning=baracuda-transformer-engine-sys: DOCS_RS=1 detected; skipping nvcc build."
        );
        return;
    }

    // Gate the actual build behind the `build-vendor` feature so a
    // bare `cargo check --workspace` doesn't try to invoke nvcc.
    if env::var_os("CARGO_FEATURE_BUILD_VENDOR").is_none() {
        println!(
            "cargo:warning=baracuda-transformer-engine-sys: `build-vendor` feature off; skipping nvcc build. \
             Enable via `--features build-vendor` (or `baracuda-kernels`'s `tensor_engine` feature)."
        );
        return;
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
    let lib_name = "baracuda_transformer_engine";
    let lib_file = lib_static_path(&out_dir, lib_name);

    // The build set is one shim TU plus the vendored cast/recipe
    // sources. Cast/recipe were chosen because:
    //   1. They are the load-bearing piece of TE's differentiated
    //      value (delayed-scaling with amax history).
    //   2. They have no cuDNN dep (only `fused_attn` does, and we
    //      skip that one).
    //   3. They have no pybind11 dep at the device-side level.
    let shim = PathBuf::from("csrc/baracuda_te_shim.cu");
    if !shim.exists() {
        panic!(
            "baracuda-transformer-engine-sys: missing shim TU at {}",
            shim.display()
        );
    }

    let mut sources: Vec<PathBuf> = vec![shim.clone()];

    // Optionally pick up additional vendored .cu files if present.
    // Phase 55 ships with the shim-only path (cast + recipe are
    // implemented inline in the shim against TE's published
    // algorithms — same pattern as Phase 49's optim, where the host-
    // side `multi_tensor_apply<T>` launcher was replaced entirely
    // rather than vendored). The hook below lets future phases drop
    // in additional .cu files without touching this build script.
    let vendor_dir = PathBuf::from("vendor/transformer-engine");
    if vendor_dir.exists() {
        collect_cu_files(&vendor_dir, &mut sources);
    }

    use baracuda_forge::KernelBuilder;
    let mut builder = KernelBuilder::new()
        .source_files(sources.iter().map(|p| p.to_string_lossy().into_owned()))
        .include_path("csrc")
        .include_path("vendor/transformer-engine")
        // expt-relaxed-constexpr — needed for the FP8 cast intrinsics
        // (`__nv_cvt_float_to_fp8` etc. from `<cuda_fp8.h>`) used
        // inside the shim's fused-amax-cast kernel.
        .arg("--expt-relaxed-constexpr")
        // expt-extended-lambda — kept for symmetry with the other
        // baracuda vendor builds; not strictly required for the
        // current shim, but keeps the door open for future
        // host-side lambda dispatch.
        .arg("--expt-extended-lambda")
        .out_dir(&out_dir);

    if cfg!(feature = "sm90a") {
        builder = builder.compute_cap(90);
    } else if cfg!(feature = "sm89") {
        builder = builder.compute_cap(89);
    } else if cfg!(feature = "sm80") {
        builder = builder.compute_cap(80);
    } else {
        println!(
            "cargo:warning=baracuda-transformer-engine-sys: no arch feature on; skipping nvcc build. \
             Enable one of `sm80`/`sm89`/`sm90a` (or `baracuda-kernels`'s `tensor_engine` feature)."
        );
        return;
    }

    builder
        .build_lib(&lib_file)
        .expect("baracuda-transformer-engine-sys: nvcc build failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static={lib_name}");

    // CUDA Runtime — the shim TU calls into `cudaMalloc` /
    // `cudaMemsetAsync` / `cudaMemcpyAsync` / kernel-launch runtime.
    println!("cargo:rustc-link-lib=dylib=cudart");

    // C++ runtime for the std::* utilities used in the shim.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn collect_cu_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_cu_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("cu") {
            out.push(path);
        }
    }
}

fn lib_static_path(out_dir: &str, name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{out_dir}/{name}.lib")
    } else {
        format!("{out_dir}/lib{name}.a")
    }
}
