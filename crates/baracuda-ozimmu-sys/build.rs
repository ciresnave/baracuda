//! Build script for `baracuda-ozimmu-sys` (Phase 44b — clean fork).
//!
//! Compiles the baracuda-owned ozIMMU implementation into a single
//! static archive. Phase 44b retired the upstream vendor tree: all
//! sources now live under `cuda/`, and the cutf submodule has been
//! eliminated entirely (the ~360 LOC of useful FP / cp_async
//! utilities were folded into `baracuda-kernels-sys`).
//!
//! Architecture coverage matches the rest of the baracuda CUDA
//! crates: a single default arch (sm_89, the RTX 4070 test target)
//! with `-gencode=` add-ons for sm_80 and sm_90a so one archive
//! runs on Ampere / Ada / Hopper without re-build. Override the
//! default arch via `CUDA_COMPUTE_CAP` per the standard
//! `baracuda-forge` mechanism.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    if env::var_os("DOCS_RS").is_some() {
        println!("cargo:warning=baracuda-ozimmu-sys: DOCS_RS=1 detected; skipping nvcc build.");
        return;
    }

    // Gate the actual nvcc invocation behind the `build-vendor` feature
    // (carried over from Phase 44 — the feature name is now slightly
    // misleading since we no longer vendor in the third-party sense,
    // but the gate semantics are the same: only build the static
    // archive when a downstream caller actually wants the Ozaki path).
    if std::env::var_os("CARGO_FEATURE_BUILD_VENDOR").is_none() {
        println!(
            "cargo:warning=baracuda-ozimmu-sys: `build-vendor` feature off; skipping nvcc build. \
             Enable via `--features build-vendor` (or `baracuda-kernels`'s `ozimmu` feature)."
        );
        return;
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
    let lib_name = "baracuda_ozimmu";
    let lib_file = lib_static_path(&out_dir, lib_name);

    // All sources live under `cuda/` after the Phase 44b clean-fork.
    let cuda_src = PathBuf::from("cuda");
    let all_sources: Vec<PathBuf> = vec![
        cuda_src.join("handle.cu"),
        cuda_src.join("config.cu"),
        cuda_src.join("split.cu"),
        cuda_src.join("gemm.cu"),
        cuda_src.join("cublas_helper.cu"),
        cuda_src.join("baracuda_shim.cu"),
    ];

    for src in &all_sources {
        if !src.exists() {
            panic!(
                "baracuda-ozimmu-sys: source file missing: {} \
                 (cuda/ tree corrupted or not yet checked in?)",
                src.display()
            );
        }
    }

    // Need to find the baracuda-kernels-sys include path so we can
    // pick up `baracuda_fp_bits.cuh`. The kernels-sys crate puts its
    // headers in `kernels/include` relative to its own crate root;
    // we resolve that via a workspace-relative path from this
    // crate's CARGO_MANIFEST_DIR.
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo"),
    );
    let kernels_sys_include = manifest_dir
        .parent()
        .expect("manifest_dir must have a parent")
        .join("baracuda-kernels-sys/kernels/include");
    if !kernels_sys_include.exists() {
        panic!(
            "baracuda-ozimmu-sys: expected `baracuda-kernels-sys/kernels/include/` not found at {}",
            kernels_sys_include.display()
        );
    }

    use baracuda_forge::KernelBuilder;
    let mut builder = KernelBuilder::new()
        .source_files(all_sources.iter().map(|p| p.to_string_lossy().into_owned()))
        .include_path("cuda/include")
        .include_path("cuda")
        .include_path(kernels_sys_include.to_string_lossy().into_owned())
        // gemm.cu uses CUB-style separable compilation; preserve
        // `-rdc=true` to allow host-device template instantiations
        // across TUs. Matches the upstream CMakeLists'
        // `CUDA_SEPARABLE_COMPILATION ON`.
        .arg("-rdc=true")
        .out_dir(&out_dir);

    // Architecture: sm_89 default (RTX 4070 test target). Add sm_80
    // + sm_90a as additional gencode targets so the same archive runs
    // across Ampere / Ada / Hopper without rebuild.
    builder = builder
        .compute_cap(89)
        .arg("-gencode=arch=compute_80,code=sm_80")
        .arg("-gencode=arch=compute_90a,code=sm_90a");

    builder
        .build_lib(&lib_file)
        .expect("baracuda-ozimmu-sys: nvcc build failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static={lib_name}");

    // Runtime libraries the static archive references.
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // C++ runtime — Linux uses libstdc++; on Windows, MSVC links
    // its own runtime automatically (no explicit flag needed).
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // CUDA device-runtime link library for `-rdc=true` separable
    // compilation. Required because gemm.cu spans multiple TUs with
    // device-side template instantiations.
    println!("cargo:rustc-link-lib=static=cudadevrt");
}

fn lib_static_path(out_dir: &str, name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{out_dir}/{name}.lib")
    } else {
        format!("{out_dir}/lib{name}.a")
    }
}
