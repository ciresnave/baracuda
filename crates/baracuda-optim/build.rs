//! Build script for `baracuda-optim` — compiles the Apex-vendored
//! optimizer functors + baracuda C-ABI shim into a single static
//! archive via baracuda-forge.
//!
//! The vendored headers live in `vendor/apex/` and are header-only
//! (`.cuh` files, instantiated through the shim TU). The build set
//! is therefore one single .cu file: `csrc/baracuda_optim_shim.cu`.
//!
//! Architecture coverage: matches baracuda-kernels-sys conventions —
//! sm_80 default, sm_89 / sm_90a as optional fanout.
//!
//! Docs.rs short-circuit: same pattern as the other vendored sys
//! crates (`DOCS_RS=1` skips the nvcc invocation).

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/apex");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    if env::var_os("DOCS_RS").is_some() {
        println!(
            "cargo:warning=baracuda-optim: DOCS_RS=1 detected; skipping nvcc build."
        );
        return;
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
    let lib_name = "baracuda_optim";
    let lib_file = lib_static_path(&out_dir, lib_name);

    let shim = "csrc/baracuda_optim_shim.cu";
    if !std::path::Path::new(shim).exists() {
        panic!("baracuda-optim: missing shim TU: {shim}");
    }

    use baracuda_forge::KernelBuilder;
    let mut builder = KernelBuilder::new()
        .source_files([shim])
        // Vendor headers (.cuh) are template-instantiated through the
        // shim; we just need their include dir on the path.
        .include_path("vendor/apex")
        // expt-relaxed-constexpr — needed for the half/bfloat16 mixed-
        // arithmetic in the Adam / SGD functors' static_cast<float>(x)
        // casts.
        .arg("--expt-relaxed-constexpr")
        // expt-extended-lambda — not strictly needed for the current
        // functor set (no host-side lambdas), but kept for symmetry
        // with the FA2 / mHC vendor builds and to future-proof against
        // adding lambda-based dispatch helpers later.
        .arg("--expt-extended-lambda")
        .out_dir(&out_dir);

    if cfg!(feature = "sm90a") {
        builder = builder.compute_cap(90);
    } else if cfg!(feature = "sm89") {
        builder = builder.compute_cap(89);
    } else if cfg!(feature = "sm80") {
        builder = builder.compute_cap(80);
    } else {
        // No arch feature on — emit the standard "no-op build" warning
        // and short-circuit. The FFI extern blocks in src/lib.rs still
        // compile-check; link will only resolve if a caller exercises
        // them (and they haven't enabled an arch feature).
        println!(
            "cargo:warning=baracuda-optim: no arch feature on; skipping nvcc build. \
             Enable one of `sm80`/`sm89`/`sm90a` (or `baracuda-kernels`'s `optim` \
             feature, which forwards `sm80` automatically)."
        );
        return;
    }

    builder
        .build_lib(&lib_file)
        .expect("baracuda-optim: nvcc build failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static={lib_name}");

    // CUDA Runtime — the shim TU references cudaGetLastError,
    // cudaMemsetAsync, and the cudart kernel-launch runtime.
    println!("cargo:rustc-link-lib=dylib=cudart");

    // C++ runtime — template-instantiated functors pull in libstdc++ /
    // MSVCP for std::memset (the per-launch metadata zero-fill).
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn lib_static_path(out_dir: &str, name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{out_dir}/{name}.lib")
    } else {
        format!("{out_dir}/lib{name}.a")
    }
}
