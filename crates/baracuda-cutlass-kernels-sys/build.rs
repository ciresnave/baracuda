//! Build script for `baracuda-cutlass-kernels-sys`.
//!
//! Compiles the curated CUTLASS template instantiations under `kernels/`
//! into a static library via `baracuda-forge`. The set of files compiled
//! is selected by Cargo features:
//!
//! - `sm80` (default): Ampere (also runs on Ada and as fallback on Hopper).
//! - `sm90a`: Hopper-specialized kernels.
//! - `cutlass-2-11`: forwarded to `baracuda-cutlass-sys`; mutually exclusive
//!   with `sm90a`.
//!
//! When no arch features are enabled the build is a no-op (the safe layer
//! will return `Error::Unsupported` at runtime).

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    // docs.rs has no nvcc and no network. baracuda-cutlass-sys already
    // emits a stub include path in DOCS_RS mode; mirror that here by
    // skipping kernel compilation entirely. The Rust source compiles
    // fine — the unresolved extern "C" symbols are only needed at link
    // time, which docs.rs doesn't perform.
    if env::var_os("DOCS_RS").is_some() {
        println!(
            "cargo:warning=baracuda-cutlass-kernels-sys: DOCS_RS=1 detected; skipping nvcc build."
        );
        return;
    }

    if cfg!(all(feature = "sm90a", feature = "cutlass-2-11")) {
        panic!(
            "baracuda-cutlass-kernels-sys: features `sm90a` and `cutlass-2-11` \
             are mutually exclusive. Hopper-specialized kernels require the \
             CUTLASS 4.x line; the 2.11 line is the CUDA-11.4 compatibility \
             path."
        );
    }

    let kernels: Vec<&'static str> = collect_kernel_files();

    if kernels.is_empty() {
        println!(
            "cargo:warning=baracuda-cutlass-kernels-sys: no kernel sources to compile (no arch feature, or kernel files not present yet)."
        );
        return;
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
    let lib_name = "baracuda_cutlass_kernels";
    let lib_file = lib_static_path(&out_dir, lib_name);

    let cutlass_include = env::var("DEP_CUTLASS_INCLUDE").expect(
        "DEP_CUTLASS_INCLUDE not set by baracuda-cutlass-sys. \
         Ensure baracuda-cutlass-sys is in the dependency tree (it is a direct \
         runtime dependency of this crate).",
    );

    use baracuda_forge::KernelBuilder;
    let mut builder = KernelBuilder::new()
        .source_files(kernels.iter().map(|k| format!("kernels/{k}")))
        .include_path(&cutlass_include)
        // Vendored CUTLASS partial-specialization headers (e.g. the SIMT
        // broadcast-epilogue routing for f32-SIMT bias kernels).
        .include_path("kernels/include")
        .out_dir(&out_dir);

    if let Ok(root) = env::var("DEP_CUTLASS_ROOT") {
        let util = PathBuf::from(&root).join("tools").join("util").join("include");
        if util.is_dir() {
            builder = builder.include_path(util);
        }
    }

    if cfg!(feature = "sm90a") {
        builder = builder.compute_cap(90).arg("-DBARACUDA_CUTLASS_HAS_SM90A");
    } else if cfg!(feature = "sm80") {
        builder = builder.compute_cap(80);
    }

    builder
        .build_lib(&lib_file)
        .expect("baracuda-cutlass-kernels-sys: nvcc build failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static={lib_name}");

    // CUTLASS GEMMs depend on the CUDA Runtime (cudart) for stream types
    // and a handful of helpers that the kernel objects reference.
    println!("cargo:rustc-link-lib=dylib=cudart");

    // C++ runtime — required because the static lib contains template
    // instantiations that pull in libstdc++ / msvcprt symbols.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn collect_kernel_files() -> Vec<&'static str> {
    let mut kernels = Vec::new();

    if cfg!(any(feature = "sm80", feature = "sm90a")) {
        // sm_80 kernels (also runs forward-compatibly on Ada and Hopper).
        if cfg!(feature = "sm80") {
            for f in &[
                "gemm_rcr_sm80.cu",
                "gemm_rcr_sm80_bias.cu",
                "gemm_rrr_sm80.cu",
                "gemm_rrr_sm80_bias.cu",
                "gemm_tf32_rcr_sm80.cu",
                "gemm_tf32_rcr_sm80_bias.cu",
                "gemm_tf32_rrr_sm80.cu",
                "gemm_tf32_rrr_sm80_bias.cu",
                "gemm_f32_simt_rcr_sm80.cu",
                "gemm_f32_simt_rcr_sm80_bias.cu",
                "gemm_f32_simt_rrr_sm80.cu",
                "gemm_f32_simt_rrr_sm80_bias.cu",
                "gemm_f64_rcr_sm80.cu",
                "gemm_f64_rcr_sm80_bias.cu",
                "gemm_f64_rrr_sm80.cu",
                "gemm_f64_rrr_sm80_bias.cu",
                // Phase 2: int8 GEMM (s8 + u8), RCR layout only, identity
                // and bias family (with f32 or i32 bias).
                // `LinearCombinationClamp` for identity,
                // `LinearCombinationBiasElementwise` for bias. RRR layout
                // is deferred — CUTLASS 4.2.0 lacks 8-bit `Congruous`
                // warp iterators, so `RowMajor × RowMajor × OpClassTensorOp`
                // can't be instantiated without vendoring a custom
                // `MmaTensorOpMultiplicandTileIterator` specialization.
                // Tracked as a follow-up for the next milestone.
                "gemm_s8_rcr_sm80.cu",
                "gemm_u8_rcr_sm80.cu",
                "gemm_s8_rcr_sm80_bias.cu",
                "gemm_u8_rcr_sm80_bias.cu",
                "gemm_batched_rcr_sm80.cu",
                "grouped_gemm_rcr_sm80.cu",
            ] {
                if std::path::Path::new(&format!("kernels/{f}")).exists() {
                    kernels.push(*f);
                }
            }
        }
        // sm_90a kernels deferred until Hopper hardware available for
        // validation. The Rust selection wiring is already in place.
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
