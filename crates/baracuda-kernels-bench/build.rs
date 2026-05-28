//! Build script for `baracuda-kernels-bench`.
//!
//! Conditionally compiles CUDA-L2 (deepreinforce-ai) HGEMM kernel
//! wrappers when the `cuda_l2` feature is enabled. The wrappers live
//! under `external/cuda-l2-probes/` (MIT licensed, mirrored from the
//! `external/cuda-l2/` git checkout). Each wrapper exposes a single
//! C ABI launcher symbol for the `gemm_vs_cuda_l2` bench.
//!
//! This crate is `publish = false`, so the build script is only
//! exercised by local `cargo bench` runs. The `cuda_l2` feature pulls
//! in `baracuda-forge` + `baracuda-cutlass-sys` (already in the
//! workspace) to get nvcc + CUTLASS headers — no new external deps.
//!
//! Without the `cuda_l2` feature this script is a no-op, and the
//! `gemm_vs_cuda_l2` bench falls back to a documentation-only mode
//! that records pre-measured numbers (see the bench file for
//! provenance).

#[cfg(feature = "cuda_l2")]
fn build_cuda_l2() {
    use baracuda_forge::KernelBuilder;

    let sources = [
        "external/cuda-l2-probes/wrapper_m128.cu",
        "external/cuda-l2-probes/wrapper_m2048.cu",
    ];

    // Skip silently if the upstream CUDA-L2 checkout is missing (e.g.
    // fresh clone without `git submodule update` or the sibling
    // `external/cuda-l2/` directory). The wrappers don't `#include`
    // the upstream sources directly — they inline the kernel bodies
    // for ODR isolation — so this check is only here as a sanity
    // hint, not a hard requirement.
    for s in &sources {
        if !std::path::Path::new(s).is_file() {
            println!(
                "cargo:warning=baracuda-kernels-bench: CUDA-L2 wrapper missing: {s} \
                 — disabling `cuda_l2` feature build"
            );
            return;
        }
    }

    let cutlass_include = std::env::var("DEP_CUTLASS_INCLUDE").expect(
        "DEP_CUTLASS_INCLUDE not set; `cuda_l2` feature requires \
         baracuda-cutlass-sys to expose CUTLASS headers.",
    );

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
    let lib_name = "baracuda_cuda_l2_probes";
    let lib_file = format!("{out_dir}/lib{lib_name}.a");
    #[cfg(target_os = "windows")]
    let lib_file = format!("{out_dir}/{lib_name}.lib");

    let mut builder = KernelBuilder::new()
        .source_files(sources.iter().copied())
        .include_path(&cutlass_include)
        .out_dir(&out_dir)
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__");

    // Same arch dispatch as baracuda-kernels-sys. Default sm_80; bump
    // to sm_89 / sm_90a if the corresponding bench-crate feature is on.
    if cfg!(feature = "sm90a") {
        builder = builder.compute_cap(90);
    } else if cfg!(feature = "sm89") {
        builder = builder.compute_cap(89);
    } else {
        builder = builder.compute_cap(80);
    }

    builder
        .build_lib(&lib_file)
        .expect("baracuda-kernels-bench: CUDA-L2 probe nvcc build failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static={lib_name}");
    // cudart for cudaFuncSetAttribute + kernel launch.
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=external/cuda-l2-probes");

    #[cfg(feature = "cuda_l2")]
    build_cuda_l2();
}
