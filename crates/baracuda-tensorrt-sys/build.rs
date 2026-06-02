//! Build script for `baracuda-tensorrt-sys`.
//!
//! Default build: a pure dynamic-loader crate — nothing is compiled, and no
//! TensorRT SDK is required.
//!
//! With the `shim` feature: compiles `shim/trt_shim.cpp` (a C++ translation
//! unit that exposes a flat `trt*` C ABI over TensorRT's C++-only runtime) and
//! statically links it. The shim does pure vtable dispatch on pointers handed
//! in from Rust, so it needs the TensorRT SDK *headers* (and the CUDA headers
//! they include for `cudaStream_t`) but links no import library — libnvinfer is
//! still loaded dynamically at runtime via libloading.

use std::env;
use std::path::PathBuf;

fn main() {
    baracuda_build::emit_rerun_hints();
    println!("cargo:rerun-if-changed=shim/trt_shim.cpp");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if env::var_os("CARGO_FEATURE_SHIM").is_none() {
        // Default path: no shim, no SDK needed.
        return;
    }

    if env::var_os("DOCS_RS").is_some() {
        println!("cargo:warning=baracuda-tensorrt-sys: DOCS_RS set; skipping C++ shim build.");
        return;
    }

    let trt_inc = find_tensorrt_include();
    let cuda_inc = find_cuda_include();

    let mut build = cc::Build::new();
    build.cpp(true).file("shim/trt_shim.cpp").include(&trt_inc);
    if let Some(cuda) = &cuda_inc {
        build.include(cuda);
    }
    // TensorRT 10 headers require C++14 or newer.
    build.flag_if_supported("-std=c++14");
    // MSVC: keep exception handling sane for the (noexcept) shim TU.
    build.flag_if_supported("/EHsc");

    // Emits `cargo:rustc-link-lib=static=baracuda_trt_shim` + the link-search
    // path, and (on Linux) the C++ runtime. No libnvinfer import lib is added:
    // the shim references none of its symbols.
    build.compile("baracuda_trt_shim");
}

/// Locate the TensorRT `include/` directory (the one containing `NvInfer.h`).
fn find_tensorrt_include() -> PathBuf {
    if let Some(dir) = env::var_os("TENSORRT_INCLUDE_DIR") {
        let p = PathBuf::from(dir);
        if p.join("NvInferRuntime.h").exists() {
            return p;
        }
    }
    if let Some(root) = env::var_os("TENSORRT_PATH") {
        let p = PathBuf::from(root).join("include");
        if p.join("NvInferRuntime.h").exists() {
            return p;
        }
    }
    panic!(
        "baracuda-tensorrt-sys: `shim` feature is on but the TensorRT SDK headers \
         were not found. Set TENSORRT_INCLUDE_DIR to the directory containing \
         NvInferRuntime.h, or TENSORRT_PATH to the SDK root (its `include/` is \
         used). See crates/baracuda-tensorrt/AUDIT.md."
    );
}

/// Best-effort CUDA `include/` directory (for `cuda_runtime_api.h`, which the
/// TensorRT headers include for `cudaStream_t`). Returns None if not found —
/// the compile then surfaces a clear missing-header error from the C++ toolchain.
fn find_cuda_include() -> Option<PathBuf> {
    if let Some(root) = env::var_os("CUDA_PATH") {
        let p = PathBuf::from(root).join("include");
        if p.join("cuda_runtime_api.h").exists() {
            return Some(p);
        }
    }
    for cand in ["/usr/local/cuda/include", "/usr/include"] {
        let p = PathBuf::from(cand);
        if p.join("cuda_runtime_api.h").exists() {
            return Some(p);
        }
    }
    None
}
