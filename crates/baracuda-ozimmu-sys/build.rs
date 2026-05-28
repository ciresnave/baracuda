//! Build script for `baracuda-ozimmu-sys`.
//!
//! Compiles the vendored ozIMMU library + the baracuda shim TU into a single
//! static archive. Patches applied to the upstream source are documented in
//! `vendor/ozimmu/VENDOR.md`.
//!
//! Architecture coverage: the upstream `CMakeLists.txt` sets
//! `CMAKE_CUDA_ARCHITECTURES 90 89 86 80`. We match the workspace pattern
//! used by `baracuda-cutlass-kernels-sys` and use a single default arch
//! (sm_89, the Phase 44 primary target on RTX 4070) with `-gencode=` add-
//! ons for sm_80 and sm_90a — so a single archive runs on Ampere, Ada, and
//! Hopper without re-build. Override the default arch via `CUDA_COMPUTE_CAP`
//! per the standard `baracuda-forge` mechanism.
//!
//! Excludes `src/cublas.cu` and `src/culip.cu` from the vendored source
//! list — those are the LD_PRELOAD interceptor + Linux-only profiler TUs
//! that the static-link integration replaces with `csrc/ozimmu_baracuda_shim.cu`.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/ozimmu");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    if env::var_os("DOCS_RS").is_some() {
        println!(
            "cargo:warning=baracuda-ozimmu-sys: DOCS_RS=1 detected; skipping nvcc build."
        );
        return;
    }

    // Gate the actual nvcc invocation behind the `build-vendor` feature.
    // When OFF (the default), this build.rs is a no-op — the FFI extern
    // blocks in `src/lib.rs` still compile and downstream crates that
    // don't opt into the ozIMMU dispatch path won't link against the
    // (unbuilt) static archive. Activating the feature is the job of
    // `baracuda-ozimmu`'s `default` feature, which the
    // `baracuda-cutlass/ozimmu` (→ `baracuda-kernels/ozimmu`) feature
    // chain pulls in.
    //
    // Why a separate feature rather than gating the whole crate behind
    // `cfg(feature = "ozimmu")` upstream: the FFI types in `src/lib.rs`
    // are useful documentation even without the vendored library being
    // present, and gating the WHOLE module body out makes the lib.rs
    // shape harder to grep.
    if std::env::var_os("CARGO_FEATURE_BUILD_VENDOR").is_none() {
        println!(
            "cargo:warning=baracuda-ozimmu-sys: `build-vendor` feature off; skipping nvcc build. \
             Enable via `--features build-vendor` (or `baracuda-kernels`'s `ozimmu` feature)."
        );
        return;
    }

    // ozIMMU's vendored sources currently rely on `__uint128_t` /
    // `__int128_t` (GCC/Clang extensions used by `cutf/experimental/fp.hpp`
    // and `vendor/ozimmu/src/split.cu`). MSVC's nvcc host compiler
    // doesn't define those types, so the Phase 44 integration only
    // builds the vendored library on Linux. Windows callers see a
    // graceful "not built" error at runtime through the FFI extern
    // blocks resolving to unresolved-symbol-at-link, which `cargo
    // check` doesn't trigger (no link step), so the workspace check
    // stays green on Windows even with the feature enabled.
    if !cfg!(target_os = "linux") {
        println!(
            "cargo:warning=baracuda-ozimmu-sys: target_os != linux; ozIMMU vendored source \
             uses `__uint128_t` (a GCC/Clang extension) that MSVC's nvcc host compiler doesn't \
             provide. The static archive is not produced — link will fail if downstream code \
             references the FFI symbols. To use the ozIMMU backend on Windows, build under \
             WSL2 (where nvcc dispatches to g++) or wait for a follow-up phase that ports \
             the int128 paths to fixed-width Win32 intrinsics."
        );
        return;
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
    let lib_name = "baracuda_ozimmu";
    let lib_file = lib_static_path(&out_dir, lib_name);

    // The vendored source set we compile. The two excluded TUs are
    // documented above and in `vendor/ozimmu/VENDOR.md`.
    let vendor_src = PathBuf::from("vendor/ozimmu/src");
    let vendored_files: Vec<PathBuf> = vec![
        vendor_src.join("handle.cu"),
        vendor_src.join("config.cu"),
        vendor_src.join("split.cu"),
        vendor_src.join("gemm.cu"),
        vendor_src.join("cublas_helper.cu"),
    ];
    let shim_file = PathBuf::from("csrc/ozimmu_baracuda_shim.cu");

    let all_sources: Vec<PathBuf> = vendored_files
        .iter()
        .cloned()
        .chain(std::iter::once(shim_file.clone()))
        .collect();

    // Verify every source exists before invoking nvcc — a missing vendor
    // file would otherwise surface as an opaque nvcc error.
    for src in &all_sources {
        if !src.exists() {
            panic!(
                "baracuda-ozimmu-sys: source file missing: {} \
                 (vendor tree corrupted or not yet checked in?)",
                src.display()
            );
        }
    }

    use baracuda_forge::KernelBuilder;
    let mut builder = KernelBuilder::new()
        .source_files(all_sources.iter().map(|p| p.to_string_lossy().into_owned()))
        .include_path("vendor/ozimmu/include")
        .include_path("vendor/ozimmu/src")
        .include_path("vendor/ozimmu/src/cutf/include")
        // Activate the direct-link patch in `vendor/ozimmu/src/utils.hpp`
        // (drops the dlfcn.h / unistd.h includes + rewrites
        // `ozIMMU_get_function_pointer` to call our shim).
        .arg("-DOZIMMU_BARACUDA_DIRECT_LINK")
        // ozIMMU's gemm.cu uses CUB-style separable compilation; keep
        // that working by passing `-rdc=true` per the upstream
        // CMakeLists' `CUDA_SEPARABLE_COMPILATION ON`. Required for
        // host-device template instantiations across TUs.
        .arg("-rdc=true")
        // Suppress noisy upstream warnings that aren't actionable for
        // us. Specifically: `field-of-reference initialization order`
        // in `handle.hpp`'s POD struct, and `narrowing conversion`
        // from `std::size_t` → `int` in `split.cu`'s legacy paths.
        .arg("-diag-suppress=20012")  // suppressed: field init order
        .arg("-diag-suppress=20013")  // suppressed: type qualifier ignored
        .out_dir(&out_dir);

    // Architecture: sm_89 default (RTX 4070 Phase 44 test target).
    // Add sm_80 + sm_90a as additional gencode targets so the same
    // archive runs across Ampere/Ada/Hopper without rebuild. Per
    // `feedback_verify_on_real_hw.md` the primary test surface is the
    // Ada 4070; the other archs are smoke-only.
    builder = builder
        .compute_cap(89)
        .arg("-gencode=arch=compute_80,code=sm_80")
        .arg("-gencode=arch=compute_90a,code=sm_90a");

    builder
        .build_lib(&lib_file)
        .expect("baracuda-ozimmu-sys: nvcc build failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static={lib_name}");

    // Runtime libraries the static archive references:
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // C++ runtime (template instantiations).
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
