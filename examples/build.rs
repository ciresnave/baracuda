//! Build script for the `baracuda-examples` crate.
//!
//! Currently only handles the `forge-hello` feature: when enabled, compiles
//! `kernels/forge_hello.cu` to PTX via `baracuda-forge`. The PTX text is
//! then `include_str!`-ed by `forge_hello.rs` at runtime.
//!
//! When the feature is off, this script is a no-op so non-CUDA developers
//! can still `cargo check --workspace`.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_FORGE_HELLO").is_some() {
        build_forge_hello();
    }
}

fn build_forge_hello() {
    use baracuda_forge::KernelBuilder;

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");

    let result = KernelBuilder::new()
        .source_files(["kernels/forge_hello.cu"])
        .out_dir(&out_dir)
        .build_ptx();

    match result {
        Ok(_) => {
            println!("cargo:rustc-cfg=forge_hello_built");
        }
        Err(e) => {
            // Fail loudly: the feature was explicitly enabled, so build
            // failure is the user's signal that their CUDA toolkit isn't
            // wired up. Don't silently skip.
            panic!(
                "forge-hello feature is enabled but the kernel failed to compile.\n\
                 Most likely your CUDA toolkit isn't on PATH or nvcc can't find a \
                 supported host compiler. Underlying error:\n  {e}"
            );
        }
    }
}
