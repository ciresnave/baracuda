//! Build script for the `baracuda-examples` crate.
//!
//! Currently only handles the `forge-hello` feature: when enabled, compiles
//! `kernels/forge_hello.cu` to PTX via `baracuda-forge`. The PTX text is
//! then `include_str!`-ed by `forge_hello.rs` at runtime.
//!
//! When the feature is off, this script is a no-op so non-CUDA developers
//! can still `cargo check --workspace`. Under `DOCS_RS=1` the kernel build is
//! skipped and an empty placeholder PTX is written (type-check only).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_FORGE_HELLO").is_some() {
        build_forge_hello();
    }
}

fn build_forge_hello() {
    use baracuda_forge::KernelBuilder;

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");

    // DOCS_RS=1 (docs.rs, and CI's no-CUDA feature check) means "type-check only,
    // no toolkit": write an empty PTX placeholder so `include_str!` resolves and
    // the example's Rust still compiles. The result is never run.
    if std::env::var_os("DOCS_RS").is_some() {
        std::fs::write(std::path::Path::new(&out_dir).join("forge_hello.ptx"), "")
            .expect("write placeholder forge_hello.ptx");
        println!(
            "cargo:warning=baracuda-examples: DOCS_RS=1 detected; skipping the nvcc build of forge_hello.cu"
        );
        return;
    }

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
