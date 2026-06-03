//! `baracuda-nccl-sys` — internal FFI / support crate.

fn main() {
    baracuda_build::emit_rerun_hints();
    println!("cargo:rerun-if-env-changed=NCCL_ROOT");
}
