fn main() {
    baracuda_build::emit_rerun_hints();
    // cuVS is dlopen'd at runtime; no link step. `CUVS_ROOT` lets a caller
    // point the loader at a non-standard RAPIDS install (mirrors NCCL_ROOT).
    println!("cargo:rerun-if-env-changed=CUVS_ROOT");
    println!("cargo:rerun-if-env-changed=RAFT_ROOT");
}
