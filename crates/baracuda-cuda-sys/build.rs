fn main() {
    // Placeholder: the real build script is wired up in day 5 once
    // baracuda-build exposes CUDA detection + bindgen helpers.
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
