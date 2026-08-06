//! The NVRTC on-demand JIT compiler — the production source → PTX
//! [`Compiler`](baracuda_kernelgen::Compiler). Carved out of
//! `baracuda-kernelgen`'s `jit.rs` (carve step 2). The whole module is behind
//! `--features nvrtc` (wired in `lib.rs`), so the per-item `#[cfg]` the source
//! carried is dropped here.

use baracuda_kernel_vocab::ArchSku;
use baracuda_kernelgen::{ArtifactKind, Compiler};

/// The production on-demand compiler: nvrtc source → PTX. Feature-gated
/// (`--features nvrtc`) because it needs the nvrtc runtime; constructed per target
/// arch (the `--gpu-architecture` flag the schedule was keyed for).
#[derive(Copy, Clone, Debug)]
pub struct NvrtcCompiler {
    arch: ArchSku,
}

impl NvrtcCompiler {
    /// A compiler targeting `arch` (the request's `target` SKU).
    #[must_use]
    pub fn new(arch: ArchSku) -> Self {
        Self { arch }
    }
}

impl Compiler for NvrtcCompiler {
    fn compile(&self, source: &str, entry: &str, _max_compile_ms: u32) -> Result<Vec<u8>, String> {
        // nvrtc has no compile-deadline API; `max_compile_ms` gates optimization
        // depth / the inward e-graph's iteration count at a coarser grain (future).
        // Use the low-level path so a compilation error surfaces the nvrtc log.
        use baracuda_nvrtc::Program;
        let name = format!("{entry}.cu");
        let prog =
            Program::new(source, &name).map_err(|e| format!("nvrtc({entry}) create: {e}"))?;
        let arch = format!("--gpu-architecture={}", arch_flag(self.arch));
        let mut opts = vec![arch];
        // fp16/bf16 kernels `#include <cuda_fp16.h>`/`<cuda_bf16.h>`; headerless
        // nvrtc has no default search path, so point it at the CUDA include dir
        // (env-detected) — without this, f16/bf16 JIT fails to find the header even
        // though the AOT (nvcc) path compiles. Harmless for header-light f32 source.
        if let Some(inc) = cuda_include_dir() {
            opts.push(format!("-I{inc}"));
        }
        let opt_refs: Vec<&str> = opts.iter().map(String::as_str).collect();
        match prog.compile_raw(&opt_refs) {
            Ok(()) => prog
                .ptx()
                .map(String::into_bytes)
                .map_err(|e| format!("nvrtc({entry}) ptx: {e}")),
            Err(e) => {
                let log = prog.log().unwrap_or_default();
                Err(format!(
                    "nvrtc({entry}): {e}\n--- nvrtc log ---\n{}",
                    log.trim()
                ))
            }
        }
    }
    fn artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::Ptx
    }
}

/// The CUDA toolkit `include/` directory (for nvrtc's `-I`), detected from the
/// usual environment vars. `None` if unset/missing — header-light (f32/f64/int)
/// kernels still compile; only the fp16/bf16 headers need it.
fn cuda_include_dir() -> Option<String> {
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(root) = std::env::var(var) {
            let inc = std::path::Path::new(&root).join("include");
            if inc.is_dir() {
                return Some(inc.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// `--gpu-architecture` flag for an [`ArchSku`].
fn arch_flag(arch: ArchSku) -> &'static str {
    match arch {
        ArchSku::Sm80 => "sm_80",
        ArchSku::Sm89 => "sm_89",
        ArchSku::Sm90a => "sm_90a",
    }
}
