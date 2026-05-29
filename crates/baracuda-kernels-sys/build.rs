//! Build script for `baracuda-kernels-sys`.
//!
//! Compiles the curated bespoke kernel sources under `kernels/` into a
//! static library via `baracuda-forge`. Mirrors the pattern of
//! `baracuda-cutlass-kernels-sys/build.rs` but does **not** depend on
//! CUTLASS — every kernel in this crate is hand-rolled on top of plain
//! CUDA PTX intrinsics (`mma.sync`, `cp.async`, `ldmatrix`).
//!
//! The set of files compiled is selected by Cargo features:
//!
//! - `sm80` (default): Ampere baseline (also runs on Ada / Hopper as
//!   forward-compatible fallback).
//! - `sm89`: Ada Lovelace specializations (adds FP8 kernels).
//! - `sm90a`: Hopper-specialized kernels.
//!
//! When no arch features are enabled the build is a no-op (the safe
//! layer will return an `Unsupported` error at runtime).

use std::env;
#[cfg(feature = "cudnn")]
use std::path::{Path, PathBuf};

/// Try to locate a directory containing the cuDNN import / shared
/// library on this host. Follows the de-facto ecosystem conventions
/// (no single official standard exists — PyTorch / TensorFlow / ONNX
/// Runtime / CMake's `FindCUDNN.cmake` all probe a similar list).
///
/// Search order (first hit wins):
///   1. `CUDNN_PATH` — canonical env override. Points at the cuDNN
///      install root (the directory containing `lib/` and `bin/`).
///   2. `CUDNN_ROOT` / `CUDNN_HOME` — historical alternates used by
///      some upstream tools; treated the same as `CUDNN_PATH`.
///   3. **Windows installer layout** (cuDNN 9+):
///      `C:\Program Files\NVIDIA\CUDNN\v<X.Y>\lib\<cuda_ver>\x64\` —
///      versioned by both cuDNN release AND target CUDA toolkit.
///   4. **Legacy "drop into CUDA toolkit" layout**: `$CUDA_PATH\lib\x64\`
///      on Windows or `$CUDA_PATH/lib64/` on Linux. This was the
///      pre-cuDNN-9 convention (tarball expanded into the CUDA dir).
///   5. **Linux distro paths**: `/usr/lib/x86_64-linux-gnu/` and
///      `/usr/local/cuda/lib64/` — debian / RPM package destinations.
///
/// Returns the directory path for `rustc-link-search=native=…`.
#[cfg(feature = "cudnn")]
fn locate_cudnn_lib_dir() -> Option<String> {
    // (1-2) Explicit env-var roots.
    for var in &["CUDNN_PATH", "CUDNN_ROOT", "CUDNN_HOME"] {
        if let Some(root) = env::var_os(var) {
            let root = PathBuf::from(root);
            if let Some(dir) = probe_install_root(&root) {
                return Some(dir);
            }
        }
    }

    // (3) Windows: standard NVIDIA installer layout.
    #[cfg(target_os = "windows")]
    {
        let std_root = PathBuf::from(r"C:\Program Files\NVIDIA\CUDNN");
        if std_root.is_dir() {
            if let Some(dir) = probe_windows_versioned_root(&std_root) {
                return Some(dir);
            }
        }
    }

    // (4) Legacy "in CUDA toolkit" layout — pre-cuDNN-9.
    if let Some(cuda_path) = env::var_os("CUDA_PATH") {
        let cuda = PathBuf::from(cuda_path);
        #[cfg(target_os = "windows")]
        {
            let x64 = cuda.join("lib").join("x64");
            if x64.join("cudnn.lib").is_file() {
                return Some(x64.to_string_lossy().into_owned());
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let lib64 = cuda.join("lib64");
            if lib64.join("libcudnn.so").is_file() || lib64.join("libcudnn.so.9").is_file() {
                return Some(lib64.to_string_lossy().into_owned());
            }
            let lib = cuda.join("lib");
            if lib.join("libcudnn.so").is_file() || lib.join("libcudnn.so.9").is_file() {
                return Some(lib.to_string_lossy().into_owned());
            }
        }
    }

    // (5) Linux distro paths.
    #[cfg(not(target_os = "windows"))]
    {
        for p in &[
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/lib64",
            "/usr/lib64",
        ] {
            let path = PathBuf::from(p);
            if path.join("libcudnn.so").is_file() || path.join("libcudnn.so.9").is_file() {
                return Some(p.to_string());
            }
        }
    }

    None
}

/// Probe a single cuDNN install root for a usable lib dir. Handles both
/// the new Windows layout (root/lib/<cuda_ver>/x64) and the simpler
/// flat layout (root/lib/x64 on Windows; root/lib on Linux).
#[cfg(feature = "cudnn")]
fn probe_install_root(root: &Path) -> Option<String> {
    let lib = root.join("lib");
    if !lib.is_dir() {
        return None;
    }
    // New Windows layout: lib/<cuda_ver>/x64/cudnn.lib
    if let Some(dir) = pick_cuda_versioned_x64_under_lib(&lib) {
        return Some(dir);
    }
    // Flat Windows layout: lib/x64/cudnn.lib
    let x64 = lib.join("x64");
    if x64.join("cudnn.lib").is_file() {
        return Some(x64.to_string_lossy().into_owned());
    }
    // Flat Linux layout: lib/libcudnn.so
    if lib.join("libcudnn.so").is_file() || lib.join("libcudnn.so.9").is_file() {
        return Some(lib.to_string_lossy().into_owned());
    }
    // Some Linux installs also use lib64 under root.
    let lib64 = root.join("lib64");
    if lib64.join("libcudnn.so").is_file() || lib64.join("libcudnn.so.9").is_file() {
        return Some(lib64.to_string_lossy().into_owned());
    }
    None
}

/// Walk a Windows `C:\Program Files\NVIDIA\CUDNN` root, pick the
/// highest cuDNN version, then the highest CUDA-toolkit sub-dir, then
/// return the `x64` lib path inside.
#[cfg(feature = "cudnn")]
fn probe_windows_versioned_root(std_root: &Path) -> Option<String> {
    let mut versions: Vec<PathBuf> = std::fs::read_dir(std_root)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.starts_with('v'))
                    .unwrap_or(false)
        })
        .collect();
    // Highest cuDNN version first.
    versions.sort();
    versions.reverse();
    for v in versions {
        if let Some(dir) = pick_cuda_versioned_x64_under_lib(&v.join("lib")) {
            return Some(dir);
        }
    }
    None
}

/// Given `.../v<X.Y>/lib/`, pick the `<cuda_ver>/x64/` sub-dir
/// containing `cudnn.lib`. Picks the highest CUDA-toolkit version.
#[cfg(feature = "cudnn")]
fn pick_cuda_versioned_x64_under_lib(lib_dir: &Path) -> Option<String> {
    if !lib_dir.is_dir() {
        return None;
    }
    let mut cuda_dirs: Vec<PathBuf> = std::fs::read_dir(lib_dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    cuda_dirs.sort();
    cuda_dirs.reverse();
    for d in cuda_dirs {
        let x64 = d.join("x64");
        if x64.join("cudnn.lib").is_file() {
            return Some(x64.to_string_lossy().into_owned());
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    // Phase 11.2 — Fuel team feedback #3. Detect Git-for-Windows' fake
    // `link.exe` (actually GNU coreutils `link`) shadowing the MSVC
    // linker on PATH and warn loudly with a fix.
    check_for_fake_link_exe();

    if env::var_os("DOCS_RS").is_some() {
        println!(
            "cargo:warning=baracuda-kernels-sys: DOCS_RS=1 detected; skipping nvcc build."
        );
        return;
    }

    let kernels = collect_kernel_files();

    if kernels.is_empty() {
        println!(
            "cargo:warning=baracuda-kernels-sys: no kernel sources to compile (no arch feature, or kernel files not present yet)."
        );
        return;
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
    let lib_name = "baracuda_kernels";
    let lib_file = lib_static_path(&out_dir, lib_name);

    use baracuda_forge::KernelBuilder;
    let mut builder = KernelBuilder::new()
        .source_files(kernels.iter().map(|k| format!("kernels/{k}")))
        // Bespoke header includes (cp.async / mma.sync helpers, dtype
        // wrappers, warp-reduction primitives, ...).
        .include_path("kernels/include")
        .watch(collect_header_files())
        .out_dir(&out_dir);

    // Phase 42 — Dao-AILab FlashAttention v2 (vendored under
    // `vendor/flash-attention/`). Gated by the `fa2` cargo feature.
    // Compiles the FA2 forward Tier-1 .cu instantiations (head_dim=128,
    // f16+bf16, causal+non-causal) plus a baracuda C-ABI launcher.
    //
    // Required include paths:
    //   1. CUTLASS (FA2 v2.8.3 uses cute::tensor + cutlass templates).
    //   2. FA2 PyTorch-free shim (replaces ATen/c10 dependencies).
    //   3. FA2 src dir (its internal cross-includes are relative).
    //
    // Compiler flags FA2 needs:
    //   * `-std=c++17` — explicit standard. baracuda-forge auto-picks
    //     c++20 for CUDA 12 toolkits; FA2's cute templates compile fine
    //     under c++17/20 so we leave auto-select. Explicit override
    //     here would conflict with the auto-detected setting.
    //   * `--use_fast_math` — already default in baracuda-forge.
    //   * `--expt-relaxed-constexpr` and `--expt-extended-lambda` —
    //     FA2's lambda-based BOOL_SWITCH chain needs these.
    //   * `-DNDEBUG` — disables CUTLASS's debug-only assertions that
    //     emit `printf` calls inside hot kernels.
    //   * `-DFLASH_NAMESPACE=baracuda_fa2` — avoids ODR collisions if
    //     a downstream binary also links the real PyTorch FA2.
    if cfg!(feature = "fa2") {
        let cutlass_include = std::env::var("DEP_CUTLASS_INCLUDE").expect(
            "DEP_CUTLASS_INCLUDE not set; `fa2` feature requires \
             baracuda-cutlass-sys to expose CUTLASS headers. \
             Make sure baracuda-cutlass-sys is in the dependency tree.",
        );
        builder = builder
            // FA2's CUTLASS templates need their internal cute/ + cutlass/
            // header tree on the include path.
            .include_path(&cutlass_include)
            // PyTorch-free shim: satisfies ATen/c10 #includes inside FA2
            // headers. Must precede any real PyTorch install on -I order
            // (forge appends paths in the order added).
            .include_path("vendor/flash-attention/shim")
            // FA2 source dir for the cross-includes between its own
            // headers and .cu files (e.g. `#include "flash.h"`).
            .include_path("vendor/flash-attention/src")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("-DNDEBUG")
            .arg("-DFLASH_NAMESPACE=baracuda_fa2");

        // FA2 forward .cu files. Phase 42 (Tier-1) shipped head_dim=128
        // only; Phase 59a expanded to the full upstream head_dim set
        // {32, 64, 96, 128, 192, 256} × {fp16, bf16} × {causal, non-causal}
        // = 24 .cu files. Upstream FA2 v2.8.3 does NOT ship 160/224/512
        // — those are Tier-3-deferred forever (no upstream sources).
        for f in &[
            // head_dim = 32 (Phase 59a)
            "vendor/flash-attention/src/flash_fwd_hdim32_fp16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim32_bf16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
            // head_dim = 64 (Phase 59a)
            "vendor/flash-attention/src/flash_fwd_hdim64_fp16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim64_bf16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
            // head_dim = 96 (Phase 59a)
            "vendor/flash-attention/src/flash_fwd_hdim96_fp16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim96_bf16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
            // head_dim = 128 (Phase 42 Tier-1)
            "vendor/flash-attention/src/flash_fwd_hdim128_fp16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim128_bf16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
            // head_dim = 192 (Phase 59a)
            "vendor/flash-attention/src/flash_fwd_hdim192_fp16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim192_bf16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
            // head_dim = 256 (Phase 59a)
            "vendor/flash-attention/src/flash_fwd_hdim256_fp16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim256_bf16_sm80.cu",
            "vendor/flash-attention/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
            // launcher (Phase 42 + extended in Phase 59a)
            "kernels/attention/fa2_launcher.cu",
        ] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    // Phase 43 — AndreSlavescu/mHC.cu (vendored under `vendor/mhc/`).
    // Gated by the `mhc` cargo feature. Compiles the baracuda C-ABI
    // launcher for the static-H FW path; the vendored kernels
    // themselves are header-only (`.cuh`) and template-instantiated
    // into the launcher TU.
    //
    // Required include paths:
    //   1. mHC's own include/ tree (mhc_types.h, utils.cuh, profiling.cuh).
    //   2. mHC's src/ tree for cross-includes between the .cuh kernels.
    //
    // Compiler flags mHC needs:
    //   * `--expt-relaxed-constexpr` — required by mHC's cooperative
    //     groups + nv_bfloat16 mixed-use.
    //   * `--expt-extended-lambda` — required by mHC's lambda usage in
    //     warp-reduction wrappers.
    //   * mHC's `fused_rmsnorm_matmul.cuh` includes `<cublasLt.h>` so
    //     the cuBLAS-Lt header dir (part of CUDA toolkit standard
    //     include) is already on -I via forge's default CUDA discovery.
    //     The launcher does not exercise this path at runtime (static
    //     H only) but the header still needs to compile.
    if cfg!(feature = "mhc") {
        builder = builder
            .include_path("vendor/mhc/include")
            .include_path("vendor/mhc/src")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda");

        // Phase 43 — mHC vendor: one launcher TU. The .cuh kernel
        // sources are template-instantiated through the launcher's
        // includes — they are not separately compiled.
        for f in &[
            "kernels/attention/mhc_launcher.cu",
        ] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    // Phase 46 — FlashInfer cherry-pick (vendored Apache-2.0). Compiles
    // 4 baracuda launcher TUs that template-instantiate the FlashInfer
    // header-only kernels for the three families in scope:
    //   1. batched paged-KV decode + paged-KV append
    //   2. sort-free top-K/top-P/min-P sampling
    //   3. cascade attention LSE merge
    if cfg!(feature = "flashinfer") {
        builder = builder
            .include_path("vendor/flashinfer/include")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("-DNDEBUG");
        // Phase 46: all four FlashInfer launchers participate. The paged
        // decode launcher previously hit an MSVC nvcc deduction issue at
        // FlashInfer's `cudaLaunchKernel((void*)kernel, …)` site under
        // the Windows per-thread-default-stream rewrite (`*_ptsz`). The
        // post-alpha.57 consolidation rebuild interposes a local wrapper
        // (`cudaLaunchKernel`-as-macro that pins to the explicit-signature
        // overload) in the launcher TU's preamble, so it compiles cleanly
        // under MSVC nvcc 12.x without touching the vendored header.
        for f in &[
            "kernels/attention/flashinfer_paged_decode_launcher.cu",
            "kernels/attention/flashinfer_paged_kv_append_launcher.cu",
            "kernels/attention/flashinfer_cascade_launcher.cu",
            "kernels/sampling/flashinfer_sampling_launcher.cu",
        ] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    // Phase 53 — bitsandbytes NF4 (vendored under
    // `vendor/bitsandbytes/`). Gated by the `bnb_nf4` cargo feature.
    // Compiles the baracuda C-ABI launcher (`nf4_launcher.cu`) that
    // template-instantiates the dequant + GEMV kernels from the vendored
    // headers (`vendor/bitsandbytes/src/nf4_*.cuh`).
    //
    // Required include paths:
    //   `vendor/bitsandbytes/src/` for the launcher's relative
    //   `#include "../../vendor/bitsandbytes/src/nf4_gemv.cuh"`.
    //
    // No special compiler flags — the NF4 kernels use only standard
    // CUDA built-ins (`__half2float`, `__bfloat162float`, warp shuffles,
    // no cooperative groups, no cp.async).
    if cfg!(feature = "bnb_nf4") {
        builder = builder.include_path("vendor/bitsandbytes/src");
        for f in &[
            "kernels/quantize/nf4_launcher.cu",
        ] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    // Phase 48 — IST-DASLab/marlin (vendored under `vendor/marlin/`,
    // Apache-2.0 with §3 patent grant). Gated by the `marlin` cargo
    // feature. Compiles both the upstream Marlin .cu (which exposes
    // the C++ `marlin_cuda(...)` host entry point) and the baracuda
    // C-ABI launcher that wraps it under
    // `baracuda_kernels_int4_marlin_gemm_*`.
    //
    // Marlin has zero external dependencies (no CUTLASS, no cuBLAS) so
    // no extra include paths needed. The kernel uses Ampere+
    // `mma.sync.m16n8k16` and `cp.async.cg.shared.global`, so we DO
    // NOT compile it under sm_90a-only builds — Marlin's `mma` and
    // `cp.async` syntax target sm_80 / sm_86 / sm_89 specifically;
    // sm_90 requires a WGMMA rewrite (Marlin v2 territory, deferred).
    if cfg!(feature = "marlin") {
        // Marlin's kernel uses constexpr `ceildiv` from inside
        // __global__ / __device__ functions; nvcc needs the
        // experimental relaxed-constexpr flag to accept this.
        // Upstream Marlin builds with the same flag.
        builder = builder.arg("--expt-relaxed-constexpr");
        for f in &[
            "vendor/marlin/src/marlin_cuda_kernel.cu",
            "kernels/quantize/marlin_launcher.cu",
        ] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    // Phase 48 — mit-han-lab/llm-awq (vendored under `vendor/awq/`,
    // MIT — no patent grant; distinct from Marlin's Apache-2.0). Gated
    // by the `awq` cargo feature. The upstream
    // `vendor/awq/src/gemm_cuda_gen.cu` has been patched to strip the
    // PyTorch host wrapper; only the `__global__ template kernel +
    // device helpers remain. The launcher inline-includes the patched
    // .cu (so the .cu is NOT listed as a standalone source) and
    // exposes `baracuda_kernels_int4_awq_gemm_*` + a workspace-size
    // query + a stub dequant entrypoint.
    //
    // AWQ's dequantize.cuh is header-only and is included transitively
    // through the GEMM .cu (not used by the launcher directly).
    //
    // No external dependencies (CUTLASS / cuBLAS / cuDNN); uses only
    // `ldmatrix.sync` + `mma.sync.m16n8k16` (Ampere+).
    if cfg!(feature = "awq") {
        builder = builder.include_path("vendor/awq/src");
        for f in &[
            "kernels/quantize/awq_launcher.cu",
        ] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    // Phase 50 — state-spaces/mamba SSD chunk-scan (Apache-2.0) +
    // Dao-AILab causal-conv1d (BSD-3-Clause). Two bespoke kernel
    // families (kernels hand-written from the algorithmic reference;
    // see `vendor/{mamba,causal-conv1d}/VENDOR.md`). Opens the
    // state-space LLM class (Mamba-2 8B, Codestral-Mamba, Falcon-Mamba,
    // Zamba2). Phase 50b adds the Mamba-1 selective_scan sibling
    // (Mamba-7B class). All gated behind the same `mamba` cargo
    // feature.
    if cfg!(feature = "mamba") {
        for f in &[
            // Phase 50 — causal-conv1d (Mamba's depthwise conv primitive).
            "kernels/conv/causal_conv1d_fp.cu",
            "kernels/conv/causal_conv1d_backward_fp.cu",
            // Phase 50 — Mamba-2 SSD chunk-scan.
            "kernels/ssd/ssd_chunk_scan_fp.cu",
            "kernels/ssd/ssd_chunk_scan_backward_fp.cu",
            // Phase 50b — Mamba-1 selective_scan.
            "kernels/ssd/selective_scan_fp.cu",
            "kernels/ssd/selective_scan_backward_fp.cu",
        ] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    // Phase 54 — xFormers cherry-pick (BSD-3-Clause). Two independent
    // bespoke kernel families ported clean-room from the xFormers
    // algorithmic reference (no source vendored verbatim; see
    // `vendor/xformers/VENDOR.md`):
    //   * `xformers_blocksparse` — block-sparse SDPA FW. Iterates only
    //     the active (q_block, k_block) pairs per a caller-supplied
    //     pattern; real speedup on long-context with sparse mask.
    //   * `xformers_sparse24` — 2:4 structured sparsity GEMM.
    //     Trailblazer uses an inflate-then-dense path (correctness
    //     first); sparse-tensor-core perf deferred to Tier 2.
    // Both features are independent and off by default.
    if cfg!(feature = "xformers_blocksparse") {
        for f in &["kernels/attention/sdpa_block_sparse_fp.cu"] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }
    if cfg!(feature = "xformers_sparse24") {
        for f in &["kernels/gemm/gemm_sparse24_fp.cu"] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    // Phase 56 — Ring Attention (sequence-parallel attention). Bespoke
    // kernel hand-ported from the algorithmic reference (Liu, Yan,
    // Abbeel 2023 — arXiv:2310.01889; upstream JAX reference at
    // https://github.com/lhao499/RingAttention under Apache-2.0). No
    // source vendored — clean-room CUDA implementation. The Rust plan
    // sits behind the `ring_attention` cargo feature on
    // `baracuda-kernels` and pulls in `baracuda-nccl` for the
    // inter-rank K/V chunk rotation. Single .cu instantiates f16 + bf16
    // step + finalize launchers plus a dtype-independent init helper.
    if cfg!(feature = "ring_attention") {
        for f in &["kernels/attention/ring_attention_kernel.cu"] {
            if std::path::Path::new(f).exists() {
                builder = builder.source_files([*f]);
            }
        }
    }

    if cfg!(feature = "sm90a") {
        builder = builder.compute_cap(90).arg("-DBARACUDA_KERNELS_HAS_SM90A");
    } else if cfg!(feature = "sm89") {
        builder = builder.compute_cap(89).arg("-DBARACUDA_KERNELS_HAS_SM89");
    } else if cfg!(feature = "sm80") {
        builder = builder.compute_cap(80);
    }

    builder
        .build_lib(&lib_file)
        .expect("baracuda-kernels-sys: nvcc build failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static={lib_name}");

    // CUDA Runtime — kernel objects reference cudart helpers (streams etc.).
    println!("cargo:rustc-link-lib=dylib=cudart");

    // cuRAND — Phase 4.5 random / sampling family routes its uniform and
    // normal generators through cuRAND. On Linux this resolves to
    // `libcurand.so`; on Windows to `curand.dll` (loader picks the
    // versioned `curand64_*.dll` from `CUDA_PATH\bin`).
    println!("cargo:rustc-link-lib=dylib=curand");

    // cuSOLVER — Milestone 6.3 dense linalg family (Cholesky / LU / QR /
    // SVD). cuSOLVER's dense API itself depends on cuBLAS for the
    // underlying BLAS calls (e.g. `geqrf` defers to `gemm` / `trsm`),
    // so both libraries must be on the link line. On Linux these
    // resolve to `libcusolver.so` / `libcublas.so`; on Windows to
    // `cusolver64_*.dll` / `cublas64_*.dll` from `CUDA_PATH\bin`.
    println!("cargo:rustc-link-lib=dylib=cusolver");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Phase 43 — mHC vendor: `fused_rmsnorm_matmul.cuh` uses cuBLAS-Lt
    // (`cublasLtMatmulPreference*`, `cublasLtMatmulAlgoGetHeuristic`).
    // Even though the static-H launcher doesn't exercise that code path
    // at runtime, the symbols are referenced by the launcher TU and
    // must resolve at link time. Only emit the link request when the
    // `mhc` feature is on so non-mHC builds don't depend on cuBLAS-Lt
    // (which is part of the cuBLAS package on CUDA 11.x+).
    #[cfg(feature = "mhc")]
    {
        println!("cargo:rustc-link-lib=dylib=cublasLt");
    }

    // cuFFT — Milestone 6.4 FFT family (FFT / IFFT / RFFT / IRFFT plus
    // bespoke fftshift / ifftshift). On Linux resolves to `libcufft.so`;
    // on Windows to `cufft64_*.dll` from `CUDA_PATH\bin`.
    println!("cargo:rustc-link-lib=dylib=cufft");

    // cuDNN — Phase 7 convolution / pooling / CTC family. Gated behind
    // the `cudnn` cargo feature because cuDNN is a separate NVIDIA
    // download not bundled with the stock CUDA toolkit; linking blindly
    // breaks builds on machines that haven't installed it. Enable the
    // feature when cuDNN is present and you want the conv / pool /
    // CTC-cuDNN plans to link.
    //
    // Search-path discovery (Windows): the NVIDIA installer places cuDNN
    // at `C:\Program Files\NVIDIA\CUDNN\v<X.Y>\lib\<cuda_ver>\x64\` — a
    // versioned directory not on the default linker search path. We
    // probe `CUDNN_PATH` first (set if the user has wired it manually);
    // otherwise we glob the standard install dir for the newest v*
    // sub-folder and the cuda-toolkit-versioned sub-sub-folder. The
    // matching `bin\<cuda_ver>\` dir holds the runtime DLLs and must be
    // on `PATH` at test-run time (the build does NOT auto-stage DLLs).
    #[cfg(feature = "cudnn")]
    {
        let cudnn_lib_dir = locate_cudnn_lib_dir();
        if let Some(dir) = cudnn_lib_dir.as_deref() {
            println!("cargo:rustc-link-search=native={}", dir);
            println!("cargo:warning=baracuda-kernels-sys: cuDNN lib dir → {}", dir);
        } else {
            println!(
                "cargo:warning=baracuda-kernels-sys: `cudnn` feature is on but no cuDNN \
                 lib directory was found via CUDNN_PATH or the standard NVIDIA Windows \
                 install layout. Linker is likely to fail on cudnn.lib. Set CUDNN_PATH \
                 to point at the cuDNN install root (the directory containing lib/ and \
                 bin/ subtrees)."
            );
        }
        println!("cargo:rustc-link-lib=dylib=cudnn");
        println!("cargo:rerun-if-env-changed=CUDNN_PATH");
        println!("cargo:rerun-if-env-changed=CUDNN_ROOT");
        println!("cargo:rerun-if-env-changed=CUDNN_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
    }

    // CUDA Driver API — the linalg/qr plan uses `cuMemcpyDtoHAsync_v2`
    // / `cuMemcpyHtoDAsync_v2` directly to round-trip the post-`geqrf`
    // matrix through host memory so we can zero its strict lower
    // triangle and stage an identity for the `ormqr` Q materialization.
    // baracuda-driver loads these via libloading at runtime; this
    // crate statically links them through the platform's `cuda` lib
    // (`libcuda.so` on Linux, `cuda.lib` referencing `nvcuda.dll` on
    // Windows — the CUDA toolkit ships both).
    println!("cargo:rustc-link-lib=dylib=cuda");

    // C++ runtime on Linux — the static lib pulls in libstdc++ via
    // template instantiations.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn collect_kernel_files() -> Vec<&'static str> {
    let mut kernels: Vec<&'static str> = Vec::new();

    // Phase 1 deliverables: int8 GEMM RRR, sm_80 baseline. Files land
    // as the kernel source is implemented; build.rs gracefully skips any
    // not-yet-present file so the workspace stays buildable during
    // Phase 0 scaffolding.
    if cfg!(any(feature = "sm80", feature = "sm89", feature = "sm90a")) {
        // Phase 3 — elementwise / shape / layout kernels. Arch-agnostic
        // pointwise compute (no tensor cores) so the same set ships
        // across every arch feature.
        for f in &[
            "elementwise/binary_add_fp.cu",
            "elementwise/binary_sub_fp.cu",
            "elementwise/binary_mul_fp.cu",
            "elementwise/binary_div_fp.cu",
            "elementwise/binary_pow_fp.cu",
            "elementwise/binary_atan2_fp.cu",
            "elementwise/binary_hypot_fp.cu",
            "elementwise/binary_copysign_fp.cu",
            "elementwise/binary_nextafter_fp.cu",
            "elementwise/binary_fmin_fp.cu",
            "elementwise/binary_fmax_fp.cu",
            "elementwise/binary_maximum_fp.cu",
            "elementwise/binary_minimum_fp.cu",
            "elementwise/binary_floor_divide_fp.cu",
            "elementwise/binary_mod_fp.cu",
            "elementwise/binary_remainder_fp.cu",
            // Phase 3.3 — integer / bool elementwise binary ops.
            // Contig-only today; strided variants deferred per the
            // milestone scope.
            "elementwise/binary_bitwise_and_int.cu",
            "elementwise/binary_bitwise_or_int.cu",
            "elementwise/binary_bitwise_xor_int.cu",
            "elementwise/binary_bitwise_left_shift_int.cu",
            "elementwise/binary_bitwise_right_shift_int.cu",
            "elementwise/binary_logical_and_bool.cu",
            "elementwise/binary_logical_or_bool.cu",
            "elementwise/binary_logical_xor_bool.cu",
            "elementwise/binary_cmp_eq_fp.cu",
            "elementwise/binary_cmp_ne_fp.cu",
            "elementwise/binary_cmp_gt_fp.cu",
            "elementwise/binary_cmp_ge_fp.cu",
            "elementwise/binary_cmp_lt_fp.cu",
            "elementwise/binary_cmp_le_fp.cu",
            "elementwise/ternary_clamp_fp.cu",
            "elementwise/ternary_fma_fp.cu",
            "elementwise/ternary_addcmul_fp.cu",
            "elementwise/ternary_addcdiv_fp.cu",
            "elementwise/where_fp.cu",
            "elementwise/where_backward_fp.cu",
            // Phase 38 (Fuel 6c.4 Gap 3) — full cond × value dtype matrix
            // for `where`. Adds U32 / I64 cond variants for the 4 fp value
            // dtypes + full int-value coverage (U8 / I8 / U32 / I16 / I32 /
            // I64) across all 3 cond dtypes + Fp8E4M3.
            "elementwise/where_dtype_fanout.cu",
            // Phase 3 Category N — shape/layout trailblazers.
            "elementwise/pad_fp.cu",
            "elementwise/concat2_fp.cu",
            "elementwise/permute_fp.cu",
            "elementwise/flip_fp.cu",
            "elementwise/roll_fp.cu",
            // Phase 13.2 — Contiguize (strided→contiguous copy). Lives
            // in its own `shape_layout/` subdir; byte-width-templated,
            // dtype-agnostic at the kernel level.
            "shape_layout/contiguize.cu",
            "elementwise/arg_reduce_fp.cu",
            "elementwise/repeat_fp.cu",
            "elementwise/reduce_var_std_fp.cu",
            "elementwise/binary_add_backward_fp.cu",
            "elementwise/binary_sub_backward_fp.cu",
            "elementwise/binary_mul_backward_fp.cu",
            "elementwise/binary_div_backward_fp.cu",
            "elementwise/binary_pow_backward_fp.cu",
            "elementwise/binary_atan2_backward_fp.cu",
            "elementwise/binary_hypot_backward_fp.cu",
            "elementwise/binary_maximum_backward_fp.cu",
            "elementwise/binary_minimum_backward_fp.cu",
            // Phase 3 backward fanout (Milestone F): ternary backwards.
            "elementwise/ternary_fma_backward_fp.cu",
            "elementwise/ternary_clamp_backward_fp.cu",
            "elementwise/ternary_addcmul_backward_fp.cu",
            "elementwise/ternary_addcdiv_backward_fp.cu",
            "elementwise/unary_sin_backward_fp.cu",
            "elementwise/unary_exp_backward_fp.cu",
            "elementwise/unary_expm1_backward_fp.cu",
            "elementwise/unary_tanh_backward_fp.cu",
            "elementwise/unary_sigmoid_backward_fp.cu",
            "elementwise/unary_sqrt_backward_fp.cu",
            "elementwise/unary_rsqrt_backward_fp.cu",
            "elementwise/unary_log_backward_fp.cu",
            "elementwise/unary_log1p_backward_fp.cu",
            "elementwise/unary_log2_backward_fp.cu",
            "elementwise/unary_log10_backward_fp.cu",
            "elementwise/unary_atan_backward_fp.cu",
            "elementwise/unary_cos_backward_fp.cu",
            "elementwise/unary_tan_backward_fp.cu",
            "elementwise/unary_sinh_backward_fp.cu",
            "elementwise/unary_cosh_backward_fp.cu",
            "elementwise/unary_asin_backward_fp.cu",
            "elementwise/unary_acos_backward_fp.cu",
            "elementwise/unary_asinh_backward_fp.cu",
            "elementwise/unary_acosh_backward_fp.cu",
            "elementwise/unary_atanh_backward_fp.cu",
            "elementwise/unary_square_backward_fp.cu",
            "elementwise/unary_cube_backward_fp.cu",
            "elementwise/unary_exp2_backward_fp.cu",
            "elementwise/unary_tanhshrink_backward_fp.cu",
            "elementwise/unary_logit_backward_fp.cu",
            "elementwise/unary_reciprocal_backward_fp.cu",
            "elementwise/unary_erf_backward_fp.cu",
            "elementwise/unary_erfc_backward_fp.cu",
            "elementwise/unary_relu_backward_fp.cu",
            "elementwise/unary_hardtanh_backward_fp.cu",
            "elementwise/unary_relu6_backward_fp.cu",
            "elementwise/unary_hardsigmoid_backward_fp.cu",
            "elementwise/unary_hardswish_backward_fp.cu",
            "elementwise/unary_softplus_backward_fp.cu",
            "elementwise/unary_silu_backward_fp.cu",
            "elementwise/unary_mish_backward_fp.cu",
            "elementwise/unary_gelu_backward_fp.cu",
            "elementwise/unary_gelu_tanh_backward_fp.cu",
            "elementwise/unary_selu_backward_fp.cu",
            "elementwise/reduce_sum_backward_fp.cu",
            "elementwise/reduce_mean_backward_fp.cu",
            "elementwise/reduce_max_min_backward_fp.cu",
            "elementwise/reduce_prod_backward_fp.cu",
            "elementwise/reduce_norm2_backward_fp.cu",
            "elementwise/reduce_logsumexp_backward_fp.cu",
            "elementwise/reduce_var_std_backward_fp.cu",
            // Phase 4 — scans (Category F).
            "elementwise/scan_cumsum_fp.cu",
            "elementwise/scan_cumprod_fp.cu",
            "elementwise/scan_cummax_fp.cu",
            "elementwise/scan_cummin_fp.cu",
            "elementwise/scan_cumprod_backward_fp.cu",
            "elementwise/scan_cummax_min_backward_fp.cu",
            "elementwise/scan_log_cumsum_exp_fp.cu",
            "elementwise/scan_log_cumsum_exp_backward_fp.cu",
            // Phase 5 — softmax family (Category H).
            "softmax/softmax_fp.cu",
            "softmax/softmax_backward_fp.cu",
            "softmax/log_softmax_fp.cu",
            "softmax/log_softmax_backward_fp.cu",
            // Milestone 5.4 — softmax family completion: GumbelSoftmax +
            // Sparsemax. GumbelSoftmax BW reuses softmax_backward_fp_kernel
            // (saved soft output), so no separate BW CU file.
            "softmax/gumbel_softmax_fp.cu",
            "softmax/sparsemax_fp.cu",
            "softmax/sparsemax_backward_fp.cu",
            // Phase 5 — normalization family (Category G). RMSNorm /
            // LayerNorm (FW + BW × 4 FP dtypes). BW launchers fire two
            // kernels: the per-cell dx kernel and (when affine grads are
            // requested) a one-block-per-feature reduction kernel for
            // dgamma / dbeta — fully deterministic, no atomic-adds.
            "norm/rms_norm_fp.cu",
            "norm/rms_norm_backward_fp.cu",
            "norm/layer_norm_fp.cu",
            "norm/layer_norm_backward_fp.cu",
            // Phase 5.1 — Norm family completion: BatchNorm / GroupNorm
            // (FW + BW × 4 FP dtypes). InstanceNorm is a Rust-side thin
            // wrapper around GroupNorm with num_groups == num_channels —
            // no separate kernel.
            "norm/batch_norm_fp.cu",
            "norm/batch_norm_backward_fp.cu",
            "norm/group_norm_fp.cu",
            "norm/group_norm_backward_fp.cu",
            // Phase 5 — loss family (Category R). MSE / NLL / CrossEntropy
            // / BCE / KLDiv (FW + BW × 4 FP dtypes). Per-cell or per-row
            // kernel + single-block tree reduction for Mean/Sum modes;
            // direct per-cell output for None mode. Deterministic
            // (no atomicAdd).
            "loss/loss_mse_fp.cu",
            "loss/loss_mse_backward_fp.cu",
            "loss/loss_bce_fp.cu",
            "loss/loss_bce_backward_fp.cu",
            "loss/loss_kl_div_fp.cu",
            "loss/loss_kl_div_backward_fp.cu",
            "loss/loss_nll_fp.cu",
            "loss/loss_nll_backward_fp.cu",
            "loss/loss_cross_entropy_fp.cu",
            "loss/loss_cross_entropy_backward_fp.cu",
            // Phase 47 — Fused Linear Cross-Entropy (Liger-Kernel
            // algorithm port). Per-chunk fused logits→grad_logits step
            // + scalar / per-row finalize + in-place scale (for BW chain
            // rule). The GEMMs are dispatched from the Rust side via
            // cuBLAS — only the per-chunk fused softmax+CE+gradient
            // kernel is bespoke work shipped here.
            "loss/flce_fp.cu",
            // Milestone 5.2 — Tier-1 losses.
            "loss/loss_l1_fp.cu",
            "loss/loss_l1_backward_fp.cu",
            "loss/loss_smooth_l1_fp.cu",
            "loss/loss_smooth_l1_backward_fp.cu",
            "loss/loss_huber_fp.cu",
            "loss/loss_huber_backward_fp.cu",
            "loss/loss_bce_with_logits_fp.cu",
            "loss/loss_bce_with_logits_backward_fp.cu",
            "loss/loss_poisson_nll_fp.cu",
            "loss/loss_poisson_nll_backward_fp.cu",
            "loss/loss_gaussian_nll_fp.cu",
            "loss/loss_gaussian_nll_backward_fp.cu",
            "loss/loss_cross_entropy_soft_fp.cu",
            "loss/loss_cross_entropy_soft_backward_fp.cu",
            // Milestone 5.3 — Tier-2 margin / embedding losses.
            "loss/loss_margin_ranking_fp.cu",
            "loss/loss_margin_ranking_backward_fp.cu",
            "loss/loss_hinge_embedding_fp.cu",
            "loss/loss_hinge_embedding_backward_fp.cu",
            "loss/loss_cosine_embedding_fp.cu",
            "loss/loss_cosine_embedding_backward_fp.cu",
            "loss/loss_triplet_margin_fp.cu",
            "loss/loss_triplet_margin_backward_fp.cu",
            "loss/loss_multi_margin_fp.cu",
            "loss/loss_multi_margin_backward_fp.cu",
            "loss/loss_multilabel_margin_fp.cu",
            "loss/loss_multilabel_margin_backward_fp.cu",
            "loss/loss_multilabel_soft_margin_fp.cu",
            "loss/loss_multilabel_soft_margin_backward_fp.cu",
            // Milestone 5.5 — CTCLoss (Phase 5 final deferral).
            "loss/loss_ctc_fp.cu",
            "loss/loss_ctc_backward_fp.cu",
            // Milestone 5.3 — PReLU FW + BW.
            "elementwise/prelu_fp.cu",
            "elementwise/prelu_backward_fp.cu",
            // Phase 3 Category N — shape/layout BWs.
            "elementwise/pad_constant_backward_fp.cu",
            "elementwise/repeat_backward_fp.cu",
            "elementwise/concat2_backward_fp.cu",
            // Phase 4 — reduction trailblazer + fanout
            // (Sum / Mean / Max / Min / Prod, each × {f32, f16, bf16, f64}).
            "elementwise/reduce_sum_fp.cu",
            "elementwise/reduce_mean_fp.cu",
            "elementwise/reduce_max_fp.cu",
            "elementwise/reduce_min_fp.cu",
            "elementwise/reduce_prod_fp.cu",
            "elementwise/reduce_norm2_fp.cu",
            "elementwise/reduce_logsumexp_fp.cu",
            // Phase 4 deferral 4.4 — heterogeneous-output reductions
            // (Any / All → Bool output; CountNonzero → i64 output).
            // 7 input dtypes each: {f32, f16, bf16, f64, i32, i64, Bool}.
            "elementwise/reduce_any_fp_int_bool.cu",
            "elementwise/reduce_all_fp_int_bool.cu",
            "elementwise/reduce_count_nonzero_fp_int_bool.cu",
            "elementwise/trace_fp.cu",
            "elementwise/unary_neg_fp.cu",
            "elementwise/unary_abs_fp.cu",
            "elementwise/unary_sign_fp.cu",
            "elementwise/unary_reciprocal_fp.cu",
            "elementwise/unary_square_fp.cu",
            "elementwise/unary_cube_fp.cu",
            // Phase 3 transcendental fanout — 12 ops × 4 FP dtypes ×
            // contig+strided. All use the f32-detour pattern for f16 /
            // bf16; f32 / f64 use the matching libm intrinsics.
            "elementwise/unary_sqrt_fp.cu",
            "elementwise/unary_rsqrt_fp.cu",
            "elementwise/unary_exp_fp.cu",
            "elementwise/unary_expm1_fp.cu",
            "elementwise/unary_log_fp.cu",
            "elementwise/unary_log1p_fp.cu",
            "elementwise/unary_sin_fp.cu",
            "elementwise/unary_cos_fp.cu",
            "elementwise/unary_tan_fp.cu",
            "elementwise/unary_sinh_fp.cu",
            "elementwise/unary_cosh_fp.cu",
            "elementwise/unary_tanh_fp.cu",
            // Phase 3 activation fanout — 10 ops × 4 FP dtypes ×
            // contig+strided. Same f32-detour pattern as the
            // transcendentals; piecewise-linear ops (hardswish /
            // hardsigmoid / hardtanh) bypass the detour's transcendental
            // cost but use it for uniform dispatch.
            "elementwise/unary_relu_fp.cu",
            "elementwise/unary_gelu_fp.cu",
            "elementwise/unary_gelu_tanh_fp.cu",
            "elementwise/unary_silu_fp.cu",
            "elementwise/unary_mish_fp.cu",
            "elementwise/unary_sigmoid_fp.cu",
            "elementwise/unary_softplus_fp.cu",
            "elementwise/unary_hardswish_fp.cu",
            "elementwise/unary_hardsigmoid_fp.cu",
            "elementwise/unary_hardtanh_fp.cu",
            // Phase 3 math / rounding fanout — 15 ops × 4 FP dtypes ×
            // contig+strided. Same f32-detour pattern as the
            // transcendentals; floor / ceil / round / trunc / frac are
            // bit-exact at f32 / f64 because they reduce to a single
            // device intrinsic with no further arithmetic. Round uses
            // the round-half-to-even convention (`rintf` / `rint`) to
            // match PyTorch and `f32::round_ties_even`.
            "elementwise/unary_cbrt_fp.cu",
            "elementwise/unary_exp2_fp.cu",
            "elementwise/unary_log2_fp.cu",
            "elementwise/unary_log10_fp.cu",
            "elementwise/unary_asin_fp.cu",
            "elementwise/unary_acos_fp.cu",
            "elementwise/unary_atan_fp.cu",
            "elementwise/unary_asinh_fp.cu",
            "elementwise/unary_acosh_fp.cu",
            "elementwise/unary_atanh_fp.cu",
            "elementwise/unary_floor_fp.cu",
            "elementwise/unary_ceil_fp.cu",
            "elementwise/unary_round_fp.cu",
            "elementwise/unary_trunc_fp.cu",
            "elementwise/unary_frac_fp.cu",
            // Phase 3 special-function / activation fanout — 8 ops × 4
            // FP dtypes × contig+strided. Erf / Erfc / Lgamma are pure
            // libdevice intrinsics; Logit / Softsign / Tanhshrink / Relu6
            // / Selu compose from libdevice transcendentals + arithmetic.
            // All use the f32-detour pattern for f16 / bf16.
            "elementwise/unary_erf_fp.cu",
            "elementwise/unary_erfc_fp.cu",
            "elementwise/unary_lgamma_fp.cu",
            "elementwise/unary_logit_fp.cu",
            "elementwise/unary_softsign_fp.cu",
            "elementwise/unary_tanhshrink_fp.cu",
            "elementwise/unary_relu6_fp.cu",
            "elementwise/unary_selu_fp.cu",
            // Phase 3 parameterized-activation fanout (hardcoded
            // defaults — LeakyRelu α=0.01, ELU α=1.0, Hardshrink λ=0.5,
            // Softshrink λ=0.5). Threshold(t, v) and PReLU need
            // distinct plan shapes (2 params / per-channel vector) and
            // ship in a later session.
            "elementwise/unary_leaky_relu_fp.cu",
            "elementwise/unary_leaky_relu_backward_fp.cu",
            "elementwise/unary_elu_fp.cu",
            "elementwise/unary_elu_backward_fp.cu",
            "elementwise/unary_hardshrink_fp.cu",
            "elementwise/unary_hardshrink_backward_fp.cu",
            "elementwise/unary_softshrink_fp.cu",
            "elementwise/unary_softshrink_backward_fp.cu",
            // Phase 3 parameterized-unary / parameterized-binary plan
            // families (Threshold / Lerp). New INSTANTIATE macros thread
            // f32 scalar params through the kernel ABI alongside the
            // tensor pointers — contig only, no strided variant.
            "elementwise/unary_threshold_fp.cu",
            "elementwise/unary_threshold_backward_fp.cu",
            // Phase 12.1 — PowI (integer-exponent power-of-x): reuses
            // the existing UNARY_PARAM_* ABI (n shipped via p0 cast to
            // int at the kernel boundary, p1 unused).
            "elementwise/unary_powi_fp.cu",
            "elementwise/unary_powi_backward_fp.cu",
            // Phase 31 — Fuel Phase 6c.2 storage.rs unblock: float-
            // exponent power, Heaviside step, exact erf-based GELU
            // alias, and the broadcast-reverse reductions used by
            // autograd's ReduceSumTo / ReduceMaxTo. ELU was modified
            // in place to thread `alpha` through (the `unary_elu_fp.cu`
            // entry above gets the breaking change).
            "elementwise/unary_powf_fp.cu",
            "elementwise/unary_step_fp.cu",
            "elementwise/unary_gelu_erf_fp.cu",
            "elementwise/reduce_to_fp.cu",
            // Phase 37 Gap 1b — integer-dtype Reduce family (single-axis
            // Sum / Min / Max / Prod, plus Argmin / Argmax i32/i64 idx).
            // Sum / Prod widen the internal accumulator to i64 / u64 and
            // narrow on store (wrap-on-overflow, matches Fuel's CPU ref).
            "elementwise/reduce_sum_int.cu",
            "elementwise/reduce_min_int.cu",
            "elementwise/reduce_max_int.cu",
            "elementwise/reduce_prod_int.cu",
            "elementwise/arg_reduce_int.cu",
            "elementwise/binary_lerp_fp.cu",
            "elementwise/binary_lerp_backward_fp.cu",
            // Phase 3 Category C′ — gated activations (FW + BW × 4
            // FP dtypes × {Glu, ReGlu, SwiGlu, GeGlu}). Plan shape:
            // input is split along `split_dim` into (a, b); output
            // `y = a · gate(b)` is half-size along that axis. Contig
            // only today; strided fanout follows the binary-strided
            // pattern.
            "elementwise/gated_swiglu_fp.cu",
            "elementwise/gated_swiglu_backward_fp.cu",
            "elementwise/gated_glu_fp.cu",
            "elementwise/gated_glu_backward_fp.cu",
            "elementwise/gated_reglu_fp.cu",
            "elementwise/gated_reglu_backward_fp.cu",
            "elementwise/gated_geglu_fp.cu",
            "elementwise/gated_geglu_backward_fp.cu",
            // Phase 4.5 — random / sampling (Category Q). Bernoulli +
            // Dropout (FW + BW) are bespoke kernels that consume a
            // cuRAND-generated uniform-rand buffer; Uniform / Normal go
            // straight through cuRAND at the safe-plan layer (no .cu).
            "random/random_bernoulli.cu",
            "random/random_dropout_fp.cu",
            "random/random_dropout_backward_fp.cu",
            // Phase 6.1 — attention positional encodings (Category K).
            // RoPE (rotary) + ALiBi (linear biases). FW + BW × 4 FP
            // dtypes. ALiBi BW uses one-block-per-head warp-shuffle
            // reduction for dslope; deterministic (no atomicAdd).
            "attention/rope_fp.cu",
            "attention/rope_backward_fp.cu",
            "attention/alibi_fp.cu",
            "attention/alibi_backward_fp.cu",
            // Milestone 6.2 — naive SDPA (FW + BW × 4 FP dtypes). Three
            // sub-kernels FW (scores / row-softmax / out), five sub-
            // kernels BW (dV / dattn / dscores=softmax_bw / dQ / dK),
            // all bundled behind a single launcher per direction.
            "attention/sdpa_fp.cu",
            "attention/sdpa_backward_fp.cu",
            // Milestone 6.5 — KV-cache append (inference-time op). Two
            // device-side copy kernels (K + V) per launcher, instantiated
            // for {f32, f16, bf16, f64}. Bit-exact (pure copy). No BW.
            "attention/kv_cache_fp.cu",
            // Milestone 6.6 — Flash Attention SDPA (FW + BW × 4 FP dtypes).
            // Tiled fused online-softmax FW kernel; 3-kernel deterministic
            // BW pipeline (D = rowsum(y ⊙ dy), then dQ per q-block, then
            // dK/dV per k-block). Br = Bc = 64, d_k = d_v ≤ 128. Saved
            // `lse` ([B, H, Q]) replaces the saved `attn` of naive SDPA.
            "attention/flash_sdpa_fp.cu",
            "attention/flash_sdpa_backward_fp.cu",
            // Phase 51 — arbitrary additive-mask attention FW. Same
            // online-softmax algorithm as flash_sdpa_fp.cu with an
            // extra `mask: f32[B, H, Q, K]` bias applied to S before
            // softmax. Unlocks spec-decode tree masks, MoE expert
            // masking, prefix-LM, sliding-window-with-sinks. FW only;
            // BW deferred (Tier 2). f32 / f16 / bf16 / f64.
            "attention/attn_arbmask_fp.cu",
            // Milestone 6.4 — cuFFT companion kernels. fftshift /
            // ifftshift (cuFFT has no native shift) at 4/8/16-byte cell
            // widths covering f32 / f64 / Complex32 / Complex64; plus
            // in-place scale-by-1/N kernels to bake PyTorch's inverse
            // FFT normalization into the output of `cufftExec{C2C,Z2Z,
            // C2R,Z2D}` in the inverse direction.
            "fft/fft_shift.cu",
            "fft/fft_shift_nd.cu",
            "fft/fft_scale.cu",
            // Milestone 6.14 — bespoke batched-`ormqr` + batched-QR
            // dense Q/R materialization. Single launch per batch
            // instead of cuSOLVER's non-batched `ormqr` looped over
            // slots (latency-dominated regime).
            "linalg/batched_ormqr.cu",
            "linalg/batched_qr_materialize.cu",
            // Milestone 6.17 — WY-blocked batched-`ormqr`. Pairs the
            // T-build kernel + V-extraction helper with cuBLAS strided-
            // batched GEMM at the safe-plan layer to lift the apply
            // step from GEMV-rates to GEMM-rates.
            "linalg/batched_ormqr_wy.cu",
            // Phase 7 Milestone 7.3 — indexing / scatter / gather family
            // (Category L). Six ops total: gather (FW+BW), scatter_add,
            // index_select (FW+BW), masked_fill (FW+BW), one_hot,
            // nonzero. Index dtype is i32 only (i64 deferred). All
            // kernels are workspace-free; BW for {gather, index_select}
            // uses atomicAdd (FP-only). masked_fill / one_hot / nonzero
            // also cover the i32 + bool element families.
            "indexing/gather.cu",
            "indexing/scatter.cu",
            "indexing/index_select.cu",
            "indexing/masked_fill.cu",
            "indexing/one_hot.cu",
            "indexing/nonzero.cu",
            // Phase 39 (Fuel 6c.4 Gap 5) — index_add (algorithmically the
            // same as `index_select_backward`, exposed under a fresh
            // non-autograd name + with f16 / bf16 dtype fanout).
            "indexing/index_add.cu",
            // Phase 7 Milestone 7.5 — embedding family (Category M).
            // `embedding` FW (f32/f64/f16/bf16) + BW (f32/f64 via
            // atomicAdd) with optional `padding_idx`. `embedding_bag`
            // FW (Sum / Mean, 4 FP dtypes) + BW (f32/f64). Max-mode is
            // deferred (needs argmax tracking).
            "embedding/embedding.cu",
            "embedding/embedding_backward.cu",
            "embedding/embedding_bag.cu",
            "embedding/embedding_bag_backward.cu",
            // Phase 25 — embedding_bag Max mode (FW + BW). FW writes
            // value + per-(b, d) contributing-row index; BW scatters
            // dout into dweight at those rows via atomicAdd.
            "embedding/embedding_bag_max.cu",
            "embedding/embedding_bag_max_backward.cu",
            // Phase 7 Milestone 7.6 — segment / scatter-reduce family
            // (Category S). Sorted + unsorted variants of sum / mean /
            // max / min / prod (FW); sum + mean BW (sorted and
            // unsorted share BW launchers — the gather access pattern
            // is identical, only the seg-ids monotonicity assumption
            // differs and that doesn't affect BW). Dtype coverage:
            // f32, f64 (atomic-FP-restricted). Single .cu ships every
            // (op × dtype) launcher via the INSTANTIATE macros in
            // `kernels/include/baracuda_segment.cuh`.
            "segment/segment.cu",
            // Phase 8 Milestone 8.1 — quantization family (Category P).
            // Per-tensor + per-channel quantize / dequantize +
            // fake_quantize (FW + BW × {f32, f64, f16, bf16} × {s8, u8}).
            // STE-based BW recomputes the in-range mask from the saved
            // input `x` — no separate mask tensor in the FW signature.
            // Sub-byte packed (s4 / u4) output is deferred.
            "quantize/per_tensor.cu",
            "quantize/per_channel.cu",
            "quantize/fake_quantize.cu",
            // Phase 8 Milestone 8.2 — per-token + per-group quantize /
            // dequantize (Category P, LLM-style activation + weight
            // compression). FW + BW × {f32, f64, f16, bf16} × {s8, u8}
            // for FW; BW is TIn-only (STE / straight-through). Sibling
            // 8.1 (per-tensor / per-channel / fake_quantize) ships its
            // own .cu files in this same directory.
            "quantize/per_token.cu",
            "quantize/per_group.cu",
            // Phase 8 Milestone 8.3 — composing quantization ops
            // (DynamicRangeQuantize + QuantizedLinear). Builds on 8.1 /
            // 8.2 primitives — see `kernels/include/baracuda_quantize_compose.cuh`.
            "quantize/compose.cu",
            // Phase 8 Milestone 8.4 — GGUF block-format dequant + MMVQ
            // (Category P). Vendored from llama.cpp via fuel-cuda-kernels.
            // Two .cu files behind a single shared header
            // (`kernels/include/baracuda_gguf.cuh`):
            //   * dequantize.cu — 11 `_run` symbols, one per block format
            //     (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + Q2_K..Q8_K). f32 output.
            //   * mmvq.cu — 11 `_run` symbols (one per block format).
            //     Phase 11.4 added a bespoke Q8_K MMVQ (upstream
            //     llama.cpp / Fuel ship only Q8_K dequant). f32
            //     activation, f32 output.
            "gguf/dequantize.cu",
            "gguf/mmvq.cu",
            // Phase 20.1 — batched MMVQ × N-experts (general-purpose
            // routing primitive). 33 quant FFI symbols (11 block formats
            // × 3 activation dtypes) + 3 pure-FP FFI symbols. Shares
            // the per-row dot-product math with `gguf/mmvq.cu`; the
            // batched template adds (token, expert, topk_weight)
            // routing semantics + atomic-vs-store dispatch.
            "gguf/mmvq_batched.cu",
            // Phase 33 — multi-M MMVQ via Q8_1 activation staging. The
            // staging kernel converts fp activations into the Q8_1 block
            // format (int8 quants + half scale + half sum); the multi-M
            // launcher reuses one weight load across `ncols_y ∈ {1,2,4,8}`
            // activation vectors → up to 8× gmem bandwidth save at
            // M=8 prefill. Q8_0 weights only in this phase; remaining
            // block formats follow in a subsequent phase.
            "gguf/quantize_q8_1.cu",
            "gguf/mmvq_multim.cu",
            // Phase 8 Milestone 8.5 — Mixture-of-Experts forward
            // (Category V). Three fused per-token-dispatch + expert-
            // matmul + accumulate kernels vendored from Fuel / attention.rs.
            //   * moe/moe_gguf.cu — scalar GGUF (Q8_0 + k-quants),
            //     f32 activations through a q8_1-staged intermediate.
            //   * moe/moe_wmma.cu — WMMA FP weights (f16, bf16).
            //   * moe/moe_wmma_gguf.cu — WMMA + GGUF combined hot path
            //     (f16/bf16 activations, Q8_0 + k-quants weights).
            // sm_70+ for the WMMA paths (already covered by every
            // arch feature this crate exposes).
            "moe/moe_gguf.cu",
            "moe/moe_wmma.cu",
            "moe/moe_wmma_gguf.cu",
            // Phase 3 fanout — cast / fill / affine. Vendored from
            // fuel-cuda-kernels (cast.cu / fill.cu / affine.cu) with
            // the standard SPDX header. Contig-only fast path; baracuda's
            // plan layer materializes strided views upstream.
            "elementwise/cast.cu",
            // Phase 13.3 — sub-byte cast paths (Bool / Fp8 / S4 / U4).
            // Bool ↔ T uses 0/non-zero truthiness; Fp8 routes through f32
            // via NVIDIA's `__nv_cvt_*_fp8` intrinsics with SATFINITE
            // semantics; S4/U4 unpack/pack handle nibble-packed storage
            // (numel must be even, packed buffer is numel/2 bytes).
            "elementwise/cast_subbyte_bool.cu",
            "elementwise/cast_subbyte_fp8.cu",
            "elementwise/cast_subbyte_s4.cu",
            "elementwise/cast_subbyte_u4.cu",
            "elementwise/fill.cu",
            "elementwise/affine.cu",
            // Phase 9 Category T — image / spatial transforms. Bespoke
            // bilinear interpolate (FW+BW), grid_sample (FW+BW) +
            // affine_grid, pixel_shuffle / pixel_unshuffle (pure layout,
            // each other's BW), roi_align / roi_pool (FW+BW), nms (no BW).
            // Trailblazer dtypes: f32 + f64 for math-bearing ops;
            // pixel_shuffle adds f16 + bf16 (memory-bound, dtype-agnostic).
            "image/interpolate.cu",
            "image/grid_sample.cu",
            "image/pixel_shuffle.cu",
            "image/roi.cu",
            "image/nms.cu",
            // Phase 19.2 — upsample nearest-2D FW + BW × 4 fp dtypes.
            // Standalone `_run` symbols complete the upsample FFI
            // surface alongside the existing `interpolate_bilinear_2d_*`
            // symbols (which are re-exported under the new
            // `upsample_bilinear_2d_*` namespace via Rust aliases
            // declared in src/lib.rs).
            "image/upsample.cu",
            // Phase 9 Milestone Category O — sort / topk / kthvalue /
            // unique / msort / histogram / bincount / searchsorted.
            // Block-bitonic primitive shared by sort + topk + msort;
            // atomic-bin accumulators for histogram + bincount; per-row
            // binary search for searchsorted; sort BW = scatter via
            // saved indices (FP only). Trailblazer caps:
            // row_len ≤ 1024, top-k ≤ 64.
            "sort/sort.cu",
            "sort/topk.cu",
            "sort/unique.cu",
            "sort/histogram.cu",
            "sort/searchsorted.cu",
            // Phase 40 (Fuel ask Gap 6b) — multi-block radix argsort
            // for `row_len > 1024`. Uses CUB's
            // `DeviceSegmentedRadixSort::SortPairs[Descending]`. Caller
            // supplies a workspace blob (query exact bytes via
            // `baracuda_kernels_argsort_<dt>_big_workspace_size`).
            // Dtype coverage: f32, f64, i32, i64.
            "sort/argsort_big.cu",
            // Phase 13.1 — WriteSlice (Category N / ShapeLayoutKind::
            // WriteSlice). Byte-width-dispatched memcpy kernel (one
            // symbol per sizeof(T) ∈ {1, 2, 4, 8, 16}) plus a
            // nibble-packed symbol for S4 / U4. Drives Fuel team's
            // persistent KV-cache append during autoregressive decoding.
            "shape_layout/write_slice.cu",
            // Phase 13.4 — Triu / Tril (upper / lower triangular masks).
            // Per-dtype fanout across {f16, bf16, f32, f64, i32, i64,
            // Bool}; one templated kernel body covers both ops via a
            // predicate functor.
            "shape_layout/triu_tril.cu",
            // Phase 16.1 — bit-exact PyTorch adaptive pooling (Avg /
            // Max, 1D / 2D / 3D, FW + BW). Single rank-agnostic kernel
            // template parameterized on (spatial_rank, in_dhw, out_dhw)
            // with per-dtype instantiations for {f16, bf16, f32, f64}.
            // Replaces the Phase 11.8 cuDNN-approximation path (uniform
            // `kernel=ceil(in/out)` / `stride=floor(in/out)`) with
            // PyTorch's non-uniform per-output-cell window convention.
            "pool/adaptive_pool.cu",
            // Phase 16.2 — LpPool 1d/2d fused bespoke kernels (FW + BW
            // × {f32, f64, f16, bf16}). cuDNN has no native LpPool; the
            // fused kernel computes `y = (Σ |x|^p)^(1/p)` per window in
            // one launch (avoids the 3-launch pow→avg_pool→pow stack +
            // the missing parameterized `Pow(p)` unary plan dependency).
            // BW uses one-thread-per-output-cell atomicAdd scatter for
            // uniformity with the rest of the pool BW family.
            "pool/lp_pool.cu",
            // Phase 16.3 — FractionalMaxPool 2-D + 3-D (FW + BW × 4 FP
            // dtypes). Bespoke kernel; cuDNN has no fractional-pool
            // primitive. Caller-provided f32 random samples
            // ([N, C, num_axes]) drive per-output-cell window
            // placement; the FW writes both `y` and a saved-indices
            // tensor (i64, argmax linear index) consumed by the BW
            // atomicAdd scatter.
            "pool/fractional_max_pool.cu",
            // Phase 19.3 — im2col / im2col1d / col2im1d bespoke
            // kernels (Category Convolution). Building blocks for
            // Fuel's conv-via-im2col-and-GEMM fallback lowering +
            // the conv-backward filter-gradient path. One .cu file
            // ships 12 `_run` symbols (3 ops × 4 FP dtypes); no
            // cuDNN dependency. col2im_1d uses atomicAdd scatter
            // (half/bf16 via `baracuda::atomic::add`).
            "conv/im2col.cu",
        ] {
            if std::path::Path::new(&format!("kernels/{f}")).exists() {
                kernels.push(*f);
            }
        }

        if cfg!(any(feature = "sm80", feature = "sm89")) {
            for f in &[
                "gemm/gemm_s8_rrr_sm80.cu",
                "gemm/gemm_u8_rrr_sm80.cu",
                "gemm/gemm_s8_rrr_sm80_bias.cu",
                "gemm/gemm_u8_rrr_sm80_bias.cu",
            ] {
                if std::path::Path::new(&format!("kernels/{f}")).exists() {
                    kernels.push(*f);
                }
            }
        }

        // Phase 2 deliverables: FP8 / int4 / bin GEMM, sm_89 (Ada
        // Lovelace) baseline. Selected only when `sm89` is enabled —
        // the FP8 m16n8k32 instruction is sm_89+ and won't assemble
        // against an `sm_80` target.
        //
        // Full FP8 SKU matrix: {E4M3, E5M2} × {RCR, RRR} × {Identity,
        // Bias, BiasRelu, BiasGelu, BiasSilu} = 20 SKUs across 8 .cu
        // files (4 Identity + 4 bias multi-instantiation files).
        if cfg!(feature = "sm89") {
            for f in &[
                "gemm/gemm_fp8_e4m3_rcr_sm89.cu",
                "gemm/gemm_fp8_e4m3_rrr_sm89.cu",
                "gemm/gemm_fp8_e5m2_rcr_sm89.cu",
                "gemm/gemm_fp8_e5m2_rrr_sm89.cu",
                "gemm/gemm_fp8_e4m3_rcr_sm89_bias.cu",
                "gemm/gemm_fp8_e4m3_rrr_sm89_bias.cu",
                "gemm/gemm_fp8_e5m2_rcr_sm89_bias.cu",
                "gemm/gemm_fp8_e5m2_rrr_sm89_bias.cu",
                // int4 GEMM — S4 / U4 × {RCR, RRR} × Identity
                // (alpha.17). The bias-family variants land in
                // subsequent fanout commits.
                // `mma.sync.m16n8k64.{s4|u4}.{s4|u4}.s32` requires
                // sm_89+. RRR header gathers two nibbles from two
                // gmem K-rows into one packed-pair smem byte (B is
                // pair-packed along N in gmem, K-contig in smem).
                "gemm/gemm_s4_rcr_sm89.cu",
                "gemm/gemm_u4_rcr_sm89.cu",
                "gemm/gemm_s4_rrr_sm89.cu",
                "gemm/gemm_u4_rrr_sm89.cu",
                // int4 bias-family multi-instantiation files. Each file
                // ships 8 `_run` symbols for the
                // `{Bias, BiasRelu, BiasGelu, BiasSilu} × {f32, i32}` bias
                // matrix at a fixed `(element, layout)` pair. The
                // Identity SKU above ships the shared `_workspace_size`
                // and `_can_implement` for each `(element, layout)`.
                "gemm/gemm_s4_rcr_sm89_bias.cu",
                "gemm/gemm_u4_rcr_sm89_bias.cu",
                "gemm/gemm_s4_rrr_sm89_bias.cu",
                "gemm/gemm_u4_rrr_sm89_bias.cu",
                // bin (B1) GEMM — RCR + RRR Identity-only. `xor.popc`
                // model; output is raw int32 popcount sum (no
                // α/β/bias/activation chain).
                // `mma.sync.m16n8k256.b1.b1.s32.xor.popc` (sm_80+;
                // gated to sm_89 here for build-set parity with
                // int4/FP8). The RRR header bit-gathers 8 K-row gmem
                // bytes into one K-pair smem byte per output column
                // (B is N-bit-packed in gmem, K-bit-packed in smem).
                "gemm/gemm_bin_rcr_sm89.cu",
                "gemm/gemm_bin_rrr_sm89.cu",
                // Phase 10 Milestone 10.3 — Flash Attention SDPA FW,
                // sm_89 specialization. Sibling of `flash_sdpa_fp.cu`
                // (sm_80 baseline). Same math, `cp.async` double-buffered
                // K/V loads + 256-thread block. f16 + bf16 only.
                "attention/flash_sdpa_sm89.cu",
            ] {
                if std::path::Path::new(&format!("kernels/{f}")).exists() {
                    kernels.push(*f);
                }
            }
        }
    }

    kernels
}

/// Walk `kernels/include` and gather every `.cuh` (or `.h`) file so the
/// builder's content-hash cache invalidates when a header changes. Without
/// this, edits to a shared header (e.g. `baracuda_loss.cuh`) don't trigger
/// a rebuild of the `.cu` files that include it.
fn collect_header_files() -> Vec<std::path::PathBuf> {
    let mut headers = Vec::new();
    let dir = std::path::Path::new("kernels/include");
    if !dir.exists() {
        return headers;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "cuh" || e == "h" || e == "hpp" || e == "cuhpp")
                .unwrap_or(false)
            {
                headers.push(path);
            }
        }
    }
    headers
}

fn lib_static_path(out_dir: &str, name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{out_dir}/{name}.lib")
    } else {
        format!("{out_dir}/lib{name}.a")
    }
}

/// Detect Git-for-Windows' fake `link.exe` (a GNU coreutils binary that
/// creates a hard link) shadowing the MSVC linker on PATH. If a developer
/// has `C:\Program Files\Git\usr\bin\` ahead of the MSVC linker on PATH,
/// `cargo build` will invoke that coreutils binary instead of `link.exe`
/// from the MSVC toolchain (or `lld-link.exe`), failing with cryptic
/// errors about unknown `/OUT:` flags. Walk PATH in order, find the first
/// `link.exe`, and warn if its path lives under `\Git\usr\bin\`.
///
/// No-op on non-Windows targets. Returns early (no warning) if no
/// `link.exe` is anywhere on PATH — the MSVC linker may still be
/// findable via the rustc / linker shim through other channels.
#[cfg(windows)]
fn check_for_fake_link_exe() {
    let Some(path) = std::env::var_os("PATH") else {
        return;
    };
    for entry in std::env::split_paths(&path) {
        let candidate = entry.join("link.exe");
        if candidate.is_file() {
            let s = candidate.to_string_lossy().to_lowercase();
            if s.contains("\\git\\usr\\bin\\") || s.contains("/git/usr/bin/") {
                println!(
                    "cargo:warning=Detected Git-for-Windows fake `link.exe` first on PATH \
                     ({}). This is NOT the MSVC linker — it's GNU coreutils `link` and will \
                     fail with a cryptic error when cargo invokes it. Fix: re-order PATH so \
                     the MSVC linker (or LLVM's lld-link.exe) appears before Git's bin \
                     directory.",
                    candidate.display()
                );
            }
            return; // first hit wins; PATH semantics
        }
    }
}

#[cfg(not(windows))]
fn check_for_fake_link_exe() {}
