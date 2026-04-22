//! Workspace task runner for baracuda.
//!
//! Subcommands:
//!
//! - `xtask regen-all`             Regenerate every committed `bindings/cuda_*.rs`.
//! - `xtask regen <lib>`           Regenerate a single `-sys` crate's bindings.
//! - `xtask build-kernels`         Recompile shipped `.ptx` fixtures via `nvcc` (planned).
//!
//! All commands require a CUDA Toolkit install discoverable via `CUDA_PATH` /
//! `CUDA_HOME` / OS defaults; see [`baracuda_build::detect_cuda`].

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("regen-all") => regen_all(&args[1..]),
        Some("regen") => regen_one(&args[1..]),
        Some("build-kernels") => {
            eprintln!("xtask build-kernels: not implemented yet (planned)");
            ExitCode::from(2)
        }
        Some(other) => {
            eprintln!("xtask: unknown subcommand: {other}");
            print_usage();
            ExitCode::from(2)
        }
        None => {
            print_usage();
            ExitCode::SUCCESS
        }
    }
}

fn print_usage() {
    println!("xtask — workspace task runner for baracuda");
    println!();
    println!("usage:");
    println!("  xtask regen-all              regenerate every -sys crate's committed bindings");
    println!("  xtask regen <lib>            regenerate only the named -sys crate (e.g. cuda, nvrtc, cublas)");
    println!("  xtask build-kernels          (planned) recompile shipped .ptx fixtures via nvcc");
}

fn regen_all(_args: &[String]) -> ExitCode {
    let install = match baracuda_build::detect_cuda() {
        Some(i) => i,
        None => {
            eprintln!(
                "xtask regen-all: no CUDA install found. Set CUDA_PATH or CUDA_HOME and retry."
            );
            return ExitCode::from(1);
        }
    };
    println!(
        "Using CUDA {} at {}",
        install
            .version
            .map(|(a, b)| format!("{a}.{b}"))
            .unwrap_or_else(|| "<unknown>".into()),
        install.root.display()
    );

    for spec in target_specs() {
        if let Err(e) = regen_one_spec(&install, *spec) {
            eprintln!("xtask regen {}: FAILED: {e}", spec.name);
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}

fn regen_one(args: &[String]) -> ExitCode {
    let Some(target) = args.first() else {
        eprintln!("xtask regen: missing library name (e.g. `xtask regen cuda`)");
        return ExitCode::from(2);
    };
    let install = match baracuda_build::detect_cuda() {
        Some(i) => i,
        None => {
            eprintln!("xtask regen: no CUDA install found. Set CUDA_PATH or CUDA_HOME and retry.");
            return ExitCode::from(1);
        }
    };
    let Some(spec) = target_specs().iter().find(|s| s.name == target).copied() else {
        eprintln!("xtask regen: unknown library '{target}'. Known:");
        for s in target_specs() {
            eprintln!("  - {}", s.name);
        }
        return ExitCode::from(2);
    };
    if let Err(e) = regen_one_spec(&install, spec) {
        eprintln!("xtask regen {}: FAILED: {e}", spec.name);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Per-`-sys` crate regeneration spec.
#[derive(Copy, Clone)]
struct Spec {
    /// Short name used in `xtask regen <name>`.
    name: &'static str,
    /// Header file under `include/` to feed bindgen.
    header: &'static str,
    /// Allowlist regex for functions (bindgen `allowlist_function`).
    allowlist_fn: &'static str,
    /// Allowlist regex for types.
    allowlist_type: &'static str,
    /// Output file relative to the workspace root.
    output: &'static str,
}

const fn target_specs() -> &'static [Spec] {
    &[Spec {
        name: "cuda",
        header: "cuda.h",
        allowlist_fn: r"^(cu|cuda)[A-Z].*",
        allowlist_type: r"^CU.*",
        output: "crates/baracuda-cuda-sys/src/bindings/generated.rs",
    }]
}

fn regen_one_spec(install: &baracuda_build::CudaInstall, spec: Spec) -> Result<(), String> {
    let header_path = install.include.join(spec.header);
    if !header_path.exists() {
        return Err(format!("header not found: {}", header_path.display()));
    }
    let workspace_root = find_workspace_root().ok_or("could not locate workspace root")?;
    let out_path = workspace_root.join(spec.output);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }

    println!("regen {} -> {}", spec.name, out_path.display());
    let bindings = baracuda_build::bindgen_builder(install)
        .header(header_path.to_string_lossy().to_string())
        .allowlist_function(spec.allowlist_fn)
        .allowlist_type(spec.allowlist_type)
        .generate()
        .map_err(|e| format!("bindgen: {e}"))?;
    bindings
        .write_to_file(&out_path)
        .map_err(|e| format!("write {}: {e}", out_path.display()))?;
    Ok(())
}

fn find_workspace_root() -> Option<PathBuf> {
    let mut dir: PathBuf = std::env::current_dir().ok()?;
    loop {
        if dir.join("Cargo.toml").exists() && is_workspace_toml(&dir.join("Cargo.toml")) {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn is_workspace_toml(path: &Path) -> bool {
    std::fs::read_to_string(path)
        .map(|s| s.contains("[workspace]"))
        .unwrap_or(false)
}
