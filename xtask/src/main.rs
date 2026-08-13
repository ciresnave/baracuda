//! Workspace task runner for baracuda.
//!
//! Subcommands:
//!
//! - `xtask regen-all`             Regenerate every committed `bindings/cuda_*.rs`.
//! - `xtask regen <lib>`           Regenerate a single `-sys` crate's bindings.
//! - `xtask build-kernels`         Recompile shipped `.ptx` fixtures via `nvcc` (planned).
//! - `xtask test-gpu [args]`       Run on-device (`#[ignore]`d) tests under the `gpu-run` lock.
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
        Some("test-gpu") => test_gpu(&args[1..]),
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
    println!(
        "  xtask regen <lib>            regenerate only the named -sys crate (e.g. cuda, nvrtc, cublas)"
    );
    println!("  xtask build-kernels          (planned) recompile shipped .ptx fixtures via nvcc");
    println!(
        "  xtask test-gpu [cargo args]  run on-device (#[ignore]d) tests under the gpu-run lock"
    );
    println!(
        "                               (compile UNLOCKED, device-run LOCKED); prefer `-p <crate>`"
    );
}

/// `xtask test-gpu [cargo test args] [-- <test-binary args>]`
///
/// Runs on-device (`#[ignore]`d) tests under the machine-wide `Global\gpu-run`
/// mutex so lock acquisition is STRUCTURAL, not a convention someone must
/// remember (the failure the postmortem indicts). Two phases:
/// 1. compile UNLOCKED — `cargo test <args> --no-run` (the lock wrapper
///    explicitly forbids serializing compile-only work);
/// 2. device-run LOCKED — `scripts/gpu-run.ps1 -Project baracuda -- cargo test
///    <args> -- --ignored <extra>`, which holds the mutex across the run.
///
/// The child's exit code is propagated verbatim, including gpu-run's `75`
/// (EX_TEMPFAIL: the lock holder looked wedged — try again later).
fn test_gpu(args: &[String]) -> ExitCode {
    // Split at the first `--`: cargo-level args before, test-binary args after.
    let (cargo_args, test_bin_args): (&[String], &[String]) =
        match args.iter().position(|a| a == "--") {
            Some(i) => (&args[..i], &args[i + 1..]),
            None => (args, &[]),
        };

    // Encourage scoping — an unfiltered run compiles+launches the entire device
    // suite (~600 files) under one lock hold. Warn, don't block.
    let scoped = cargo_args.iter().any(|a| {
        matches!(
            a.as_str(),
            "-p" | "--package" | "--test" | "--bin" | "--example"
        )
    });
    if !scoped {
        eprintln!(
            "xtask test-gpu: no -p/--test filter — this builds AND launches the whole device \
             suite under one lock. Prefer `-p <crate>` / `--test <name>`."
        );
    }

    // Phase 1 — compile UNLOCKED (never hold the GPU mutex during a build).
    match std::process::Command::new("cargo")
        .arg("test")
        .args(cargo_args)
        .arg("--no-run")
        .status()
    {
        Ok(s) if s.success() => {}
        Ok(s) => return ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("xtask test-gpu: could not spawn `cargo test --no-run`: {e}");
            return ExitCode::from(1);
        }
    }

    // The device-run command shared by both the locked (Windows) and the
    // unlocked-fallback paths: `cargo test <cargo_args> -- --ignored <extra>`.
    let cargo_test_run = |cmd: &mut std::process::Command| {
        cmd.arg("cargo")
            .arg("test")
            .args(cargo_args)
            .arg("--")
            .arg("--ignored")
            .args(test_bin_args);
    };

    // On-device declare-and-report wiring (baracuda-driver `test-support`): FAIL
    // LOUD if a `require!`d resource is absent here — this is the box where the
    // evidence must exist — and collect `require_optional!` declared skips into a
    // log we tally after the run, so a would-be silent skip is reported instead.
    let skip_log = std::env::temp_dir().join("baracuda-gpu-test-skips.log");
    let _ = std::fs::remove_file(&skip_log);
    let set_env = |cmd: &mut std::process::Command| {
        cmd.env("BARACUDA_GPU_REQUIRED", "1")
            .env("BARACUDA_SKIP_LOG", &skip_log);
    };

    // Off-Windows (the 4070 box is Windows-only) there is no lock to take —
    // run the device tests directly, with a warning, rather than fail.
    if !cfg!(windows) {
        eprintln!(
            "xtask test-gpu: the gpu-run lock is Windows-only; running device tests WITHOUT it."
        );
        let mut cmd = std::process::Command::new("cargo");
        // First token is the program, so drop the leading "cargo" the helper adds.
        cmd.arg("test")
            .args(cargo_args)
            .arg("--")
            .arg("--ignored")
            .args(test_bin_args);
        set_env(&mut cmd);
        let code = match cmd.status() {
            Ok(s) => s.code().unwrap_or(1),
            Err(e) => {
                eprintln!("xtask test-gpu: could not spawn `cargo test`: {e}");
                1
            }
        };
        tally_skips(&skip_log);
        return ExitCode::from(code as u8);
    }

    // Phase 2 — device-run LOCKED via scripts/gpu-run.ps1.
    let Some(root) = find_workspace_root() else {
        eprintln!("xtask test-gpu: could not locate workspace root");
        return ExitCode::from(1);
    };
    let script = root.join("scripts").join("gpu-run.ps1");
    if !script.exists() {
        eprintln!(
            "xtask test-gpu: lock wrapper not found at {}",
            script.display()
        );
        return ExitCode::from(1);
    }

    // Prefer PowerShell 7+ (`pwsh`); fall back to Windows `powershell`.
    for (i, shell) in ["pwsh", "powershell"].iter().enumerate() {
        let mut cmd = std::process::Command::new(shell);
        cmd.arg("-File")
            .arg(&script)
            .arg("-Project")
            .arg("baracuda")
            .arg("--");
        cargo_test_run(&mut cmd);
        set_env(&mut cmd);
        match cmd.status() {
            Ok(s) => {
                tally_skips(&skip_log);
                return ExitCode::from(s.code().unwrap_or(1) as u8);
            }
            // Shell not on PATH: try the next candidate; any other error is real.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && i == 0 => continue,
            Err(e) => {
                eprintln!("xtask test-gpu: could not spawn {shell}: {e}");
                return ExitCode::from(1);
            }
        }
    }
    eprintln!("xtask test-gpu: neither `pwsh` nor `powershell` found on PATH");
    ExitCode::from(1)
}

/// Read the declared-skip log written by `require_optional!` (via
/// `BARACUDA_SKIP_LOG`) and report what was skipped — so an absent optional
/// runtime on the box is STATED, never a silent `ok`. A `require!` (critical)
/// absence would already have failed the run (BARACUDA_GPU_REQUIRED), so what
/// lands here is the optional-runtime skips.
fn tally_skips(skip_log: &Path) {
    let Ok(contents) = std::fs::read_to_string(skip_log) else {
        return;
    };
    let rows: Vec<&str> = contents
        .lines()
        .filter(|l| l.starts_with("SKIP-DECLARED"))
        .collect();
    if rows.is_empty() {
        return;
    }
    eprintln!(
        "xtask test-gpu: {} test(s) declared a resource skippable and skipped it on this box \
         (require_optional! — an optional runtime/hardware was absent here):",
        rows.len()
    );
    for row in rows {
        // "SKIP-DECLARED\t<test>\t<reason>"
        let mut cols = row.split('\t').skip(1);
        let test = cols.next().unwrap_or("<?>");
        let reason = cols.next().unwrap_or("<?>");
        eprintln!("  - {test}: {reason}");
    }
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
