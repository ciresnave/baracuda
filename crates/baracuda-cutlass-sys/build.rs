//! Acquire CUTLASS headers and emit Cargo metadata for downstream consumers.
//!
//! Resolution order:
//! 1. `CUTLASS_DIR` env var → use that as a local checkout, skip fetch.
//! 2. `BARACUDA_CUTLASS_COMMIT` env var → sparse-checkout that commit.
//! 3. `cutlass-2-11` feature → CUTLASS v2.11.0 (CUDA 11.4-compatible).
//! 4. Default → CUTLASS v4.2.0 (CUDA 12+).
//!
//! Caching: sparse checkouts live under `$BARACUDA_CUTLASS_HOME/checkouts/`
//! (default `~/.baracuda-cutlass-sys/checkouts/`), keyed by version or commit
//! prefix. File-locked for concurrent-build safety.

use fs2::FileExt;
use std::env;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const CUTLASS_REPO: &str = "https://github.com/NVIDIA/cutlass.git";
const CUTLASS_DEFAULT_TAG: &str = "v4.2.0";
const CUTLASS_2_11_TAG: &str = "v2.11.0";
const CUTLASS_INCLUDE_PATHS: &[&str] = &["include", "tools/util/include"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    println!("cargo:rerun-if-env-changed=BARACUDA_CUTLASS_COMMIT");
    println!("cargo:rerun-if-env-changed=BARACUDA_CUTLASS_HOME");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    // Phase 11.2 — Fuel team feedback #3. Detect Git-for-Windows' fake
    // `link.exe` (actually GNU coreutils `link`) shadowing the MSVC
    // linker on PATH and warn loudly with a fix.
    check_for_fake_link_exe();

    // docs.rs has no network access and a read-only $HOME, so the normal
    // git-sparse-checkout path is impossible. Short-circuit to an empty
    // include dir under OUT_DIR — rustdoc only needs the Rust sources to
    // compile, and downstream build scripts gate their nvcc invocations
    // on DOCS_RS in the same way.
    if env::var_os("DOCS_RS").is_some() {
        if let Err(e) = run_docs_rs_stub() {
            eprintln!("baracuda-cutlass-sys: docs.rs stub failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    if let Err(e) = run() {
        eprintln!("\n========================================");
        eprintln!("baracuda-cutlass-sys: failed to acquire CUTLASS");
        eprintln!("========================================");
        eprintln!("{e}");
        eprintln!("\nFixes:");
        eprintln!("  • Set CUTLASS_DIR=/path/to/cutlass to use a local checkout.");
        eprintln!("  • Set BARACUDA_CUTLASS_COMMIT=<sha> to pin a specific commit.");
        eprintln!("  • Enable feature `cutlass-2-11` if you need a CUDA 11.4-compatible CUTLASS.");
        eprintln!("  • Ensure `git` is on PATH and the build host has network access.");
        eprintln!("========================================\n");
        std::process::exit(1);
    }
}

fn run_docs_rs_stub() -> Result<(), String> {
    let out_dir = env::var("OUT_DIR")
        .map_err(|_| "OUT_DIR must be set by cargo".to_string())?;
    let include = PathBuf::from(&out_dir).join("cutlass-stub-include");
    fs::create_dir_all(&include)
        .map_err(|e| format!("create stub include dir {}: {e}", include.display()))?;

    println!("cargo:warning=baracuda-cutlass-sys: DOCS_RS=1 detected; emitting empty include stub (no CUTLASS fetch).");
    emit_cargo_keys(&PathBuf::from(&out_dir), &include, "docs.rs-stub");
    Ok(())
}

fn run() -> Result<(), String> {
    if let Ok(custom) = env::var("CUTLASS_DIR") {
        let root = PathBuf::from(&custom);
        let include = root.join("include");
        if !include.is_dir() {
            return Err(format!(
                "CUTLASS_DIR='{custom}' has no include/ subdirectory"
            ));
        }
        println!("cargo:warning=baracuda-cutlass-sys: using CUTLASS_DIR={}", root.display());
        emit_cargo_keys(&root, &include, "CUTLASS_DIR");
        return Ok(());
    }

    let target = pick_target();
    let cache_root = cache_dir()?;
    let cache_key = format!("cutlass-{}", target.cache_suffix());
    let dep_dir = cache_root.join(&cache_key);

    fs::create_dir_all(&cache_root)
        .map_err(|e| format!("create cache dir {}: {e}", cache_root.display()))?;

    let lock_path = cache_root.join(format!("{cache_key}.lock"));
    let lock_file = File::create(&lock_path)
        .map_err(|e| format!("create lock file {}: {e}", lock_path.display()))?;
    lock_file
        .lock_exclusive()
        .map_err(|e| format!("acquire lock on {}: {e}", lock_path.display()))?;

    let result = fetch_with_lock(&dep_dir, &target);

    let _ = FileExt::unlock(&lock_file);

    let dep_dir = result?;
    let include = dep_dir.join("include");
    if !include.is_dir() {
        return Err(format!(
            "CUTLASS sparse checkout at {} has no include/ subdirectory",
            dep_dir.display()
        ));
    }

    println!(
        "cargo:warning=baracuda-cutlass-sys: using {} at {}",
        target.label(),
        dep_dir.display()
    );
    emit_cargo_keys(&dep_dir, &include, target.label());

    Ok(())
}

#[derive(Debug)]
enum Target {
    Tag(&'static str),
    Commit(String),
}

impl Target {
    fn cache_suffix(&self) -> String {
        match self {
            Target::Tag(t) => t.trim_start_matches('v').replace('.', "_"),
            Target::Commit(c) => c[..16.min(c.len())].to_string(),
        }
    }

    fn label(&self) -> &str {
        match self {
            Target::Tag(t) => t,
            Target::Commit(c) => c.as_str(),
        }
    }
}

fn pick_target() -> Target {
    if let Ok(commit) = env::var("BARACUDA_CUTLASS_COMMIT") {
        return Target::Commit(commit);
    }

    if cfg!(feature = "cutlass-2-11") {
        return Target::Tag(CUTLASS_2_11_TAG);
    }

    Target::Tag(CUTLASS_DEFAULT_TAG)
}

fn fetch_with_lock(dep_dir: &Path, target: &Target) -> Result<PathBuf, String> {
    if dep_dir.join("include").is_dir() && current_ref_matches(dep_dir, target) {
        println!(
            "cargo:warning=baracuda-cutlass-sys: cache hit at {}",
            dep_dir.display()
        );
        return Ok(dep_dir.to_path_buf());
    }

    if !dep_dir.exists() {
        match target {
            Target::Tag(tag) => clone_tag(dep_dir, tag)?,
            Target::Commit(_) => clone_default_branch(dep_dir)?,
        }
    }

    setup_sparse_checkout(dep_dir)?;

    if let Target::Commit(commit) = target {
        cleanup_git_locks(dep_dir);
        fetch_commit(dep_dir, commit)?;
        checkout_commit(dep_dir, commit)?;
    }

    Ok(dep_dir.to_path_buf())
}

fn current_ref_matches(dep_dir: &Path, target: &Target) -> bool {
    let want = match target {
        Target::Tag(t) => (*t).to_string(),
        Target::Commit(c) => c.clone(),
    };

    let head = match Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(dep_dir)
        .output()
    {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        }
        _ => return false,
    };

    if head == want {
        return true;
    }

    if matches!(target, Target::Tag(_)) {
        let tag_resolved = Command::new("git")
            .args(["rev-list", "-n", "1", &want])
            .current_dir(dep_dir)
            .output();
        if let Ok(out) = tag_resolved {
            if out.status.success() {
                let resolved = String::from_utf8_lossy(&out.stdout).trim().to_string();
                return resolved == head;
            }
        }
    }

    false
}

fn clone_tag(dep_dir: &Path, tag: &str) -> Result<(), String> {
    println!(
        "cargo:warning=baracuda-cutlass-sys: cloning {} {}",
        CUTLASS_REPO, tag
    );

    let dep_dir_str = dep_dir
        .to_str()
        .ok_or_else(|| "non-UTF-8 cache path".to_string())?;

    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--no-recurse-submodules",
            "--branch",
            tag,
            CUTLASS_REPO,
            dep_dir_str,
        ])
        .status()
        .map_err(|e| git_invocation_error("clone", e))?;

    if !status.success() {
        return Err(format!("git clone {tag} failed with status {status}"));
    }

    Ok(())
}

fn clone_default_branch(dep_dir: &Path) -> Result<(), String> {
    println!(
        "cargo:warning=baracuda-cutlass-sys: cloning {} (default branch, will checkout commit)",
        CUTLASS_REPO
    );

    let dep_dir_str = dep_dir
        .to_str()
        .ok_or_else(|| "non-UTF-8 cache path".to_string())?;

    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--no-recurse-submodules",
            CUTLASS_REPO,
            dep_dir_str,
        ])
        .status()
        .map_err(|e| git_invocation_error("clone", e))?;

    if !status.success() {
        return Err(format!("git clone failed with status {status}"));
    }

    Ok(())
}

fn setup_sparse_checkout(dep_dir: &Path) -> Result<(), String> {
    let mut args = vec!["sparse-checkout", "set"];
    for path in CUTLASS_INCLUDE_PATHS {
        args.push(path);
    }

    let status = Command::new("git")
        .args(&args)
        .current_dir(dep_dir)
        .status()
        .map_err(|e| git_invocation_error("sparse-checkout set", e))?;

    if !status.success() {
        return Err(format!("git sparse-checkout failed with status {status}"));
    }

    Ok(())
}

fn fetch_commit(dep_dir: &Path, commit: &str) -> Result<(), String> {
    println!(
        "cargo:warning=baracuda-cutlass-sys: fetching commit {commit}"
    );

    let status = Command::new("git")
        .args(["fetch", "--no-recurse-submodules", "origin", commit])
        .current_dir(dep_dir)
        .status()
        .map_err(|e| git_invocation_error("fetch", e))?;

    if !status.success() {
        return Err(format!("git fetch {commit} failed with status {status}"));
    }

    Ok(())
}

fn checkout_commit(dep_dir: &Path, commit: &str) -> Result<(), String> {
    let status = Command::new("git")
        .args(["checkout", commit])
        .current_dir(dep_dir)
        .status()
        .map_err(|e| git_invocation_error("checkout", e))?;

    if !status.success() {
        return Err(format!("git checkout {commit} failed with status {status}"));
    }

    Ok(())
}

fn cleanup_git_locks(dep_dir: &Path) {
    let git_dir = dep_dir.join(".git");
    for name in ["index.lock", "HEAD.lock", "config.lock"] {
        let lock = git_dir.join(name);
        if let Ok(meta) = lock.metadata() {
            if let Ok(modified) = meta.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    if elapsed.as_secs() > 600 {
                        println!(
                            "cargo:warning=baracuda-cutlass-sys: removing stale git lock {}",
                            lock.display()
                        );
                        let _ = fs::remove_file(&lock);
                    }
                }
            }
        }
    }
}

fn cache_dir() -> Result<PathBuf, String> {
    if let Ok(custom) = env::var("BARACUDA_CUTLASS_HOME") {
        return Ok(PathBuf::from(custom).join("checkouts"));
    }

    let home = env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .map_err(|_| "neither HOME nor USERPROFILE is set".to_string())?;

    Ok(PathBuf::from(home)
        .join(".baracuda-cutlass-sys")
        .join("checkouts"))
}

fn emit_cargo_keys(root: &Path, include: &Path, version_label: &str) {
    println!("cargo:root={}", root.display());
    println!("cargo:include={}", include.display());
    println!("cargo:include_dir={}", include.display());
    println!("cargo:version={version_label}");

    println!(
        "cargo:rustc-env=BARACUDA_CUTLASS_INCLUDE_DIR={}",
        include.display()
    );
    println!(
        "cargo:rustc-env=BARACUDA_CUTLASS_ROOT={}",
        root.display()
    );
    println!("cargo:rustc-env=BARACUDA_CUTLASS_VERSION={version_label}");
}

fn git_invocation_error(operation: &str, err: io::Error) -> String {
    if err.kind() == io::ErrorKind::NotFound {
        format!(
            "git {operation} failed: git executable not found in PATH. \
             Install git and retry. Original error: {err}"
        )
    } else {
        format!("git {operation} failed: {err}")
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
    let Some(path) = env::var_os("PATH") else {
        return;
    };
    for entry in env::split_paths(&path) {
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
