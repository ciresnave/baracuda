//! External dependency management (CUTLASS, custom git repos).
//!
//! Handles fetching header-only C++ dependencies via git, with sparse
//! checkout, content-addressed caching under `~/.baracuda-forge/git/checkouts/`,
//! and file-locked concurrent-build safety.
//!
//! For the safe Rust-side CUTLASS pin, see the future `baracuda-cutlass-sys`
//! crate (Phase 2 of the integration plan).

use crate::error::{Error, Result};
use fs2::FileExt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const ANSI_RED_BOLD: &str = "\x1b[1;31m";
const ANSI_RESET: &str = "\x1b[0m";

/// Well-known CUTLASS repository configuration.
const CUTLASS_REPO: &str = "https://github.com/NVIDIA/cutlass.git";
const CUTLASS_DEFAULT_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";
const CUTLASS_INCLUDE_PATHS: &[&str] = &["include", "tools/util/include"];

/// External dependency configuration.
#[derive(Debug, Clone)]
pub struct ExternalDependency {
    /// Name of the dependency.
    pub name: String,
    /// Git repository URL.
    pub repo_url: String,
    /// Commit hash to checkout.
    pub commit: String,
    /// Include paths within the repo (relative to repo root).
    pub include_paths: Vec<String>,
    /// Additional sparse-checkout paths to fetch alongside includes.
    pub extra_paths: Vec<String>,
    /// Whether to allow git submodule recursion.
    pub recurse_submodules: bool,
}

impl ExternalDependency {
    /// Create a CUTLASS dependency with default or custom commit.
    pub fn cutlass(commit: Option<&str>) -> Self {
        Self {
            name: "cutlass".to_string(),
            repo_url: CUTLASS_REPO.to_string(),
            commit: commit.unwrap_or(CUTLASS_DEFAULT_COMMIT).to_string(),
            include_paths: CUTLASS_INCLUDE_PATHS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            extra_paths: Vec::new(),
            recurse_submodules: true,
        }
    }

    /// Create a custom git dependency.
    pub fn git(
        name: &str,
        repo_url: &str,
        commit: &str,
        include_paths: Vec<&str>,
        extra_paths: Vec<&str>,
        recurse_submodules: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            repo_url: repo_url.to_string(),
            commit: commit.to_string(),
            include_paths: include_paths.iter().map(|s| s.to_string()).collect(),
            extra_paths: extra_paths.iter().map(|s| s.to_string()).collect(),
            recurse_submodules,
        }
    }

    fn sparse_paths(&self) -> Vec<&str> {
        let mut paths = Vec::with_capacity(self.include_paths.len() + self.extra_paths.len());
        for path in &self.include_paths {
            paths.push(path.as_str());
        }
        for path in &self.extra_paths {
            if !self.include_paths.iter().any(|p| p == path) {
                paths.push(path.as_str());
            }
        }
        paths
    }

    /// Fetch the dependency to the cache directory.
    ///
    /// Uses sparse checkout to only fetch include directories. Caches under
    /// `~/.baracuda-forge/git/checkouts/{name}-{commit_prefix}/` to avoid
    /// re-cloning on subsequent builds. Uses file locking to prevent
    /// concurrent builds from conflicting.
    pub fn fetch(&self, out_dir: &Path) -> Result<PathBuf> {
        let cache_dir = forge_git_cache_dir(out_dir)?;

        let commit_prefix = &self.commit[..16.min(self.commit.len())];
        let cache_key = format!("{}-{}", self.name, commit_prefix);
        let dep_dir = cache_dir.join(&cache_key);

        let lock_path = cache_dir.join(format!("{}.lock", cache_key));
        let lock_file = File::create(&lock_path)
            .map_err(|e| Error::GitOperationFailed(format!("Failed to create lock file: {}", e)))?;

        lock_file
            .lock_exclusive()
            .map_err(|e| Error::GitOperationFailed(format!("Failed to acquire lock: {}", e)))?;

        let result = self.fetch_with_lock(&dep_dir);

        // UFCS pins us to fs2's trait method even on Rust 1.89+, where
        // std::fs::File grew its own consuming `unlock` that would otherwise
        // win method resolution and break our 1.75 MSRV.
        let _ = FileExt::unlock(&lock_file);

        result
    }

    fn fetch_with_lock(&self, dep_dir: &PathBuf) -> Result<PathBuf> {
        if dep_dir.join("include").exists() {
            if let Ok(current_commit) = self.get_current_commit(dep_dir) {
                if current_commit == self.commit {
                    println!(
                        "cargo:warning=Using cached {} at {}",
                        self.name,
                        dep_dir.display()
                    );
                    return Ok(dep_dir.clone());
                }
            }
        }

        if !dep_dir.exists() {
            self.clone_repo(dep_dir)?;
        }

        self.setup_sparse_checkout(dep_dir)?;
        self.checkout_commit(dep_dir)?;

        println!(
            "cargo:warning=Cached {} at {}",
            self.name,
            dep_dir.display()
        );

        Ok(dep_dir.clone())
    }

    /// Get include path arguments for nvcc.
    pub fn include_args(&self, base_dir: &Path) -> Vec<String> {
        let mut args = Vec::new();

        args.push(format!("-I{}", base_dir.display()));

        for include_path in &self.include_paths {
            let full_path = base_dir.join(include_path);
            if full_path.exists() {
                args.push(format!("-I{}", full_path.display()));
            }
        }

        args
    }

    fn get_current_commit(&self, dir: &PathBuf) -> Result<String> {
        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(dir)
            .output()
            .map_err(|e| git_command_error("rev-parse", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn clone_repo(&self, target_dir: &Path) -> Result<()> {
        println!("cargo:warning=Cloning {} from {}", self.name, self.repo_url);

        let target_dir_str = target_dir
            .to_str()
            .ok_or_else(|| Error::GitOperationFailed("Invalid path encoding".to_string()))?;

        let mut cmd = Command::new("git");
        cmd.args(["clone", "--depth", "1", "--filter=blob:none", "--sparse"]);
        if !self.recurse_submodules {
            cmd.arg("--no-recurse-submodules");
        }
        let status = cmd
            .arg(&self.repo_url)
            .arg(target_dir_str)
            .status()
            .map_err(|e| git_command_error("clone", e))?;

        if !status.success() {
            return Err(Error::GitOperationFailed(format!(
                "git clone failed with status: {}",
                status
            )));
        }

        Ok(())
    }

    fn setup_sparse_checkout(&self, dir: &PathBuf) -> Result<()> {
        let mut args = vec!["sparse-checkout", "set"];
        for path in self.sparse_paths() {
            args.push(path);
        }

        let status = Command::new("git")
            .args(&args)
            .current_dir(dir)
            .status()
            .map_err(|e| git_command_error("sparse-checkout", e))?;

        if !status.success() {
            return Err(Error::GitOperationFailed(format!(
                "git sparse-checkout failed with status: {}",
                status
            )));
        }

        Ok(())
    }

    fn checkout_commit(&self, dir: &PathBuf) -> Result<()> {
        self.cleanup_git_locks(dir);

        println!(
            "cargo:warning=Fetching {} commit {}",
            self.name, self.commit
        );

        let mut cmd = Command::new("git");
        cmd.arg("fetch");
        if !self.recurse_submodules {
            cmd.arg("--no-recurse-submodules");
        }
        let status = cmd
            .args(["origin", &self.commit])
            .current_dir(dir)
            .status()
            .map_err(|e| git_command_error("fetch", e))?;

        if !status.success() {
            return Err(Error::GitOperationFailed(format!(
                "git fetch failed with status: {}",
                status
            )));
        }

        let status = Command::new("git")
            .args(["checkout", &self.commit])
            .current_dir(dir)
            .status()
            .map_err(|e| git_command_error("checkout", e))?;

        if !status.success() {
            return Err(Error::GitOperationFailed(format!(
                "git checkout failed with status: {}",
                status
            )));
        }

        Ok(())
    }

    fn cleanup_git_locks(&self, dir: &Path) {
        let git_dir = dir.join(".git");
        let lock_files = [
            git_dir.join("index.lock"),
            git_dir.join("HEAD.lock"),
            git_dir.join("config.lock"),
        ];

        for lock_file in &lock_files {
            if lock_file.exists() {
                if let Ok(metadata) = lock_file.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(elapsed) = modified.elapsed() {
                            if elapsed.as_secs() > 600 {
                                println!(
                                    "cargo:warning=Removing stale git lock file: {}",
                                    lock_file.display()
                                );
                                let _ = std::fs::remove_file(lock_file);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Dependency manager for handling multiple external dependencies.
#[derive(Debug, Clone, Default)]
pub struct DependencyManager {
    dependencies: Vec<ExternalDependency>,
    local_includes: Vec<PathBuf>,
}

impl DependencyManager {
    /// Create a new dependency manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add CUTLASS dependency.
    pub fn with_cutlass(mut self, commit: Option<&str>) -> Self {
        self.dependencies.push(ExternalDependency::cutlass(commit));
        self
    }

    /// Add a custom git dependency.
    pub fn with_git_dependency(
        mut self,
        name: &str,
        repo: &str,
        commit: &str,
        include_paths: Vec<&str>,
        extra_paths: Vec<&str>,
        recurse_submodules: bool,
    ) -> Self {
        self.dependencies.push(ExternalDependency::git(
            name,
            repo,
            commit,
            include_paths,
            extra_paths,
            recurse_submodules,
        ));
        self
    }

    /// Add a local include path.
    pub fn with_local_include<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.local_includes.push(path.into());
        self
    }

    /// Fetch all dependencies and return include arguments.
    ///
    /// CUTLASS is special-cased: if cargo set `DEP_CUTLASS_INCLUDE` (which it
    /// does whenever the consuming crate also depends on `baracuda-cutlass-sys`,
    /// since that crate has `links = "cutlass"`), forge uses those headers
    /// directly and skips its own git fetch. This lets users opt into the
    /// version-pinned, feature-flagged `baracuda-cutlass-sys` flow without
    /// changing their `KernelBuilder` calls.
    pub fn fetch_all(&self, out_dir: &Path) -> Result<Vec<String>> {
        let mut include_args = Vec::new();

        for local in &self.local_includes {
            if local.exists() {
                include_args.push(format!("-I{}", local.display()));
            }
        }

        for dep in &self.dependencies {
            if dep.name == "cutlass" {
                if let Some(env_args) = cutlass_args_from_env() {
                    println!(
                        "cargo:warning=baracuda-forge: using CUTLASS from baracuda-cutlass-sys (DEP_CUTLASS_INCLUDE)"
                    );
                    include_args.extend(env_args);
                    continue;
                }
            }
            let dep_dir = dep.fetch(out_dir)?;
            include_args.extend(dep.include_args(&dep_dir));
        }

        Ok(include_args)
    }

    /// Fetch a specific dependency and return its checkout root.
    pub fn fetch_dependency(&self, name: &str, out_dir: &Path) -> Result<PathBuf> {
        let dep = self
            .dependencies
            .iter()
            .find(|d| d.name == name)
            .ok_or_else(|| Error::GitOperationFailed(format!("Unknown dependency: {name}")))?;
        dep.fetch(out_dir)
    }

    /// Check if CUTLASS is enabled.
    pub fn has_cutlass(&self) -> bool {
        self.dependencies.iter().any(|d| d.name == "cutlass")
    }
}

/// Return `-I` args for CUTLASS based on env vars set by `baracuda-cutlass-sys`,
/// or `None` if it isn't in the build graph.
///
/// `baracuda-cutlass-sys` declares `links = "cutlass"`, so cargo sets
/// `DEP_CUTLASS_INCLUDE` (and `DEP_CUTLASS_ROOT`) in dependent crates'
/// build-script environments. We use `INCLUDE` for the primary `-I` and
/// (if present) `ROOT/tools/util/include` for the secondary one CUTLASS
/// often expects.
fn cutlass_args_from_env() -> Option<Vec<String>> {
    let include = std::env::var("DEP_CUTLASS_INCLUDE").ok()?;
    let root = std::env::var("DEP_CUTLASS_ROOT").ok();
    Some(cutlass_args_from_paths(&include, root.as_deref()))
}

fn cutlass_args_from_paths(include: &str, root: Option<&str>) -> Vec<String> {
    let mut args = vec![format!("-I{include}")];
    if let Some(root) = root {
        let util = Path::new(root).join("tools").join("util").join("include");
        if util.is_dir() {
            args.push(format!("-I{}", util.display()));
        }
    }
    args
}

/// Try to resolve CUTLASS from cargo's git checkouts directory.
pub fn resolve_cutlass_from_cargo_checkouts() -> Option<PathBuf> {
    let checkouts_dir = cargo_git_checkouts_dir().ok()?;

    let search_patterns = ["candle-flash-attn-*", "cutlass-*"];

    for pattern in search_patterns {
        let full_pattern = format!("{}/{}", checkouts_dir.display(), pattern);
        if let Ok(entries) = glob::glob(&full_pattern) {
            for entry in entries.flatten() {
                for subdir in ["cutlass", ""] {
                    let cutlass_path = if subdir.is_empty() {
                        entry.clone()
                    } else {
                        entry.join(subdir)
                    };

                    if cutlass_path.join("include").exists() {
                        return Some(cutlass_path);
                    }

                    if let Ok(subdirs) = std::fs::read_dir(&entry) {
                        for subentry in subdirs.flatten() {
                            let check_path = if subdir.is_empty() {
                                subentry.path()
                            } else {
                                subentry.path().join(subdir)
                            };

                            if check_path.join("include").exists() {
                                return Some(check_path);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Get the global cache directory for baracuda-forge git checkouts.
///
/// Priority:
/// 1. `$BARACUDA_FORGE_HOME/git/checkouts/` if set.
/// 2. `~/.baracuda-forge/git/checkouts/` if `HOME` is set.
/// 3. `$CARGO_HOME/git/checkouts/` (reuses Cargo's cache directory).
/// 4. `<fallback_dir>/git_cache` as last resort.
fn forge_git_cache_dir(fallback_dir: &Path) -> Result<PathBuf> {
    let cache_dir = if let Ok(home) = std::env::var("BARACUDA_FORGE_HOME") {
        PathBuf::from(home).join("git").join("checkouts")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".baracuda-forge")
            .join("git")
            .join("checkouts")
    } else if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        PathBuf::from(cargo_home).join("git").join("checkouts")
    } else {
        fallback_dir.join("git_cache")
    };

    std::fs::create_dir_all(&cache_dir).map_err(|e| {
        Error::GitOperationFailed(format!(
            "Failed to create cache dir {}: {}",
            cache_dir.display(),
            e
        ))
    })?;

    Ok(cache_dir)
}

fn cargo_git_checkouts_dir() -> Result<PathBuf> {
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        return Ok(PathBuf::from(cargo_home).join("git").join("checkouts"));
    }

    if let Ok(home) = std::env::var("HOME") {
        return Ok(PathBuf::from(home)
            .join(".cargo")
            .join("git")
            .join("checkouts"));
    }

    Err(Error::InvalidConfig(
        "Neither CARGO_HOME nor HOME is set".to_string(),
    ))
}

fn git_command_error(operation: &str, err: io::Error) -> Error {
    let mut message = format!("git {operation} failed: {err}");

    if err.kind() == io::ErrorKind::NotFound {
        let install_hint = format!("{ANSI_RED_BOLD}Please install git and retry.{ANSI_RESET}");
        message = format!(
            "git {operation} failed: git executable not found in PATH. {install_hint} Original error: {err}"
        );
    }

    Error::GitOperationFailed(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn cutlass_args_include_only_when_no_root() {
        let args = cutlass_args_from_paths("/cutlass/include", None);
        assert_eq!(args, vec!["-I/cutlass/include".to_string()]);
    }

    #[test]
    fn cutlass_args_skip_util_dir_when_missing() {
        let tmp = std::env::temp_dir().join(format!(
            "baracuda-forge-cutlass-args-{}-missing",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("include")).unwrap();
        let include = tmp.join("include").to_string_lossy().to_string();
        let root = tmp.to_string_lossy().to_string();

        let args = cutlass_args_from_paths(&include, Some(&root));
        assert_eq!(args.len(), 1);
        assert!(args[0].starts_with("-I"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cutlass_args_add_util_dir_when_present() {
        let tmp = std::env::temp_dir().join(format!(
            "baracuda-forge-cutlass-args-{}-present",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        let util = tmp.join("tools").join("util").join("include");
        fs::create_dir_all(&util).unwrap();
        fs::create_dir_all(tmp.join("include")).unwrap();
        let include = tmp.join("include").to_string_lossy().to_string();
        let root = tmp.to_string_lossy().to_string();

        let args = cutlass_args_from_paths(&include, Some(&root));
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], format!("-I{include}"));
        assert!(args[1].contains("tools"));
        assert!(args[1].contains("util"));

        let _ = fs::remove_dir_all(&tmp);
    }
}
