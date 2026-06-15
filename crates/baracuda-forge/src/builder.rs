//! Main kernel builder implementation.

use crate::compute_cap::{ComputeCapability, GpuArch};
use crate::dependency::DependencyManager;
use crate::error::{Error, Result};
use crate::hash::{hash_args, hash_paths, BuildCache};
use crate::parallel::ParallelConfig;
use crate::source::SourceSelector;
use crate::toolkit::CudaToolkit;

use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

/// Main builder for CUDA kernel compilation.
#[derive(Debug)]
pub struct KernelBuilder {
    toolkit: Option<CudaToolkit>,
    compute_cap: ComputeCapability,
    sources: SourceSelector,
    dependencies: DependencyManager,
    parallel: ParallelConfig,
    out_dir: PathBuf,
    extra_args: Vec<String>,
    incremental: bool,
    /// Explicit C++ standard for `-std=`. `None` means auto-select from the
    /// detected toolkit version: c++20 for CUDA >= 12.0, c++17 otherwise.
    cpp_std: Option<String>,
}

impl Default for KernelBuilder {
    fn default() -> Self {
        let out_dir = std::env::var("OUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("target/debug"));

        Self {
            toolkit: None,
            compute_cap: ComputeCapability::default(),
            sources: SourceSelector::default(),
            dependencies: DependencyManager::default(),
            parallel: ParallelConfig::default(),
            out_dir,
            extra_args: Vec::new(),
            incremental: true,
            cpp_std: None,
        }
    }
}

impl KernelBuilder {
    /// Create a new kernel builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    // ========== Source Selection ==========

    /// Add a directory to search for `.cu` files (recursive).
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// KernelBuilder::new().source_dir("src/kernels");
    /// ```
    pub fn source_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.sources = self.sources.add_directory(dir);
        self
    }

    /// Add specific kernel files.
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// KernelBuilder::new().source_files(["src/kernels/hello.cu", "src/kernels/world.cu"]);
    /// ```
    pub fn source_files<I, P>(mut self, files: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        self.sources = self.sources.add_files(files);
        self
    }

    /// Add kernel files matching a glob pattern.
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// KernelBuilder::new().source_glob("src/**/*.cu");
    /// ```
    pub fn source_glob(mut self, pattern: &str) -> Self {
        self.sources = self.sources.add_glob(pattern);
        self
    }

    /// Exclude files matching patterns.
    pub fn exclude(mut self, patterns: &[&str]) -> Self {
        self.sources = self.sources.exclude(patterns);
        self
    }

    /// Add paths to watch for changes (headers, etc.).
    pub fn watch<I, P>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        self.sources = self.sources.watch(paths);
        self
    }

    // ========== Compute Capability ==========

    /// Set the default compute capability (numeric, auto-selects suffix for sm_90+).
    pub fn compute_cap(mut self, cap: usize) -> Self {
        self.compute_cap = self.compute_cap.with_default(cap);
        self
    }

    /// Set the default compute capability with explicit arch string (e.g., `"90a"`, `"100a"`).
    pub fn compute_cap_arch(mut self, arch: &str) -> Self {
        self.compute_cap = self.compute_cap.with_default_arch(arch);
        self
    }

    /// Set compute cap override for specific kernels (numeric).
    ///
    /// Pattern can use wildcards: `"sm90_*.cu"`, `"*_hopper.cu"`.
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// KernelBuilder::new()
    ///     .source_glob("src/**/*.cu")
    ///     .with_compute_override("sm90_*.cu", 90)   // Hopper kernels
    ///     .with_compute_override("sm80_*.cu", 80);  // Ampere kernels
    /// ```
    pub fn with_compute_override(mut self, pattern: &str, cap: usize) -> Self {
        self.compute_cap = self.compute_cap.with_override(pattern, cap);
        self
    }

    /// Set compute cap override with explicit arch string.
    pub fn with_compute_override_arch(mut self, pattern: &str, arch: &str) -> Self {
        self.compute_cap = self.compute_cap.with_override_arch(pattern, arch);
        self
    }

    /// Get the current default compute capability (base number only).
    pub fn get_compute_cap(&self) -> Option<usize> {
        self.compute_cap.get_default().ok().map(|a| a.base)
    }

    /// Set compute capability (mutable reference version).
    pub fn set_compute_cap(&mut self, cap: usize) {
        self.compute_cap = ComputeCapability::new().with_default(cap);
    }

    /// Require explicit compute capability (fail fast if not set).
    ///
    /// Use this for Docker builds or CI environments where `nvidia-smi` is
    /// unavailable. The build fails immediately if `CUDA_COMPUTE_CAP` is not
    /// set and no compute capability was explicitly configured.
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// # fn build() -> Result<(), baracuda_forge::Error> {
    /// // In a Docker build, fail at build time if CUDA_COMPUTE_CAP wasn't
    /// // baked into the image:
    /// KernelBuilder::new()
    ///     .require_explicit_compute_cap()?
    ///     .source_dir("src/kernels")
    ///     .build_lib("libkernels.a")?;
    /// # Ok(()) }
    /// ```
    pub fn require_explicit_compute_cap(self) -> Result<Self> {
        if self.compute_cap.get_default().is_ok() {
            return Ok(self);
        }

        if std::env::var("CUDA_COMPUTE_CAP").is_ok() {
            return Ok(self);
        }

        Err(Error::ComputeCapDetectionFailed(
            "Explicit compute capability required but not set. \
            Either call .compute_cap(N) on the builder or set CUDA_COMPUTE_CAP environment variable. \
            This is required for Docker builds where nvidia-smi is unavailable.".to_string()
        ))
    }

    // ========== External Dependencies ==========

    /// Add CUTLASS dependency.
    ///
    /// `commit` pins a specific CUTLASS commit hash. Pass `None` to use the
    /// built-in default. When the consuming crate also depends on
    /// `baracuda-cutlass-sys`, that crate's pinned version wins automatically
    /// via cargo's `links` mechanism — forge then skips its own git fetch.
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// # fn build() -> Result<(), baracuda_forge::Error> {
    /// KernelBuilder::new()
    ///     .source_dir("src/kernels")
    ///     .with_cutlass(None)
    ///     .arg("-DUSE_CUTLASS")
    ///     .build_lib("libkernels.a")?;
    /// # Ok(()) }
    /// ```
    pub fn with_cutlass(mut self, commit: Option<&str>) -> Self {
        self.dependencies = self.dependencies.with_cutlass(commit);
        self
    }

    /// Add a custom git dependency.
    ///
    /// If `recurse_submodules` is false, clone/fetch adds `--no-recurse-submodules`.
    pub fn with_git_dependency(
        mut self,
        name: &str,
        repo: &str,
        commit: &str,
        include_paths: Vec<&str>,
        extra_paths: Vec<&str>,
        recurse_submodules: bool,
    ) -> Self {
        self.dependencies = self.dependencies.with_git_dependency(
            name,
            repo,
            commit,
            include_paths,
            extra_paths,
            recurse_submodules,
        );
        self
    }

    /// Fetch a configured git dependency and return its checkout root.
    pub fn fetch_git_dependency(&self, name: &str) -> Result<PathBuf> {
        self.dependencies.fetch_dependency(name, &self.out_dir)
    }

    /// Add a local include path.
    pub fn include_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.dependencies = self.dependencies.with_local_include(path);
        self
    }

    // ========== Parallel Configuration ==========

    /// Set the percentage of available threads to use (0.0 - 1.0).
    pub fn thread_percentage(mut self, percentage: f32) -> Self {
        self.parallel = self.parallel.with_percentage(percentage);
        self
    }

    /// Set the maximum number of threads.
    pub fn max_threads(mut self, max: usize) -> Self {
        self.parallel = self.parallel.with_max_threads(max);
        self
    }

    /// Set patterns for files that should use nvcc's `--threads=N` flag.
    pub fn nvcc_thread_patterns<S: AsRef<str>>(
        mut self,
        patterns: &[S],
        num_nvcc_threads: usize,
    ) -> Self {
        self.parallel = self
            .parallel
            .with_nvcc_thread_patterns(patterns, num_nvcc_threads);
        self
    }

    // ========== Build Configuration ==========

    /// Set the output directory.
    pub fn out_dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.out_dir = dir.into();
        self
    }

    /// Add an extra nvcc argument.
    pub fn arg(mut self, arg: &str) -> Self {
        self.extra_args.push(arg.to_string());
        self
    }

    /// Add multiple extra nvcc arguments.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for arg in args {
            self.extra_args.push(arg.as_ref().to_string());
        }
        self
    }

    /// Disable incremental builds.
    pub fn no_incremental(mut self) -> Self {
        self.incremental = false;
        self
    }

    /// Set explicit CUDA toolkit path.
    pub fn cuda_root<P: AsRef<Path>>(mut self, path: P) -> Self {
        if let Ok(toolkit) = CudaToolkit::from_nvcc_path(path.as_ref().join("bin").join("nvcc")) {
            self.toolkit = Some(toolkit);
        }
        self
    }

    /// Set the C++ standard passed to nvcc as `-std=<standard>`.
    ///
    /// Pass values like `"c++17"`, `"c++20"`. When unset (the default), the
    /// builder selects automatically from the detected toolkit version:
    /// `c++20` for CUDA >= 12.0, `c++17` otherwise.
    ///
    /// If your `extra_args` already contains a `-std=` argument, this method's
    /// value is ignored (your explicit `-std=` wins).
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// // Force c++17 even on CUDA 12+, e.g. for code that must compile
    /// // against both 11.x and 12.x toolkits:
    /// KernelBuilder::new().cpp_std("c++17");
    /// ```
    pub fn cpp_std(mut self, standard: &str) -> Self {
        self.cpp_std = Some(standard.to_string());
        self
    }

    // ========== Build Methods ==========

    /// Build a static library from all kernel sources.
    ///
    /// `out_file` is typically `format!("{}/libkernels.a", env!("OUT_DIR"))`.
    /// Pair with `cargo:rustc-link-search` and `cargo:rustc-link-lib` to wire
    /// the library into the resulting Rust binary.
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// let out_dir = std::env::var("OUT_DIR").unwrap();
    /// KernelBuilder::new()
    ///     .source_dir("src/kernels")
    ///     .arg("-O3")
    ///     .build_lib(format!("{out_dir}/libkernels.a"))
    ///     .unwrap();
    /// println!("cargo:rustc-link-search={out_dir}");
    /// println!("cargo:rustc-link-lib=kernels");
    /// ```
    pub fn build_lib<P: Into<PathBuf>>(&self, out_file: P) -> Result<()> {
        let out_file = out_file.into();

        let toolkit = match &self.toolkit {
            Some(t) => t.clone(),
            None => CudaToolkit::detect()?,
        };

        let _ = self.parallel.init_thread_pool();

        println!(
            "cargo:warning=Using {} threads for compilation",
            self.parallel.thread_count()
        );

        std::fs::create_dir_all(&self.out_dir)?;

        let kernel_files = self.sources.resolve()?;
        if kernel_files.is_empty() {
            println!("cargo:warning=No kernel files found");
            return Ok(());
        }

        for file in &kernel_files {
            println!("cargo:rerun-if-changed={}", file.display());
        }
        for path in self.sources.watch_paths() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

        let dep_args = self.dependencies.fetch_all(&self.out_dir)?;

        let mut cache = if self.incremental {
            BuildCache::load(&self.out_dir)
        } else {
            BuildCache::default()
        };

        let cpp_std_arg = self.resolve_cpp_std_arg(&toolkit);

        let mut all_args = Vec::new();
        if let Some(std_arg) = &cpp_std_arg {
            all_args.push(std_arg.clone());
        }
        all_args.extend(self.extra_args.iter().cloned());
        all_args.extend(dep_args.clone());
        let args_hash = hash_args(&all_args);

        let watch_hash = hash_paths(self.sources.watch_paths());

        let mut compile_jobs: Vec<(PathBuf, PathBuf, GpuArch)> = Vec::new();
        let mut all_obj_files: Vec<PathBuf> = Vec::new();

        for kernel_file in &kernel_files {
            let filename = kernel_file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let gpu_arch = self.compute_cap.get_for_file(filename)?;

            let obj_file = self.object_file_path(kernel_file);
            all_obj_files.push(obj_file.clone());

            if self.incremental
                && !cache.needs_rebuild(
                    kernel_file,
                    &obj_file,
                    &gpu_arch.to_nvcc_arch(),
                    &args_hash,
                    &watch_hash,
                )
            {
                continue;
            }

            compile_jobs.push((kernel_file.clone(), obj_file, gpu_arch));
        }

        if compile_jobs.is_empty() && out_file.exists() {
            println!("cargo:warning=All library kernels up-to-date, skipping compilation");
            return Ok(());
        }

        println!(
            "cargo:warning=Compiling {} of {} kernels",
            compile_jobs.len(),
            kernel_files.len()
        );

        let target = std::env::var("TARGET").ok();
        let is_msvc = target.as_ref().is_some_and(|t| t.contains("msvc"));
        let ccbin_env = std::env::var("NVCC_CCBIN").ok();
        let nvcc_threads = self.parallel.nvcc_threads();

        let had_error = AtomicBool::new(false);

        compile_jobs.par_iter().try_for_each(
            |(kernel_file, obj_file, gpu_arch)| -> Result<()> {
                if had_error.load(Ordering::Relaxed) {
                    return Ok(());
                }

                let gencode_arg = gpu_arch.to_gencode_arg();

                let mut command = Command::new(&toolkit.nvcc_path);
                command
                    .arg(&gencode_arg)
                    .arg("-c")
                    .arg("-o")
                    .arg(obj_file)
                    .args(["--default-stream", "per-thread"]);

                if let Some(std_arg) = &cpp_std_arg {
                    command.arg(std_arg);
                }

                for arg in &self.extra_args {
                    command.arg(arg);
                }

                for arg in &dep_args {
                    command.arg(arg);
                }

                if self.dependencies.has_cutlass() {
                    command.arg("-DUSE_CUTLASS");
                }

                if let Some(ccbin) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin]);
                }

                if !is_msvc {
                    command.arg("-Xcompiler").arg("-fPIC");
                } else {
                    command.arg("-D_USE_MATH_DEFINES");
                    msvc_cccl_args(&mut command);
                }

                if let Some(threads) = nvcc_threads {
                    let filename = kernel_file.to_string_lossy();
                    if self.parallel.should_use_nvcc_threads(&filename) {
                        command.arg(format!("--threads={}", threads));
                    }
                }

                command.arg(kernel_file);

                let output = command
                    .spawn()
                    .map_err(|e| Error::NvccNotFound(format!("Failed to spawn nvcc: {}", e)))?
                    .wait_with_output()
                    .map_err(|e| Error::CompilationFailed {
                        path: kernel_file.clone(),
                        message: e.to_string(),
                    })?;

                if !output.status.success() {
                    had_error.store(true, Ordering::Relaxed);
                    return Err(Error::CompilationFailed {
                        path: kernel_file.clone(),
                        message: format!(
                            "nvcc error:\n{}\n{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        ),
                    });
                }

                Ok(())
            },
        )?;

        if self.incremental {
            for (kernel_file, obj_file, gpu_arch) in &compile_jobs {
                cache.update(
                    kernel_file,
                    obj_file,
                    &gpu_arch.to_nvcc_arch(),
                    &args_hash,
                    &watch_hash,
                )?;
            }
            cache.save(&self.out_dir)?;
        }

        // Linking with many objects can exceed Windows' 32 KiB
        // command-line limit (one large baracuda-kernels-sys build
        // pushed past this with 217 .cu files, and `nvcc --lib`
        // doesn't accept response files — it errors with
        // `Don't know what to do with '@file'`).
        //
        // On MSVC hosts we work around this by invoking the MSVC
        // archiver (`lib.exe`) directly with the object list passed
        // via a response file (`@file`), which `lib.exe` natively
        // supports for arguments of arbitrary length. nvcc's `--lib`
        // would have shelled out to `lib.exe` anyway, so this
        // preserves the same archive format. We discover `lib.exe`
        // by querying `vswhere.exe` for the most recent MSVC install
        // (matching the layout the Rust MSVC toolchain target uses).
        //
        // On non-MSVC hosts argv limits are much higher (~2 MiB on
        // Linux) so we keep the simpler nvcc --lib path there.
        if is_msvc {
            archive_with_msvc_lib(&out_file, &all_obj_files, &self.out_dir)?;
        } else {
            let mut command = Command::new(&toolkit.nvcc_path);
            command
                .arg("--lib")
                .arg("-o")
                .arg(&out_file)
                .args(&all_obj_files);

            let output = command
                .spawn()
                .map_err(|e| Error::NvccNotFound(format!("Failed to spawn nvcc for linking: {}", e)))?
                .wait_with_output()
                .map_err(|e| Error::LinkingFailed(e.to_string()))?;

            if !output.status.success() {
                return Err(Error::LinkingFailed(format!(
                    "nvcc linking error:\n{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
        }

        Ok(())
    }

    /// Build PTX files from all kernel sources.
    ///
    /// Each `.cu` source produces a `<stem>.ptx` text file in the configured
    /// `out_dir`. The returned [`PtxOutput`] can write a Rust source file
    /// that exposes each PTX as a `pub const &str` for runtime loading via
    /// `baracuda-driver`'s `Module::load_ptx`.
    ///
    /// ```no_run
    /// # use baracuda_forge::KernelBuilder;
    /// # fn build() -> Result<(), baracuda_forge::Error> {
    /// let output = KernelBuilder::new()
    ///     .source_glob("src/**/*.cu")
    ///     .build_ptx()?;
    /// output.write("src/kernels.rs")?;
    /// # Ok(()) }
    /// ```
    pub fn build_ptx(&self) -> Result<PtxOutput> {
        let toolkit = match &self.toolkit {
            Some(t) => t.clone(),
            None => CudaToolkit::detect()?,
        };

        let _ = self.parallel.init_thread_pool();
        std::fs::create_dir_all(&self.out_dir)?;

        let kernel_files = self.sources.resolve()?;

        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            toolkit.include_dir.display()
        );

        for file in &kernel_files {
            println!("cargo:rerun-if-changed={}", file.display());
        }
        for path in self.sources.watch_paths() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

        let dep_args = self.dependencies.fetch_all(&self.out_dir)?;
        let ccbin_env = std::env::var("NVCC_CCBIN").ok();
        let is_msvc = std::env::var("TARGET").ok().is_some_and(|t| t.contains("msvc"));
        let nvcc_threads = self.parallel.nvcc_threads();
        let watch_hash = hash_paths(self.sources.watch_paths());
        let mut cache = BuildCache::load(&self.out_dir);

        let cpp_std_arg = self.resolve_cpp_std_arg(&toolkit);

        let mut all_args = Vec::new();
        if let Some(std_arg) = &cpp_std_arg {
            all_args.push(std_arg.clone());
        }
        all_args.extend(self.extra_args.iter().cloned());
        all_args.extend(dep_args.clone());
        let args_hash = hash_args(&all_args);

        let mut compile_jobs = Vec::new();
        for kernel_file in &kernel_files {
            let filename = kernel_file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let gpu_arch = self.compute_cap.get_for_file(filename)?;

            let output_file = self
                .out_dir
                .join(kernel_file.with_extension("ptx").file_name().unwrap());

            if self.incremental
                && !cache.needs_rebuild(
                    kernel_file,
                    &output_file,
                    &gpu_arch.to_nvcc_arch(),
                    &args_hash,
                    &watch_hash,
                )
            {
                continue;
            }

            compile_jobs.push((kernel_file, output_file, gpu_arch));
        }

        if compile_jobs.is_empty() {
            println!("cargo:warning=All PTX kernels up-to-date, skipping compilation");
            return Ok(PtxOutput {
                paths: kernel_files,
                out_dir: self.out_dir.clone(),
            });
        }

        println!(
            "cargo:warning=Compiling {} of {} PTX kernels",
            compile_jobs.len(),
            kernel_files.len()
        );

        compile_jobs.par_iter().try_for_each(
            |(kernel_file, _output_file, gpu_arch)| -> Result<()> {
                let gencode_arg = gpu_arch.to_gencode_arg();

                let mut command = Command::new(&toolkit.nvcc_path);
                command
                    .arg(&gencode_arg)
                    .arg("--ptx")
                    .args(["--default-stream", "per-thread"])
                    .args(["--output-directory", &self.out_dir.to_string_lossy()]);

                if let Some(std_arg) = &cpp_std_arg {
                    command.arg(std_arg);
                }

                for arg in &self.extra_args {
                    command.arg(arg);
                }
                for arg in &dep_args {
                    command.arg(arg);
                }
                if let Some(ccbin) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin]);
                }

                if is_msvc {
                    msvc_cccl_args(&mut command);
                }

                if let Some(threads) = nvcc_threads {
                    let file_path = kernel_file.to_string_lossy();
                    if self.parallel.should_use_nvcc_threads(&file_path) {
                        command.arg(format!("--threads={}", threads));
                    }
                }

                command.arg(kernel_file);

                let output = command
                    .spawn()
                    .map_err(|e| Error::NvccNotFound(format!("Failed to spawn nvcc: {}", e)))?
                    .wait_with_output()
                    .map_err(|e| Error::CompilationFailed {
                        path: kernel_file.to_path_buf(),
                        message: e.to_string(),
                    })?;

                if !output.status.success() {
                    return Err(Error::CompilationFailed {
                        path: kernel_file.to_path_buf(),
                        message: format!(
                            "nvcc error:\n{}\n{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        ),
                    });
                }

                Ok(())
            },
        )?;

        if self.incremental {
            for kernel_file in &kernel_files {
                let filename = kernel_file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                let gpu_arch = self.compute_cap.get_for_file(filename)?;
                let output_file = self
                    .out_dir
                    .join(kernel_file.with_extension("ptx").file_name().unwrap());

                cache.update(
                    kernel_file,
                    &output_file,
                    &gpu_arch.to_nvcc_arch(),
                    &args_hash,
                    &watch_hash,
                )?;
            }
            cache.save(&self.out_dir)?;
        }

        Ok(PtxOutput {
            paths: kernel_files,
            out_dir: self.out_dir.clone(),
        })
    }

    /// Resolve the `-std=<standard>` argument: explicit override, then auto from
    /// toolkit version, then `None` if neither is available *and* the user
    /// already supplied a `-std=` in `extra_args`.
    fn resolve_cpp_std_arg(&self, toolkit: &CudaToolkit) -> Option<String> {
        if self.extra_args.iter().any(|a| a.starts_with("-std=")) {
            return None;
        }

        if let Some(s) = &self.cpp_std {
            return Some(format!("-std={s}"));
        }

        let standard = match toolkit.version {
            Some((major, _)) if major >= 12 => "c++20",
            _ => "c++17",
        };
        Some(format!("-std={standard}"))
    }

    fn object_file_path(&self, kernel_file: &Path) -> PathBuf {
        let mut hasher = DefaultHasher::new();
        kernel_file.display().to_string().hash(&mut hasher);
        let hash = hasher.finish();

        let stem = kernel_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("kernel");

        self.out_dir.join(format!("{}-{:x}.o", stem, hash))
    }
}

/// Append the nvcc flags CCCL-heavy translation units require on MSVC hosts.
///
/// CUDA 12.5+ (and every CUDA 13.x, including the 13.3 the Fuel team hit)
/// bundles a CCCL whose `<cuda/std/__cccl/preprocessor.h>` opens with a hard
/// `#error` (MSVC `fatal error C1189`) when the host `cl.exe` is driving its
/// legacy *traditional* preprocessor:
///
/// ```text
/// #if defined(_MSC_VER) && !defined(__clang__)
/// #  if (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL == 1) \
///     && !defined(CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
/// #    error MSVC/cl.exe with traditional preprocessor is used ...
/// ```
///
/// CUTLASS (here) and cub/thrust (in `baracuda-kernels-sys`) pull this header
/// in transitively, so *every* CCCL-touching `.cu` fails to compile. We pass
/// `-Xcompiler /Zc:preprocessor`, which flips `cl.exe` to its standard-
/// conforming preprocessor (defining `_MSVC_TRADITIONAL=0`). That is both the
/// fix CCCL's own message recommends and the one CUTLASS's variadic-macro-heavy
/// headers actually need — unlike defining
/// `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING`, which only silences the
/// guard while leaving the non-conformant preprocessor (and its latent macro-
/// expansion bugs) in place. Verified against nvcc 13.3 + MSVC 19.5x: the flag
/// clears the error in both the host and device front-end passes.
///
/// No-op gate: callers invoke this only on MSVC targets. `/Zc:preprocessor`
/// needs VS 2019 16.5+, which every CUDA-12/13-supported MSVC comfortably
/// exceeds, so it is always safe to pass here.
fn msvc_cccl_args(command: &mut Command) {
    command.arg("-Xcompiler").arg("/Zc:preprocessor");
}

/// Locate the MSVC archiver (`lib.exe`) at build time.
///
/// We prefer the host-architecture build that matches `target_arch`
/// (since that's what nvcc would have picked). The discovery walks
/// the same paths cc-rs / cargo's MSVC setup walks, in priority order:
///
/// 1. `BARACUDA_FORGE_LIB_EXE` env var (escape hatch).
/// 2. `vswhere.exe` (the canonical MS-supplied locator).
/// 3. PATH (`lib.exe` may already be on PATH in dev-shell builds).
///
/// Returns the absolute path on success; an `Err` describing what
/// was tried otherwise.
fn find_msvc_lib_exe() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("BARACUDA_FORGE_LIB_EXE") {
        let pb = PathBuf::from(&p);
        if pb.exists() {
            return Ok(pb);
        }
        return Err(Error::LinkingFailed(format!(
            "BARACUDA_FORGE_LIB_EXE points to a non-existent file: {}",
            p
        )));
    }

    // Probe vswhere at its fixed install location.
    let vswhere = PathBuf::from(
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
    );
    if vswhere.exists() {
        let output = Command::new(&vswhere)
            .args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ])
            .output()
            .ok();
        if let Some(out) = output {
            if out.status.success() {
                let install_path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !install_path.is_empty() {
                    let install = PathBuf::from(&install_path);
                    let version_file = install.join(
                        r"VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt",
                    );
                    if let Ok(ver) = std::fs::read_to_string(&version_file) {
                        let ver = ver.trim();
                        // Match host arch — assume x64 (current target arch on
                        // every CUDA-supported Windows host today).
                        let lib = install
                            .join("VC")
                            .join("Tools")
                            .join("MSVC")
                            .join(ver)
                            .join("bin")
                            .join("Hostx64")
                            .join("x64")
                            .join("lib.exe");
                        if lib.exists() {
                            return Ok(lib);
                        }
                    }
                }
            }
        }
    }

    // Fall back to PATH lookup.
    if Command::new("lib.exe").arg("/?").output().is_ok() {
        return Ok(PathBuf::from("lib.exe"));
    }

    Err(Error::LinkingFailed(
        "could not locate MSVC `lib.exe`. Set `BARACUDA_FORGE_LIB_EXE` to its \
         full path, or run from a Visual Studio Developer Command Prompt so \
         `lib.exe` is on PATH."
            .to_string(),
    ))
}

/// Invoke `lib.exe` to assemble a static archive from `obj_files`,
/// passing the object list via a response file (`@file`) to avoid
/// Windows' command-line-length limit.
fn archive_with_msvc_lib(
    out_file: &Path,
    obj_files: &[PathBuf],
    out_dir: &Path,
) -> Result<()> {
    let lib_exe = find_msvc_lib_exe()?;

    let response_file = out_dir.join(".lib_response.txt");
    {
        let mut f = std::fs::File::create(&response_file).map_err(|e| {
            Error::LinkingFailed(format!(
                "failed to create lib response file {}: {}",
                response_file.display(),
                e
            ))
        })?;
        for obj in obj_files {
            // Each path on its own line, surrounded by quotes so any
            // embedded spaces survive lib.exe's response-file parser.
            writeln!(f, "\"{}\"", obj.display()).map_err(|e| {
                Error::LinkingFailed(format!(
                    "failed to write lib response file: {}",
                    e
                ))
            })?;
        }
    }

    let mut command = Command::new(&lib_exe);
    command
        .arg("/NOLOGO")
        .arg(format!("/OUT:{}", out_file.display()))
        .arg(format!("@{}", response_file.display()));

    let output = command
        .spawn()
        .map_err(|e| {
            Error::NvccNotFound(format!(
                "Failed to spawn {} for linking: {}",
                lib_exe.display(),
                e
            ))
        })?
        .wait_with_output()
        .map_err(|e| Error::LinkingFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(Error::LinkingFailed(format!(
            "{} archiving error:\n{}\n{}",
            lib_exe.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(())
}

/// Output from PTX compilation.
pub struct PtxOutput {
    paths: Vec<PathBuf>,
    #[allow(dead_code)]
    out_dir: PathBuf,
}

impl PtxOutput {
    /// Write a Rust source file with `const` declarations for each PTX file.
    pub fn write<P: AsRef<Path>>(&self, out: P) -> Result<()> {
        let mut file = std::fs::File::create(out.as_ref())?;

        for kernel_path in &self.paths {
            let name = kernel_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("KERNEL");

            writeln!(
                file,
                r#"pub const {}: &str = include_str!(concat!(env!("OUT_DIR"), "/{}.ptx"));"#,
                name.to_uppercase().replace(['.', '-'], "_"),
                name
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toolkit_with_version(version: Option<(u32, u32)>) -> CudaToolkit {
        CudaToolkit {
            nvcc_path: PathBuf::from("/dev/null"),
            include_dir: PathBuf::from("/dev/null"),
            lib_dir: PathBuf::from("/dev/null"),
            version,
        }
    }

    #[test]
    fn cpp_std_auto_selects_cpp20_for_cuda_12() {
        let b = KernelBuilder::new();
        let arg = b.resolve_cpp_std_arg(&toolkit_with_version(Some((12, 6))));
        assert_eq!(arg.as_deref(), Some("-std=c++20"));
    }

    #[test]
    fn cpp_std_auto_selects_cpp17_for_cuda_11() {
        let b = KernelBuilder::new();
        let arg = b.resolve_cpp_std_arg(&toolkit_with_version(Some((11, 8))));
        assert_eq!(arg.as_deref(), Some("-std=c++17"));
    }

    #[test]
    fn cpp_std_auto_selects_cpp17_when_version_unknown() {
        let b = KernelBuilder::new();
        let arg = b.resolve_cpp_std_arg(&toolkit_with_version(None));
        assert_eq!(arg.as_deref(), Some("-std=c++17"));
    }

    #[test]
    fn cpp_std_explicit_override_wins() {
        let b = KernelBuilder::new().cpp_std("c++17");
        let arg = b.resolve_cpp_std_arg(&toolkit_with_version(Some((12, 6))));
        assert_eq!(arg.as_deref(), Some("-std=c++17"));
    }

    #[test]
    fn cpp_std_extra_arg_disables_auto() {
        let b = KernelBuilder::new().arg("-std=c++14");
        let arg = b.resolve_cpp_std_arg(&toolkit_with_version(Some((12, 6))));
        assert_eq!(arg, None);
    }
}
