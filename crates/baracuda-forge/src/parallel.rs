//! Parallel build configuration.

use glob::Pattern;
use std::path::Path;
use std::str::FromStr;

/// Parallel build configuration.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    thread_percentage: f32,
    max_threads: Option<usize>,
    min_threads: usize,
    nvcc_thread_file_patterns: Vec<String>,
    num_nvcc_threads: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            thread_percentage: 0.5,
            max_threads: None,
            min_threads: 1,
            nvcc_thread_file_patterns: vec!["flash_api".to_string(), "cutlass".to_string()],
            num_nvcc_threads: Some(2),
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the percentage of available threads to use (clamped to 0.0..=1.0).
    pub fn with_percentage(mut self, percentage: f32) -> Self {
        self.thread_percentage = percentage.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum number of threads.
    pub fn with_max_threads(mut self, max: usize) -> Self {
        self.max_threads = Some(max.max(1));
        self
    }

    /// Set the minimum number of threads.
    pub fn with_min_threads(mut self, min: usize) -> Self {
        self.min_threads = min.max(1);
        self
    }

    /// Set patterns for files that should use nvcc's internal `--threads=N` flag.
    ///
    /// Replaces the default patterns (`"flash_api"`, `"cutlass"`).
    pub fn with_nvcc_thread_patterns<S: AsRef<str>>(
        mut self,
        patterns: &[S],
        num_nvcc_threads: usize,
    ) -> Self {
        self.nvcc_thread_file_patterns = patterns.iter().map(|s| s.as_ref().to_string()).collect();
        self.num_nvcc_threads = if num_nvcc_threads > 0 {
            Some(num_nvcc_threads)
        } else {
            None
        };
        self
    }

    /// Check if a file matches any of the thread patterns.
    ///
    /// Supports glob patterns (e.g. `"gemm_*.cu"`) and substring matching.
    pub fn should_use_nvcc_threads(&self, path_str: &str) -> bool {
        let path = Path::new(path_str);
        let filename_component = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

        self.nvcc_thread_file_patterns.iter().any(|pattern| {
            if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
                if let Ok(compiled) = Pattern::new(pattern) {
                    if !pattern.contains('/')
                        && !pattern.contains('\\')
                        && compiled.matches(filename_component)
                    {
                        return true;
                    }

                    if compiled.matches(path_str) {
                        return true;
                    }
                }
            }
            path_str.contains(pattern)
        })
    }

    /// Calculate the number of threads to use.
    pub fn thread_count(&self) -> usize {
        if let Ok(env_threads) = std::env::var("BARACUDA_FORGE_THREADS") {
            if let Ok(n) = usize::from_str(&env_threads) {
                return n.max(1);
            }
        }

        if let Ok(env_threads) = std::env::var("RAYON_NUM_THREADS") {
            if let Ok(n) = usize::from_str(&env_threads) {
                return n.max(1);
            }
        }

        // Cargo passes `-j` to build scripts as NUM_JOBS. Honour it as an upper
        // bound so a consumer running concurrent CUDA builds can cap this pool
        // with the vocabulary they already have (see `resolve_thread_count`).
        let num_jobs = std::env::var("NUM_JOBS")
            .ok()
            .and_then(|v| usize::from_str(&v).ok());

        self.resolve_thread_count(self.detect_available_threads(), num_jobs)
    }

    /// Resolve the pool size from the detected core count and cargo's `NUM_JOBS`
    /// (`-j`) when set. Pure (reads no environment) so the NUM_JOBS policy is
    /// unit-testable without env-var races.
    ///
    /// `NUM_JOBS` is an UPPER CAP, never a floor: cargo defaults it to the
    /// logical-CPU count when `-j` is omitted, so treating it as the primary
    /// value would silently raise the default from 50% to 100% of the box — the
    /// opposite of what a `-j` cap is for, and the exact oversubscription that
    /// makes ptxas fail with "Memory allocation failure". Capping means `-j 4`
    /// actually limits nvcc (the good-citizen case under concurrent builds),
    /// while an omitted `-j` leaves the memory-safe 50% default untouched.
    /// `BARACUDA_FORGE_THREADS` (checked first, in `thread_count`) remains the
    /// way to raise ABOVE the default.
    fn resolve_thread_count(&self, available: usize, num_jobs: Option<usize>) -> usize {
        let mut calculated = if let Some(max) = self.max_threads {
            max.min(available)
        } else {
            (available as f32 * self.thread_percentage).ceil() as usize
        };

        if let Some(n) = num_jobs {
            calculated = calculated.min(n.max(1));
        }

        calculated.max(self.min_threads).min(available)
    }

    /// Initialize the rayon thread pool with configured settings.
    pub fn init_thread_pool(&self) -> Result<(), rayon::ThreadPoolBuildError> {
        let thread_count = self.thread_count();
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build_global()
    }

    /// Get thread count for nvcc's `--threads` argument.
    pub fn nvcc_threads(&self) -> Option<usize> {
        self.num_nvcc_threads
    }

    fn detect_available_threads(&self) -> usize {
        if let Ok(parallelism) = std::thread::available_parallelism() {
            return parallelism.get();
        }
        num_cpus::get_physical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ParallelConfig::default();
        assert_eq!(config.thread_percentage, 0.5);
        assert!(config.max_threads.is_none());
    }

    #[test]
    fn num_jobs_caps_but_never_raises_the_default() {
        let cfg = ParallelConfig::default(); // 50% of cores, min_threads 1
        // `-j` omitted: cargo sets NUM_JOBS = core count, so the 50% default holds.
        assert_eq!(cfg.resolve_thread_count(16, Some(16)), 8);
        // `-j 4` (good citizen under concurrent builds) actually caps the pool.
        assert_eq!(cfg.resolve_thread_count(16, Some(4)), 4);
        // `-j` above the 50% default does NOT raise it — the memory-safe
        // heuristic wins; BARACUDA_FORGE_THREADS is the knob to go higher.
        assert_eq!(cfg.resolve_thread_count(16, Some(100)), 8);
        // No NUM_JOBS at all: unchanged 50% default.
        assert_eq!(cfg.resolve_thread_count(16, None), 8);
        // The min_threads floor still applies under an aggressive cap.
        assert_eq!(cfg.resolve_thread_count(16, Some(1)), 1);
    }

    #[test]
    fn test_percentage_clamping() {
        let config = ParallelConfig::new().with_percentage(1.5);
        assert_eq!(config.thread_percentage, 1.0);

        let config = ParallelConfig::new().with_percentage(-0.5);
        assert_eq!(config.thread_percentage, 0.0);
    }

    #[test]
    fn test_thread_patterns() {
        let config = ParallelConfig::default();
        assert!(config.should_use_nvcc_threads("flash_api.cu"));
        assert!(config.should_use_nvcc_threads("src/flash_api_v2.cu"));
        assert!(config.should_use_nvcc_threads("cutlass_gemm.cu"));
        assert!(!config.should_use_nvcc_threads("simple.cu"));

        let config = ParallelConfig::new().with_nvcc_thread_patterns(&["gemm_*.cu", "special"], 4);
        assert!(config.should_use_nvcc_threads("gemm_fp16.cu"));
        assert!(config.should_use_nvcc_threads("src/gemm_int8.cu"));
        assert!(config.should_use_nvcc_threads("special_kernel.cu"));
        assert!(!config.should_use_nvcc_threads("flash_api.cu"));
    }

    #[test]
    fn test_glob_vs_substring() {
        let config = ParallelConfig::new().with_nvcc_thread_patterns(&["*gemm*.cu"], 2);
        assert!(config.should_use_nvcc_threads("/path/to/my_gemm_kernel.cu"));
        assert!(!config.should_use_nvcc_threads("/path/to/other.cu"));
    }
}
