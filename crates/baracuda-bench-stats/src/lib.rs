//! Driver-free pieces of board #162's kernel performance baseline design
//! (`docs/design/2026-10-09-kernel-perf-baseline-design.md`): timing
//! statistics, machine-state record types, and the baseline key. No CUDA
//! dependency — this crate is CI-executed on a GPU-less runner, unlike
//! `baracuda-kernels-bench`'s own device tests. The capture logic that
//! actually RUNS `nvidia-smi`/`powercfg` and produces a [`MachineState`]
//! lives in `baracuda-kernels-bench`, which has the device context; this
//! crate owns the data shape and the pure parsing of their text output.

#![deny(missing_docs)]

use std::cmp::Ordering;

/// Best/worst/mean/median/p99/spread over one kernel's timing samples, in
/// nanoseconds. CireSnave (board 162, 2026-10-09): best/worst/average alone
/// hide bimodality, so every field here is kept, not reduced to one number.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimingStats {
    /// Fastest sample (best case).
    pub min_ns: f64,
    /// Slowest sample (worst case).
    pub max_ns: f64,
    /// Arithmetic mean over all samples.
    pub mean_ns: f64,
    /// Median (50th percentile); the average of the two middle samples on
    /// an even-sized input.
    pub median_ns: f64,
    /// 99th percentile, nearest-rank method: `rank = ceil(0.99 * n)`,
    /// clamped to `[1, n]`, 1-indexed into the ascending-sorted samples.
    pub p99_ns: f64,
    /// Population standard deviation (divides by `n`, not `n - 1` — this
    /// describes the observed sample itself, not an inferred population).
    pub stddev_ns: f64,
    /// Number of samples the other fields were computed over.
    pub sample_count: u32,
}

/// Why [`compute_stats`] could not produce a [`TimingStats`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatsError {
    /// The input slice had zero samples — there is no honest "fastest" or
    /// "mean" over nothing, so this is an error, not a zeroed-out struct.
    EmptySamples,
}

impl std::fmt::Display for StatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySamples => write!(f, "compute_stats: no samples to summarize"),
        }
    }
}

impl std::error::Error for StatsError {}

/// Compute [`TimingStats`] over `samples_ns` (nanoseconds, any order).
///
/// # Errors
///
/// [`StatsError::EmptySamples`] if `samples_ns` is empty.
pub fn compute_stats(samples_ns: &[f64]) -> Result<TimingStats, StatsError> {
    if samples_ns.is_empty() {
        return Err(StatsError::EmptySamples);
    }
    let mut sorted: Vec<f64> = samples_ns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = sorted.len();

    let min_ns = sorted[0];
    let max_ns = sorted[n - 1];
    let mean_ns = sorted.iter().sum::<f64>() / n as f64;
    let median_ns = if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let variance = sorted.iter().map(|x| (x - mean_ns).powi(2)).sum::<f64>() / n as f64;
    let stddev_ns = variance.sqrt();
    let p99_rank = ((0.99 * n as f64).ceil() as usize).clamp(1, n);
    let p99_ns = sorted[p99_rank - 1];

    Ok(TimingStats {
        min_ns,
        max_ns,
        mean_ns,
        median_ns,
        p99_ns,
        stddev_ns,
        sample_count: n as u32,
    })
}

/// One `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` line,
/// parsed. Each numeric field is `None` when `nvidia-smi` reports it as
/// unsupported (`[N/A]` / `N/A`) — an absent reading, not a zero.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NvidiaSmiSample {
    /// GPU name, e.g. `"NVIDIA GeForce RTX 4070 Laptop GPU"`.
    pub device_name: String,
    /// Driver version string, e.g. `"581.08"`.
    pub driver_version: String,
    /// SM (core) clock, MHz.
    pub sm_clock_mhz: Option<u32>,
    /// Memory clock, MHz.
    pub mem_clock_mhz: Option<u32>,
    /// Power limit, watts.
    pub power_limit_w: Option<f64>,
    /// GPU temperature, degrees Celsius.
    pub temperature_c: Option<u32>,
}

/// Parse one CSV line produced by `nvidia-smi
/// --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu
/// --format=csv,noheader,nounits` (field order fixed to that query).
///
/// # Errors
///
/// Returns `Err` with a short description if the line doesn't split into
/// exactly 6 comma-separated fields.
pub fn parse_nvidia_smi_query_csv(line: &str) -> Result<NvidiaSmiSample, String> {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    let [name, driver, sm_clk, mem_clk, power, temp] = fields.as_slice() else {
        return Err(format!(
            "expected 6 CSV fields (name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu), got {}: {line:?}",
            fields.len()
        ));
    };
    Ok(NvidiaSmiSample {
        device_name: (*name).to_string(),
        driver_version: (*driver).to_string(),
        sm_clock_mhz: parse_na(sm_clk),
        mem_clock_mhz: parse_na(mem_clk),
        power_limit_w: parse_na(power),
        temperature_c: parse_na(temp),
    })
}

/// Parse an `nvidia-smi` numeric field, treating `N/A` / `[N/A]` (its
/// unsupported-query marker) as absent rather than a parse failure.
fn parse_na<T: std::str::FromStr>(field: &str) -> Option<T> {
    let trimmed = field.trim();
    if trimmed.eq_ignore_ascii_case("n/a") || trimmed.eq_ignore_ascii_case("[n/a]") {
        return None;
    }
    trimmed.parse().ok()
}

/// Parse the active power-plan name out of `powercfg /getactivescheme`'s
/// output (Windows), e.g. `"Power Scheme GUID: <guid>  (Balanced)"` ->
/// `Some("Balanced")`. `None` if the expected `(...)` suffix isn't found —
/// an unrecognized format is reported as absent, not guessed at.
#[must_use]
pub fn parse_windows_power_plan(output: &str) -> Option<String> {
    let open = output.rfind('(')?;
    let close = output.rfind(')')?;
    if close <= open {
        return None;
    }
    Some(output[open + 1..close].to_string())
}

/// The machine-state block recorded alongside every captured [`TimingStats`]
/// — CireSnave (board 162): "test both on a noisy PC and with as many
/// things paused or shut down as possible for a clean run" means a noisy and
/// a quiet run must be DISTINGUISHABLE after the fact from what they
/// recorded, not from a trust-me label. Every field is `Option` (except the
/// identity fields, which the capturer always knows) because an environment
/// that can't answer a query records that absence honestly rather than
/// guessing.
///
/// This struct is pure data — produced by `baracuda-kernels-bench`'s device-
/// context capture code, which calls [`parse_nvidia_smi_query_csv`] and
/// [`parse_windows_power_plan`] on real command output. It carries no
/// capture logic itself, so it has no device dependency and is constructible
/// (and testable) anywhere, including this driver-free crate.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MachineState {
    /// Stable, human-assigned id for the reporting box (e.g.
    /// `"cires-laptop-4070"`) — NOT auto-derived from hardware/BIOS strings,
    /// which aren't stable across an OS reinstall.
    pub machine_id: String,
    /// Full GPU name, e.g. `"NVIDIA GeForce RTX 4070 Laptop GPU"`.
    pub device_name: String,
    /// `(major, minor)` compute capability, e.g. `[8, 9]` for sm_89.
    pub device_capability: [u32; 2],
    /// Driver version string.
    pub driver_version: String,
    /// CUDA toolkit version actually used to build the kernel under test,
    /// e.g. `"13.0"` — distinct from the driver's max-supported version.
    pub cuda_toolkit_version: String,
    /// SM (core) clock at capture time, MHz.
    pub sm_clock_mhz: Option<u32>,
    /// Memory clock at capture time, MHz.
    pub mem_clock_mhz: Option<u32>,
    /// Power limit at capture time, watts.
    pub power_limit_w: Option<f64>,
    /// GPU temperature at capture time, degrees Celsius.
    pub temperature_c: Option<u32>,
    /// Count of other processes holding a CUDA context on this GPU at
    /// capture time (may include the capturing process itself).
    pub other_gpu_process_count: Option<u32>,
    /// Active OS power plan/scheme name (Windows: `powercfg
    /// /getactivescheme`'s parenthesized name, e.g. `"Balanced"`). `None`
    /// on a platform without this concept or when it couldn't be read.
    pub power_plan: Option<String>,
    /// Capture time, Unix seconds (UTC) — matches this crate family's
    /// existing convention (`HwStamp::captured_unix_s`,
    /// `baracuda-kernels-bench/src/lib.rs`'s `current_hwstamp`); avoids
    /// needing a date-formatting dependency for what SystemTime already
    /// gives for free.
    pub captured_unix_s: u64,
}

/// A kernel performance baseline's identity: which kernel, at which dtype
/// and shape class, on which target architecture, **on which machine**.
///
/// The `machine_id` component is the board-162 extension over the existing
/// `PytorchBaselineEntry` shape (`crates/baracuda-kernels-bench/src/lib.rs`):
/// CireSnave's P40-vs-4070-vs-4060 point means architecture token alone is
/// not enough once more than one box reports a baseline for the same
/// `cuda:smNN` target.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BaselineKey {
    /// Kernel/op family, e.g. `"gemm"`.
    pub kernel: String,
    /// Element dtype label, e.g. `"f16"`.
    pub dtype: String,
    /// Shape-class descriptor, e.g. `"M128_N4096_K4096"`.
    pub shape_class: String,
    /// Target architecture token, e.g. `"cuda:sm89"`.
    pub target_arch: String,
    /// Which machine this baseline was captured on (see [`MachineState::machine_id`]).
    pub machine_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_ascending_samples_report_known_stats() {
        let samples: Vec<f64> = (1..=10).map(f64::from).collect();
        let stats = compute_stats(&samples).expect("non-empty samples");
        assert_eq!(stats.sample_count, 10);
        assert_eq!(stats.min_ns, 1.0);
        assert_eq!(stats.max_ns, 10.0);
        assert_eq!(stats.mean_ns, 5.5);
        assert_eq!(stats.median_ns, 5.5); // even count: average of 5 and 6
    }

    #[test]
    fn empty_samples_is_an_error_not_a_zeroed_struct() {
        assert_eq!(compute_stats(&[]), Err(StatsError::EmptySamples));
    }

    #[test]
    fn single_sample_has_zero_spread() {
        let stats = compute_stats(&[42.0]).expect("non-empty");
        assert_eq!(stats.sample_count, 1);
        assert_eq!(stats.min_ns, 42.0);
        assert_eq!(stats.max_ns, 42.0);
        assert_eq!(stats.mean_ns, 42.0);
        assert_eq!(stats.median_ns, 42.0);
        assert_eq!(stats.p99_ns, 42.0);
        assert_eq!(stats.stddev_ns, 0.0);
    }

    #[test]
    fn p99_of_one_to_one_hundred_is_the_99th_smallest() {
        let samples: Vec<f64> = (1..=100).map(f64::from).collect();
        let stats = compute_stats(&samples).expect("non-empty");
        assert_eq!(stats.p99_ns, 99.0);
    }

    #[test]
    fn stddev_matches_a_hand_computed_population_value() {
        // 1..=10: mean 5.5, population variance 8.25, stddev sqrt(8.25).
        let samples: Vec<f64> = (1..=10).map(f64::from).collect();
        let stats = compute_stats(&samples).expect("non-empty");
        assert!((stats.stddev_ns - 8.25_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn unsorted_input_gives_the_same_stats_as_sorted() {
        let sorted: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shuffled: Vec<f64> = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        assert_eq!(
            compute_stats(&sorted).unwrap(),
            compute_stats(&shuffled).unwrap()
        );
    }

    #[test]
    fn parses_a_real_nvidia_smi_csv_line() {
        let line = "NVIDIA GeForce RTX 4070 Laptop GPU, 581.08, 1500, 8001, 80.00, 55";
        let sample = parse_nvidia_smi_query_csv(line).expect("well-formed line");
        assert_eq!(sample.device_name, "NVIDIA GeForce RTX 4070 Laptop GPU");
        assert_eq!(sample.driver_version, "581.08");
        assert_eq!(sample.sm_clock_mhz, Some(1500));
        assert_eq!(sample.mem_clock_mhz, Some(8001));
        assert_eq!(sample.power_limit_w, Some(80.00));
        assert_eq!(sample.temperature_c, Some(55));
    }

    #[test]
    fn na_fields_become_none_not_a_parse_error() {
        let line = "NVIDIA GeForce RTX 4070 Laptop GPU, 581.08, [N/A], 8001, [N/A], 55";
        let sample = parse_nvidia_smi_query_csv(line).expect("still well-formed");
        assert_eq!(sample.sm_clock_mhz, None);
        assert_eq!(sample.power_limit_w, None);
        assert_eq!(sample.mem_clock_mhz, Some(8001));
    }

    #[test]
    fn too_few_csv_fields_is_a_parse_error() {
        let line = "NVIDIA GeForce RTX 4070 Laptop GPU, 581.08";
        assert!(parse_nvidia_smi_query_csv(line).is_err());
    }

    #[test]
    fn parses_the_active_windows_power_plan_name() {
        let output = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\r\n";
        assert_eq!(
            parse_windows_power_plan(output),
            Some("Balanced".to_string())
        );
    }

    #[test]
    fn power_plan_output_with_no_parens_is_none() {
        assert_eq!(parse_windows_power_plan("garbage, no parens here"), None);
    }

    #[test]
    fn machine_state_round_trips_through_json() {
        let state = MachineState {
            machine_id: "cires-laptop-4070".to_string(),
            device_name: "NVIDIA GeForce RTX 4070 Laptop GPU".to_string(),
            device_capability: [8, 9],
            driver_version: "581.08".to_string(),
            cuda_toolkit_version: "13.0".to_string(),
            sm_clock_mhz: Some(1500),
            mem_clock_mhz: Some(8001),
            power_limit_w: Some(80.0),
            temperature_c: Some(55),
            other_gpu_process_count: Some(0),
            power_plan: Some("Balanced".to_string()),
            captured_unix_s: 1_760_000_000,
        };
        let json = serde_json::to_string(&state).expect("serialize");
        let back: MachineState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(state, back);
    }

    #[test]
    fn baseline_keys_with_different_machine_ids_are_distinct() {
        let base = BaselineKey {
            kernel: "gemm".to_string(),
            dtype: "f16".to_string(),
            shape_class: "M128_N4096_K4096".to_string(),
            target_arch: "cuda:sm89".to_string(),
            machine_id: "cires-laptop-4070".to_string(),
        };
        let other_machine = BaselineKey {
            machine_id: "cires-desktop-4060".to_string(),
            ..base.clone()
        };
        assert_ne!(base, other_machine);

        let mut seen = std::collections::HashSet::new();
        seen.insert(base.clone());
        assert!(!seen.contains(&other_machine));
        seen.insert(other_machine.clone());
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn baseline_keys_with_same_fields_are_equal_and_hash_equal() {
        let a = BaselineKey {
            kernel: "gemm".to_string(),
            dtype: "f16".to_string(),
            shape_class: "M128_N4096_K4096".to_string(),
            target_arch: "cuda:sm89".to_string(),
            machine_id: "cires-laptop-4070".to_string(),
        };
        let b = a.clone();
        assert_eq!(a, b);
        let mut seen = std::collections::HashSet::new();
        seen.insert(a);
        assert!(seen.contains(&b));
    }
}
