//! Board #162 — machine-state capture
//! (`docs/design/2026-10-09-kernel-perf-baseline-design.md`).
//!
//! This module is the device/host-dependent half: it shells out to
//! `nvidia-smi` (and, on Windows, `powercfg`) and hands the raw text to
//! `baracuda-bench-stats`'s pure parsers. It has the same CI status as this
//! crate's other on-device tests (`tests/*_ondevice.rs`): CI only
//! TYPE-CHECKS it (`cargo check --tests`, see `.github/workflows/ci.yml`'s
//! `arch-gated-check` job) because running it needs a real GPU + driver;
//! it is run for real on the maintainer's box. The portable parsing logic
//! these functions call IS run for real in CI, in `baracuda-bench-stats`.

use baracuda_bench_stats::{
    MachineState, NvidiaSmiSample, parse_nvidia_smi_query_csv, parse_windows_power_plan,
};
use std::process::Command;

/// Run `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` for GPU 0
/// and parse the result.
///
/// # Errors
///
/// A `String` describing why: `nvidia-smi` isn't on `PATH`, exited non-zero,
/// or produced output [`parse_nvidia_smi_query_csv`] couldn't parse.
pub fn capture_nvidia_smi_sample() -> Result<NvidiaSmiSample, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("failed to run nvidia-smi: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout
        .lines()
        .next()
        .ok_or_else(|| "nvidia-smi produced no output".to_string())?;
    parse_nvidia_smi_query_csv(first_line)
}

/// Count processes currently holding a CUDA context on GPU 0, via
/// `nvidia-smi --query-compute-apps=pid --format=csv,noheader`.
///
/// `None` if `nvidia-smi` can't be run. **May include the capturing process
/// itself** if it already holds a context when this runs — not subtracted
/// out, since a caller that hasn't yet created its context and one that has
/// need different corrections, and guessing which is worse than reporting
/// the raw count honestly.
#[must_use]
pub fn capture_other_gpu_process_count() -> Option<u32> {
    let output = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(stdout.lines().filter(|l| !l.trim().is_empty()).count() as u32)
}

/// Read the active Windows power plan via `powercfg /getactivescheme`.
/// `None` on a non-Windows OS, or if the command fails.
#[must_use]
pub fn capture_power_plan() -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("powercfg")
            .arg("/getactivescheme")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        parse_windows_power_plan(&String::from_utf8_lossy(&output.stdout))
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

/// Capture a full [`MachineState`] for `machine_id`.
///
/// `device_name`, `device_capability`, and `cuda_toolkit_version` come from
/// the caller's already-open CUDA context (e.g. `Device::name`,
/// `Device::compute_capability`, `version()`) rather than being re-queried
/// here, so this function needs no `Context`/`Device` of its own — every
/// other field is best-effort (`nvidia-smi`/`powercfg`), recorded as `None`
/// on failure rather than guessed at.
#[must_use]
pub fn capture_machine_state(
    machine_id: &str,
    device_name: &str,
    device_capability: [u32; 2],
    cuda_toolkit_version: &str,
) -> MachineState {
    let smi = capture_nvidia_smi_sample().ok();
    let captured_unix_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    MachineState {
        machine_id: machine_id.to_string(),
        device_name: device_name.to_string(),
        device_capability,
        driver_version: smi
            .as_ref()
            .map(|s| s.driver_version.clone())
            .unwrap_or_default(),
        cuda_toolkit_version: cuda_toolkit_version.to_string(),
        sm_clock_mhz: smi.as_ref().and_then(|s| s.sm_clock_mhz),
        mem_clock_mhz: smi.as_ref().and_then(|s| s.mem_clock_mhz),
        power_limit_w: smi.as_ref().and_then(|s| s.power_limit_w),
        temperature_c: smi.as_ref().and_then(|s| s.temperature_c),
        other_gpu_process_count: capture_other_gpu_process_count(),
        power_plan: capture_power_plan(),
        captured_unix_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ⚠️ These tests need a real NVIDIA driver (`nvidia-smi` on `PATH`), and
    // the power-plan test needs Windows. Same CI status as this crate's
    // other `_ondevice` tests: CI only `cargo check --tests`s this module
    // (see the module doc); these run for real on the maintainer's box.

    #[test]
    fn captures_a_real_nvidia_smi_sample() {
        let sample = capture_nvidia_smi_sample().expect("nvidia-smi must be on PATH on this box");
        assert!(!sample.device_name.is_empty());
        assert!(!sample.driver_version.is_empty());
    }

    #[test]
    fn captures_other_gpu_process_count_without_erroring() {
        // Can't assert a specific count (depends on what else is running),
        // but it must at least succeed on a box with nvidia-smi.
        assert!(capture_other_gpu_process_count().is_some());
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn captures_a_real_windows_power_plan() {
        let plan = capture_power_plan().expect("powercfg must work on Windows");
        assert!(!plan.is_empty());
    }

    #[test]
    fn capture_machine_state_fills_in_identity_fields_even_if_smi_fails() {
        let state = capture_machine_state("test-machine", "Test Device", [8, 9], "13.0");
        assert_eq!(state.machine_id, "test-machine");
        assert_eq!(state.device_name, "Test Device");
        assert_eq!(state.device_capability, [8, 9]);
        assert_eq!(state.cuda_toolkit_version, "13.0");
        assert!(state.captured_unix_s > 0);
    }
}
