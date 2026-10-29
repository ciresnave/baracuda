//! Workspace task runner for baracuda.
//!
//! Subcommands:
//!
//! - `xtask regen-all`             Regenerate every committed `bindings/cuda_*.rs`.
//! - `xtask regen <lib>`           Regenerate a single `-sys` crate's bindings.
//! - `xtask build-kernels`         Recompile shipped `.ptx` fixtures via `nvcc` (planned).
//! - `xtask test-gpu [args]`       Run on-device tests under the `gpu-run` lock.
//!
//! All commands require a CUDA Toolkit install discoverable via `CUDA_PATH` /
//! `CUDA_HOME` / OS defaults; see [`baracuda_build::detect_cuda`].

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("-h" | "--help" | "help") => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some("-V" | "--version") => {
            println!("xtask {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
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
    println!("  xtask test-gpu [cargo args]  run on-device tests under the gpu-run lock");
    println!(
        "                               (compile UNLOCKED, device-run LOCKED); prefer `-p <crate>`"
    );
}

/// `xtask test-gpu [cargo test args] [-- <test-binary args>]`
///
/// Runs on-device tests (`device-tests` feature / legacy `#[ignore]`) under the machine-wide `Global\gpu-run`
/// mutex so lock acquisition is STRUCTURAL, not a convention someone must
/// remember (the failure the postmortem indicts). Two phases:
/// 1. compile UNLOCKED — `cargo test <args> --no-run` (the lock wrapper
///    explicitly forbids serializing compile-only work);
/// 2. device-run LOCKED — `scripts/gpu-run.ps1 -Project baracuda -- cargo test
///    <args> -- --include-ignored <extra>`, which holds the mutex across the run.
///
/// Writes a machine-readable run record (pass/fail/skip counts, skips by reason,
/// commit, host, exit code) to `$BARACUDA_RESULTS_FILE` or
/// `target/device-test-results.json`. A run that passed zero tests exits non-zero.
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

    // `--include-ignored` runs the `#[ignore]`d tests of crates not yet migrated
    // AND the ordinary (feature-gated `device-tests`) tests of migrated ones.
    let test_args = |cmd: &mut std::process::Command| {
        cmd.arg("test")
            .args(cargo_args)
            .arg("--")
            .arg("--include-ignored")
            .args(test_bin_args);
    };

    let started = unix_now();
    // Off-Windows (the 4070 box is Windows-only) there is no lock to take —
    // run the device tests directly, with a warning, rather than fail.
    let (code, lines) = if !cfg!(windows) {
        eprintln!(
            "xtask test-gpu: the gpu-run lock is Windows-only; running device tests WITHOUT it."
        );
        let mut cmd = std::process::Command::new("cargo");
        test_args(&mut cmd);
        set_env(&mut cmd);
        match run_tee(&mut cmd) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("xtask test-gpu: could not spawn `cargo test`: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
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
        let mut result = None;
        for (i, shell) in ["pwsh", "powershell"].iter().enumerate() {
            let mut cmd = std::process::Command::new(shell);
            cmd.arg("-File")
                .arg(&script)
                .arg("-Project")
                .arg("baracuda")
                .arg("--")
                .arg("cargo");
            test_args(&mut cmd);
            set_env(&mut cmd);
            match run_tee(&mut cmd) {
                Ok(r) => {
                    result = Some(r);
                    break;
                }
                // Shell not on PATH: try the next candidate; any other error is real.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && i == 0 => continue,
                Err(e) => {
                    eprintln!("xtask test-gpu: could not spawn {shell}: {e}");
                    return ExitCode::from(1);
                }
            }
        }
        match result {
            Some(r) => r,
            None => {
                eprintln!("xtask test-gpu: neither `pwsh` nor `powershell` found on PATH");
                return ExitCode::from(1);
            }
        }
    };

    let skips = tally_skips(&skip_log);
    let summary = RunSummary::from_output(&lines);
    let mut exit = code;
    // A run that executed nothing asserts nothing: never report it as a pass.
    if exit == 0 && summary.passed == 0 {
        eprintln!(
            "xtask test-gpu: exit 0 but 0 tests passed — a run that ran nothing is not a pass \
             (wrong -p/--test filter, or a missing `--features device-tests`?)"
        );
        exit = 1;
    }
    write_results(
        &results_path(),
        &RunRecord {
            started_unix: started,
            finished_unix: unix_now(),
            args: cargo_args.join(" "),
            exit_code: exit,
            summary,
            skips,
        },
    );
    ExitCode::from(exit as u8)
}

/// Where the machine-readable run record goes: `$BARACUDA_RESULTS_FILE`, else
/// `target/device-test-results.json` under the workspace root.
fn results_path() -> PathBuf {
    if let Some(p) = std::env::var_os("BARACUDA_RESULTS_FILE") {
        return PathBuf::from(p);
    }
    find_workspace_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("target")
        .join("device-test-results.json")
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Run `cmd`, echoing its stdout as it arrives (so a live run looks the same) while
/// keeping every line for the results record. Returns the exit code (1 if killed
/// by a signal) and the captured stdout lines. stderr is inherited.
fn run_tee(cmd: &mut std::process::Command) -> std::io::Result<(i32, Vec<String>)> {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    let mut child = cmd.stdout(Stdio::piped()).spawn()?;
    let mut lines = Vec::new();
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            println!("{line}");
            lines.push(line);
        }
    }
    let status = child.wait()?;
    Ok((status.code().unwrap_or(1), lines))
}

/// Totals summed over every libtest `test result:` line in a run.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
struct RunSummary {
    passed: u64,
    failed: u64,
    ignored: u64,
}

impl RunSummary {
    /// Parse `test result: ok. 3 passed; 0 failed; 1 ignored; ...` lines.
    fn from_output(lines: &[String]) -> Self {
        let mut s = Self::default();
        for l in lines {
            let Some(rest) = l.trim_start().strip_prefix("test result:") else {
                continue;
            };
            for part in rest.split(';') {
                let part = part
                    .trim()
                    .trim_start_matches("ok.")
                    .trim_start_matches("FAILED.");
                let mut it = part.trim().splitn(2, ' ');
                let (Some(n), Some(label)) = (it.next(), it.next()) else {
                    continue;
                };
                let Ok(n) = n.parse::<u64>() else { continue };
                match label.trim() {
                    "passed" => s.passed += n,
                    "failed" => s.failed += n,
                    "ignored" => s.ignored += n,
                    _ => {}
                }
            }
        }
        s
    }
}

struct RunRecord {
    started_unix: u64,
    finished_unix: u64,
    args: String,
    exit_code: i32,
    summary: RunSummary,
    /// `(test, reason)` for every `require_optional!` skip.
    skips: Vec<(String, String)>,
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

impl RunRecord {
    /// Hand-rolled JSON (xtask carries no serde): small, flat, stable keys.
    fn to_json(&self, commit: &str, host: &str) -> String {
        let mut by_reason: std::collections::BTreeMap<&str, u64> = Default::default();
        for (_, reason) in &self.skips {
            *by_reason.entry(reason.as_str()).or_default() += 1;
        }
        let reasons = by_reason
            .iter()
            .map(|(r, n)| format!("{}: {n}", json_str(r)))
            .collect::<Vec<_>>()
            .join(", ");
        let skipped = self
            .skips
            .iter()
            .map(|(t, r)| format!("{{\"test\": {}, \"reason\": {}}}", json_str(t), json_str(r)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{{\n  \"schema\": \"baracuda-device-test-results-v1\",\n  \"status\": {},\n  \
             \"exit_code\": {},\n  \"started_unix\": {},\n  \"finished_unix\": {},\n  \
             \"commit\": {},\n  \"host\": {},\n  \"args\": {},\n  \"passed\": {},\n  \
             \"failed\": {},\n  \"ignored\": {},\n  \"skipped_declared\": {},\n  \
             \"skips_by_reason\": {{{}}},\n  \"skips\": [{}]\n}}\n",
            json_str(if self.exit_code == 0 { "pass" } else { "fail" }),
            self.exit_code,
            self.started_unix,
            self.finished_unix,
            json_str(commit),
            json_str(host),
            json_str(&self.args),
            self.summary.passed,
            self.summary.failed,
            self.summary.ignored,
            self.skips.len(),
            reasons,
            skipped,
        )
    }
}

fn write_results(path: &Path, rec: &RunRecord) {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_owned());
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::write(path, rec.to_json(&commit, &host)) {
        Ok(()) => eprintln!(
            "xtask test-gpu: results -> {} ({} passed, {} failed, {} declared skips)",
            path.display(),
            rec.summary.passed,
            rec.summary.failed,
            rec.skips.len()
        ),
        Err(e) => eprintln!("xtask test-gpu: could not write {}: {e}", path.display()),
    }
    if let Some(summary_path) = std::env::var_os("GITHUB_STEP_SUMMARY")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(summary_path)
    {
        use std::io::Write as _;
        let _ = writeln!(
            f,
            "### Device tests: {} passed, {} failed, {} declared skips (exit {})",
            rec.summary.passed,
            rec.summary.failed,
            rec.skips.len(),
            rec.exit_code
        );
    }
}

/// Read the declared-skip log written by `require_optional!` (via
/// `BARACUDA_SKIP_LOG`), report what was skipped — so an absent optional
/// runtime on the box is STATED, never a silent `ok` — and return the
/// `(test, reason)` rows. A `require!` (critical) absence would already have
/// failed the run (BARACUDA_GPU_REQUIRED), so what lands here is the
/// optional-runtime skips.
fn tally_skips(skip_log: &Path) -> Vec<(String, String)> {
    let Ok(contents) = std::fs::read_to_string(skip_log) else {
        return Vec::new();
    };
    let rows: Vec<(String, String)> = contents
        .lines()
        .filter(|l| l.starts_with("SKIP-DECLARED"))
        .map(|row| {
            // "SKIP-DECLARED\t<test>\t<reason>"
            let mut cols = row.split('\t').skip(1);
            (
                cols.next().unwrap_or("<?>").to_owned(),
                cols.next().unwrap_or("<?>").to_owned(),
            )
        })
        .collect();
    if rows.is_empty() {
        return rows;
    }
    eprintln!(
        "xtask test-gpu: {} test(s) declared a resource skippable and skipped it on this box \
         (require_optional! — an optional runtime/hardware was absent here):",
        rows.len()
    );
    for (test, reason) in &rows {
        eprintln!("  - {test}: {reason}");
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(s: &str) -> String {
        s.to_owned()
    }

    #[test]
    fn summary_sums_libtest_result_lines() {
        let out = vec![
            l("running 3 tests"),
            l(
                "test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.1s",
            ),
            l(
                "test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s",
            ),
            l("test not_a_result_line ... ok"),
        ];
        assert_eq!(
            RunSummary::from_output(&out),
            RunSummary {
                passed: 5,
                failed: 1,
                ignored: 1
            }
        );
    }

    #[test]
    fn summary_of_empty_output_is_zero() {
        assert_eq!(RunSummary::from_output(&[]), RunSummary::default());
    }

    #[test]
    fn json_escapes_and_groups_skips_by_reason() {
        let rec = RunRecord {
            started_unix: 1,
            finished_unix: 2,
            args: "-p \"x\"".to_owned(),
            exit_code: 0,
            summary: RunSummary {
                passed: 4,
                failed: 0,
                ignored: 0,
            },
            skips: vec![
                (l("a"), l("no NVSHMEM")),
                (l("b"), l("no NVSHMEM")),
                (l("c"), l("no sparse")),
            ],
        };
        let j = rec.to_json("abc", "host");
        assert!(j.contains("\"status\": \"pass\""));
        assert!(j.contains("\"skipped_declared\": 3"));
        assert!(j.contains("\"no NVSHMEM\": 2"));
        assert!(j.contains("\"no sparse\": 1"));
        assert!(j.contains("\"args\": \"-p \\\"x\\\"\""));
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
