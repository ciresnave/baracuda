//! `kernelgen` CLI basics: `--help` / `--version` succeed, an unknown flag or
//! extra positional is a usage error, and none of them write anything.
//! Names the crate (`baracuda_cuda_emit`) so test-crate-locality sees it.

use std::path::PathBuf;
use std::process::Command;

fn run_in_empty_dir(tag: &str, args: &[&str]) -> (std::process::Output, usize) {
    let _ = std::any::type_name::<baracuda_cuda_emit::Cuda>();
    let dir: PathBuf =
        std::env::temp_dir().join(format!("kernelgen-cli-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_kernelgen"))
        .args(args)
        .current_dir(&dir)
        .output()
        .expect("spawn kernelgen");
    let n = std::fs::read_dir(&dir).unwrap().count();
    let _ = std::fs::remove_dir_all(&dir);
    (out, n)
}

#[test]
fn version_and_help_exit_zero_and_write_nothing() {
    for flag in ["--version", "-V", "--help", "-h"] {
        let (out, files) = run_in_empty_dir("ok", &[flag]);
        assert!(out.status.success(), "{flag}: {:?}", out.status);
        assert_eq!(files, 0, "{flag} wrote files");
        assert!(!out.stdout.is_empty(), "{flag} printed nothing");
    }
}

#[test]
fn unknown_flag_and_extra_args_exit_two_and_write_nothing() {
    for args in [&["--bogus"][..], &["-x"], &["a", "b", "c"]] {
        let (out, files) = run_in_empty_dir("bad", args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert_eq!(files, 0, "{args:?} wrote files");
        assert!(!out.stderr.is_empty(), "{args:?} printed no error");
    }
}

#[test]
fn positional_out_dir_still_generates() {
    let (out, files) = run_in_empty_dir("gen", &["outdir"]);
    assert!(out.status.success(), "{:?}", out.status);
    assert_eq!(files, 1, "expected exactly the out dir under cwd");
}
