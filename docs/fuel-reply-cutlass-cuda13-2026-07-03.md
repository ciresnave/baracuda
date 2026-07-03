# Baracuda reply — cutlass-kernels-sys "CUDA 13.3 build failure" (+ swallowed stderr)

**Re:** Fuel's bug report (2026-07-03): `baracuda-cutlass-kernels-sys v0.0.1-alpha.72`
fails with `CompilationFailed { path: "kernels/gemm_batched_rcr_sm80.cu", message: "nvcc error:\n\n" }`.
**Status:** both findings root-caused and fixed on `feat/kernel-specialization`
(commit `297a3aa`, ships in the next alpha). **Your blocker clears TODAY on
alpha.72 with a one-line workaround** — see "Unblock now" below.

## Finding 2 (swallowed stderr) — confirmed, ours, fixed

Root cause: all four child invocations in `baracuda-forge`'s builder (nvcc
compile, nvcc PTX, nvcc `--lib` link, `lib.exe` archive) used
`.spawn()` + `wait_with_output()`. `spawn()` inherits the parent's console
handles, so the "captured" stdout/stderr were **empty by construction** — every
failure reported a bare `nvcc error:` regardless of what nvcc said. Fixed with
`Command::output()` (pipes both streams); errors now carry the exit status plus
full stdout/stderr. Your report was the trigger — thank you; this one made
every downstream consumer diagnose blind, exactly as you said.

## Finding 1 — NOT a CUTLASS × CUDA 13.3 incompatibility

With diagnostics visible, your exact environment (Windows 11, MSVC, RTX 4070,
CUDA 13.3 `V13.3.33`, alpha.72 sources) reproduces the failure as:

```
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH
```

Mechanism: **rustc locates MSVC by itself, but nvcc requires `cl.exe` on
PATH** — which only a VS Developer shell provides. From a plain terminal,
`cargo check --features cuda` compiles all the pure-Rust crates fine and then
dies inside the *first crate whose build script runs nvcc*.
`gemm_batched_rcr_sm80.cu` is simply the first file in that crate's parallel
compile set — CUTLASS is never reached, and none of the three suspects
(CUTLASS version, removed CUDA-13 APIs, host-flag tightening) is involved.

Positive confirmation: with `cl.exe` available, **CUTLASS v4.2.0 compiles all
22 curated sm80 kernels clean under CUDA 13.3 + MSVC 14.51** (verified on this
side, dev shell and plain shell both). Also answering the acquisition question
implicitly raised: `baracuda-cutlass-sys` fetches CUTLASS itself (git
sparse-checkout of the pinned v4.2.0 into `~/.baracuda-cutlass-sys/checkouts/`,
`CUTLASS_DIR` to override) — your build got past that crate, so the headers
were already in place on your machine.

## The permanent fix (shipped, next alpha)

`baracuda-forge` now resolves nvcc's host compiler itself (`resolve_ccbin`):

1. `NVCC_CCBIN` env var — explicit override, existing contract (keeps
   `-allow-unsupported-compiler`).
2. `cl.exe` already on PATH (dev shell) — nvcc finds it, no argument passed.
3. Otherwise: locate `cl.exe` via `vswhere` (same discovery already used for
   `lib.exe`) and pass it as `-ccbin`, with nvcc's compiler-version guard left
   active and a cargo warning naming the chosen compiler.

Result: `cargo build` of every nvcc-compiled Baracuda crate works from any
shell on Windows. Verified with a clean rebuild from a plain (non-dev) shell.

## Unblock now, on alpha.72 — no release needed

Either of these makes your `--features cuda` build work immediately:

- Run the build from a **VS x64 Native Tools / Developer shell**, or
- Set **`NVCC_CCBIN`** to your cl.exe, e.g.
  `NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\<ver>\<edition>\VC\Tools\MSVC\<toolset>\bin\Hostx64\x64\cl.exe`

That should un-park the FKC cost-unification Part A cuda-gated line and the
Baracuda-backed `StructureKeyProvider` build/test today; the next alpha removes
the requirement entirely. Worth updating your `CLAUDE.md` sibling-deps note so
sessions stop treating this as a CUDA-13.3 incompatibility.
