# sm_61 NVRTC toolchain selection (board #106, B1 (a)+(b))

Empirical findings, not static analysis — every claim below is a real NVRTC
compile attempt run on this box, not inferred from `nvcc --list-gpu-arch` or
changelog text alone (though that cross-check is included too).

## Toolkit-per-architecture matrix (measured)

Two CUDA toolkits are installed on this box: 12.9.2 and 13.3 (the PATH
default). A minimal `extern "C" __global__ void k(int *x) { x[0] = 1; }`
kernel was compiled via `baracuda-nvrtc::Program` with each toolkit's NVRTC
loaded through the new `BARACUDA_NVRTC_PATH` override (see below), fresh
process per attempt, minimal `PATH` (no ambient CUDA bin dir):

| target       | via 12.9.2's NVRTC | via 13.3's NVRTC |
|--------------|---------------------|-------------------|
| `sm_61`      | COMPILE OK          | FAIL: `NVRTC_ERROR_INVALID_OPTION` — "invalid value for --gpu-architecture" |
| `sm_89`      | COMPILE OK          | COMPILE OK        |

Matches `nvcc --list-gpu-arch` for both toolkits: 12.9.2 lists
`compute_61`, 13.3 does not (13.x dropped Pascal/Maxwell/Volta). So:

- **sm_61 → must use the 12.9.2 (or any pre-13.x) toolkit's NVRTC.**
- **sm_80 and newer → either toolkit's NVRTC works**; no reason to force a
  specific one.

## A real gotcha this uncovered: pointing at the right DLL was not enough

The first attempt — `BARACUDA_NVRTC_PATH` pointed straight at
`nvrtc64_120_0.dll` from the 12.9.2 install, nothing else changed — failed
on *every* target, including `sm_89`, with:

```
NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
nvrtc: error: failed to open nvrtc-builtins64_129.dll.
  Make sure that nvrtc-builtins64_129.dll is installed correctly.
```

...despite `nvrtc-builtins64_129.dll` sitting in the exact same directory as
`nvrtc64_120_0.dll`. Root cause (confirmed, not assumed): NVRTC resolves its
own builtins DLL with a *runtime* `LoadLibrary` call made at actual compile
time, not as a static import of `nvrtc64_*.dll` itself — so loading
`nvrtc64_120_0.dll` by absolute path puts `nvrtc64` itself in memory, but
does nothing for a lookup `nvrtc64` performs on its own, later, by bare
filename.

Tried `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` first (the flag that's supposed to
make a loaded module's own directory searchable for *its* dependencies) —
verified it does **not** fix this, consistent with that flag governing a
module's static import resolution, not a dependent module's own later
runtime `LoadLibrary` calls. The fix that actually worked, and is what
`Library::open_at` now does on Windows: prepend the loaded DLL's parent
directory to the process `PATH`, once, only after that DLL has already
loaded successfully (never for a failed/speculative candidate — several
other `-sys` crates' search loops call `open_at` once per guess, and a
failed guess's directory has no business ending up on `PATH`).

## What's in `BARACUDA_NVRTC_PATH`

`baracuda-nvrtc-sys::nvrtc()` now checks `BARACUDA_NVRTC_PATH` before its
hardcoded candidate list (`baracuda_core::Library::open_with_env_override`).
If set, that exact path is tried exclusively — no fallback to the hardcoded
candidates — and a set-but-unusable override surfaces as
`LoaderError::EnvOverrideUnusable` rather than silently resolving to
whichever hardcoded candidate the box happens to have. Tests:
`baracuda-core::loader::tests::env_override_*` (generic helper, mutation-
checked: reverting the "fail loudly" behavior to a silent fallback breaks
`env_override_set_but_unusable_fails_loudly_without_falling_back`) and
`baracuda-nvrtc-sys::tests::env_override_wiring` (the NVRTC-specific wiring).

## Scope note

This covers B1 parts (a) (`BARACUDA_NVRTC_PATH`) and (b) (sm_61 toolchain
selection) only. Part (c) — honoring `KernelPlan::half_arith()`'s `ViaF32`
decision — needs Unpopped's `half_arith()`, merged on Unpopped's `main` but
not yet published to crates.io (board item 136); per the PM's 2026-10-07
ruling it is NOT git-rev-pinned (CireSnave: "I don't even like depending on
things on GitHub if I can avoid it") and instead waits on that publish.
