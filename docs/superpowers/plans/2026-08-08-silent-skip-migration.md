# Plan: migrate the 40 silent-skip on-device tests to `require!` / `require_optional!`

## Context
The test-integrity audit (2026-08-08, chain-wide via the KISS Architect) found **43
`#[test]` fns** that clear their gate, *run*, and report `ok` having asserted nothing
— a runtime early-return on an absent device/toolkit/runtime, invisible to a
`filtered out` count. The 3 default-run sites + **5b** (the vacuous `relu_add`
"correctness guard") are fixed (commit `90573472`). The `require!` primitive is built
(commit `98992ce8`). This plan migrates the remaining **40 `#[ignore]`-gated sites**.

## The mechanism (already built — `baracuda-driver` `test-support` feature)
- **`require!(probe, "reason")`** — for CORE resources that MUST exist on the box (a
  bare CUDA device/context/stream). Absent + `BARACUDA_GPU_REQUIRED` (set by
  `cargo gpu-test`) → **PANIC** (absence-would-have-failed). Absent otherwise →
  declare a tool-discoverable `SKIP-DECLARED` and return.
- **`require_optional!(probe, "reason")`** — for OPTIONAL runtimes/hardware that may
  be absent even on the box. Always declare-and-skip; never fails loud.
- Both replace `if probe().is_err() { return }` / `let Some(x) = probe() else { return }`.
- `probe` is any `Result` or `Option` (a `Present` trait normalizes both).

## Per-crate work (~14 crates, 40 sites)
For each test-crate that has sites:
1. Add the feature in `[dev-dependencies]`:
   `baracuda-driver = { workspace = true, features = ["test-support"] }`
   (crates already dev-dep the driver; this adds the feature).
2. Replace each silent-skip idiom with `require!` (core GPU) or `require_optional!`
   (optional runtime), categorized from the audit's site list:
   - **CORE → `require!`**: bare-CUDA-device probes — `baracuda-driver` wave17/19/22/28
     smokes, `baracuda-runtime` wave3/5 smokes, `baracuda-cutensor` contract_gemm
     (device/context/stream/install must exist on the box).
   - **OPTIONAL → `require_optional!`**: `baracuda-nccl` (single_gpu / unique_id /
     ring_attention), `baracuda-nvshmem` smoke, `baracuda-megatron` &
     `baracuda-optim` multi-rank (≥2 GPUs), `baracuda-cufile`, `baracuda-tensorrt`,
     `baracuda-nvcomp`, `baracuda-cvcuda` — runtimes/extra hardware NOT guaranteed on
     the box.
   - **variant_gate** (`if stamp.arch != Sm89 { eprintln!("skipping"); return }`): the
     box IS sm89, so on the box this MUST run → `require!` on an arch-match probe
     (returns `Some(())` iff `stamp.arch == Sm89`), so a non-sm89 box declares-skip
     but the 4070 fails-loud if the arch probe somehow mismatches.

## The 4 Tier-B sites (no `assert!` — the op under test is `.unwrap()`-validated)
`baracuda-driver` wave5 `vmm_end_to_end`, wave28 `ctx_record_and_wait_event`,
`baracuda-nvshmem` `symmetric_heap_alloc_and_self_put`, `baracuda-cvcuda`
`resize_tensor_smoke`. After migrating the skip to `require!`/`require_optional!`,
the `.unwrap()`/`.expect()` on the device op IS the assertion (a failing op panics),
so no added `assert!` is needed — just confirm the op still runs on the box.

## Verification (per the evidence standard)
Each per-crate run: `cargo gpu-test -p <crate>` on the 4070, and STATE:
- the exact command + feature flags,
- the ran-count (`test result: ok. N passed; 0 failed; 0 filtered out`),
- the declared-skip tally that `cargo gpu-test` prints.
A test that now **FAILS** on the box — a mis-categorized core resource (`require!`
where the resource is genuinely absent), or a real regression the old silent skip
hid — is a finding; report it with the site + the cause.

## Out of scope (separate follow-ups)
- **5a "tolerant" no-assert tests** (`baracuda-driver` wave18/wave28,
  `baracuda-runtime` wave5) — a different class: no early-return, assert nothing even
  on the happy path. Sweep separately.
- **The tautological-assertion axis** (assertions that cannot fail) — the KISS
  Architect flagged it "when I have room"; its own audit, since "an assertion that
  can't fail and an absent assertion are the same defect."
