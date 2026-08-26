# CI coverage matrix — what no CI job touches

**Status:** audit (the deliverable is the enumeration, including the rows deliberately left uncovered)
**Measured at:** `b508cad5` (main), 2026-08-20. Every figure here is a claim about that ref — re-derive before citing.
**Method:** surveyed `.github/workflows/` (one file, `ci.yml`), the 70 workspace crates, and each crate's `tests/`. The question asked of every row is **not "does it pass" but "can it fail, and has it ever"** — a row that has never produced a negative is indistinguishable from one switched off, and that is the version counted as coverage.

---

## 1. What CI is

**One workflow (`ci.yml`), one job (`Build without CUDA installed`), matrix `[ubuntu-latest, windows-latest]`.** Toolchain `dtolnay/rust-toolchain@stable`. Steps:

| step | scope |
|------|-------|
| `rustfmt --check` (chunked) | whole workspace `*.rs` minus `tools/` |
| `cargo build/clippy/clippy --tests/test/doc` | `baracuda-cuda-emit` only (default features) |
| `clippy/test --features cuvs` | `baracuda-cuvs-sys`, `baracuda-cuvs` |
| `clippy/clippy --tests/test --features seam` | `baracuda-cuda-emit` |
| `clippy/clippy --tests/test --features convert` | `baracuda-cuda-emit` |

**There is NO CUDA/GPU/self-hosted runner.** `ci.yml` comments said the CUDA crates and cuda-emit's real `nvrtc` feature are "validated on the CUDA runner" — **no such runner existed in the workflow.** That validation happens only on the maintainer's local RTX 4070 box. (Fixed in this PR: the comments now say "validated locally on the 4070 box; there is no CI CUDA runner.")

**This is a distinct, sharper shape than "a check that runs but cannot fail" — name it separately.** A non-signalling check comes in four kinds, worst last:

| shape | what it looks like |
|-------|--------------------|
| ALWAYS-PENDING | never blocks, never passes — visibly stuck |
| ALWAYS-NEUTRAL | completes, gets counted as coverage (e.g. Codeac `neutral` on every commit) |
| ABSENT ENTIRELY | looks like a repo nobody has broken |
| **DOCUMENTED-ABSENT** | **looks like a repo somebody IS checking — and a reader has been told so** |

The first three are silent; **DOCUMENTED-ABSENT actively answers the question a careful reader would ask.** Someone auditing coverage reads the comment, concludes the CUDA path is validated in CI, and stops looking. That is why fixing the comment ranks ABOVE the coverage gap itself: 2-of-70-in-CI is a known, budgetable state — a comment that misrepresents it is what keeps it unknown.

---

## 2. The coverage matrix

Axes: **crate × feature × target-kind × platform × toolchain.**

> **LEGEND — two kinds of UNCOVERED, and stamping them the same hides which is actionable.**
> - **CONSTRAINT ("needs-CUDA-can't"):** the code cannot build without the CUDA toolkit (a `build.rs` that invokes `nvcc`, or `cc`-built CUDA/TensorRT headers). No CI runner has the toolkit, so it is uncovered *by hardware*, not by omission. Not fixable here.
> - **OMISSION ("could-build-doesn't"):** the code builds without the toolkit and simply isn't built by any job. **Fixable** — and dangerous precisely because a matrix that stamps it identically to a CONSTRAINT tells a reader it has the same, unfixable status.
> The original version of this row (below, struck) conflated them; the correction is the substance of the feature-coverage PR.

### Crate (70 total) — CORRECTED
- **COVERED without the toolkit — far more than first stated.** Only **7 crates actually invoke `nvcc`** (`baracuda-kernels-sys`, `baracuda-cutlass-sys`, `baracuda-cutlass-kernels-sys`, `baracuda-ozimmu-sys`, `baracuda-transformer-engine-sys`, `baracuda-kernels-bench`, and `baracuda-optim` *only with an `sm8x/9x` arch feature*), plus `baracuda-tensorrt-sys` *only with its `cc`-built `shim`*. **CONSTRAINT** = those. **Every other `-sys`/wrapper crate is lazy-libloading** (its `build.rs` emits only rerun hints; symbols resolve at runtime), so it builds CUDA-free — it was **OMISSION**, not constraint.
- ~~UNCOVERED (68): every `-sys` binding + wrapper crate … compile `.cu` via `nvcc`, so they cannot build on a no-CUDA runner.~~ **This was wrong** — it attributed CONSTRAINT to ~61 crates that are OMISSION. Corrected by measuring the build scripts (nvcc/`cc`/`links` greps), not assuming the stack needs the toolkit.

### Feature — CORRECTED (this row under-counted the dimension it named)
- **COVERED (before this PR):** default, `seam`, `convert`, `cuvs`.
- **OMISSION, now COVERED by this PR:** the CUDA-free feature-gated code that no job built — `baracuda-types` (`half-crate`/`f8-crate`/`num-complex-crate`/`derive`), `half-crate` on `cudnn`/`nccl`/`megatron`/`cuvs`, the `baracuda` umbrella's 20 `pub use` gates, `baracuda-driver/test-support`, `baracuda-runtime/driver-interop`, `baracuda-cuda-emit/nvrtc` (clippy/build; its *tests* need a GPU), `baracuda-optim/distributed_optim`. Now gated by a `clippy --all-features` step over the CUDA-free surface + the `feature-coverage` guard.
- **CONSTRAINT (still uncovered by design):** every feature routing through the 7 nvcc crates (kernels' `fa2`/`flashinfer`/`marlin`/… , cutlass' `sm8x`/`ozimmu`, optim's `sm8x/9x`) and `tensorrt-sys/shim`. Recorded in the guard's `CUDA_ONLY`/`CUDA_FEATURES` lists, not swept.
- **The old row said only `nvrtc` + `--all-features` were uncovered** — it named the Feature axis and then enumerated only the CUDA half of it, which is more dangerous than an uncovered axis because the `--features` steps present made the dimension *look* handled.

### Target-kind (for the covered crates)
- **COVERED:** `lib`, `--tests` (added in #21 after a lib inline-test lint rode in unnoticed), integration `tests/*.rs`.
- **PARTIAL:** `cargo doc` runs for **`baracuda-cuda-emit` only** (`-p baracuda-cuda-emit --no-deps`). The other 69 crates' rustdoc — including intra-doc links — is **never built by CI.** (See §3.1.)

### Platform
- **COVERED:** `ubuntu-latest`, `windows-latest` (no-CUDA).
- **UNCOVERED:** any CUDA/GPU platform (no runner).

### Toolchain — **UNPINNED AXIS**
`rust-toolchain.toml` = `channel = "stable"` — a **moving alias, not a pin.** This box resolves it to **1.97.1** (2026-07-14); CI's `@stable` resolves **1.98.0** (2026-08-18). So local `cargo clippy`/`build` checks a **different diagnostic set** than the gate: a lint that postdates 1.97.1 reds CI on a locally-green tree. Exposure is **lint/warning reproducibility only, not numerics** — and the reason is SCOPE, not a Rust guarantee. Rust/LLVM do NOT promise bit-for-bit FP reproducibility across compiler versions in general (codegen and FP contraction can change results). Here the numerics are insulated from rustc on both sides: (a) Baracuda's KERNELS are CUDA compiled by **nvcc** — rustc never touches them, so a rustc bump cannot change what a kernel computes; (b) the host-side reference values the tests compare against are **exact IEEE-754 ops only** (relu / add / max / integer — no transcendentals compared bit-exact), and Rust does not fast-math or FP-contract these by default, so they are bit-stable across the 1.97.1↔1.98.0 versions in play. So the skew is a diagnostic-set difference, not a numeric one — scoped, not asserted as a language guarantee. **Recorded, not fixed** — a portfolio-wide pin decision is with the maintainer; fixing it here alone is the one-decision-three-ways trap this audit exists to prevent.

---

## 3. Vacuous-green shapes (measured)

Each is a place where a check completes without being able to signal — "a surface or context no CI job exercises."

### 3.1 Broken intra-doc links — 69 crates ungated
CI runs `cargo doc` for `baracuda-cuda-emit` only. Every other crate's rustdoc, including `[Type]` intra-doc links that render as nothing on docs.rs when the target is private or renamed, is never built by CI. This is the shape Unpopped found across their own published crates. **`baracuda-kernels-types` is `pub use unpopped_vocab::*`, so its re-exported docs inherit unpopped's repairs on the next bump — but the other 68 are on their own.**
- **Fix:** add `cargo doc --workspace --no-deps` (or a per-crate loop for the no-CUDA subset) + a `[workspace.lints.rustdoc]` deny on `broken_intra_doc_links` + `private_intra_doc_links`. Born-red control: a deliberate `[NonexistentType]` proving it fires.

### 3.2 Tests in the wrong crate — 3 files
A test in `crates/A/tests/` runs under `cargo test -p A`, never under `-p B`, regardless of whose code it exercises. A split moves code and leaves tests where they were. Measured (comment-stripped, per the caveat that a crate name mentioned only in a `//` comment is not usage):
- `baracuda-cuda-emit/tests/backend_declines.rs` — tests `CpuC`/`Slang` declines via `try_generate`; uses **no** `baracuda_cuda_emit` export. It sits in cuda-emit's tests but exercises `unpopped-cpu-c`/`unpopped-slang`/`unpopped`.
- `baracuda-runtime/tests/external_smoke.rs`
- `baracuda-types-derive/tests/derive_device_repr.rs` (proc-macro tests conventionally use the consumer crate — likely legitimate; classify before acting).
- **NOTE ON A DISCREPANCY:** a relayed figure put this at "baracuda 64." My primary-source measurement at `b508cad5` is **3** (comment-stripped). The two are different metrics; cite the measured 3, not the relay.
- **Fix (guard):** walk `crates/*/tests/*.rs`, strip `//` comments, fail any file that never names its own crate — with two honesty checks: assert it scanned > 0 files (an empty walk ≠ clean), and assert the detector still fires on a legitimately-exempt file (the exemption can't go vacuous).

### 3.3 Cross-crate `#[should_panic]` pins
`baracuda-cuda-emit`'s test module holds `#[should_panic]` tests that pin panics in **unpopped's** `plan.rs` (e.g. `rowreduce_forward_reduced_ref_panics`). These are green while the upstream defect is open and **red the moment it is fixed** — a `#[should_panic]` is the one artifact whose *passing* documents a defect, and every mechanism this project has for surfacing problems keys on red. Not a gap to close; a polarity to be aware of when the upstream fix lands (the pin flips to a failure and someone deletes it deliberately). Tracked with unpopped 0.6.0.

---

## 4. Instrument liveness — can each row fail, has it ever

| row | can fail? | has it? |
|-----|-----------|---------|
| rustfmt `--check` | yes | yes (historically red on style-edition-2024 backlog) |
| clippy default / seam / convert (lib) | yes | yes |
| clippy `--tests` (default/seam/convert) | yes | **yes — caught the `single_element_loop` at cuda.rs:10702 (#21)** |
| `test --features convert` | yes | **yes — caught `tests/convert.rs` not compiling against the 0.3.0 API (#24 predecessor)** |
| `cargo doc -p cuda-emit` | yes | not observed to fail |
| Codeac | yes | measured green on 8 recent commits — **one of the three portfolio Codeac instances that still signals** (some others conclude `neutral` on every commit) |
| the "CUDA runner" rows | **no — the runner does not exist** | never runs |

The last row is the point: a row that can never fail is not coverage. The CUDA surface's real liveness is the local 4070, and that is a maintainer-run gate, not a CI one.

---

## 5. Recommendations — gate, or record as uncovered

1. **Doc-links (§3.1): GATE.** `cargo doc --workspace --no-deps` on the no-CUDA subset + the rustdoc-deny lint. Small, born-red-able.
2. **Test-in-wrong-crate (§3.2): GATE.** The walk-guard with the two honesty checks. Reclassify `backend_declines.rs` (move to `unpopped-conformance`, or keep with an explicit "consumer-side contract" justification — it is a judgment, not a mechanical move).
3. **Toolchain axis (§2): RECORD, do not fix.** Pending the portfolio pin ruling.
4. **CUDA surface (68 crates, nvrtc, on-device): RECORD as intentionally CI-uncovered** — validated on the local 4070 by design (no-CUDA runners cannot build it). **TOP-PRIORITY sub-item, APPLIED IN THIS PR:** fixed the `ci.yml` DOCUMENTED-ABSENT comment so the coverage claim matches reality. This ranks above the coverage gap: the gap is a known, budgetable state; the false comment is what kept it unknown.
5. **Cross-crate `#[should_panic]` (§3.3): LEAVE, note the polarity.** Flips red when unpopped 0.6.0 fixes the pinned defect; that is the design.

**What this audit does NOT do:** it does not run `cargo doc --workspace` (needs the no-CUDA subset resolved) or build the guards. It enumerates. The guards are follow-up work; naming the uncovered rows is the output, so the next gap is read rather than rediscovered.
