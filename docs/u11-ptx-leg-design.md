# Design note: the PTX leg of Unpopped's typed `Extern` (ROADMAP U11 step 3)

Status: **proposal, nothing built.** Task from the PM (2026-11-04): consume Unpopped's `ExternDecl` in `baracuda-cuda-emit` and
compare one real single-instruction PTX asm body with its reference on the device runner. Facts below were read at Unpopped
`origin/main` 34de309 (0.43.2) and baracuda `origin/main` 45ed52b9; crates.io was read 2026-11-04 and again 2026-11-05 (after the PM published
Unpopped 0.43.2).

## What exists

- **Unpopped has steps 1 and 2** (the options note, `docs/options-u11-extern-and-inline-asm.md`, still says "nothing built"; it is stale on this).
  0.43.0: `ExternDecl { name, params, results, effects, bodies, reference }`, `Effects { pure, reads, writes, sync, volatile }`,
  `TargetBody { target, body }`, `BodyText::Asm { template, constraints }` / `Symbol { library, symbol }`, `Module.externs`, and
  `Stmt::ExternCall { decl, args, results }`, with a checker (`kernel_extern.rs`). 0.43.1: the oracle and cpu-c run an `ExternCall`
  through its `reference`; a declaration with no reference traps. Asm placeholders are `{N}`, parameters first, then results.
- **`baracuda-cuda-emit` has no kernel-layer emitter.** It implements the plan-level `Backend` (`Cuda`: elementwise / contract plans to `.cu`,
  NVRTC compile via `NvrtcCompiler`); it never touches `Module`, `KernelFn` or `Stmt` (0 matches for `Stmt::`, `Module`, `KernelFn`
  in `crates/baracuda-cuda-emit/src` at 45ed52b9; the same `git grep` finds `impl Backend for Cuda`). Only Unpopped's cpu-c has `lower_module`.
- **The device-test pattern exists**: `tests/typed_exprs.rs` generates with `Cuda`, compiles with NVRTC, runs on the device and compares
  with Unpopped's oracle, under `--features nvrtc,device-tests`, run by the self-hosted workflow (`.github/workflows/device-tests.yml`,
  `runs-on: [self-hosted, windows, gpu, baracuda]`).

## Blockers that are not design questions

1. **Unpopped 0.43.2 is published (2026-11-05, read back on crates.io for `unpopped`, `-vocab`, `-cpu-c`); baracuda is still on 0.40.0.**
   baracuda may not depend on a path or git version (CireSnave 2026-10-07), so the whole family moves together: `unpopped`, `-vocab`,
   `-cpu-c`, and `baracuda-cuda-emit` / `-parse` 0.40.0 -> 0.43.0 (MAJOR.MINOR follow Unpopped). That move is its own PR, and it is **not** the
   first PR: it follows blocker 2.
2. **A version cycle with Fuel, between the family move and the seam crates.** `baracuda-cuda-seam` depends on `fuel-kernel-seam` (newest
   0.36.7, requires `unpopped-vocab ^0.40.0`), so a 0.43 move leaves two copies of `unpopped-vocab` in its graph; the `vulkane-slang-emit` /
   `-parse` 0.20.2 dev-dependencies also require `^0.40.0`. Fuel's own `fuel-dispatch` jit uses baracuda's seam crates (alpha.80, vocab
   `^0.40.0`), so Fuel on 0.43 gets two copies until baracuda ships 0.43. The publish order is being agreed between the Fuel and baracuda lanes
   and the PM; this note does not fix it.
3. **Existing emit paths.** Unpopped 0.43.2 made `ReduceOp` `#[non_exhaustive]`: `baracuda-cuda-emit` fails with 13 E0004 in `src/cuda.rs`.
   The PM approved a wildcard arm at each site that takes the existing decline path (a typed error, never a panic), with no emitted-output change.
4. **The seam hold** (CireSnave 2026-10-28, "no new CUDA parser/emitter FEATURES until the seam settles"). PM ruling 2026-11-05: the hold is a pin on
   Fuel's `baracuda-cuda-emit` dependency; a new default-off feature that adds `extern_asm_cuda` and changes no existing emitted output or
   Fuel-facing surface is not a seam change. If the build touches existing emit paths beyond item 3, stop and ask.

## Proposal (narrow): spell the asm body, not the module

Do **not** build a CUDA `lower_module` (a whole second kernel-layer emitter; Q2). Add one function and one test.

1. `baracuda_cuda_emit::extern_asm_cuda(decl: &ExternDecl, args: &[String], results: &[String]) -> Result<String, ExternError>`:
   finds the body with `target == "cuda-ptx"` and returns one CUDA statement, `asm volatile("...");` (`volatile` as declared;
   a `pure` decl uses plain `asm`). Refuses with a typed error naming the declaration when there is no `cuda-ptx` body, the body is a
   `Symbol` (v1: asm only), or the constraint count is not `params + results`. It never falls back to the reference silently
   (the Unpopped options note requires this).
2. **Operand numbering differs and is the main thing to test.** Unpopped numbers parameters first, then results (`{0}` is param 0).
   CUDA inline asm numbers outputs first (`%0` is the first output). So parameter `{i}` becomes `%(results + i)` and result `{params + j}`
   becomes `%j`; the constraint string (`"=r,r"`) splits at commas, `=`-prefixed entries become the output list. Example, `popc.b32`:
   params `[U32]`, results `[U32]`, template `popc.b32 {1}, {0};`, constraints `=r,r` -> `asm volatile("popc.b32 %0, %1;" : "=r"(r) : "r"(x));`.
3. **The device leg** (`tests/extern_ptx.rs`, `nvrtc,device-tests`): build an Unpopped `Module` with one `ExternDecl` (a `cuda-ptx` body and a
   `reference` kernel) and a kernel that applies it per element; run the module through Unpopped's oracle (`run_module`, which runs the
   reference); build the `.cu` for the same kernel by hand around `extern_asm_cuda`'s one statement; compile with NVRTC; run on the
   device; compare byte for byte. Two cases, both exact (no tolerance): `fma.rn.f32` (reference: the kernel layer's fused multiply-add if
   `ScalarExpr` has one; a mul-then-add reference rounds twice and would not match, so without one this case is dropped) and `popc.b32` (reference: a 32-step shift-and-add loop; the IR has no
   popcount, 0 matches for `popc` in `ir.rs` / `kernel.rs`). Inputs include the edge values (NaN payloads, +-0, subnormals, 0, `u32::MAX`).
4. **Mutants** (each must make the test fail): swap `%0`/`%1`; drop `volatile` on a volatile decl; use `fma.rz` for `fma.rn`; a reference off by one bit.
5. Until the runner has run it, the doc string of `extern_asm_cuda` says in one line that the asm body is unexercised.

What this proves: the spelling of operands and the one instruction agree with the reference on one device. What it does not prove:
asm in general, other targets, memory-touching asm (v1 has no pointer operands), or a module-level emission path.

## Decisions for the PM

| # | Question | Recommendation |
|---|---|---|
| Q1 | Is "spell one asm body" exempt from the 2026-10-28 emitter-features hold? | **Answered 2026-11-05 (PM): yes**, as a new default-off feature with no change to existing emit output or Fuel-facing surface. |
| Q2 | Narrow `extern_asm_cuda` or a full CUDA `lower_module`? | Narrow. `lower_module` is the seam's job, and the hold covers it more clearly. |
| Q3 | Who publishes Unpopped 0.43.x, and when? | **Answered 2026-11-05 (PM): the PM; published 0.43.2.** The family move is not the first PR (blocker 2). |
| Q4 | Cases | `fma.rn.f32` + `popc.b32`, both exact. |
| Q5 | Version | **Answered (PM): alpha.92** for this docs PR. |

## Build order, once approved and unblocked

1. Fuel's seam crates and Vulkane's dev-dependencies on `unpopped-vocab` 0.43.2, in the order the Fuel / baracuda / PM plan sets; then the
baracuda family move 0.40.0 -> 0.43.x with the 13 `ReduceOp` wildcard arms (own PR, no emitted-output change). 2. `extern_asm_cuda` + unit tests (operand renumbering, each refusal, mutants) with no GPU.
3. `tests/extern_ptx.rs` on the device under `scripts/gpu-run.ps1`, then on the self-hosted runner.
