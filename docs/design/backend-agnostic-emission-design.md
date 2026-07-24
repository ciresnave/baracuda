# Backend-agnostic emission — design + plan (2026-07-10)

Feasibility scout done. This records the load-bearing decisions for making non-CUDA backend emission first-class, starting with a CPU C-emitter. Goal: prove the `KernelPlan` IR is genuinely backend-neutral, give a GPU-free execution path, and add a third independent leg to the correctness triangle (CUDA emitter ↔ CPU-C emitter ↔ Rust oracle).

## Verdict up front

**The expression seam is already the correct, sufficient factoring; the schedule skeletons are legitimately backend-divergent (parallel tree vs serial loop) and should NOT be force-factored.** A `CpuC` backend = reuse `backend.rs` wholesale + a CPU-C `Lowering` + fresh serial-loop skeletons, authored against the oracle's `eval_*`.

## What's already backend-neutral (reuse verbatim)

- **The `Backend` trait** (`backend.rs:113`) requires only `name` / `lower(plan)->GeneratedKernel` / `supports_dtype`; `lower_variants` defaults to empty. `GeneratedKernel = {name, source}` — "source in the backend's language", nothing CUDA-bound. `supports_dtype` lets a CPU backend honestly decline `F16`/`Bf16` in v1.
- **The `Lowering` seam** (`backend.rs:143`) + `lower_expr` (`209`) / `lower_dag` (`241`) / `lower_dag_multi` / `lower_dag_all` / `const_lit` (`188`): the shared driver emits ONLY portable infix (`(a + b)`) + the six seam closures + already-valid-C constants (`NAN`/`INFINITY`/decimal). Zero CUDA in the driver. **A CPU-C backend reuses all of this unchanged.**
- **Speller portability** (the closures cuda.rs injects): `binary_f32/f64` (`powf/atan2f/copysignf/nextafterf/fmaxf/fminf/fmodf` — all C99 `<math.h>`), `binary_int`, `select_f32/f64` — **all portable C as-is**. `unary_f32/f64` is ~all C99 EXCEPT `rsqrt` (CUDA intrinsic → spell `1.0f/sqrtf(x)`). The ONLY genuinely CUDA-only atoms are `rsqrt` (trivial) and the f16/bf16 promote/demote intrinsics (`__half2float` etc.) — and the half codec already exists in pure Rust in `oracle.rs` (`f16_to_f64:323`, `bf16_to_f64:339`, `f32_to_f16_bits:344`, `f32_to_bf16_bits:397`), so v1 can either reuse that or decline the halves.

## What's CUDA-specific (rewrite serial)

cuda.rs has ~226 `blockIdx/threadIdx/gridDim/blockDim/__global__/__shared__/__shfl*/__syncthreads/block_*` occurrences — the parallel harness. Per-Access: (a) expression via `lower_dag` = REUSE; (b) index/offset/axis math (`offset_expr:6016`, unravel `981-985`, `rr_role`) = REUSE (portable content, currently private in cuda.rs); (c) launch/parallel harness = REWRITE serial.

- **Elementwise** (`emit_strided:849`, `emit_scalar:795`): the ENTIRE CUDA-specificity is the 3-line grid-stride prologue (`cuda.rs:976-978`) + the `extern "C" __global__` header. Body + unravel + `offset_expr` are already seam/helper-mediated. **→ trivial first target: a flat `for (i=0;i<n;i++)` + a plain `void` signature.**
- **Reduction** (`emit_reduction:2187`): the *general* path is ALREADY a serial nest (`emit_reduced_nest`, documented to match the oracle order) — strip the grid-stride wrapper → CPU-ready. The *InnerContig* fast path is the parallel block-tree (a CUDA-only default).
- **RowReduce / Scan / Window**: staged-fold / serial-base / per-output-window shapes; re-spell serial. **Contraction / RowSort**: most parallel (register-tiled GEMM, smem bitonic) — a CPU version is a from-scratch naive rewrite; lowest reuse.

## Plan (revised — the skeleton-factor step is DROPPED per the verdict)

1. **CpuC Elementwise emitter** — a new `cpu_c.rs` module: a `CpuC` `Backend` whose `lower` handles `Schedule::Elementwise` (+ Strided) via a CPU-C `Lowering` (reuse `binary_*`/`select_*`/`binary_int`/`const_lit`; twin `unary_*` with `rsqrt→1/sqrt`; decline f16/bf16 in v1) + a serial `for` loop + a plain `void name(const float* in0, ..., float* out, long long n, ...)` signature + `#include <math.h>`. Reuse `lower_dag` + the unravel/`offset_expr` math (promote those to `pub(crate)`). Author it against `oracle.rs`'s `eval_elementwise` (which already encodes the exact serial nest).
2. **Validate**: an emission golden (structure) + a compile-and-run cross-check on this box (cc/cl compiles the emitted C; run it; compare vs `oracle::evaluate`) — a genuine CUDA↔CPU-C↔oracle triangulation, since the oracle shares zero lowering code (`oracle.rs:9-21`, confirmed).
3. **Backend-authoring doc**: grounded in the real CpuC impl — "to add a backend, implement name/lower/supports_dtype, reuse the seam + lower_dag, spell the schedule for your execution model, validate against the oracle." Note explicitly: DO NOT hoist schedule skeletons into a shared trait (forced abstraction); the seam is the factoring.
4. Extend Access coverage (Reduction next — the general serial nest is closest) as consumers warrant.

## Anchors
`backend.rs`: trait `113`, `GeneratedKernel 16`, `Lowering 143`, `lower_expr 209`, `lower_dag 241`, `const_lit 188`. `cuda.rs`: `emit_strided 849`, grid-stride prologue `976-978`, unravel `981-985`, `offset_expr 6016`, `unary_f32 6118`/`f64 6168`, `binary_f32 6246`/`f64 6307`, `select_f32 6486`, `emit_reduction 2187`, `emit_reduced_nest 2644`. `oracle.rs`: `eval 794`, `eval_elementwise 1075`, `eval_reduction 1133`, `read_strided 974`, independence doc `9-21`, half codec `323/339/344/397`. Reference: [[kernelgen-ir-frontier]], [[cuda-box-local-validation]] (compile+run validation is local).

---

# Backend-authoring guide (grounded in CpuC, VALIDATED)

Plan steps 1–2 are **DONE**: `CpuC` (crates/baracuda-kernelgen/src/cpu_c.rs, commit 3bebce5c) is the first non-CUDA backend, and its emitted C was **compiled GPU-free and executed on this box** — `in0*in1+in0` produced `[5,12,21,-32,-3.5]` == the oracle/definition, proving the IR is backend-neutral *by execution*, not just design. This section is step 3: how to add the next backend.

## To add a backend

1. **Implement the three required `Backend` methods** (`lower_variants` defaults empty):
   - `name(&self) -> &str` — a short id.
   - `supports_dtype(&self, dt) -> bool` — the JIT trust boundary's typed-decline gate. Decline honestly (CpuC declines f16/bf16 = no CPU half codec yet, and u32 = index/address-only). AOT `lower` may still panic on a dtype it can't spell; add a `supports_dtype` assert at the top of `lower` as the backstop (the 0a "gate every layer" lesson).
   - `lower(&self, plan) -> GeneratedKernel { name, source }` — dispatch on `plan.schedule`; emit `source` in your backend's language for the schedules you serve, and **panic clearly** on the rest (AOT authoring is trusted; a panic is the honest boundary).
2. **REUSE the expression seam — do not re-implement it.** The body math lowers through the SAME neutral `lower_dag`/`lower_expr` over a `Lowering` whose six closures you provide. Reuse the CUDA spellers verbatim where they're portable (they mostly are — `binary_f32/f64`, `binary_int`, `select_f32/f64`, `const_lit`); write a twin ONLY for the atoms that differ in your target (CpuC's sole new atom is `rsqrt` → `1.0f/sqrtf(x)`). Promote a reused CUDA fn to `pub(crate)` **body-untouched** so every CUDA golden stays byte-identical.
3. **Spell the SCHEDULE for your execution model — this is where backends legitimately diverge.** CUDA's is a parallel grid-stride + block-tree harness; CpuC's is a serial `for (long long i = 0; i < n; ++i)`. **Do NOT hoist schedule skeletons into a shared trait** — that is a forced abstraction (parallel tree vs serial loop have no worth-sharing common shape). The seam is the factoring; the schedule is the divergence. Author each schedule against the oracle's matching `eval_*` (`oracle.rs`), which already encodes the serial loop nest + fold order + index math your emitter must reproduce.
4. **VALIDATE by triangulation.** An emission golden pins the structure. The load-bearing check is compile-and-run vs the oracle: the oracle shares ZERO lowering code (`oracle.rs:9-21`), so agreement of `<your backend's compiled output>` with `oracle::evaluate(&plan, ...)` is a genuine third leg (CUDA emitter ↔ your emitter ↔ oracle). On this box (see [[cuda-box-local-validation]]): `nvcc -ccbin "<VC/Tools/MSVC/<ver>/bin/Hostx64/x64>" -x c emitted.c -o e.exe` compiles portable C via MSVC (cl isn't on PATH; nvcc `-ccbin` at the cl dir sets up the MSVC env).

## Extend CpuC (as consumers warrant)
Reduction next — its *general* path is already a serial nest matching the oracle order, so the CPU version strips the grid-stride wrapper. Then RowReduce (staged serial folds). Contraction/RowSort are from-scratch naive rewrites (lowest reuse) — do them only if a consumer needs GPU-free execution of those. f16/bf16 need a CPU half codec (the oracle's is reusable). Multi-output Elementwise is the N-store follow-up.
